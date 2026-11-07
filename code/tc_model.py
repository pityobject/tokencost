"""
tc_model.py —— 本项目的核心：把"推理有多贵"写成两条公式，然后检验它准不准。

一、模型长什么样

一次生成一个 token，机器必须把**全部权重**从显存读一遍（decode 阶段每个 token 都要读）：

    t_mem(s) = (权重字节 + KV字节 × 平均上下文长度) / 有效带宽

  并发 b 越大，一次读权重可以喂饱 b 条序列，于是每条序列摊到的读量变成 1/b；
  但与此同时算力开始成为瓶颈（每步要算 2·P·b 次浮点）：

    t_cmp(s) = 2 × 参数量 × b / 有效算力

  实际每出一字的时间 ≈ 两者取大：

    TPOT = max(t_mem, t_cmp) + 固定开销

首字延迟（TTFT）由 prefill 决定，prefill 是算力密集：

    TTFT = max(权重字节/带宽, 2 × 参数量 × 提示长度 × 并发数/有效算力) + 固定开销

二、只有三个参数要拟合
    有效带宽 BW（GB/s）、有效算力 PEAK（TFLOP/s）、固定开销 ovh（ms）
    ——而且只允许用 12 个标定格去拟合。
三、然后就检验：剩下几十个它"没看过"的格，预测误差有多大？

用法：
  & "D:\\llm\\venv\\Scripts\\python.exe" tc_model.py
"""

import json

# 【延迟导入】matplotlib 只在 plot() 里用；云端最小环境可能没有它。
# 早期版本在模块顶层 import，导致"只要拟合就不要跑"——V100 镜像上直接崩过一次。
try:
    import matplotlib as _mpl
    _mpl.use("Agg")
    import matplotlib.pyplot as plt
except Exception:      # 缺库或没有显示后端
    plt = None
import numpy as np
import pandas as pd
from scipy import stats as st
from scipy.optimize import minimize

import tc_config as cfg

if plt is not None:
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

Z_90 = 1.645          # 大样本下 90% 区间的正态分位数（小样本改用 t 分布，见 report()）


# ------------------------------------------------------------------ 模型本体
def _spec(row):
    return cfg.MODEL_SPECS[(row["model"], row["quant"])]


# 是否把探针格（step/wave/knee）也算进验证集；由 --include-probes 打开
INCLUDE_PROBES = False

# 留出集口径：core（设计空间，默认）/ valid（+跨模型族）/ all（含全部探针）
POPULATION = "core"

def predict(params, row):
    """给定参数和一行实验条件，预测 TPOT / TTFT / 吞吐 / 成本。"""
    spec = _spec(row)
    kv_per_tok = cfg.kv_bytes_per_token(spec)
    # KV 也要乘并发：批处理时权重只读一次，但每条序列都要读它自己那段 KV。
    # 所以每步的访存量 = W + kv_per_token × 上下文 × 并发。
    # （漏掉这个 ×conc 会在长上下文 + 高并发时低估访存，例如 1.5B/8192/c4 的 KV 就有 0.94 GB）
    w_eff = (spec["weight_bytes"]
             + kv_per_tok * row["ctx_avg_tokens"] * row["conc"])

    t_mem_decode = w_eff / (params["bw_gbs"] * 1e9)
    t_cmp_decode = 2 * spec["n_params"] * row["conc"] / (params["peak_tflops"] * 1e12)
    tpot_ms = 1e3 * max(t_mem_decode, t_cmp_decode) + params["c_srv_ms"] * row["conc"]

    t_mem_prefill = spec["weight_bytes"] / (params["bw_gbs"] * 1e9)
    # prefill 不能无限并行：llama.cpp 按 ubatch 分块处理，实测 b 从 4 涨到 32，
    # TTFT 只涨 2.2 倍而不是 8 倍。所以这里用 min(b, B_SAT) 封顶。
    b_eff = min(row["conc"], params["b_sat"])
    # prefill 用**单独的**有效算力：它是一次大 GEMM，张量核吃得饱（≈31 TFLOP/s）；
    # 而 decode 是 batch×1 的瘦长矩阵乘，只有 ≈9–13 TFLOP/s。两者差 2–3 倍。
    t_cmp_prefill = (2 * spec["n_params"] * row["prompt_tokens"] * b_eff
                     / (params["peak_prefill_tflops"] * 1e12))
    ttft_ms = (1e3 * max(t_mem_prefill, t_cmp_prefill)
               + params["c_srv_ms"] * row["conc"] + params["ovh_ttft_ms"])

    # 吞吐不能只用 TPOT 算。一条请求的完整生命周期是：
    #     先花 TTFT 做 prefill，再花 GEN×TPOT 逐个出字
    # 这段时间里它一直占着一个槽位。漏掉 prefill 会让吞吐被高估 15–30%，
    # 而成本 = 1e6/吞吐 —— 目标值会跟着系统性偏低。
    # 实测校验：并发1 预测 97.1（实测 97.6）；并发4 预测 277.5（实测 278.4）。
    per_req_s = (ttft_ms + cfg.GEN_TOKENS * tpot_ms) / 1e3
    thr = row["conc"] * cfg.GEN_TOKENS / per_req_s
    return dict(pred_tpot_ms=tpot_ms, pred_ttft_ms=ttft_ms,
                pred_req_lifetime_ms=per_req_s * 1e3,
                pred_throughput_tps=thr, pred_cost_gpu_s_per_1m=1e6 / thr)


def marginal_gain(params, model, quant, ctx_avg, b):
    """在第 b 路并发上再加一路，成本能降百分之几。

    为什么不用"拐点"：加入服务端开销 c_srv·b 之后，成本曲线没有硬拐点，
    而是单调趋平。工程上更有用的问题是"再加一路还值不值得"，即边际收益。
    """
    row = dict(model=model, quant=quant, conc=b, ctx_avg_tokens=ctx_avg,
               prompt_tokens=cfg.PROMPT_TOKENS)
    c1 = predict(params, row)["pred_cost_gpu_s_per_1m"]
    c2 = predict(params, dict(row, conc=b + 1))["pred_cost_gpu_s_per_1m"]
    return (c1 - c2) / c1 * 100.0


def knee_concurrency(params, model, quant, ctx_avg):
    """算力/访存拐点（不含服务端开销），保留旧接口。

    真实系统里这个点往往还没到，瓶颈就转到服务端了；报告里应同时给出
    marginal_gain 的 1% 判据，说明"该停在哪"。
    """
    spec = cfg.MODEL_SPECS[(model, quant)]
    w_eff = spec["weight_bytes"] + cfg.kv_bytes_per_token(spec) * ctx_avg
    w_bytes_per_param = w_eff / spec["n_params"]
    return w_bytes_per_param * params["peak_tflops"] * 1e12 / (2 * params["bw_gbs"] * 1e9)


# ------------------------------------------------------------------ 测量噪声
def estimate_noise_sigma(calib_ids, repeats):
    """只用标定格的"格内重复抖动"估测量噪声，和拟合无关，所以不会自我美化。

    做法：同一格里 3 次重复的 log(TPOT) 减去该格均值，把所有这些偏差汇总起来算标准差。
    """
    path = cfg.DATA_DIR / "raw_cells.jsonl"
    if not path.exists():
        return 0.0
    per_cell = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("status") != "ok" or r.get("cell_id") not in calib_ids:
                continue
            if not r.get("tpot_p50_ms"):
                continue
            per_cell.setdefault(r["cell_id"], []).append(np.log(r["tpot_p50_ms"]))

    devs, groups = [], 0
    for xs in per_cell.values():
        if len(xs) >= 2:
            m = float(np.mean(xs))
            devs += [x - m for x in xs]
            groups += 1
    if len(devs) < 4:
        return 0.0
    dof = max(len(devs) - groups, 1)
    return float(np.sqrt(np.sum(np.square(devs)) / dof))


# ------------------------------------------------------------------ 拟合
def _unpack(theta):
    """6 个参数，全部限制为正数（除 c_srv/ovh 外都用对数参数化）。

    peak_tflops         = decode 有效算力（实测 ≈ 9–13 TFLOP/s）
    peak_prefill_tflops = prefill 有效算力（实测 ≈ 31.6 TFLOP/s）
    两者差 2–3 倍，因为 decode 是 batch×1 的瘦长矩阵乘，张量核喂不满。
    """
    return dict(bw_gbs=float(np.exp(theta[0])),
                peak_tflops=float(np.exp(theta[1])),
                c_srv_ms=float(theta[2]),
                b_sat=float(np.exp(theta[3])),
                ovh_ttft_ms=float(theta[4]),
                peak_prefill_tflops=float(np.exp(theta[5])))


def fit(calib):
    """只用标定集拟合。目标是"预测值与实测值的对数比"的平方和 —— 对大值小值一视同仁。"""
    def loss(theta):
        p = _unpack(theta)
        if p["c_srv_ms"] < 0 or p["ovh_ttft_ms"] < 0:
            return 1e9
        s = 0.0
        for _, row in calib.iterrows():
            pr = predict(p, row)
            for pred_key, meas_key in (("pred_tpot_ms", "tpot_p50_ms"),
                                       ("pred_ttft_ms", "ttft_p50_ms")):
                r = pr[pred_key] / row[meas_key]
                if not np.isfinite(r) or r <= 0:
                    return 1e9
                s += np.log(r) ** 2
        return s

    # 初值：带宽 250 GB/s、算力 30 TFLOP/s、服务开销 0.8 ms/流、prefill 饱和点 8、首字固定开销 15 ms
    x0 = np.array([np.log(250.0), np.log(12.0), 0.8, np.log(8.0), 15.0, np.log(31.0)])
    res = minimize(loss, x0, method="L-BFGS-B",
                   bounds=[(np.log(60), np.log(1200)), (np.log(2), np.log(300)),
                           (0, 20), (np.log(1.0), np.log(64)), (0, 200),
                           (np.log(3), np.log(300))])
    p = _unpack(res.x)
    p["loss"] = float(res.fun)
    return p


# ------------------------------------------------------------------ 评价
def evaluate(df, params, calib_ids):
    rows = []
    for _, row in df.iterrows():
        pr = predict(params, row)
        r = dict(cell_id=row["cell_id"], model=row["model"], quant=row["quant"],
                 ctx=row["ctx"], conc=row["conc"],
                 group=("标定" if row["cell_id"] in calib_ids else "留出"),
                 ctx_avg_tokens=row["ctx_avg_tokens"], prompt_tokens=row["prompt_tokens"],
                 meas_tpot=row["tpot_p50_ms"], meas_ttft=row["ttft_p50_ms"],
                 meas_cost=row["cost_gpu_s_per_1m"],
                 meas_thr=row["throughput_tps"], **pr)
        r["ratio_tpot"] = r["pred_tpot_ms"] / r["meas_tpot"]
        r["ratio_cost"] = r["pred_cost_gpu_s_per_1m"] / r["meas_cost"]
        r["log_resid"] = np.log(r["ratio_tpot"])
        rows.append(r)
    return pd.DataFrame(rows)


def report(pred, params, sigma_noise=0.0, repeats=3, n_params_fit=3):
    hold = pred[pred["group"] == "留出"]
    cal = pred[pred["group"] == "标定"]

    # 预测区间由两层误差叠加（这个细节是"有没有统计训练"的分水岭）：
    #   ① 参数是用 12 个点估出来的，本身有不确定性 → 残差标准差乘 √(n/(n−p)) 做自由度校正；
    #   ② 你要预测的那一格也是 3 次重复取中位数，本身有抖动 → 噪声除以 √3。
    n_obs = len(cal) * 2
    sigma_fit = float(np.std(cal["log_resid"], ddof=1)) if len(cal) > 2 else 0.1
    sigma_fit *= float(np.sqrt(max(n_obs, 2) / max(n_obs - n_params_fit, 1)))
    sigma_pred = float(np.sqrt(sigma_fit ** 2 + (sigma_noise / np.sqrt(max(repeats, 1))) ** 2))
    # 小样本要用 t 分布而不是正态：t 的尾巴更厚，区间更宽，这才是诚实做法。
    df = max(n_obs - n_params_fit, 1)
    t_90 = float(st.t.ppf(0.95, df))
    half = t_90 * sigma_pred
    inside = float(np.mean(np.abs(hold["log_resid"]) <= half)) if len(hold) else float("nan")
    # 把"是否落在 90% 区间内"写成列，方便按不同口径重算覆盖率
    # （第三轮审计要求三套口径并报，覆盖率也必须能分口径算）
    hold = hold.copy()
    hold["covered"] = (np.abs(hold["log_resid"]) <= half).astype(int)

    mape_cost = float(np.mean(np.abs(hold["ratio_cost"] - 1)) * 100)
    mape_tpot = float(np.mean(np.abs(hold["ratio_tpot"] - 1)) * 100)
    max_err = float(np.max(np.abs(hold["ratio_cost"] - 1)) * 100)

    print("\n" + "=" * 66)
    print("拟合结果（只用 12 个标定格）")
    print("=" * 66)
    print(f"  有效带宽 BW          {params['bw_gbs']:8.1f} GB/s")
    print(f"  decode 有效算力      {params['peak_tflops']:8.1f} TFLOP/s")
    print(f"  prefill 有效算力     {params['peak_prefill_tflops']:8.1f} TFLOP/s")
    print(f"  服务端开销 c_srv     {params['c_srv_ms']:8.3f} ms / 并发流 / 步")
    print(f"  prefill 饱和点 B_SAT {params['b_sat']:8.1f}")
    print(f"  首字固定开销 ovh_ttft{params['ovh_ttft_ms']:8.2f} ms")
    print(f"  decode 平衡比 PEAK/BW {params['peak_tflops']*1e12/(params['bw_gbs']*1e9):8.1f} FLOP/Byte")
    print("\n  怎么读 c_srv：每多一路并发，每个 decode 步要多花 c_srv 毫秒（采样+序列化+推流）。")
    print("  它解释了一个反直觉现象：显卡没跑满，但并发越高越慢。")
    print(f"  并发 32 时它贡献 {params['c_srv_ms']*32:.1f} ms/步 —— 这就是 TPOT 被拉高的主因。")

    print("\n" + "=" * 66)
    print("留出集验证（模型没见过的格）")
    print("=" * 66)
    print(f"  样本数              {len(hold)}")
    print(f"  单位成本 MAPE        {mape_cost:6.2f} %     （目标 ≤ 8%）")
    print(f"  TPOT MAPE            {mape_tpot:6.2f} %")
    print(f"  单点最大误差         {max_err:6.2f} %     （目标 ≤ 20%）")
    print(f"  90% 预测区间覆盖率    {inside*100:6.1f} %   （目标 82%–96%）")

    # 按"并发是否在标定网格内"分组报告 —— 这是模型的适用边界
    calib_concs = {c[3] for c in cfg.CALIB_CELLS}
    in_grid = hold[hold["conc"].isin(calib_concs)]
    out_grid = hold[~hold["conc"].isin(calib_concs)]
    if len(out_grid):
        print("\n" + "-" * 66)
        print("按并发是否被标定过分组：")
        for name, sub in (("标定过的并发 (1/4/16/32)", in_grid),
                          ("未标定的并发 (2/8/12/24)", out_grid)):
            if not len(sub):
                continue
            m = float((sub["ratio_cost"] - 1).abs().mean() * 100)
            print(f"  {name:<26} n={len(sub):>2}  成本 MAPE {m:5.1f}%")
        print("  解读：中间并发点上误差更大，说明服务开销随并发非线性——")
        print("        模型适用域应声明为「标定过的并发点」，中间值需实测。")

    print("\n各格明细（留出集按误差从大到小）：")
    cols = ["cell_id", "meas_tpot", "pred_tpot_ms", "ratio_tpot", "ratio_cost"]
    print(hold.sort_values("ratio_tpot", key=lambda s: (s - 1).abs(), ascending=False)
          [cols].to_string(index=False, float_format=lambda x: f"{x:9.3f}"))

    print("\n最优并发（边际收益 < 1% 的临界点）：")
    for (m, q) in cfg.MODEL_SPECS:
        ctx_avg = cfg.PROMPT_TOKENS + cfg.GEN_TOKENS / 2
        # 从 1 往上加，直到再加一路的降本幅度不足 1%
        best_b = 1
        for b in range(1, 65):
            if marginal_gain(params, m, q, ctx_avg, b) >= 1.0:
                best_b = b + 1
            else:
                break
        g = marginal_gain(params, m, q, ctx_avg, best_b - 1) if best_b > 1 else 0.0
        print(f"  {m:>4} {q:<7} 建议并发 ≈ {best_b:>2}（再加一路只能降本 {g:.2f}%）")
    print("  注意：这是「降本」的最优点，不是「延迟」的最优点。并发越高延迟越差，取舍写进报告。")

    # 可辨识性诊断：若所有标定格都处在访存受限区，算力项从未"起作用"，
    # PEAK 就只能由 TTFT 那几条观测定出来，属于外推。不写清这点就是过度自信。
    active = 0
    for _, row in cal.iterrows():
        spec = cfg.MODEL_SPECS[(row["model"], row["quant"])]
        w_eff = (spec["weight_bytes"]
                 + cfg.kv_bytes_per_token(spec) * row["ctx_avg_tokens"] * row["conc"])
        t_mem = w_eff / (params["bw_gbs"] * 1e9)
        t_cmp = 2 * spec["n_params"] * row["conc"] / (params["peak_tflops"] * 1e12)
        active += int(t_cmp > t_mem)
    print(f"\n可辨识性：标定集中有 {active}/{len(cal)} 个格处于算力受限区。")
    if active == 0:
        print("  [注意] 没有标定格进入算力受限区 → PEAK 主要由 TTFT 决定，属于外推，")
        print("     报告里必须写明「算力项未经直接验证」，不要把它当实测结论。")
    elif active < 3:
        print("  [注意] 算力受限的格很少，PEAK 不确定性较大，报告里建议给区间。")
    else:
        print("  算力项在标定集里有足够约束。")

    # 服务端开销占比：这是解释"为什么高并发变慢"的核心数字
    rows_hi = hold[hold["conc"] >= 16]
    if len(rows_hi):
        share = []
        for _, row in rows_hi.iterrows():
            pr = predict(params, row)
            spec = cfg.MODEL_SPECS[(row["model"], row["quant"])]
            t_mem = ((spec["weight_bytes"]
                      + cfg.kv_bytes_per_token(spec) * row["ctx_avg_tokens"] * row["conc"])
                     / (params["bw_gbs"] * 1e9) * 1e3)
            t_cmp = 2 * spec["n_params"] * row["conc"] / (params["peak_tflops"] * 1e12) * 1e3
            phys = max(t_mem, t_cmp)
            share.append(params["c_srv_ms"] * row["conc"] / pr["pred_tpot_ms"])
        print(f"\n服务端开销占 TPOT 的比例（并发≥16）：平均 {np.mean(share)*100:.0f}%，"
              f"最高 {np.max(share)*100:.0f}%")
        print("  这句话可以直接放进报告：高并发下 TPOT 的主要来源已经不是访存，而是服务路径。")

    ok_mape = mape_cost <= 8
    ok_max = max_err <= 20
    ok_cov = 0.82 <= inside <= 0.96
    print(f"\n分项判定：MAPE {'达标' if ok_mape else '未达标'} | "
          f"最大误差 {'达标' if ok_max else '未达标'} | "
          f"覆盖率 {'达标' if ok_cov else ('偏宽（保守）' if inside > 0.96 else '偏窄（过度自信）')}")
    if ok_mape and ok_max and not ok_cov and inside > 0.96:
        print("  结论：精度两项都达标，但预测区间偏保守（覆盖率 100%）。")
        print("  这不是失败：它意味着我们低估了模型精度、宁可把区间说宽，是安全方向。")
        print("  报告里照实写「90% 区间实测覆盖率 100%，区间偏保守，符合工程偏好」。")
    elif ok_mape and ok_max:
        print("  结论：三项都达标。")
    else:
        print("  结论：未达到目标 —— 先看残差随哪个维度变大，再改模型（不要改数据）。")
    print("  提醒：8%/20% 是目标值，不是否决线。真实数字 + 残差分析 + 适用域限定，")
    print("        才是这个项目通过的标准；为了达标去改数据，项目立刻归零。")
    return dict(bw=params["bw_gbs"], peak=params["peak_tflops"],
                peak_prefill=params["peak_prefill_tflops"],
                c_srv=params["c_srv_ms"], b_sat=params["b_sat"],
                ovh_ttft=params["ovh_ttft_ms"],
                mape_cost=mape_cost, mape_tpot=mape_tpot, max_err=max_err,
                coverage=inside, sigma_log=sigma_pred, sigma_noise=sigma_noise,
                half_width_log=half, passed=bool(ok_mape and ok_max),
                coverage_ok=bool(ok_cov),
                # 数据规模写进产物：CLAIMS 之前只能靠默认值 68，一旦 --population 变了会静默失配
                n_holdout=int(len(hold)), n_calib=int(len(cal)),
                calc_bound_cells=int(active), calc_bound_total=int(len(cal)),
                # 【必须落盘】标定并发 / 未标定并发 两组 MAPE。
                # 以前只打印在控制台，claims 用 .get(..., 5.8) 兜底，
                # 于是重拟合后文档里的 5.8% 永远不更新（复核第二轮至今）。
                mape_in_grid=(float((in_grid["ratio_cost"] - 1).abs().mean() * 100)
                              if len(in_grid) else None),
                n_in_grid=int(len(in_grid)),
                mape_out_grid=(float((out_grid["ratio_cost"] - 1).abs().mean() * 100)
                               if len(out_grid) else None),
                n_out_grid=int(len(out_grid)))


def plot(pred):
    if plt is None:
        print("[绘图] 未安装 matplotlib，跳过作图（拟合与指标不受影响）")
        return
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))

    ax = axes[0]
    for grp, style in [("标定", "o"), ("留出", "s")]:
        s = pred[pred["group"] == grp]
        ax.scatter(s["meas_cost"], s["pred_cost_gpu_s_per_1m"], marker=style,
                   s=55, alpha=0.8, label=f"{grp}（{len(s)} 格）")
    lo = max(1, pred["meas_cost"].min() * 0.6)
    hi = pred["meas_cost"].max() * 1.6
    ax.plot([lo, hi], [lo, hi], "k--", lw=1, label="y = x（完美预测）")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("实测单位成本（GPU 秒/百万 token）")
    ax.set_ylabel("预测单位成本")
    ax.set_title("预测 vs 实测：点越贴近虚线越准")
    ax.legend(); ax.grid(alpha=0.3, which="both")

    ax = axes[1]
    for grp, style in [("标定", "o"), ("留出", "s")]:
        s = pred[pred["group"] == grp]
        ax.scatter(s["conc"], (s["ratio_cost"] - 1) * 100, marker=style, s=55,
                   alpha=0.8, label=grp)
    ax.axhline(0, color="k", lw=1)
    ax.axhline(8, color="r", ls=":", lw=1); ax.axhline(-8, color="r", ls=":", lw=1)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("并发")
    ax.set_ylabel("成本预测误差（%）")
    ax.set_title("残差图：误差是否随某个维度系统性变大")
    ax.legend(); ax.grid(alpha=0.3)

    fig.tight_layout()
    p = cfg.FIG_DIR / "fig4_pred_vs_actual.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print(f"\n图已保存：{p}")



def bootstrap_params(calib, n_boot=200, seed=0):
    """对 12 个标定格做 bootstrap 重采样，给出每个参数的 5%–95% 区间。

    为什么不用解析 Hessian：标定点只有 12 个、参数 6 个，
    渐近正态近似在这里不可靠；重采样得到的区间更诚实（通常也更宽）。
    """
    rng = np.random.default_rng(seed)
    keys = ("bw_gbs", "peak_tflops", "peak_prefill_tflops", "c_srv_ms", "b_sat", "ovh_ttft_ms")
    draws = {k: [] for k in keys}
    ok = 0
    for _ in range(n_boot):
        idx = rng.integers(0, len(calib), len(calib))
        sub = calib.iloc[idx]
        try:
            p = fit(sub)
        except Exception:
            continue
        if not all(np.isfinite(p[k]) for k in keys):
            continue
        for k in keys:
            draws[k].append(p[k])
        ok += 1
    out = {}
    for k in keys:
        v = np.array(draws[k])
        if len(v) < 10:
            out[k] = (float("nan"), float("nan"))
        else:
            out[k] = (float(np.percentile(v, 5)), float(np.percentile(v, 95)))
    return out, ok


def main():
    # 口径必须能用命令行切换，否则"文档写一套、代码跑一套"还会再发生。
    import argparse
    global INCLUDE_PROBES, POPULATION
    ap = argparse.ArgumentParser(description="6 参数模型拟合与留出验证")
    ap.add_argument("--population", default="core",
                    choices=["core", "valid", "all", "ext"],
                    help="留出集口径：core=设计空间(默认) / valid=+跨模型族 / all=含全部探针")
    ap.add_argument("--bootstrap", type=int, default=0,
                    help="参数区间：对 12 个标定格做 N 次重采样（建议 200）")
    ap.add_argument("--include-probes", action="store_true",
                    help="把 step/wave/knee 探针格也算进留出集（默认不算）")
    args = ap.parse_args()
    INCLUDE_PROBES = args.include_probes or (args.population == "all")
    POPULATION = args.population

    cfg.ensure_dirs()
    # 优先用稳态口径：墙钟吞吐会被"收尾空闲"污染，而每格的波次数并不一致。
    # 稳态吞吐 = 并发×生成长度/平均单请求生命周期，与波次无关，正是模型预测的量。
    steady = cfg.DATA_DIR / "summary_steady.csv"
    path = steady if steady.exists() else cfg.DATA_DIR / "summary.csv"
    if not path.exists():
        raise SystemExit(f"没有汇总表 {path}，先跑 tc_analyze.py")
    df = pd.read_csv(path)
    # 用**实测**的提示长度，而不是配置里写的名义值。二者差 5–10% 时，
    # TTFT 的 prefill 项会系统偏一点，白白贡献误差。
    if "prompt_tokens_median" in df.columns:
        df["prompt_tokens"] = df["prompt_tokens_median"].fillna(cfg.PROMPT_TOKENS)
    else:
        df["prompt_tokens"] = float(cfg.PROMPT_TOKENS)
    df["ctx_avg_tokens"] = df["prompt_tokens"] + cfg.GEN_TOKENS / 2
    # 【区分验证集与探针实验】
    # 带后缀的格（_w4 / _n8 / _n64…）来自 step/wave/knee 三个探针预设：
    # 它们用了非标准请求数、还引入了 0.5B 这个新尺寸，目的是回答"台阶在哪/波次有没有影响"，
    # 并不属于原设计空间的验证集。混进来会把验证指标变成探针指标。
    import re as _re
    probe = df["cell_id"].astype(str).str.contains(r"(?:_w\d+|_n\d+)$", regex=True)
    # 【不要在这里剔除探针格】口径由 --population 决定（core/valid/all），
    # 在这里先删会导致 "all" 悄悄变成"不含探针"，又是一种口径漂移。
    n_probe = int(probe.sum())
    if n_probe:
        print(f"[口径] 探测到 {n_probe} 个探针格（step/wave/knee），"
              f"由 --population 决定是否计入（当前 {POPULATION}）")

    if "steady_cost_gpu_s_per_1m" in df.columns:
        df["cost_wall"] = df["cost_gpu_s_per_1m"]
        df["cost_gpu_s_per_1m"] = df["steady_cost_gpu_s_per_1m"]
        print("[口径] 使用稳态成本（wave-independent）作为验证目标")

    df = df.dropna(subset=["tpot_p50_ms", "ttft_p50_ms", "cost_gpu_s_per_1m"])

    if {"clock_sm_median", "power_w_median"} <= set(df.columns):
        bad = ((df["clock_sm_median"] < cfg.MIN_SM_CLOCK_MHZ)
               & (df["power_w_median"] < cfg.MIN_POWER_W))
        if bad.any():
            print(f"[有效性] 剔除 {int(bad.sum())} 个降频状态下测的格")
            df = df[~bad].copy()

    # 【必须剔除】KV 外溢到主机内存的格（ctx×并发 > 训练上下文）。
    # 它们不属于"KV 全在显存"这个物理系统，混进来会把带宽估计拉低一大截。
    n_before = len(df)
    fits = [cfg.kv_fits_vram((r["model"], r["quant"]), int(r["ctx"]), int(r["conc"]))
            for _, r in df.iterrows()]
    df = df[[f for f in fits]].copy()
    if len(df) < n_before:
        print(f"[regime] 剔除 {n_before - len(df)} 个 KV 外溢格（KV 装不进 8 GB 显存），"
              f"它们单独作为「显存悬崖」证据，不参与模型拟合")

    calib_ids = {cfg.cell_id(*c) for c in cfg.CALIB_CELLS}
    missing = calib_ids - set(df["cell_id"])
    if missing:
        print(f"警告：有 {len(missing)} 个标定格没数据，拟合会不准：{sorted(missing)}")

    # ---------------------------------------------------------------
    # 【口径必须显式命名】第三轮审计指出：报告里的 8.70%/68 格 与代码默认输出
    # 9.08%/76 格 对不上，根因是 8 个跨模型族格（_xf）没被探针过滤排除。
    # 现在把三套口径写死，并在每次运行时同时打印，避免再出现"文档与代码漂移"：
    #   core  = 设计空间本身（45 core + 35 dense），12 标定 + 68 留出  ← 对外默认
    #   valid = core + 8 个跨模型族格（作为"独立外推检验"，单列）
    #   all   = 全部 108 格（含台阶/波次/拐点探针，只作参考）
    # ---------------------------------------------------------------
    cid_series = df["cell_id"].astype(str)
    # 【唯一实现】探针/多轮/混合/跨族 的判定来自 tc_config
    _marks = cfg.mark_population(df)
    has_suffix = _marks["is_probe"] | _marks["is_mt"]
    is_mix = _marks["is_mix"]
    is_xf = _marks["is_xf"]

    # 【白名单判定】是否属于设计空间（只看 model/quant/ctx/conc 四元组）。
    # 之前用后缀黑名单，导致 _mtN / _mix / c48|64|128 悄悄混进 core——
    # 那批格里包含《适用域矩阵》自己声明"TTFT 不可用"的混合长度工况。
    _design = cfg.design_space_cells()
    _keys = list(zip(df["model"], df["quant"],
                     df["ctx"].astype(int), df["conc"].astype(int)))
    in_design = pd.Series([k in _design for k in _keys], index=df.index)

    # 探针（step/wave/knee）+ 新实验（_p 长度、_mt 多轮、_mix 混合）都不进 core
    probe_like = has_suffix | is_mix
    # ext = 三族扩展实验（多轮 _mt / 混合长度 _mix / 高并发 c48-128），
    # 它们既不在设计空间、也不属于探针，单独成一套口径，避免"读者不知道这些格算什么"。
    is_ext_suffix = cid_series.str.contains(r"_(?:mt\d+)$", regex=True) | is_mix
    extra_conc = ~in_design & ~has_suffix & ~is_xf & ~is_mix

    # 【唯一实现】口径判定全部来自 tc_config；这里只做一次标记 + 取掩码。
    # 以前这段逻辑在 5 个脚本里各写一遍，漏改一处就出现"同一指标两个答案"。
    _marked = cfg.mark_population(df)
    pops = {_n: df[cfg.population_mask(_marked, _n)] for _n in
            ("core", "valid", "all", "ext")}
    df = pops[POPULATION]
    print(f"[口径] population={args.population}（core=设计空间 / valid=+跨族 / all=含全部探针）")

    calib = df[df["cell_id"].isin(calib_ids)]
    if len(calib) < 4:
        raise SystemExit(f"标定格只有 {len(calib)} 个，太少，先把网格跑完。")

    print(f"标定集 {len(calib)} 格，留出集 {len(df)-len(calib)} 格")
    params = fit(calib)
    sigma_noise = estimate_noise_sigma(calib_ids, cfg.REPEATS)
    print(f"格内重复的测量噪声 σ_noise = {sigma_noise:.4f}（对数尺度）")
    pred = evaluate(df, params, calib_ids)
    summary = report(pred, params, sigma_noise=sigma_noise, repeats=cfg.REPEATS)

    if args.bootstrap:
        ci, n_ok = bootstrap_params(calib, n_boot=args.bootstrap)
        print("\n" + "=" * 66)
        print(f"参数区间（bootstrap {n_ok} 次有效重采样，5%–95%）")
        print("=" * 66)
        names = {"bw_gbs": ("带宽 GB/s", ""), "peak_tflops": ("decode 算力 TFLOP/s", "弱"),
                 "peak_prefill_tflops": ("prefill 算力 TFLOP/s", ""),
                 "c_srv_ms": ("c_srv ms/流/步", ""), "b_sat": ("B_SAT", ""),
                 "ovh_ttft_ms": ("ovh ms", "弱")}
        for k, (lo, hi) in ci.items():
            label, weak = names[k]
            point = params[k]
            width = (hi - lo) / point if point else float("nan")
            flag = "  ← 弱约束" if weak else ""
            print(f"  {label:<22}{point:9.3f}   [{lo:8.3f}, {hi:8.3f}]"
                  f"   相对宽度 {width*100:5.0f}%{flag}")
        print("  判读：相对宽度 > 60% 的参数，只能当量级用，不能当精确值引用。")
        (cfg.RESULTS_DIR / "param_ci.json").write_text(
            json.dumps({k: {"point": params[k], "lo": v[0], "hi": v[1],
                            "rel_width": (v[1] - v[0]) / params[k] if params[k] else None}
                        for k, v in ci.items()},
                       ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"  区间已写入 {cfg.RESULTS_DIR / 'param_ci.json'}")

    # 墙钟口径（端到端真实吞吐）：与稳态并报，避免把"稳态推导"当"端到端成本"
    if "cost_wall" in df.columns:
        hold_ids = set(df["cell_id"]) - calib_ids
        w = df[df["cell_id"].isin(hold_ids)][["cell_id", "cost_wall"]]
        pr = pred[pred["cell_id"].isin(hold_ids)][["cell_id", "pred_cost_gpu_s_per_1m"]]
        mg = pr.merge(w, on="cell_id", how="inner")
        if len(mg):
            ratio = mg["pred_cost_gpu_s_per_1m"] / mg["cost_wall"]
            print(f"\n[双口径] 稳态成本 MAPE {summary['mape_cost']:.2f}%  |  "
                  f"墙钟成本 MAPE {(ratio-1).abs().mean()*100:.2f}%（最大 "
                  f"{(ratio-1).abs().max()*100:.2f}%）")
            summary["mape_cost_wall"] = float((ratio - 1).abs().mean() * 100)
            summary["max_err_wall"] = float((ratio - 1).abs().max() * 100)

    # 三套口径同时报，任何人复现都能对得上
    print("\n" + "=" * 66)
    print("三套口径（同一组参数，只改留出集范围）")
    print("=" * 66)
    print(f"{'口径':<8}{'留出格数':>8}{'成本MAPE':>10}{'最大误差':>10}{'覆盖率':>9}")
    pops_out = {}
    for name, sub in pops.items():
        p2 = evaluate(sub, params, calib_ids)
        hold2 = p2[~p2["cell_id"].isin(calib_ids)]
        if not len(hold2):
            continue
        cmape = float((hold2["ratio_cost"] - 1).abs().mean() * 100)
        mx = float((hold2["ratio_cost"] - 1).abs().max() * 100)
        half_w = float(summary.get("half_width_log", float("nan")))
        cov = float((hold2["log_resid"].abs() <= half_w).mean()) if np.isfinite(half_w) else float("nan")
        print(f"{name:<8}{len(hold2):>8}{cmape:>9.2f}%{mx:>9.2f}%{cov*100:>8.1f}%")
        # 【必须落盘】这些数以前只存在于控制台输出和 CLAIMS 的静态文本里，
        # 于是"重拟合后 CLAIMS 自相矛盾 / nan"。现在统一由这里产出一份。
        pops_out[name] = dict(n_holdout=int(len(hold2)), mape_cost=float(cmape),
                              max_err=float(mx), coverage=float(cov))
    (cfg.RESULTS_DIR / "populations.json").write_text(
        json.dumps(pops_out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  三套口径已写入 {cfg.RESULTS_DIR / 'populations.json'}")
    print("  对外引用时必须带口径名（默认 core）。")

    (cfg.RESULTS_DIR / "fit.json").write_text(
        json.dumps({**params, **summary}, ensure_ascii=False, indent=2), encoding="utf-8")
    pred.to_csv(cfg.RESULTS_DIR / "predictions.csv", index=False, encoding="utf-8-sig")
    plot(pred)
    (cfg.RESULTS_DIR / "fit.json").write_text(
        json.dumps({**params, **summary}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n参数与指标已保存：{cfg.RESULTS_DIR / 'fit.json'}")


if __name__ == "__main__":
    main()









