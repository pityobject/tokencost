"""
tc_recommend.py —— 项目的最终产物：一台"配置推荐器"。

输入：你的 SLA（首字延迟、出字速度）+ 请求长度分布
输出：在满足 SLA 的所有配置里，单位成本最低的那个，以及它的预测区间

它为什么值钱：
  工程师平时做容量规划，是"跑一堆 benchmark，然后人肉挑一个"。
  这个工具把"跑 benchmark"换成了"用标定过的模型算"，3 秒出结论，还给误差范围。

用法：
  & "D:\\llm\\venv\\Scripts\\python.exe" tc_recommend.py --ttft 600 --tpot 45
  & "D:\\llm\\venv\\Scripts\\python.exe" tc_recommend.py --ttft 600 --tpot 45 --prompt-mean 1200 --prompt-sigma 0.8
"""

import argparse
import json

from pathlib import Path

import numpy as np

import tc_config as cfg
from tc_model import predict

Z_90 = 1.645


# ---------------------------------------------------------------- 钱（W2）
# 成本单位一直是「GPU 秒 / 百万 token」——工程上准确，但对业务方不直观。
# 这里把 GPU·秒折成「元 / 百万 token」。
#
# **不设默认价格**：价格是外部假设，默认塞进去会让数字看起来比实际更确定。
# 观察到的现实差距：本地电费口径约 0.06 元/小时，云端 A10 按量约 10.5 元/小时，
# 同一份 GPU·秒换算成钱会差 175 倍——所以"多少钱"必须和"什么价"一起说。
PRICE_CNY_PER_HOUR = {
    "laptop": 0.06,      # 95 W × 0.6 元/度（不含机器折旧）
    "a10": 10.5,         # 云端按量参考价（会变，仅作量级参考）
    "4090": 3.0,
    "a100_40g": 18.0,
}


def resolve_price(spec):
    """--gpu-price 接受数字或预设名，返回 (元/小时, 说明)。"""
    if spec is None:
        return None, ""
    key = str(spec).strip().lower()
    if key in PRICE_CNY_PER_HOUR:
        return PRICE_CNY_PER_HOUR[key], f"预设 {key}"
    try:
        return float(key), "自定义"
    except ValueError:
        raise SystemExit(
            f"--gpu-price 需要数字（元/小时）或预设名：{sorted(PRICE_CNY_PER_HOUR)}")


def to_cny(cost_gpu_s_per_1m, price_per_hour):
    """GPU·秒/百万token → 元/百万token。"""
    if price_per_hour is None:
        return float("nan")
    return cost_gpu_s_per_1m / 3600.0 * price_per_hour


# ---------------------------------------------------------------- 实测尾部（A3）
def load_measured_tails():
    """从 results/quantile_tails.csv 读每格的 p90/p50，按并发取中位。

    返回 {"ttft": {conc: ratio}, "tpot": {conc: ratio}}；没有文件则返回 {}。
    """
    fp = cfg.RESULTS_DIR / "quantile_tails.csv"
    if not fp.exists():
        return {}
    import pandas as pd
    t = pd.read_csv(fp)
    out = {}
    for metric in ("ttft", "tpot"):
        col = f"{metric}_p90_over_p50"
        if col not in t.columns:
            continue
        by = t.groupby("conc")[col].median()
        out[metric] = {int(k): float(v) for k, v in by.items() if float(v) == float(v)}
    return out


def measured_tail_factor(tails, metric, conc):
    """按并发对数插值取实测 p90/p50；超出范围用端点值。"""
    m = tails.get(metric) or {}
    if not m:
        return 1.0
    ks = sorted(m)
    if conc <= ks[0]:
        return m[ks[0]]
    if conc >= ks[-1]:
        return m[ks[-1]]
    import numpy as np
    xs = np.log(np.array(ks, dtype=float))
    ys = np.array([m[k] for k in ks], dtype=float)
    return float(np.interp(np.log(float(conc)), xs, ys))


def cvar(x, q=0.9):
    xs = np.sort(np.asarray(x))[::-1]
    k = max(1, int(round(len(xs) * (1 - q))))
    return float(xs[:k].mean())


def simulate(params, model, quant, conc, prompt_mean, prompt_sigma,
             ctx=None, n=4000, seed=42):
    """蒙特卡洛：对请求长度分布抽样，算出这一格的延迟分布与成本。

    这里用蒙特卡洛而不是直接代公式，是因为延迟对请求长度是分段的 max(...)，
    不是线性的，只能靠抽样拿尾部统计量（CVaR）。

    ctx：该格的每槽位上下文上限。**必须传**——否则长尾抽样会给出"比上下文还长"的
    请求，把尾部延迟抬到不可能的高度，导致整个场景被判成"无可行配置"。
    真实系统里这种请求要么被拒绝、要么被截断，不会按原长度计费。
    """
    rng = np.random.default_rng(seed)
    L = rng.lognormal(mean=np.log(max(prompt_mean, 1.0)), sigma=prompt_sigma, size=n)
    if ctx is not None:
        L = np.minimum(L, max(ctx - cfg.GEN_TOKENS, 1))   # 超出槽位的部分不可行

    tpot, ttft = [], []
    for l in L:
        row = dict(model=model, quant=quant, conc=conc,
                   ctx_avg_tokens=float(l) + cfg.GEN_TOKENS / 2,
                   prompt_tokens=float(l))
        pr = predict(params, row)
        tpot.append(pr["pred_tpot_ms"])
        ttft.append(pr["pred_ttft_ms"])

    tpot = np.array(tpot)
    ttft = np.array(ttft)
    # 单位成本按"期望 token 产出速度"算。同样要算上 prefill：
    # 一条请求占用槽位的时间 = TTFT + 生成长度×TPOT
    tpot_mean = float(tpot.mean())
    ttft_mean = float(ttft.mean())
    per_req_s = (ttft_mean + cfg.GEN_TOKENS * tpot_mean) / 1e3
    thr = conc * cfg.GEN_TOKENS / per_req_s
    return dict(
        tpot_p50=float(np.median(tpot)), tpot_cvar90=cvar(tpot, 0.9),
        ttft_p50=float(np.median(ttft)), ttft_cvar90=cvar(ttft, 0.9),
        throughput_tps=thr, cost_gpu_s_per_1m=1e6 / thr,
        tpot_mean=tpot_mean,
    )


def main():
    ap = argparse.ArgumentParser(description="给定 SLA，推荐成本最低的推理配置")
    ap.add_argument("--ttft", type=float, default=600.0, help="首字延迟上限（毫秒，按 CVaR90 判定）")
    ap.add_argument("--tpot", type=float, default=45.0, help="出字间隔上限（毫秒，按 CVaR90 判定）")
    ap.add_argument("--calib", default=None,
                    help="用自己的标定结果（tc_calibrate.py 的产物）代替 fit.json")
    ap.add_argument("--prompt-mean", type=float, default=cfg.PROMPT_TOKENS, help="平均提示长度（token）")
    ap.add_argument("--prompt-sigma", type=float, default=0.5, help="提示长度的对数标准差（0.5 ≈ 长度差异较大）")
    ap.add_argument("--sigma-log", type=float, default=None,
                    help="模型误差的对数标准差；不填则从 results/fit.json 读")
    # 质量底线默认 6 个百分点 ≈ 140 题基准的一个置信区间半宽。
    # 定得比测量噪声还紧（比如 3pp），等于在统计上分不清的差异上做硬决策——
    # 实测：3pp 时只剩 7B 的 1 格可选，6pp 时可选集恢复到 9 格以上。
    ap.add_argument("--quality-sweep", action="store_true",
                    help="输出质量损失→最便宜配置的权衡表（推荐对外展示这个）")
    ap.add_argument("--quality-basis", choices=["point", "lcb"], default="lcb",
                    help="质量约束用点估计(point)还是 Wilson 下界(lcb，默认，更保守)")
    ap.add_argument("--min-acc-drop", type=float, default=6.0,
                    help="允许比最高准确率低几个百分点（质量下限）。默认 6 ≈ 一个 CI 半宽")
    ap.add_argument("--tail", choices=["assumed", "measured"], default="assumed",
                    help="尾部口径：assumed=对数正态假设（历史行为）；"
                         "measured=用实测 p90/p50 随并发变化")
    ap.add_argument("--gpu-price", default=None,
                    help="GPU 价格（元/小时），接受数字或预设：laptop / a10 / 4090 / a100_40g。"
                         "不给则只报 GPU·秒——价格是外部假设，不设默认值")
    ap.add_argument("--top", type=int, default=12)
    args = ap.parse_args()

    # 质量表（如果跑过 tc_eval.py）
    quality = {}
    quality_lo = {}     # Wilson 下界（--quality-basis lcb 时使用）
    qpath = cfg.DATA_DIR / "quality.csv"
    if qpath.exists():
        import pandas as pd
        qdf = pd.read_csv(qpath).dropna(subset=["acc"])
        quality = {(r["model"], r["quant"]): float(r["acc"]) for _, r in qdf.iterrows()}
        if "acc_lo" in qdf.columns:
            quality_lo = {(r["model"], r["quant"]): float(r["acc_lo"])
                          for _, r in qdf.iterrows() if r["acc_lo"] == r["acc_lo"]}
        # 【口径】默认用 Wilson 下界做质量约束：140 题的区间半宽约 ±6pp，
        # 用点估计会出现"差 0.43pp 就被剪掉/放行"的抖动（第三轮审计 C 条）。
        if args.quality_basis == "lcb" and quality_lo:
            quality = quality_lo
            print(f"[质量] 使用 Wilson 下界做约束（比点估计保守）")
        elif args.quality_basis == "lcb":
            print("[质量] 质量表里没有 acc_lo 列，回退到点估计")
    else:
        print("[提示] 没找到 data/quality.csv，本次不做质量约束。建议先跑 tc_eval.py。")
    best_acc = max(quality.values()) if quality else None

    # --calib：用用户自标定的参数（tc_calibrate.py 产物，键名与 fit.json 一致）
    fit_path = Path(args.calib) if args.calib else (cfg.RESULTS_DIR / "fit.json")
    if not Path(fit_path).exists():
        raise SystemExit(f"没有 {fit_path}；仓库标定先跑 tc_model.py，"
                         f"自标定先跑 tc_calibrate.py")
    if args.calib:
        print(f"[标定] 使用 {fit_path}（自标定），不是仓库的 fit.json")
    fit = json.loads(Path(fit_path).read_text(encoding="utf-8"))
    # 与 tc_model.fit() 的输出对齐（5 参数）
    params = dict(bw_gbs=fit["bw"], peak_tflops=fit["peak"],
                  c_srv_ms=fit["c_srv"], b_sat=fit["b_sat"],
                  ovh_ttft_ms=fit["ovh_ttft"],
                  peak_prefill_tflops=fit.get("peak_prefill", fit["peak"]))
    if args.sigma_log is not None:
        half_log = Z_90 * args.sigma_log
    else:
        half_log = fit.get("half_width_log", Z_90 * fit.get("sigma_log", 0.1))
    tails = load_measured_tails() if args.tail == "measured" else {}
    if args.tail == "measured" and not tails:
        raise SystemExit("[尾部] 需要 results/quantile_tails.csv（先跑 tc.py tails）")

    price, price_src = resolve_price(args.gpu_price)
    print("=" * 78)
    print(f"尾部口径：{args.tail}"
          + ("（实测 p90/p50 随并发变化）" if args.tail == "measured"
             else "（对数正态假设，历史行为）"))
    print(f"输入：首字延迟 ≤ {args.ttft:.0f} ms，出字间隔 ≤ {args.tpot:.0f} ms（均按 CVaR90 判定）")
    print(f"请求长度：均值 {args.prompt_mean:.0f} token，对数标准差 {args.prompt_sigma}")
    print(f"模型参数：带宽 {params['bw_gbs']:.0f} GB/s，算力 {params['peak_tflops']:.0f} TFLOP/s，"
          f"服务开销 {params['c_srv_ms']:.2f} ms/流，prefill 饱和 {params['b_sat']:.0f}，"
          f"首字固定 {params['ovh_ttft_ms']:.1f} ms；90% 预测区间 ±{half_log*100:.1f}%")
    print("=" * 78)

    # 请求长度的 95 分位：用来检查"这一格的每槽位上下文装不装得下这种请求"。
    # 不检查这一条，推荐器会推 ctx=512 的配置去跑 2048 token 的长文档——物理上根本塞不进去。
    p95_prompt = float(np.exp(np.log(max(args.prompt_mean, 1.0)) + 1.645 * args.prompt_sigma))
    need_ctx = p95_prompt + cfg.GEN_TOKENS
    print(f"上下文需求：提示长度 95 分位约 {p95_prompt:.0f} token，加生成长度共需 {need_ctx:.0f} token")

    rows = []
    for c in cfg.all_cells():
        sim = simulate(params, c["model"], c["quant"], c["conc"],
                       args.prompt_mean, args.prompt_sigma, ctx=c["ctx"])
        # 把模型自身的不确定性加到延迟上：90% 预测区间的上界
        ttft_hi = sim["ttft_cvar90"] * np.exp(half_log)
        tpot_hi = sim["tpot_cvar90"] * np.exp(half_log)
        acc = quality.get((c["model"], c["quant"]), float("nan"))
        ok_quality = True if best_acc is None else (acc >= best_acc - args.min_acc_drop / 100.0)
        ok_feasible = c["feasible"]
        ok_ctx_fit = c["ctx"] >= need_ctx          # 每槽位上下文必须装得下这种请求
        ok_sla = (ttft_hi <= args.ttft) and (tpot_hi <= args.tpot)
        rows.append(dict(cell_id=cfg.cell_id(c["model"], c["quant"], c["ctx"], c["conc"]),
                         model=c["model"], quant=c["quant"], ctx=c["ctx"], conc=c["conc"],
                         vram_est_gb=c["vram_est_gb"], feasible=ok_feasible,
                         ctx_fits=ok_ctx_fit,
                         acc=acc, meets_quality=ok_quality,
                         ttft_hi_ms=ttft_hi, tpot_hi_ms=tpot_hi,
                         throughput_tps=sim["throughput_tps"],
                         cost=sim["cost_gpu_s_per_1m"], meets_sla=ok_sla))

    import pandas as pd
    df = pd.DataFrame(rows)
    valid = df[df["feasible"] & df["ctx_fits"] & df["meets_sla"] & df["meets_quality"]] \
        .sort_values("cost")

    if valid.empty:
        print("\n没有任何配置同时满足上下文、显存、质量与 SLA。最接近的三个（按 SLA 违反程度）：")
        df["violation"] = (df["ttft_hi_ms"] / args.ttft).clip(lower=1) * \
                          (df["tpot_hi_ms"] / args.tpot).clip(lower=1)
        print(df[df["feasible"] & df["ctx_fits"]].nsmallest(3, "violation")
              [["cell_id", "ttft_hi_ms", "tpot_hi_ms", "cost"]]
              .to_string(index=False, float_format=lambda x: f"{x:9.1f}"))
        print("\n建议：放宽 SLA，或降低请求长度假设，或换更大的显存。")
        return

    best = valid.iloc[0]
    worst = valid.iloc[-1]
    print(f"\n推荐配置：{best['cell_id']}")
    print(f"  预测首字延迟（90% 置信上限）  {best['ttft_hi_ms']:7.1f} ms   （上限 {args.ttft:.0f}）")
    print(f"  预测出字间隔（90% 置信上限）  {best['tpot_hi_ms']:7.1f} ms   （上限 {args.tpot:.0f}）")
    print(f"  预测吞吐                      {best['throughput_tps']:7.1f} token/s")
    print(f"  预测成本                      {best['cost']:7.0f} GPU 秒 / 百万 token")
    if price is not None:
        cny = to_cny(best["cost"], price)
        print(f"  折算成钱                      {cny:7.2f} 元 / 百万 token"
              f"（按 {price:g} 元/小时，{price_src}）")
    print(f"\n对照：满足 SLA 的最贵配置是 {worst['cell_id']}，成本 {worst['cost']:.0f}"
          f"（是推荐的 {worst['cost']/best['cost']:.2f} 倍）")

    print(f"\nSLA 内成本最低的前 {args.top} 个配置：")
    show = valid.head(args.top).copy()
    cols = ["cell_id", "vram_est_gb", "ttft_hi_ms", "tpot_hi_ms", "throughput_tps", "cost"]
    if price is not None:
        show["cost_cny"] = show["cost"].map(lambda c: to_cny(c, price))
        cols.append("cost_cny")
    print(show[cols].to_string(index=False, float_format=lambda x: f"{x:9.2f}"))

    print("\n被判出局的原因分布：")
    print(f"  显存装不下                {int((~df['feasible']).sum())} 格")
    print(f"  每槽位上下文装不下请求    {int((df['feasible'] & ~df['ctx_fits']).sum())} 格")
    print(f"  质量不达标                {int((df['feasible'] & df['ctx_fits'] & ~df['meets_quality']).sum())} 格")
    print(f"  其余条件够但不满足 SLA    {int((df['feasible'] & df['ctx_fits'] & df['meets_quality'] & ~df['meets_sla']).sum())} 格")
    print("（这行统计就是'约束在剪设计空间'的直接证据，写报告时用得上。）")

    # ------------------------------------------------------------------
    # 质量—成本权衡表
    #
    # 为什么必须给这张表而不是一个答案：扩大题集到 400 题后，
    # 7B 的质量显著优于其他组合（91.0% vs 84.2%，McNemar/Holm 后 p<0.01），
    # 于是任何"不显著劣于最好"的约束都会把候选缩成只剩 7B，成本跳 5 倍。
    # 这说明"能接受多少质量损失"是**业务决策**，模型只负责把汇率算清楚。
    # ------------------------------------------------------------------
    if args.quality_sweep and best_acc is not None:
        # 【唯一实现】第三轮复核指出：本函数与 tc_threshold_sensitivity.py
        # 各自实现了一遍"SLA 过滤 + 候选集"，于是同一问题给出两套答案
        # （半宽差 36.5 倍、候选集也不同）。现在两者都调 tc_common.rank()。
        import tc_common as tc
        print("\n" + "=" * 76)
        print("质量—成本权衡表：允许的质量损失 → 满足 SLA 的最便宜配置")
        print("=" * 76)
        print(f"{'质量损失上限':>12}{'配置':<28}{'成本':>10}{'实际质量':>10}")
        prev = None
        sweep_rows = []
        for drop in (0, 2, 4, 6, 7.5, 8, 10, 15, 20, 30):
            pick, _ = tc.rank(params, fit, quality_drop_pp=drop,
                              prompt_mean=args.prompt_mean, prompt_sigma=args.prompt_sigma,
                              ttft_cap=args.ttft, tpot_cap=args.tpot,
                              basis=args.quality_basis)
            if pick is None:
                continue
            mark = "" if prev is None or float(pick["cost"]) >= prev else "   ← 成本降档"
            print(f"{drop:>10} pp  {pick['cell_id']:<28}"
                  f"{float(pick['cost']):>10.0f}{float(pick['acc'])*100:>9.1f}%{mark}")
            prev = float(pick["cost"])
            sweep_rows.append(dict(drop=drop, cell_id=pick["cell_id"],
                                   cost=float(pick["cost"]), acc=float(pick["acc"])))
        import pandas as _pd
        _pd.DataFrame(sweep_rows).to_csv(cfg.RESULTS_DIR / "recommendation_sweep.csv",
                                         index=False, encoding="utf-8-sig")
        print("\n  读法：每一行是一个业务决策——'我愿意损失 N 个百分点质量'。")
        print("        跳跃点就是多花多少钱能买到多少质量；最终选哪行是业务方定，不是模型定。")

    out = cfg.RESULTS_DIR / "recommendation.csv"
    df.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"\n全量推演结果：{out}")


if __name__ == "__main__":
    main()



