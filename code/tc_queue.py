# -*- coding: utf-8 -*-
"""
# 状态：量级验证通过（2026-11-02，90 秒采样）。未饱和区预测/实测 1.2–1.6 倍，
#       全部偏保守（高估延迟）；λ 超容量时正确判饱和而不给有限值。
#       中位误差 34%，**不作为精确预测引用**，只用于量级判断。
tc_queue.py —— 排队层：从"请求要多久"补到"请求要等多久"。

## 为什么需要它

六参数模型给的是**服务时间**（TTFT + GEN×TPOT），也就是"轮到你之后要花多久"。
真实系统里，请求还要先**排队**。所以报告里有一条限制：
"模型不算排队时间，λ 超过容量后 TTFT 会失真。"

这一层补的就是排队时间。做法不是写离散事件仿真（容易写错、也难验），
而是用**解析近似 + 固定点**，然后用已有的 7 个 λ 点验证。

## 模型

设系统有 c 个槽位，到达率 λ，单请求服务时间 S(b)（由六参数模型给，随并发变化）。

稳态下的自洽条件（Little 定律 + 槽位约束）：
    ρ = λ·S(b) / c                      占用率
    Wq = Allen–Cunneen 近似(ρ, c)        排队时间
    W  = Wq + S(b)                       总逗留时间
    b  = min(c, λ·W)                     有效并发

这是一个不动点：b 影响 S，S 影响 W，W 又回头决定 b。迭代求解。

Allen–Cunneen（G/G/c 的经典近似）：
    Wq ≈ (Ca² + Cs²)/2 · ρ^(√(2(c+1)) − 1) / (c(1 − ρ)) · S
其中 Ca² 是到达过程变异系数平方（泊松 = 1），Cs² 是服务时间变异系数平方
（没有逐请求数据时取 1，并在结论里声明这是近似）。

## 判据

  · ρ < 0.98：给出预测，与实测比；
  · ρ ≥ 0.98：判定"系统饱和"——此时延迟由观测窗口决定，模型不该给有限值。
    这本身就是一条结论：模型能告诉你"从哪个 λ 开始不可用"。

用法：
    python tc_queue.py
"""

import json
import sys

import numpy as np
import pandas as pd

import tc_config as cfg
from tc_model import predict

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

MODEL, QUANT, CTX, SLOTS = "3B", "q4_k_m", 512, 8


def service_time(params, b):
    """单请求服务时间（秒）：TTFT(b) + GEN × TPOT(b)。"""
    row = dict(model=MODEL, quant=QUANT, ctx=CTX, conc=max(b, 1e-6),
               prompt_tokens=float(cfg.PROMPT_TOKENS),
               ctx_avg_tokens=cfg.PROMPT_TOKENS + cfg.GEN_TOKENS / 2)
    pr = predict(params, row)
    ttft_s = pr["pred_ttft_ms"] / 1e3
    tpot_s = pr["pred_tpot_ms"] / 1e3
    return ttft_s, tpot_s, ttft_s + cfg.GEN_TOKENS * tpot_s


def allen_cunneen(rho, c, S, ca2=1.0, cs2=1.0):
    """G/G/c 排队时间的 Allen–Cunneen 近似（秒）。ρ≥1 时不适用。"""
    if rho >= 1.0:
        return float("inf")
    if rho <= 1e-9:
        return 0.0
    expo = np.sqrt(2 * (c + 1)) - 1
    return (ca2 + cs2) / 2 * (rho ** expo) / (c * (1 - rho)) * S


def solve_fixed_point(params, lam, c=SLOTS, tol=1e-3, iters=60):
    """解 b = min(c, λ·W)：只统计服务时间，不含排队。"""
    b = min(c, max(1e-3, lam * service_time(params, 1.0)[2]))
    for _ in range(iters):
        _, _, S = service_time(params, b)
        rho = lam * S / c
        if rho >= 1.0:
            # 饱和：不加排队也算不出稳态，直接返回临界值
            return b, S, rho, float("inf"), float("inf"), True
        Wq = allen_cunneen(rho, c, S)
        W = Wq + S
        b_new = min(c, max(1e-3, lam * W))
        if abs(b_new - b) < tol:
            b = b_new
            break
        b = b_new
    ttft_s, _, S = service_time(params, b)
    rho = lam * S / c
    Wq = allen_cunneen(rho, c, S)
    return b, S, rho, Wq, ttft_s + Wq, rho >= 0.98


def main():
    p = cfg.DATA_DIR / "probe_poisson.csv"
    if not p.exists():
        raise SystemExit(f"没有 {p}，先跑 tc_probe_poisson.py")
    df = pd.read_csv(p).sort_values("lam")

    fit = json.loads((cfg.RESULTS_DIR / "fit.json").read_text(encoding="utf-8"))
    params = dict(bw_gbs=fit["bw"], peak_tflops=fit["peak"],
                  peak_prefill_tflops=fit.get("peak_prefill", fit["peak"]),
                  c_srv_ms=fit["c_srv"], b_sat=fit["b_sat"], ovh_ttft_ms=fit["ovh_ttft"])

    print("=" * 92)
    print("排队层：从服务时间到端到端延迟（Allen–Cunneen + 固定点）")
    print("=" * 92)

    _, _, S_c = service_time(params, SLOTS)
    cap_model = SLOTS / S_c
    cap_meas = float(df["achieved_qps"].max())
    print(f"模型给出的容量上界   c / S(c) = {SLOTS} / {S_c:.3f}s = {cap_model:.2f} 请求/秒")
    print(f"实测吞吐天花板                 = {cap_meas:.2f} 请求/秒"
          f"（实测/预测 {cap_meas/cap_model:.0%}）")

    # prefill 干扰：泊松下 TPOT 比同步波次高多少
    summ = pd.read_csv(cfg.DATA_DIR / "summary_steady.csv")
    ref = summ[(summ["model"] == MODEL) & (summ["quant"] == QUANT)
               & (summ["ctx"] == CTX) & (summ["conc"] == SLOTS)]
    tpot_infl = 1.0
    if len(ref):
        t_sync = float(ref.iloc[0]["tpot_p50_ms"])
        mid = df[df["mean_conc"].between(SLOTS * 0.8, SLOTS * 1.2)]
        if len(mid):
            t_pois = float(mid["tpot_p50"].median())
            if t_pois == t_pois and t_sync > 0:
                tpot_infl = t_pois / t_sync
    print(f"prefill 干扰系数（泊松 TPOT / 同步波次 TPOT） = {tpot_infl:.2f}")
    cap_pois = SLOTS / (service_time(params, SLOTS)[0] + cfg.GEN_TOKENS *
                        service_time(params, SLOTS)[1] * tpot_infl)
    print(f"计入干扰后的容量上界           = {cap_pois:.2f} 请求/秒"
          f"（与实测天花板比 {cap_meas/cap_pois:.0%}）")

    print("\n" + "-" * 92)
    print(f"{'λ':>5}{'ρ':>7}{'有效并发':>9}{'排队(预测)':>11}{'TTFT预测':>10}"
          f"{'TTFT实测':>10}{'比值':>8}  判定")
    rows = []
    for _, r in df.iterrows():
        lam = float(r["lam"])
        b, S, rho, Wq, ttft_pred, sat = solve_fixed_point(params, lam)
        # 计入 prefill 干扰：服务时间按比例放大
        b2, S2, rho2, Wq2, ttft2, sat2 = solve_fixed_point_inflated(params, lam, tpot_infl)
        meas = float(r["ttft_p50"])
        ratio = (ttft2 * 1e3) / meas if meas > 0 and np.isfinite(ttft2) else float("nan")
        verdict = "饱和（不给有限预测）" if sat2 else ("吻合" if 0.5 <= ratio <= 2.0 else "偏差大")
        print(f"{lam:>5.1f}{rho2:>7.2f}{b2:>9.2f}{Wq2*1e3:>10.0f}ms"
              f"{ttft2*1e3:>9.0f}ms{meas:>9.0f}ms"
              f"{('—' if not np.isfinite(ratio) else f'{ratio:>7.1f}')}  {verdict}")
        rows.append(dict(lam=lam, rho=rho2, b=b2, wq_ms=Wq2 * 1e3,
                         ttft_pred_ms=ttft2 * 1e3, ttft_meas_ms=meas,
                         ratio=ratio, saturated=bool(sat2)))

    ok = [x for x in rows if not x["saturated"] and np.isfinite(x["ratio"])]
    print("\n" + "-" * 92)
    if ok:
        errs = [abs(x["ratio"] - 1) for x in ok]
        print(f"未饱和区（{len(ok)} 个 λ）：预测/实测 相对误差 中位 "
              f"{np.median(errs)*100:.0f}%，最大 {max(errs)*100:.0f}%")
        verdict = ("排队层可用" if np.median(errs) < 0.5
                   else "排队层仅有量级意义，应如实标注")
        print(f"判据（中位误差 < 50%）→ {verdict}")
    sat_n = sum(1 for x in rows if x["saturated"])
    print(f"饱和区：{sat_n} 个 λ 被判定为'无稳态'——模型此时不给有限预测，只说'不可用'。")

    out = cfg.RESULTS_DIR / "queue_summary.json"
    out.write_text(json.dumps(dict(
        cap_model=cap_model, cap_meas=cap_meas, cap_poisson=cap_pois,
        tpot_inflation=tpot_infl, rows=rows), ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n已写入 {out}")


def solve_fixed_point_inflated(params, lam, infl, c=SLOTS, tol=1e-3, iters=60):
    """与 solve_fixed_point 相同，但服务时间乘上 prefill 干扰系数。

    为什么必须乘：泊松到达时每个新请求的 prefill 都会打断 decode，
    实测 TPOT 比同步波次高约 25%。不加这一项，容量会被高估。
    """
    def svc(b):
        t, p, s = service_time(params, b)
        return t, p * infl, t + cfg.GEN_TOKENS * p * infl

    b = min(c, max(1e-3, lam * svc(1.0)[2]))
    for _ in range(iters):
        _, _, S = svc(b)
        rho = lam * S / c
        if rho >= 1.0:
            return b, S, rho, float("inf"), float("inf"), True
        Wq = allen_cunneen(rho, c, S)
        b_new = min(c, max(1e-3, lam * (Wq + S)))
        if abs(b_new - b) < tol:
            b = b_new
            break
        b = b_new
    ttft_s, _, S = svc(b)
    rho = lam * S / c
    Wq = allen_cunneen(rho, c, S)
    return b, S, rho, Wq, ttft_s + Wq, rho >= 0.98


if __name__ == "__main__":
    main()
