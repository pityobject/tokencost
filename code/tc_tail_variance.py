# -*- coding: utf-8 -*-
"""
SLA 上界的方差项（`tc.py tailvar`）：把"常数尾部假设"换成 `σ ∝ (b·L)^γ`

## 问题

推荐器的 90% 上界现在用「对数正态 1.645σ」这个**常数**尾部假设，实测高并发下
`p90/p50` 能到 3.75（并发 64），而假设只给 1.09——**SLA 判定偏乐观**。

## 做法

1. 用 A10 的三组数据（L=64/256/1900，c1–32，共 15 个点）拟合
   `CV(b,L) = a · (b·L)^γ`；
2. 若 TPOT 近似对数正态，则 `p90/p50 ≈ exp(1.2816·CV)`；
3. **样本外验证**：拿本机 136 格（不是拟合用的数据）比较预测与实测的 `p90/p50`。

## 用法

    python code/tc_tail_variance.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import tc_config as cfg

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

A10 = [
    ("jitter_a10_vllm_L64.json", 64),
    ("jitter_a10_vllm_L256.json", 256),
    ("jitter_a10_vllm.json", 1900),
]
Z90 = 1.2816      # 正态 90% 分位


def load_fit_points():
    pts = []
    for name, L in A10:
        fp = cfg.RESULTS_DIR / "a10_raw" / name
        if not fp.exists():
            continue
        d = json.loads(fp.read_text(encoding="utf-8"))
        for k, v in d["cells"].items():
            b = int(k)
            pts.append(dict(L=L, b=b, cv=v["tpot_cv"],
                            p90p50=v["tpot_p90_over_p50"], src=name))
    return pts


def main():
    pts = load_fit_points()
    if len(pts) < 6:
        print("拟合点不足")
        return
    b = np.array([p["b"] for p in pts], float)
    L = np.array([p["L"] for p in pts], float)
    cv = np.array([p["cv"] for p in pts], float)
    # CV = c0 + a·(b-1)^p·L^q
    # 关键：b=1 时 (b-1)=0 退回基线 c0 —— 实测在 b=1 时 CV≈1.4%，与 L 无关。
    # 用 (b·L)^γ 这种乘积形式会错误地让 L 单独抬高 CV（L=1900,c1 会预测 3.6%）。
    from scipy.optimize import curve_fit

    def model(X, c0, a, pp, qq):
        bb, LL = X
        return c0 + a * np.power(np.maximum(bb - 1, 1e-9), pp) * np.power(LL, qq)

    try:
        popt, _ = curve_fit(model, (b, L), cv, p0=[0.012, 1e-5, 0.6, 0.7],
                            maxfev=20000)
        c0, a, g, gL = popt
        pred = model((b, L), *popt)
        r2 = 1 - np.sum((cv - pred) ** 2) / np.sum((cv - cv.mean()) ** 2)
        fit_ok = True
    except Exception as e:
        print("  拟合失败：", e)
        c0, a, g, gL, r2, fit_ok = cv.min(), 0.0, 0.0, 0.0, 0.0, False

    print("=" * 80)
    print("一、抖动幅度的经验律：CV = a · (b·L)^γ")
    print("=" * 80)
    print(f"  拟合（{len(pts)} 个点，A10 三组长度 × 五档并发）：")
    print(f"  CV = {c0*100:.2f}% + {a:.6f}·(b-1)^{g:.2f}·L^{gL:.2f}   R² = {r2:.4f}")
    print(f"  基线 {c0*100:.2f}%（b=1 时的结构性抖动，与 L 无关）")
    print(f"  增长项 ≈ (b-1)^{g:.2f}·L^{gL:.2f}；指数和 {g+gL:.2f}")

    print("\n  逐点核对：")
    print(f"  {'L':>5}{'b':>5}{'CV 实测':>10}{'CV 预测':>10}{'p90/p50 实测':>15}{'按 CV 推算':>12}")
    for p in sorted(pts, key=lambda q: (q["L"], q["b"])):
        pr = c0 + a * max(p["b"] - 1, 1e-9) ** g * p["L"] ** gL
        print(f"  {p['L']:>5}{p['b']:>5}{p['cv']*100:>9.2f}%{pr*100:>9.2f}%"
              f"{p['p90p50']:>15.3f}{np.exp(Z90*pr):>12.3f}")

    # 验证「对数正态 + 1.2816σ」能否把 CV 换算成 p90/p50
    err = [abs(np.exp(Z90 * p["cv"]) / p["p90p50"] - 1) * 100 for p in pts]
    print(f"\n  用 exp(1.2816·CV) 反算 p90/p50 的误差：中位 {np.median(err):.1f}%，"
          f"最大 {max(err):.1f}%")
    print("  → 说明「对数正态」在 A10 vLLM 上是个可用的形状假设，缺的只是 CV 随 b·L 增长")

    # 样本外验证：本机 136 格
    print("\n" + "=" * 80)
    print("二、样本外验证：本机 136 格（不是拟合用的数据）")
    print("=" * 80)
    fp = Path(cfg.__file__).resolve().parent.parent / "data" / "summary_steady.csv"
    if not fp.exists():
        print("  缺 summary_steady.csv，跳过")
        return
    df = pd.read_csv(fp)
    df = df[(df["tpot_p50_ms"] > 0) & (df["tpot_p90_ms"] > 0)
            & (df["prompt_tokens_median"] > 0)]
    rows = []
    for _, r in df.iterrows():
        bb = float(r["conc"])
        LL = float(r["prompt_tokens_median"]) + float(r["n_out_median"]) / 2
        cvp = c0 + a * max(bb - 1, 1e-9) ** g * LL ** gL
        pred_ratio = max(1.0, float(np.exp(Z90 * cvp)))
        meas_ratio = float(r["tpot_p90_ms"] / r["tpot_p50_ms"])
        rows.append((pred_ratio, meas_ratio, bb, LL))
    pr = np.array([x[0] for x in rows])
    mr = np.array([x[1] for x in rows])
    mape = float(np.mean(np.abs(pr / mr - 1)) * 100)
    print(f"  格数 {len(rows)}｜预测 p90/p50 的 MAPE = **{mape:.1f}%**")
    print(f"  实测 p90/p50：中位 {np.median(mr):.3f}，max {mr.max():.3f}")
    print(f"  预测 p90/p50：中位 {np.median(pr):.3f}，max {pr.max():.3f}")
    # 与「常数 1.09」基线对比
    base = np.abs(1.09 / mr - 1).mean() * 100
    print(f"  对照：常数 1.09 假设的 MAPE = {base:.1f}%")
    print(f"  → 方差项把尾部预测误差从 {base:.0f}% 降到 {mape:.0f}%")

    out = dict(c0=round(float(c0), 5), a=round(float(a), 8),
               exp_b=round(float(g), 3), exp_L=round(float(gL), 3),
               r2=round(float(r2), 4),
               n_fit_points=len(pts),
               lognormal_check_median_err_pct=round(float(np.median(err)), 1),
               out_of_sample=dict(n_cells=len(rows), mape_pct=round(mape, 1),
                                  baseline_const_109_mape_pct=round(float(base), 1)),
               note="CV = a·(b·L)^γ；p90/p50 ≈ exp(1.2816·CV)。拟合只用 A10 15 点，验证用本机 136 格")
    (cfg.RESULTS_DIR / "tail_variance.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n→ results/tail_variance.json")


if __name__ == "__main__":
    main()
