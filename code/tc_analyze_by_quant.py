# -*- coding: utf-8 -*-
"""
A-2 按量化档分别拟合：`c_srv` 到底是不是"部署栈属性"？

## 问题

现在 6 个参数是**全数据混合拟合**的，等于隐含假设"Q2 和 Q8 的每步服务开销相同"。
但低比特解码（反量化）可能带来额外开销。若 `c_srv` 随量化档显著变化，
那它就不只是栈属性，还是**精度属性**——这直接影响"换量化能省多少"的结论。

## 产出

- `results/params_by_quant.csv`：每个量化档的 6 个参数 + 格数 + 拟合误差
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import tc_config as cfg
import tc_model as tm

sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def prep(df):
    df = df.copy()
    if "prompt_tokens" not in df.columns:
        df["prompt_tokens"] = df["prompt_tokens_median"]
    if "ctx_avg_tokens" not in df.columns:
        df["ctx_avg_tokens"] = df["prompt_tokens"] + cfg.GEN_TOKENS / 2
    return df


def fit_subset(df):
    p = tm.fit(df)
    res_t, res_p = [], []
    for _, r in df.iterrows():
        pr = tm.predict(p, r)
        if not np.isfinite(pr["pred_ttft_ms"]):
            continue
        res_t.append(abs(np.log(pr["pred_ttft_ms"] / r["ttft_p50_ms"])))
        res_p.append(abs(np.log(pr["pred_tpot_ms"] / r["tpot_p50_ms"])))
    return p, float(np.mean(res_t) * 100), float(np.mean(res_p) * 100)


def main():
    df = pd.read_csv(cfg.DATA_DIR / "summary_steady.csv")
    df = df[df["cell_id"].str.contains("_p", regex=False) == False]
    df = prep(df)
    print(f"总格数 {len(df)}，量化档：{sorted(df['quant'].unique())}\n")

    rows = []
    for q, g in df.groupby("quant"):
        if len(g) < 8:
            print(f"  跳过 {q}（只有 {len(g)} 格，不足以拟合 6 参数）")
            continue
        p, et, ep = fit_subset(g)
        rows.append(dict(quant=q, n_cells=len(g),
                         bw=round(p["bw_gbs"], 1),
                         peak=round(p["peak_tflops"], 2),
                         peak_prefill=round(p["peak_prefill_tflops"], 2),
                         c_srv=round(p["c_srv_ms"], 3),
                         b_sat=round(p["b_sat"], 1),
                         ovh=round(p["ovh_ttft_ms"], 2),
                         ttft_mape=round(et, 1), tpot_mape=round(ep, 1)))

    out = pd.DataFrame(rows).sort_values("quant")
    out.to_csv(cfg.RESULTS_DIR / "params_by_quant.csv", index=False, encoding="utf-8")

    print(f"  {'量化':<10}{'格数':>5}{'BW':>8}{'c_srv':>8}{'b_sat':>7}"
          f"{'PEAK_pre':>10}{'TTFT%':>7}{'TPOT%':>7}")
    for r in rows:
        print(f"  {r['quant']:<10}{r['n_cells']:>5}{r['bw']:>8.1f}"
              f"{r['c_srv']:>8.3f}{r['b_sat']:>7.1f}"
              f"{r['peak_prefill']:>10.1f}{r['ttft_mape']:>7.1f}{r['tpot_mape']:>7.1f}")

    if len(out) >= 2:
        cs = out["c_srv"]
        bw = out["bw"]
        print(f"\n=== 判读 ===")
        print(f"  c_srv 范围 {cs.min():.3f}–{cs.max():.3f}"
              f"（最大/最小 = {cs.max()/max(cs.min(), 1e-9):.2f}×）")
        print(f"  BW    范围 {bw.min():.1f}–{bw.max():.1f}"
              f"（最大/最小 = {bw.max()/max(bw.min(), 1e-9):.2f}×）")
        if cs.max() / max(cs.min(), 1e-9) > 1.3:
            print("  → c_srv 随量化档变化明显：每步开销不只属于部署栈，**也受精度影响**")
        else:
            print("  → c_srv 在各量化档间接近：每步开销主要是**栈属性**，对精度不敏感")
        if bw.max() / max(bw.min(), 1e-9) > 1.15:
            print("  → 注意：BW 也随量化变化，可能混入了模型尺寸差异（不同档的格来自不同模型）")
    print("\n→ results/params_by_quant.csv")


if __name__ == "__main__":
    main()
