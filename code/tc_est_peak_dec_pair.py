# -*- coding: utf-8 -*-
"""
PEAK_dec 的成对差分估计（`tc.py peakdec --pair`）
对同一并发 b，用两个不同规模的模型：

    TPOT_A - TPOT_B = max(mem_A, cmp_A) - max(mem_B, cmp_B)   （c_srv·b 抵消）

若两者都已进入算力区（cmp 主导），则

    ΔTPOT = 2·(P_A - P_B)·b / PEAK_dec
    ⇒ PEAK_dec = 2·(P_A - P_B)·b / ΔTPOT

好处：`c_srv` 完全消掉，不需要它准确；坏处：必须确认两者都在算力区，
否则差值是访存项之差（那要另一套公式）。

用法：
    python code/tc_est_peak_dec.py --pair --csv data/summary_steady.csv \
        --a 7B/q4_k_m --b 1.5B/q4_k_m --conc 32
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import tc_config as cfg

sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def n_params(model, quant):
    spec = cfg.MODEL_SPECS.get((model, quant))
    return float(spec["n_params"]) if spec else None


def weight_bytes(model, quant):
    spec = cfg.MODEL_SPECS.get((model, quant))
    return float(spec["weight_bytes"]) if spec else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", action="store_true")
    ap.add_argument("--csv", default=str(Path(cfg.__file__).parent.parent /
                                        "data" / "summary_steady.csv"))
    ap.add_argument("--a", default="7B/q4_k_m")
    ap.add_argument("--b", default="1.5B/q4_k_m")
    ap.add_argument("--conc", type=int, default=32)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    df = df[df["conc"] == args.conc]
    ma, qa = args.a.split("/")
    mb, qb = args.b.split("/")
    A = df[(df["model"] == ma) & (df["quant"] == qa)]
    B = df[(df["model"] == mb) & (df["quant"] == qb)]
    if A.empty or B.empty:
        print(f"缺数据：{args.a} {len(A)} 行 / {args.b} {len(B)} 行")
        return

    # 配对：同一 ctx 下比较（越接近越好）
    PA = cfg.MODEL_SPECS[(ma, qa)]
    PB = cfg.MODEL_SPECS[(mb, qb)]
    rows = []
    for _, ra in A.iterrows():
        cand = B[(B["ctx"] == ra["ctx"])]
        if cand.empty:
            continue
        rb = cand.iloc[(cand["prompt_tokens_median"] - ra["prompt_tokens_median"]).abs().argsort()[:1]].iloc[0]
        tpot_a, tpot_b = float(ra["tpot_p50_ms"]), float(rb["tpot_p50_ms"])
        rows.append(dict(ctx=ra["ctx"], L=float(ra["prompt_tokens_median"]),
                         tpot_a=tpot_a, tpot_b=tpot_b, dt=tpot_a - tpot_b))
    if not rows:
        print("没有可配对的 ctx")
        return

    dP = (PA["n_params"] - PB["n_params"])
    print("=" * 80)
    print(f"成对差分：{args.a} vs {args.b} @ 并发 {args.conc}")
    print("=" * 80)
    print(f"  参数量差 ΔP = {(dP/1e9):.2f} B")
    print(f"  {'ctx':>6}{'L':>8}{'TPOT_A':>10}{'TPOT_B':>10}{'ΔTPOT':>10}"
          f"{'PEAK_dec 估计':>15}")
    ests = []
    for r in rows:
        if r["dt"] <= 5:      # 差值太小，估计会爆
            continue
        peak = 2 * dP * args.conc / (r["dt"] * 1e-3) / 1e12
        ests.append(peak)
        print(f"  {int(r['ctx']):>6}{r['L']:>8.0f}{r['tpot_a']:>10.1f}"
              f"{r['tpot_b']:>10.1f}{r['dt']:>10.1f}{peak:>13.1f} TFLOP/s")

    if not ests:
        print("\n  ❌ ΔTPOT 太小（<5 ms），无法稳定估计")
        return
    est = float(np.median(ests))
    print(f"\n  中位估计 PEAK_dec ≈ **{est:.1f} TFLOP/s**"
          f"（范围 {min(ests):.1f}–{max(ests):.1f}，n={len(ests)}）")
    print(f"  当前拟合值 {cfg.MODEL_SPECS and '13.157'}")
    print("\n  前提：两个模型在该并发下都必须**算力受限**，否则差值来自访存项。")
    print("  校验办法：若各 ctx 的估计值稳定（离散小），说明确实在算力区。")

    out = dict(mode="pairwise", a=args.a, b=args.b, conc=args.conc,
               delta_params=dP, estimates_tflops=[round(x, 2) for x in ests],
               median_tflops=round(est, 2),
               range=[round(min(ests), 2), round(max(ests), 2)],
               per_ctx=rows,
               note="c_srv 在差分中抵消；要求两者都在算力区")
    fp = Path(args.out) if args.out else (cfg.RESULTS_DIR / "peak_dec_pairwise.json")
    fp.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n→ {fp}")


if __name__ == "__main__":
    main()
