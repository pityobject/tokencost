# -*- coding: utf-8 -*-
"""tc_analyze_poisson.py —— 容量曲线：区分吞吐天花板与延迟膝点，并量化 prefill 干扰。

原来三个口径问题：
① 基线用了所有 λ 里 p90 的最小值（λ=1.0 的 110ms），
   而 λ=0.5 只有 15 个样本、p90=243ms，纯噪声更大 → 基线不稳；
② 把"p90 恶化前能撑住的 QPS"当成吞吐上限，但**延迟膝点远早于吞吐天花板**，
   两者混为一谈会把容量低估一半；
③ 与模型对比时拿"可持续 QPS"去比"服务能力"，口径不对称。

新口径：
  · 基线 = 低负载（λ ≤ 1）的 p90 中位数；
  · **吞吐天花板** = 实测达成的最大 QPS；
  · **延迟膝点** = p90 首次超过基线 3 倍的 λ；
  · 与模型比吞吐天花板，不拿 SLA 点去比服务能力。
"""
import json
import sys

import pandas as pd

import tc_config as cfg
from tc_model import predict

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

MODEL, QUANT, CTX, SLOTS = "3B", "q4_k_m", 512, 8


def main():
    p = cfg.DATA_DIR / "probe_poisson.csv"
    if not p.exists():
        raise SystemExit(f"没有 {p}，先跑 tc_probe_poisson.py")
    df = pd.read_csv(p)

    fit = json.loads((cfg.RESULTS_DIR / "fit.json").read_text(encoding="utf-8"))
    params = dict(bw_gbs=fit["bw"], peak_tflops=fit["peak"],
                  peak_prefill_tflops=fit.get("peak_prefill", fit["peak"]),
                  c_srv_ms=fit["c_srv"], b_sat=fit["b_sat"], ovh_ttft_ms=fit["ovh_ttft"])

    print("=" * 88)
    print("容量曲线分析（区分吞吐天花板与延迟膝点）")
    print("=" * 88)

    low = df[df["lam"] <= 1.0]
    base_p90 = float(low["ttft_p90"].median()) if len(low) else float(df["ttft_p90"].min())
    df["p90_ratio"] = df["ttft_p90"] / base_p90

    print(f"基线（λ≤1 的 p90 中位）: {base_p90:.0f} ms\n")
    print(f"{'λ':>5}{'达成QPS':>9}{'达成率':>7}{'平均并发':>9}{'TTFT p50':>10}"
          f"{'TTFT p90':>10}{'p90/基线':>10}{'TPOT':>8}")
    for _, r in df.iterrows():
        print(f"{r['lam']:>5.1f}{r['achieved_qps']:>9.2f}{r['achieved_qps']/r['lam']:>7.0%}"
              f"{r['mean_conc']:>9.2f}{r['ttft_p50']:>10.0f}{r['ttft_p90']:>10.0f}"
              f"{r['p90_ratio']:>9.1f}×{r['tpot_p50']:>8.1f}")

    cap = df.loc[df["achieved_qps"].idxmax()]
    knee = df[df["p90_ratio"] >= 3.0]
    print("\n" + "-" * 88)
    print("① 吞吐天花板（实测达成的最大 QPS）")
    print(f"   λ={cap['lam']:.1f} 时达成 {cap['achieved_qps']:.2f} 请求/秒"
          f"（平均并发 {cap['mean_conc']:.1f}）")
    print("② 延迟膝点（p90 TTFT 超过基线 3 倍）")
    if len(knee):
        k = knee.iloc[0]
        ok_max = df[df["p90_ratio"] < 3.0]["lam"].max()
        print(f"   λ={k['lam']:.1f} 时 p90={k['ttft_p90']:.0f} ms（基线的 {k['p90_ratio']:.1f} 倍）")
        print("   → 吞吐还能涨，但延迟已经崩了；对延迟敏感的业务可用区间只能到 "
              f"λ≈{ok_max:.1f}")
    else:
        print("   未出现 3 倍恶化")

    row = dict(model=MODEL, quant=QUANT, ctx=CTX, conc=SLOTS,
               prompt_tokens=float(cfg.PROMPT_TOKENS),
               ctx_avg_tokens=cfg.PROMPT_TOKENS + cfg.GEN_TOKENS / 2)
    pr = predict(params, row)
    req_s = (pr["pred_ttft_ms"] + cfg.GEN_TOKENS * pr["pred_tpot_ms"]) / 1e3
    qps_pred = SLOTS / req_s
    ratio = cap["achieved_qps"] / qps_pred
    print("\n" + "-" * 88)
    print("③ 模型预测 vs 实测（都按吞吐上限口径）")
    print(f"   模型：b={SLOTS} 稳态 → 单请求 {req_s*1000:.0f} ms → 服务能力 {qps_pred:.2f} 请求/秒")
    print(f"   实测：吞吐天花板 {cap['achieved_qps']:.2f} 请求/秒")
    print(f"   实测/预测 = {ratio:.2f}")
    if 0.6 <= ratio <= 1.0:
        print(f"   判读：模型给的是**上界**，实测约为其 {ratio:.0%}，"
              "差距来自排队与 prefill 干扰。可用于容量规划，但要乘折减系数。")
    elif ratio > 1.0:
        print("   判读：实测超过预测，模型在 b=槽位 处偏保守。")
    else:
        print("   判读：实测明显低于预测，瓶颈不止服务时间。")

    summ = pd.read_csv(cfg.DATA_DIR / "summary_steady.csv")
    ref = summ[(summ["model"] == MODEL) & (summ["quant"] == QUANT)
               & (summ["ctx"] == CTX) & (summ["conc"] == SLOTS)]
    if len(ref):
        tpot_sync = float(ref.iloc[0]["tpot_p50_ms"])
        mid = df[df["mean_conc"].between(SLOTS * 0.8, SLOTS * 1.2)]
        tpot_pois = float(mid["tpot_p50"].median()) if len(mid) else float("nan")
        if tpot_pois == tpot_pois:
            print("\n" + "-" * 88)
            print("④ prefill 干扰：同为并发≈8，泊松到达 vs 同步波次")
            print(f"   同步波次 TPOT {tpot_sync:.1f} ms ｜ 泊松到达 TPOT {tpot_pois:.1f} ms"
                  f"  →  高 {tpot_pois/tpot_sync-1:+.0%}")
            print("   原因：同步波次里 prefill 只在开头发生一次；泊松到达时每个新请求都要插入")
            print("         prefill，打断正在跑的 decode。这是模型未刻画的一项开销。")

    out = cfg.RESULTS_DIR / "poisson_summary.json"
    out.write_text(json.dumps(dict(
        base_p90_ms=base_p90,
        capacity_qps=float(cap["achieved_qps"]),
        capacity_lambda=float(cap["lam"]),
        model_qps_pred=float(qps_pred),
        ratio=float(ratio),
        knee_lambda=float(knee.iloc[0]["lam"]) if len(knee) else None,
        rows=df.to_dict(orient="records")), ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n已写入 {out}")


if __name__ == "__main__":
    main()
