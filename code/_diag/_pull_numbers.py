# -*- coding: utf-8 -*-
"""从 summary.csv 提取报告要用的关键行，保证引用的是真实数字。"""
import pathlib
import sys

import pandas as pd

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
df = pd.read_csv(pathlib.Path(r"C:\Users\王钟垚\Desktop\find_career\tokencost\data\summary.csv"))

cols = ["cell_id", "ttft_p50_ms", "tpot_p50_ms", "throughput_tps",
        "cost_gpu_s_per_1m", "temp_c_median", "clock_sm_median", "util_pct_median"]
print("=== 关键格 ===")
for cid in ["1.5B_q4_k_m_ctx512_c1", "1.5B_q4_k_m_ctx512_c4",
            "1.5B_q4_k_m_ctx512_c16", "1.5B_q4_k_m_ctx512_c32",
            "3B_q4_k_m_ctx2048_c1", "3B_q4_k_m_ctx2048_c4",
            "3B_q4_k_m_ctx512_c32", "7B_q4_k_m_ctx2048_c1", "7B_q4_k_m_ctx2048_c16"]:
    r = df[df["cell_id"] == cid]
    if len(r):
        r = r.iloc[0]
        print(f"{cid:<26} TTFT {r['ttft_p50_ms']:6.1f}  TPOT {r['tpot_p50_ms']:5.2f}  "
              f"吞吐 {r['throughput_tps']:6.0f}  成本 {r['cost_gpu_s_per_1m']:8.0f}  "
              f"util {r['util_pct_median']:3.0f}%  温度 {r['temp_c_median']:4.1f}")

print("\n=== 全表统计 ===")
print(f"格数 {len(df)}   温度 {df['temp_c_median'].min():.0f}-{df['temp_c_median'].max():.0f} °C")
print(f"SM 时钟 {df['clock_sm_median'].min():.0f}-{df['clock_sm_median'].max():.0f} MHz")
print(f"成本区间 {df['cost_gpu_s_per_1m'].min():.0f} ~ {df['cost_gpu_s_per_1m'].max():.0f} GPU·s/百万token")
print(f"吞吐区间 {df['throughput_tps'].min():.0f} ~ {df['throughput_tps'].max():.0f} tok/s")

# 并发-成本降幅（每个模型组合）
print("\n=== 并发 1→32 的成本降幅 ===")
for (m, q), sub in df.groupby(["model", "quant"]):
    s = sub.sort_values("conc")
    c1 = s[s["conc"] == 1]["cost_gpu_s_per_1m"]
    if len(c1):
        base = c1.iloc[0]
        best = s["cost_gpu_s_per_1m"].min()
        print(f"  {m:>5} {q:<7} 成本 {base:7.0f} → {best:6.0f}（降为 1/{base/best:.2f}）")
