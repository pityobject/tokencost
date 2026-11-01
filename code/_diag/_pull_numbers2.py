# -*- coding: utf-8 -*-
"""从 raw_cells.jsonl 提取报告要用的真实数字（含 GPU 利用率）。"""
import json
import pathlib
import statistics as st
import sys
from collections import defaultdict

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
p = pathlib.Path(r"C:\Users\王钟垚\Desktop\find_career\tokencost\data\raw_cells.jsonl")
rows = [json.loads(l) for l in p.open(encoding="utf-8") if l.strip()]
ok = [r for r in rows if r.get("status") == "ok"]
g = defaultdict(list)
for r in ok:
    g[r["cell_id"]].append(r)


def med(cid, f):
    return st.median(x[f] for x in g[cid] if x.get(f) is not None)


print("=== 关键格（中位数）===")
print(f"{'cell':<26}{'TTFT':>8}{'TPOT':>7}{'吞吐':>8}{'成本':>9}{'util':>6}{'温度':>6}{'MHz':>7}")
for cid in ["1.5B_q4_k_m_ctx512_c1", "1.5B_q4_k_m_ctx512_c4",
            "1.5B_q4_k_m_ctx512_c16", "1.5B_q4_k_m_ctx512_c32",
            "3B_q4_k_m_ctx2048_c1", "3B_q4_k_m_ctx2048_c4",
            "3B_q4_k_m_ctx512_c32", "7B_q4_k_m_ctx2048_c1", "7B_q4_k_m_ctx2048_c16"]:
    if cid not in g:
        continue
    print(f"{cid:<26}{med(cid,'ttft_p50_ms'):>8.1f}{med(cid,'tpot_p50_ms'):>7.2f}"
          f"{med(cid,'throughput_tps'):>8.0f}{med(cid,'cost_gpu_s_per_1m'):>9.0f}"
          f"{med(cid,'util_pct_median'):>5.0f}%{med(cid,'temp_c_median'):>6.1f}"
          f"{med(cid,'clock_sm_median'):>7.0f}")

print("\n=== 全表 ===")
t = [r["temp_c_median"] for r in ok if r.get("temp_c_median") is not None]
c = [r["cost_gpu_s_per_1m"] for r in ok]
th = [r["throughput_tps"] for r in ok]
print(f"格数 {len(g)}  行数 {len(ok)}")
print(f"温度 {min(t):.0f}-{max(t):.0f} °C   成本 {min(c):.0f}-{max(c):.0f}   吞吐 {min(th):.0f}-{max(th):.0f}")

print("\n=== 各模型组合：并发 1→32 成本降幅 ===")
for (m, q) in sorted({(r["model"], r["quant"]) for r in ok}):
    sub = [cid for cid in g if cid.startswith(f"{m}_{q}_")]
    c1 = [med(cid, "cost_gpu_s_per_1m") for cid in sub if cid.endswith("_c1")]
    best = min(med(cid, "cost_gpu_s_per_1m") for cid in sub)
    if c1:
        print(f"  {m:>5} {q:<7} {st.median(c1):7.0f} → {best:6.0f}  （1/{st.median(c1)/best:.2f}）")
