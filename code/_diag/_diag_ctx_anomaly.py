# -*- coding: utf-8 -*-
"""对比：同一模型/并发下，ctx 2048 vs 4096 的运行条件（温度/频率/显存/功耗）。"""
import json
import pathlib
import statistics as st
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = pathlib.Path(r"C:\Users\王钟垚\Desktop\find_career\tokencost")
rows = [json.loads(l) for l in (ROOT / "data/raw_cells.jsonl").open(encoding="utf-8") if l.strip()]
ok = [r for r in rows if r.get("status") == "ok"]
g = {}
for r in ok:
    g.setdefault(r["cell_id"], []).append(r)


def m(cid, f):
    v = [x[f] for x in g.get(cid, []) if x.get(f) is not None]
    return st.median(v) if v else float("nan")


print(f"{'cell':<28}{'TPOT':>7}{'吞吐':>7}{'温度':>6}{'SM MHz':>8}{'显存MB':>8}{'功耗W':>7}{'util':>6}")
for m_ in ["7B", "3B", "1.5B"]:
    for c in [1, 4]:
        for ctx in [512, 1024, 2048, 4096, 8192]:
            cid = f"{m_}_q4_k_m_ctx{ctx}_c{c}"
            if cid not in g:
                continue
            print(f"{cid:<28}{m(cid,'tpot_p50_ms'):>7.1f}{m(cid,'throughput_tps'):>7.0f}"
                  f"{m(cid,'temp_c_median'):>6.1f}{m(cid,'clock_sm_median'):>8.0f}"
                  f"{m(cid,'mem_used_mb_median'):>8.0f}{m(cid,'power_w_median'):>7.1f}"
                  f"{m(cid,'util_pct_median'):>5.0f}%")
    print()

print("=== 看日志里这些格的 n_ctx 与 KV 分配 ===")
for cid in ["7B_q4_k_m_ctx4096_c1", "7B_q4_k_m_ctx2048_c1", "7B_q4_k_m_ctx1024_c4"]:
    log = ROOT / "logs" / f"server_{cid}.log"
    if not log.exists():
        continue
    txt = log.read_text(encoding="utf-8", errors="replace")
    for line in txt.splitlines():
        if any(k in line for k in ("n_ctx_slot", "KV", "kv cache", "flash", "compute buffer")):
            print(f"  [{cid}] {line.strip()[:120]}")
