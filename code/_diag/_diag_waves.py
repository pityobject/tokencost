# -*- coding: utf-8 -*-
"""查 c8/c12/c24 为什么吞吐非单调：看每格的请求数、波次、空闲时间。"""
import json
import pathlib
import statistics as st
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = pathlib.Path(__file__).resolve().parents[2]
import tc_config as cfg  # noqa: E402

rows = [json.loads(l) for l in (ROOT / "data/raw_cells.jsonl").open(encoding="utf-8") if l.strip()]
ok = [r for r in rows if r.get("status") == "ok"]
g = {}
for r in ok:
    g.setdefault(r["cell_id"], []).append(r)


def m(cid, f):
    v = [x[f] for x in g.get(cid, []) if x.get(f) is not None]
    return st.median(v) if v else float("nan")


print(f"{'cell':<28}{'请求':>5}{'波次':>6}{'span秒':>8}{'token':>7}{'吞吐':>7}{'每槽t/s':>9}{'平均耗时ms':>11}")
for ctx in [512]:
    for c in [1, 2, 4, 8, 12, 16, 24, 32]:
        cid = f"1.5B_q4_k_m_ctx{ctx}_c{c}"
        if cid not in g:
            continue
        n_req = m(cid, "requests")
        print(f"{cid:<28}{n_req:>5.0f}{n_req/c:>6.2f}{m(cid,'measure_span_s'):>8.2f}"
              f"{m(cid,'total_tokens'):>7.0f}{m(cid,'throughput_tps'):>7.0f}"
              f"{m(cid,'per_slot_tps'):>9.0f}{m(cid,'mean_total_ms'):>11.0f}")

print("\n每格实际发多少请求（requests_per_repeat 的配置）")
for c in [1, 2, 4, 8, 12, 16, 24, 32]:
    print(f"  并发 {c:>2}: {cfg.requests_per_repeat(c):>3} 个请求 → 波次 {cfg.requests_per_repeat(c)/c:.2f}")

print("\n推断：波次差异会让'测量窗口'包含不同比例的收尾空闲，吞吐随之失真。")
print("      而 TPOT 会被'新请求的 prefill 与旧请求的 decode 重叠'污染，波次越多污染越重。")
