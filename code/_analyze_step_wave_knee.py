# -*- coding: utf-8 -*-
"""汇总 step / wave / knee 三组实验，回答问题：
1) 台阶在哪个并发？是台阶还是平滑曲线？
2) 波次数到底影不影响 TPOT？（假设检验）
3) 0.5B 的吞吐在哪里饱和？是算力受限还是服务受限？
"""
import json
import pathlib
import statistics as st
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = pathlib.Path(__file__).resolve().parents[1]
import tc_config as cfg  # noqa: E402

rows = [json.loads(l) for l in (ROOT / "data/raw_cells.jsonl").open(encoding="utf-8") if l.strip()]
ok = [r for r in rows if r.get("status") == "ok"]
g = {}
for r in ok:
    g.setdefault(r["cell_id"], []).append(r)


def m(cid, f):
    v = [x[f] for x in g.get(cid, []) if x.get(f) is not None]
    return st.median(v) if v else float("nan")


print("=" * 78)
print("① step：并发 4→16，固定 4 波（3B-Q4 / ctx512）")
print("=" * 78)
print(f"{'并发':>4}{'请求':>6}{'TPOT ms':>9}{'吞吐 t/s':>10}{'每槽 t/s':>10}{'TTFT ms':>9}   与上一档的 TPOT 增幅")
prev = None
for conc in [4, 5, 6, 7, 8, 10, 12, 16]:
    cid = f"3B_q4_k_m_ctx512_c{conc}_w4"
    if cid not in g:
        print(f"{conc:>4}    （未跑）")
        continue
    t = m(cid, "tpot_p50_ms")
    d = f"{(t/prev-1)*100:+6.1f}%" if prev else "   —"
    print(f"{conc:>4}{4*conc:>6}{t:>9.1f}{m(cid,'throughput_tps'):>10.0f}"
          f"{m(cid,'per_slot_tps'):>10.0f}{m(cid,'ttft_p50_ms'):>9.1f}   {d}")
    prev = t

print("\n对照：同配置但现在用默认请求数的老格（c4=24 请求/6 波，c8=32/4 波，c16=48/3 波）")
for conc in [4, 8, 16]:
    cid = f"3B_q4_k_m_ctx512_c{conc}"
    if cid in g:
        print(f"  老 c{conc:<2} TPOT {m(cid,'tpot_p50_ms'):5.1f} ms   新 c{conc}_w4 TPOT "
              f"{m(f'3B_q4_k_m_ctx512_c{conc}_w4','tpot_p50_ms'):5.1f} ms")

print()
print("=" * 78)
print("② wave：并发固定 8，只改请求数（1/2/4/8 波）—— 检验'波次假设'")
print("=" * 78)
print(f"{'请求数':>6}{'波次':>6}{'TPOT ms':>9}{'吞吐 t/s':>10}{'TTFT ms':>9}")
vals = []
for n in [8, 16, 32, 64]:
    cid = f"3B_q4_k_m_ctx512_c8_n{n}"
    if cid not in g:
        continue
    t = m(cid, "tpot_p50_ms")
    vals.append(t)
    print(f"{n:>6}{n/8:>6.1f}{t:>9.1f}{m(cid,'throughput_tps'):>10.0f}{m(cid,'ttft_p50_ms'):>9.1f}")
if vals:
    spread = (max(vals) - min(vals)) / st.mean(vals) * 100
    print(f"\n  TPOT 极差 {max(vals)-min(vals):.1f} ms（相对 {spread:.1f}%）")
    print("  结论：" + ("波次几乎不影响 → 台阶是真·并发效应，不是测量假象"
                        if spread < 8 else "波次有影响，需固定波次再比较"))

print()
print("=" * 78)
print("③ knee：0.5B 的饱和点在哪？算力受限还是服务受限？")
print("=" * 78)
print(f"{'并发':>4}{'TPOT ms':>9}{'吞吐 t/s':>10}{'c_srv×并发':>11}{'访存下限 t/s':>13}{'算力下限 t/s':>13}")
spec = cfg.MODEL_SPECS[("0.5B", "q4_k_m")]
BW, PEAKD, CSRV = 249.0e9, 13.2e12, 0.766
best = (0, 0)
for conc in [1, 2, 4, 8, 16, 32, 48, 64]:
    cid = f"0.5B_q4_k_m_ctx512_c{conc}_w4"
    if cid not in g:
        continue
    t, thr = m(cid, "tpot_p50_ms"), m(cid, "throughput_tps")
    if thr > best[1]:
        best = (conc, thr)
    w_eff = spec["weight_bytes"] + cfg.kv_bytes_per_token(spec) * 320 * conc
    mem_tps = conc / (w_eff / BW)
    cmp_tps = conc / (2 * spec["n_params"] * conc / PEAKD)
    print(f"{conc:>4}{t:>9.1f}{thr:>10.0f}{CSRV*conc:>11.1f}{mem_tps:>13.0f}{cmp_tps:>13.0f}")
print(f"\n  吞吐峰值：并发 {best[0]} 时 {best[1]:.0f} t/s")
print("  判读：如果公式项远大于实测，说明既不是访存也不是算力在限——是服务路径。")
