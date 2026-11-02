# -*- coding: utf-8 -*-
"""把 c_srv 拆成三层：
   ① 模型 eval（服务端自己记的 per-token 时间）
   ② decode 循环内的额外开销 = ① − max(访存时间, 算力时间)
   ③ 交付/HTTP 开销 = 客户端 TPOT − ①

依据：llama-server 每个任务结束会打印
   "eval time = 5783.42 ms / 128 tokens ( 45.54 ms per token)"
这是服务端自己量的 decode 时间，与客户端测的 TPOT 对比就能定位开销在哪一层。

不新增任何实验，只解析已有日志。
"""
import json
import pathlib
import re
import statistics as st
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = pathlib.Path(__file__).resolve().parents[1]
import tc_config as cfg  # noqa: E402

PAT = re.compile(r"eval time =\s+([\d.]+) ms /\s+(\d+) tokens \(\s*([\d.]+) ms per token")

rows = [json.loads(l) for l in (ROOT / "data/raw_cells.jsonl").open(encoding="utf-8") if l.strip()]
ok = [r for r in rows if r.get("status") == "ok"]
g = {}
for r in ok:
    g.setdefault(r["cell_id"], []).append(r)


def med(cid, f):
    v = [x[f] for x in g.get(cid, []) if x.get(f) is not None]
    return st.median(v) if v else float("nan")


def server_tpot(cid, gen=cfg.GEN_TOKENS):
    """从服务端日志取"正式负载轮"（生成 token 数 = GEN）的每 token eval 时间中位数。"""
    log = ROOT / "logs" / f"server_{cid}.log"
    if not log.exists():
        return float("nan")
    vals = []
    with log.open(encoding="utf-8", errors="replace") as f:
        for line in f:
            m = PAT.search(line)
            if m and int(m.group(2)) == gen:
                vals.append(float(m.group(3)))
    return st.median(vals) if vals else float("nan")


def gpu_time(key, conc, ctx_avg=320):
    """模型认为的纯硬件时间：max(访存, 算力)。"""
    s = cfg.MODEL_SPECS[key]
    w_eff = s["weight_bytes"] + cfg.kv_bytes_per_token(s) * ctx_avg * conc
    t_mem = w_eff / (249.1e9) * 1e3
    t_cmp = 2 * s["n_params"] * conc / (13.16e12) * 1e3
    return max(t_mem, t_cmp), t_mem, t_cmp


print("=" * 92)
print("c_srv 三向分解（3B-Q4，ctx512）")
print("=" * 92)
print(f"{'并发':>4}{'客户端TPOT':>11}{'服务端eval':>11}{'硬件下限':>10}"
      f"{'decode内开销':>13}{'交付开销':>10}{'占TPOT':>9}")
rows_out = []
for conc in [1, 4, 8, 16, 32]:
    cid = f"3B_q4_k_m_ctx512_c{conc}"
    if cid not in g:
        continue
    t_client = med(cid, "tpot_p50_ms")
    t_srv = server_tpot(cid)
    t_hw, t_mem, t_cmp = gpu_time(("3B", "q4_k_m"), conc)
    in_decode = t_srv - t_hw
    delivery = t_client - t_srv
    rows_out.append((conc, t_client, t_srv, t_hw, in_decode, delivery))
    print(f"{conc:>4}{t_client:>11.1f}{t_srv:>11.1f}{t_hw:>10.1f}"
          f"{in_decode:>13.1f}{delivery:>10.1f}{in_decode/t_client*100:>8.0f}%")

if rows_out:
    print("\n解读：")
    print("  · decode 内开销 = 服务端 eval − 硬件下限 → 采样、批构造、同步等每步固定工作")
    print("  · 交付开销      = 客户端 TPOT − 服务端 eval → HTTP/SSE/客户端解析")
    print("  · 若交付开销 ≈ 0，说明 c_srv 几乎全部在 decode 循环内部，不在网络层")

    # 拟合 c_srv：用 (TPOT - 硬件下限)/并发
    print("\n每并发流每步的额外开销（= c_srv 的实测来源）：")
    print(f"{'并发':>4}{'总额外开销/并发':>18}{'其中decode内':>14}{'其中交付':>12}")
    for conc, tc, ts, thw, ind, dl in rows_out:
        print(f"{conc:>4}{(tc-thw)/conc:>18.3f}{(ts-thw)/conc:>14.3f}{dl/conc:>12.3f}")

    out = ROOT / "data" / "csrv_decomposition.csv"
    import csv
    with out.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["conc", "client_tpot_ms", "server_eval_ms", "hw_floor_ms",
                    "in_decode_overhead_ms", "delivery_overhead_ms"])
        w.writerows(rows_out)
    print(f"\n已写入 {out}")
