"""
tc_probe_kvquant.py —— KV cache 量化：把"显存悬崖"推远多少？（v2：同批交替对照）

## v1 的设计错误

第一版只跑 q8_0，然后拿"旧数据集里的 f16 数值"去比。
但那份旧数据已经被归档删除，而且跨批次比较会混入机器状态漂移——
上一轮 c_srv 探针就是栽在"跨变体固定顺序"上。

## v2 的做法

**同一格，交替跑 f16 与 q8_0，各 2 轮。**
顺序按格子奇偶交替（第一格 f16 先、第二格 q8 先），把顺序效应平摊掉。

对每个格报告：吞吐、TPOT、以及"q8 相对 f16 的变化"。

## 预期

| 格 | f16 状态 | q8_0 预期 |
|---|---|---|
| 1.5B-Q4 ctx8192 并发32 | KV 7.5 GB → 外溢 | KV 3.76 GB → 装得下，吞吐应大幅回升 |
| 3B-Q4 ctx8192 并发16 | KV 4.8 GB → 刚好装下 | 松快，可能略升 |
| 7B-Q4 ctx8192 并发16 | KV 7.7 GB → 外溢 | 3.85 + 权重 4.68 = 8.5 GB → 仍外溢（对照组） |

用法：
  & "D:\\llm\\venv\\Scripts\\python.exe" tc_probe_kvquant.py
"""

import csv
import sys
import time

import tc_config as cfg
import tc_loadgen as lg
import tc_server as srv

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

CELLS = [
    ("1.5B", "q4_k_m", 8192, 32),
    ("3B",   "q4_k_m", 8192, 16),
    ("7B",   "q4_k_m", 8192, 16),
]
OUT = cfg.DATA_DIR / "probe_kvquant.csv"
REPEATS = 2


def main():
    import asyncio
    rows = []
    print("=" * 84)
    print("KV 量化对照（同格交替：f16 ↔ q8_0）")
    print("=" * 84)

    for idx, (model, quant, ctx, conc) in enumerate(CELLS):
        key = (model, quant)
        cid = f"{model}_{quant}_ctx{ctx}_c{conc}"
        # 奇偶交替：第 0 格先 f16，第 1 格先 q8_0，第 2 格先 f16……
        order = ["f16", "q8_0"] if idx % 2 == 0 else ["q8_0", "f16"]
        print(f"\n[{idx+1}/{len(CELLS)}] {cid}   顺序：{order[0]} → {order[1]}")
        for kv in order:
            kv_type = None if kv == "f16" else "q8_0"
            proc = None
            t0 = time.time()
            try:
                proc, log = srv.start(key, ctx, conc, tag=f"kv_{kv}_{cid}",
                                      kv_cache_type=kv_type, timeout_s=240)
                time.sleep(1.5)
                url = f"{cfg.BASE_URL}/v1/chat/completions"
                asyncio.run(lg.run_load(url, conc, max(2, conc), cfg.PROMPT_TOKENS, 16,
                                        timeout_s=600))
                time.sleep(1.0)
                for rep in range(REPEATS):
                    n_req = cfg.requests_per_repeat(conc)
                    req_rows = asyncio.run(lg.run_load(url, conc, n_req, cfg.PROMPT_TOKENS,
                                                       cfg.GEN_TOKENS, timeout_s=600))
                    agg = lg.aggregate(req_rows, conc)
                    rows.append(dict(cell_id=cid, model=model, quant=quant, ctx=ctx,
                                     conc=conc, kv_type=kv, repeat=rep,
                                     weight_bytes=cfg.MODEL_SPECS[key]["weight_bytes"],
                                     vram_est_gb=round(cfg.estimate_vram_gb(key, ctx, conc), 2),
                                     prompt_tokens=cfg.PROMPT_TOKENS, **agg))
                    print(f"    {kv:<5} 第{rep+1}轮  TPOT {agg['tpot_p50_ms']:7.1f} ms  "
                          f"吞吐 {agg['throughput_tps']:6.0f} t/s")
            except Exception as e:
                print(f"    {kv:<5} 失败：{str(e)[:100]}")
            finally:
                if proc is not None:
                    srv.stop(proc)
                print(f"      用时 {(time.time()-t0)/60:.1f} min")
                time.sleep(1.0)

    if not rows:
        raise SystemExit("没有有效数据")
    keys = list(rows[0].keys())
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with OUT.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f"\n已写入 {OUT}（{len(rows)} 行）")

    # 就地打印对照
    import collections
    import statistics as st
    g = collections.defaultdict(dict)
    for r in rows:
        g[r["cell_id"]].setdefault(r["kv_type"], []).append(r)
    print("\n" + "=" * 84)
    print(f"{'格':<26}{'f16 吞吐':>10}{'q8 吞吐':>10}{'变化':>8}"
          f"{'f16 TPOT':>10}{'q8 TPOT':>9}{'变化':>8}")
    def med(rowlist, key):
        vals = [r[key] for r in rowlist if isinstance(r.get(key), (int, float))]
        return st.median(vals) if vals else float("nan")

    for cid, d in g.items():
        if "f16" not in d or "q8_0" not in d:
            continue
        a = {k: med(d["f16"], k) for k in ("throughput_tps", "tpot_p50_ms")}
        b = {k: med(d["q8_0"], k) for k in ("throughput_tps", "tpot_p50_ms")}
        dt = b["throughput_tps"] / a["throughput_tps"] - 1
        dp = b["tpot_p50_ms"] / a["tpot_p50_ms"] - 1
        print(f"{cid:<26}{a['throughput_tps']:>10.0f}{b['throughput_tps']:>10.0f}{dt*100:>7.0f}%"
              f"{a['tpot_p50_ms']:>10.1f}{b['tpot_p50_ms']:>9.1f}{dp*100:>7.0f}%")


if __name__ == "__main__":
    main()

