"""
tc_probe_cpu.py —— 迁移性实验：换一台"机器"（纯 CPU 后端）再标定一次。

## 为什么做

报告里有一句核心主张：**"模型结构跟硬件无关，搬到 A100 或昇腾只需重标定 BW 和 PEAK"**。
现在这只是主张，没有任何证据。CPU 后端正好是一个真正不同的 (BW, PEAK) 点：
  带宽  249 → 约 30–60 GB/s（低 5–8 倍）
  算力  31.9 → 约 0.3–0.8 TFLOP/s（低 40–100 倍）

## 怎么做

标定格（用来重新拟合两个常数）：1.5B 与 3B × ctx512 × 并发 1/4
验证格（不参与拟合）      ：1.5B 与 3B × ctx512 × 并发 8

然后回答三个问题：
  ① 只用 BW/PEAK 两个常数重标定，留出格还能不能预测？
  ② CPU 上拟合出的 c_srv 与 GPU 上的 0.766 差多少？（若接近 → c_srv 与计算后端无关）
  ③ 同一模型 CPU vs GPU 的吞吐比是多少？

## 时间

CPU 慢，所以每格只跑 2 轮、请求数压到 8/16。预计 15–25 分钟。

用法：
  & "D:\\llm\\venv\\Scripts\\python.exe" tc_probe_cpu.py
"""

import csv
import json
import sys
import time

import tc_config as cfg
import tc_loadgen as lg
import tc_server as srv

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# (模型, 量化, 上下文, 并发, 是否标定)
CELLS = [
    ("1.5B", "q4_k_m", 512, 1, True),
    ("1.5B", "q4_k_m", 512, 4, True),
    ("3B",   "q4_k_m", 512, 1, True),
    ("3B",   "q4_k_m", 512, 4, True),
    ("1.5B", "q4_k_m", 512, 8, False),   # 留出：验证
    ("3B",   "q4_k_m", 512, 8, False),
]
OUT = cfg.DATA_DIR / "probe_cpu.csv"
REPEATS = 2


def n_req_for(conc):
    """CPU 上要控制工作量：并发 1 跑 8 个请求，并发 4/8 跑 16 个。"""
    return 8 if conc == 1 else 16


def main():
    import asyncio
    rows = []
    print("=" * 76)
    print("CPU 迁移实验（-ngl 0，纯 CPU 后端）")
    print("=" * 76)
    for model, quant, ctx, conc, is_calib in CELLS:
        key = (model, quant)
        cid = f"{model}_{quant}_ctx{ctx}_c{conc}"
        n_req = n_req_for(conc)
        t0 = time.time()
        proc = None
        try:
            proc, log = srv.start(key, ctx, conc, tag=f"cpu_{cid}", ngl=0, timeout_s=600)
            time.sleep(1.0)
            url = f"{cfg.BASE_URL}/v1/chat/completions"
            # 预热
            asyncio.run(lg.run_load(url, conc, max(2, conc), cfg.PROMPT_TOKENS, 16, timeout_s=900))
            time.sleep(1.0)
            for rep in range(REPEATS):
                req_rows = asyncio.run(lg.run_load(url, conc, n_req, cfg.PROMPT_TOKENS,
                                                   cfg.GEN_TOKENS, timeout_s=1800))
                agg = lg.aggregate(req_rows, conc)
                row = dict(cell_id=cid, model=model, quant=quant, ctx=ctx, conc=conc,
                           group=("标定" if is_calib else "留出"),
                           weight_bytes=cfg.MODEL_SPECS[key]["weight_bytes"],
                           n_params=cfg.MODEL_SPECS[key]["n_params"],
                           prompt_tokens=cfg.PROMPT_TOKENS, repeat=rep, backend="cpu",
                           **agg)
                rows.append(row)
                print(f"  {cid:<26} 第{rep+1}轮  TTFT {agg['ttft_p50_ms']:8.1f} ms  "
                      f"TPOT {agg['tpot_p50_ms']:7.2f} ms  吞吐 {agg['throughput_tps']:6.1f} t/s")
        except Exception as e:
            print(f"  {cid:<26} 失败：{e}")
        finally:
            if proc is not None:
                srv.stop(proc)
            print(f"    用时 {(time.time()-t0)/60:.1f} min\n")
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
    print(f"已写入 {OUT}（{len(rows)} 行）")
    print("下一步： python tc_analyze_cpu.py")


if __name__ == "__main__":
    main()
