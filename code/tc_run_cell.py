"""
tc_run_cell.py —— 跑完"一格"实验（一个固定组合：模型+量化+上下文+并发）。

一格的流程（每一步都有原因，别省）：
  1. 启动 llama-server（上下文和并发槽位只能在启动时定）
  2. 等 /health 就绪
  3. 预热：发 2 个请求丢掉 —— 第一次推理要编译 CUDA kernel、预热缓存，特别慢
  4. 正式跑 REPEATS 轮，每轮 requests_per_repeat 个请求
  5. 每轮汇总成一行指标；同时记录 GPU 温度/频率
  6. 关掉服务

用法（单格手动跑，调试用）：
  & "D:\\llm\\venv\\Scripts\\python.exe" tc_run_cell.py 3B q4_k_m 2048 4
"""

import argparse
import asyncio
import json
import statistics
import sys
import time
import traceback

import tc_config as cfg
import tc_loadgen as lg
import tc_server as srv
from tc_gpu import GpuMonitor

RAW_CELLS = cfg.DATA_DIR / "raw_cells.jsonl"
RAW_REQS = cfg.DATA_DIR / "raw_requests.jsonl"


def _append_jsonl(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _client_cpu_seconds():
    """本进程累计 CPU 时间（秒）。用来判断"是不是客户端自己成了瓶颈"。"""
    try:
        import psutil
        t = psutil.Process().cpu_times()
        return t.user + t.system
    except Exception:
        return None


def run_cell(model, quant, ctx, conc, repeats=None, save_requests=True,
             warmup=2, settle_s=2.0, verbose=True, requests=None, suffix=""):
    """跑一格，返回这一格每一轮的指标列表。失败时返回带 status='failed' 的单行。"""
    key = (model, quant)
    repeats = repeats or cfg.REPEATS
    # requests 显式给定时用它：做"波次对照"和"固定波次"实验必须绕开默认值
    n_req = requests if requests else cfg.requests_per_repeat(conc)
    cid = cfg.cell_id(model, quant, ctx, conc, suffix)

    spec = cfg.MODEL_SPECS[key]
    # decode 阶段平均要读多长的 KV cache：提示长度 + 生成到一半
    ctx_avg = cfg.PROMPT_TOKENS + cfg.GEN_TOKENS / 2.0

    base = dict(
        cell_id=cid, model=model, quant=quant, ctx=ctx, conc=conc,
        weight_bytes=spec["weight_bytes"], n_params=spec["n_params"],
        ctx_avg_tokens=ctx_avg,
        vram_est_gb=round(cfg.estimate_vram_gb(key, ctx, conc), 2),
        requests=n_req, gen_tokens=cfg.GEN_TOKENS,
    )

    proc, log_path = None, None
    rows = []
    try:
        t_start = time.time()
        proc, log_path = srv.start(key, ctx, conc, tag=cid)
        if verbose:
            print(f"    服务就绪 {time.time()-t_start:.1f}s  日志 {log_path.name}")
        time.sleep(settle_s)          # 让显存分配、频率稳定下来

        url = f"{cfg.BASE_URL}/v1/chat/completions"
        # 预热（结果丢弃）
        if warmup:
            asyncio.run(lg.run_load(url, conc, max(warmup, 1) * conc,
                                    cfg.PROMPT_TOKENS, 16, timeout_s=180))
            time.sleep(1.0)

        with GpuMonitor(interval_s=2.0) as mon:
            for r in range(repeats):
                t0 = time.perf_counter()
                cpu0 = _client_cpu_seconds()
                req_rows = asyncio.run(lg.run_load(
                    url, conc, n_req, cfg.PROMPT_TOKENS, cfg.GEN_TOKENS,
                    timeout_s=max(300, 20 * n_req)))
                cpu1 = _client_cpu_seconds()
                dur = max(time.perf_counter() - t0, 1e-9)
                # 客户端占了多少 CPU（单核百分比）。高并发时如果它贴近 100%，
                # 说明测出来的是"我的 Python 脚本能跑多快"，而不是"显卡能跑多快"。
                client_cpu_pct = ((cpu1 - cpu0) / dur * 100.0) if cpu0 is not None else None
                agg = lg.aggregate(req_rows, conc)
                row = dict(base, repeat=r, status="ok",
                           # 注意：这里的 wall_s 是"含客户端开销的整轮耗时"，
                           # 与 agg 里的 measure_span_s（纯测量窗口）含义不同，故意分开命名
                           wall_s=round(dur, 2), client_cpu_pct=client_cpu_pct, **agg)
                rows.append(row)
                if save_requests:
                    _append_jsonl(RAW_REQS, [dict(base, repeat=r, **x) for x in req_rows])
                if verbose:
                    print(f"    第{r+1}/{repeats}轮  "
                          f"TTFT {agg.get('ttft_p50_ms', float('nan')):7.1f}ms  "
                          f"TPOT {agg.get('tpot_p50_ms', float('nan')):6.1f}ms  "
                          f"吞吐 {agg.get('throughput_tps', float('nan')):7.1f} tok/s"
                          + (f"  客户端CPU {client_cpu_pct:.0f}%" if client_cpu_pct else ""))
                    if client_cpu_pct and client_cpu_pct > 90:
                        print("      [注意] 客户端 CPU 接近单核饱和：这一格的吞吐可能被我的脚本限制，"
                              "不是模型的上限。报告里要标注。")

        gpu = mon.summary()
        for row in rows:
            row.update(gpu)

        # 当场判定这次测量是否有效：时钟与功耗双低 = 降频
        clk = gpu.get('clock_sm_median', 1e9)
        pwr = gpu.get('power_w_median', 1e9)
        valid = not (clk < cfg.MIN_SM_CLOCK_MHZ and pwr < cfg.MIN_POWER_W)
        for row in rows:
            row['valid'] = valid
        if not valid:
            print(f"    [无效] 这一格在降频状态下测得（SM {clk:.0f} MHz / {pwr:.0f} W）——"
                  f"没插电或电源计划不是「最佳性能」。数据作废，重跑前先修环境。")
    except Exception as e:
        # 失败也要落盘：这些"跑不了"的格本身就是重要证据（显存约束剪枝）
        tail = ""
        if log_path and log_path.exists():
            tail = log_path.read_text(encoding="utf-8", errors="replace")[-800:]
        rows = [dict(base, repeat=-1, status="failed",
                     error=f"{type(e).__name__}: {e}", server_log_tail=tail)]
        if verbose:
            print(f"    这个格失败了：{e}")
            if "显存" in str(e) or "启动即退出" in str(e):
                print(f"    → 属于显存约束剪枝（估计需要 {base['vram_est_gb']} GB），不是你的错。")
    finally:
        if proc is not None:
            srv.stop(proc)

    _append_jsonl(RAW_CELLS, rows)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("quant")
    ap.add_argument("ctx", type=int)
    ap.add_argument("conc", type=int)
    ap.add_argument("--repeats", type=int, default=1, help="调试时用 1 轮")
    args = ap.parse_args()

    cfg.ensure_dirs()
    print(f"=== 单格实验：{args.model} {args.quant} ctx={args.ctx} 并发={args.conc} ===")
    rows = run_cell(args.model, args.quant, args.ctx, args.conc, repeats=args.repeats)
    print("\n结果：")
    for r in rows:
        for k, v in r.items():
            if k in ("server_log_tail",):
                continue
            print(f"  {k:<24} {v if isinstance(v, str) else (f'{v:.4f}' if isinstance(v, float) else v)}")


if __name__ == "__main__":
    main()
