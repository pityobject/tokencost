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
import hashlib
import json
import statistics
import sys
import time
import traceback

import tc_config as cfg
import tc_gpu as tc_gpu_mod
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
             warmup=2, settle_s=2.0, verbose=True, requests=None, suffix="",
             prompt_tokens=None, length_mix=None):
    """跑一格，返回这一格每一轮的指标列表。失败时返回带 status='failed' 的单行。"""
    key = (model, quant)
    repeats = repeats or cfg.REPEATS
    # requests 显式给定时用它：做"波次对照"和"固定波次"实验必须绕开默认值
    n_req = requests if requests else cfg.requests_per_repeat(conc)
    # 提示长度可覆盖：TTFT 的 prefill 项随 L 线性增长，而原数据集只有一种长度，
    # 不做这个维度就永远在把"长文档场景"当外推。
    prompt_tokens = int(prompt_tokens or cfg.PROMPT_TOKENS)

    # 【E2】混合请求长度：{长度: 权重}。每个请求独立抽样，与固定长度形成单变量对照。
    if isinstance(length_mix, str):
        length_mix = lg.parse_length_mix(length_mix)
    if length_mix:
        # 【指纹必须进 id】不同混合分布若共用 `_mix` 后缀，两次实验会写进同一格、
        # 后者覆盖前者的身份（2026-11-05 实际发生过）。
        key_s = ",".join(f"{k}:{v:.4f}" for k, v in sorted(length_mix.items()))
        fp = hashlib.sha1(key_s.encode()).hexdigest()[:4]
        suffix = (suffix or "") + f"_mix{fp}"
        if verbose:
            mean_L = sum(k * v for k, v in length_mix.items())
            print(f"    [混合长度] {length_mix}（加权平均 {mean_L:.0f} token，指纹 {fp}）",
                  flush=True)
        prompt_tokens = int(round(mean_L))   # 用于显存/上下文校验与记录

    if prompt_tokens != cfg.PROMPT_TOKENS and not suffix:
        suffix = f"_p{prompt_tokens}"        # 自动加后缀，避免覆盖已有格
    cid = cfg.cell_id(model, quant, ctx, conc, suffix)

    # 【守卫】同一 id 不允许承载不同的混合分布（否则身份会被覆盖）
    if length_mix:
        raw = cfg.DATA_DIR / "raw_cells.jsonl"
        if raw.exists():
            import json as _json
            want = ",".join(f"{k}:{v:.4f}" for k, v in sorted(length_mix.items()))
            for line in raw.open(encoding="utf-8"):
                line = line.strip()
                if not line:
                    continue
                try:
                    r = _json.loads(line)
                except Exception:
                    continue
                if r.get("cell_id") == cid and r.get("length_mix") not in (None, want):
                    raise SystemExit(
                        f"[冲突] {cid} 已存在，但混合分布不同："
                        f"已有 {r.get('length_mix')}，本次 {want}。"
                        f"指纹机制本应避免这种情况，请检查 make_prompt 或后缀逻辑。")

    # 【防呆（已纠正）】llama.cpp 的 --ctx-size 会按 --parallel 平分；
    # 而本项目的启动脚本传的是 `ctx × 并发`，所以每槽恰好 = ctx（日志里的 n_ctx_slot 就是它）。
    # 于是：每槽够不够看 `prompt+生成 <= ctx`；总显存看 `ctx × 并发`。
    # 真正的风险是 KV 总量 = ctx × 并发，超显存就会静默降级（本项目的"性能悬崖"）：
    # ctx 24576 × 8 槽 = 7.25 GB KV → 整体慢 5 倍且不报错。
    need = prompt_tokens + cfg.GEN_TOKENS
    if need > int(ctx):
        raise SystemExit(
            f"[无效实验] prompt {prompt_tokens} + 生成 {cfg.GEN_TOKENS} = {need} token，"
            f"超过 ctx={ctx}，服务端会截断。")
    spec_kv = cfg.kv_bytes_per_token(cfg.MODEL_SPECS[key])
    kv_gb = spec_kv * int(ctx) * max(int(conc), 1) / 1e9
    w_gb = cfg.MODEL_SPECS[key]["weight_bytes"] / 1e9
    if kv_gb + w_gb > 6.5:
        print(f"    ⚠️ 显存预警：KV≈{kv_gb:.1f} GB + 权重≈{w_gb:.1f} GB，"
              f"在 8 GB 卡上很可能触发静默降级（吞吐掉数倍且不报错）。"
              f"建议减小 ctx 或并发。", flush=True)

    spec = cfg.MODEL_SPECS[key]
    # decode 阶段平均要读多长的 KV cache：提示长度 + 生成到一半
    ctx_avg = prompt_tokens + cfg.GEN_TOKENS / 2.0

    base = dict(
        cell_id=cid, model=model, quant=quant, ctx=ctx, conc=conc,
        weight_bytes=spec["weight_bytes"], n_params=spec["n_params"],
        ctx_avg_tokens=ctx_avg,
        vram_est_gb=round(cfg.estimate_vram_gb(key, ctx, conc), 2),
        requests=n_req, gen_tokens=cfg.GEN_TOKENS,
        prompt_tokens_nominal=prompt_tokens,
    **({"length_mix": ",".join(f"{k}:{v:.4f}" for k, v in sorted(length_mix.items()))}
       if length_mix else {}),
    )

    proc, log_path = None, None
    rows = []
    try:
        t_start = time.time()
        # 【背景噪声闸门】此刻服务还没起，GPU 上的占用全部来自"别的程序"：
        # 壁纸引擎、浏览器硬件加速、QQ 音乐、游戏启动器、甚至**别的 DSH 会话**
        #（它们可能自己起了 llama-server —— 那是显存 + SM + CPU 三重争用）。
        # 记录进每一行，事后可以筛出被污染的格。
        try:
            bg = tc_gpu_mod.idle_baseline()
        except Exception as _e:
            bg = dict(bg_level="unknown",
                      bg_notes=f"采样异常：{type(_e).__name__}: {str(_e)[:120]}",
                      bg_mem_mb=None, bg_util_p95=None, bg_clock_mhz=None)
        if bg.get("bg_level") in ("warn", "bad"):
            print(f"    ⚠️ 背景占用（{bg.get('bg_level')}）：{bg.get('bg_notes')}", flush=True)
            if bg.get("bg_level") == "bad":
                print("       建议先关掉：别的 DSH 会话 / 浏览器 / 游戏 / 壁纸引擎，再重跑这一格。",
                      flush=True)
        # 首次加载要把权重读进显存：3B 约 5–15 秒，7B 常要 30–60 秒。
        # 不加这句提示很容易被误判成卡死、然后被手动中断——中断就白跑了（无日志、无落盘）。
        if verbose:
            est = "30–60 秒" if model in ("7B", "Llama1B") else "5–15 秒"
            print(f"    正在启动 llama-server（{model} 首次加载约 {est}，请勿中断）...", flush=True)
        proc, log_path = srv.start(key, ctx, conc, tag=cid)
        if verbose:
            print(f"    服务就绪 {time.time()-t_start:.1f}s  日志 {log_path.name}")
        time.sleep(settle_s)          # 让显存分配、频率稳定下来

        url = f"{cfg.BASE_URL}/v1/chat/completions"
        # 预热（结果丢弃）
        if warmup:
            asyncio.run(lg.run_load(url, conc, max(warmup, 1) * conc,
                                    prompt_tokens, 16, timeout_s=180,
                                    length_mix=length_mix))
            time.sleep(1.0)

        # 【E1】监控粒度为"每一轮"，不是整个 cell。
        # 原因：只有逐轮的温度/时钟才能区分两种情况——
        #   ① 热状态：同一批内越跑越慢（温度上升、时钟下降）
        #   ② 批次状态：批内平稳、批与批之间跳变（显存碎片/驱动状态）
        # 之前 GpuMonitor 包住整个 cell，所有轮共享一个 summary，这两种情况分不开。
        for r in range(repeats):
            with GpuMonitor(interval_s=1.0) as mon:
                t0 = time.perf_counter()
                cpu0 = _client_cpu_seconds()
                req_rows = asyncio.run(lg.run_load(
                    url, conc, n_req, prompt_tokens, cfg.GEN_TOKENS,
                    timeout_s=max(300, 20 * n_req), length_mix=length_mix,
                    seed=r))
                cpu1 = _client_cpu_seconds()
                dur = max(time.perf_counter() - t0, 1e-9)
                # 客户端占了多少 CPU（单核百分比）。高并发时如果它贴近 100%，
                # 说明测出来的是"我的 Python 脚本能跑多快"，而不是"显卡能跑多快"。
                client_cpu_pct = ((cpu1 - cpu0) / dur * 100.0) if cpu0 is not None else None
                agg = lg.aggregate(req_rows, conc)
                gpu = mon.summary()
            # 【W1】时钟相对判据：记录可达参考值下的比值，不做静默剔除
            try:
                clk = gpu.get("clock_sm_median") if isinstance(gpu, dict) else None
                ratio, gate, note = tc_gpu_mod.clock_gate(clk, model)
                gpu["clock_ref_mhz"] = (tc_gpu_mod.clock_reference() or {}).get(model)
                gpu["clock_ratio"] = ratio
                gpu["clock_gate"] = gate
                gpu["clock_note"] = note
                if gate in ("warn", "bad"):
                    print(f"    ⚠️ 第{r+1}轮时钟偏低（{gate}）：{note}", flush=True)
            except Exception as _e:
                gpu["clock_gate"] = "unknown"
                gpu["clock_note"] = f"判定失败：{type(_e).__name__}: {str(_e)[:80]}"
            row = dict(base, repeat=r, status="ok", **bg, **gpu,
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
                      f"  温度 {gpu.get('temp_c_median', float('nan')):.0f}°C"
                      f"  时钟 {gpu.get('clock_sm_median', float('nan')):.0f}MHz"
                      + (f"  客户端CPU {client_cpu_pct:.0f}%" if client_cpu_pct else ""))
                if client_cpu_pct and client_cpu_pct > 90:
                    print("      [注意] 客户端 CPU 接近单核饱和：这一格的吞吐可能被我的脚本限制，"
                          "不是模型的上限。报告里要标注。")

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



# ================================================================ 多轮（E3）
def run_cell_turns(model, quant, ctx, conc, turns, prompt_tokens=None,
                   verbose=True, requests=None, save_requests=False):
    """多轮：一条会话跑 turns 轮，每轮产出一行（repeat = 轮次）。

    与单轮的差别：每轮输入包含前面全部历史，**KV 随之累积**。
    这正是模型的单轮假设要接受检验的地方。
    """
    key = (model, quant)
    prompt_tokens = int(prompt_tokens or cfg.PROMPT_TOKENS)
    n_sess = requests if requests else cfg.requests_per_repeat(conc)
    cid = cfg.cell_id(model, quant, ctx, conc, f"_mt{turns}")

    # 上下文守卫：最坏情况 = 首轮 + (turns-1) 轮的增长（生成 + 追问）
    worst = prompt_tokens + (turns - 1) * (cfg.GEN_TOKENS + 24) + cfg.GEN_TOKENS
    if worst > int(ctx):
        raise SystemExit(
            f"[无效实验] {turns} 轮后前缀最坏约 {worst} token，超过每槽 ctx={ctx}。"
            f"请提高 ctx 或减少轮数。")

    base = dict(cell_id=cid, model=model, quant=quant, ctx=ctx, conc=conc,
                weight_bytes=0, n_params=0,
                ctx_avg_tokens=prompt_tokens + cfg.GEN_TOKENS / 2,
                vram_est_gb=round(cfg.estimate_vram_gb(key, ctx, conc), 2),
                requests=n_sess, gen_tokens=cfg.GEN_TOKENS,
                prompt_tokens_nominal=prompt_tokens, turns=turns)

    proc = None
    rows = []
    try:
        proc, log_path = srv.start(key, ctx, conc, tag=cid)
        if verbose:
            print(f"    [多轮] {turns} 轮 × {n_sess} 条会话，逐轮记录前缀长度", flush=True)
        time.sleep(2.0)
        url = f"{cfg.BASE_URL}/v1/chat/completions"
        res = asyncio.run(lg.run_turns(url, conc, n_sess, turns,
                                       prompt_tokens, cfg.GEN_TOKENS,
                                       verbose=verbose))
        for r in res:
            if not r.get("ok"):
                rows.append(dict(base, repeat=r["turn"], status="failed",
                                 error="该轮没有成功的请求"))
                continue
            rows.append(dict(base, repeat=r["turn"], status="ok",
                             prompt_tokens_median=r.get("prefix_tokens_median"),
                             ttft_p50_ms=r["ttft_p50_ms"],
                             tpot_p50_ms=r["tpot_p50_ms"],
                             n_ok=r["n_ok"]))
            if verbose:
                print(f"    第{r['turn']}轮  前缀≈{r.get('prefix_tokens_median')} token  "
                      f"TTFT {r['ttft_p50_ms']:7.1f}ms  TPOT {r['tpot_p50_ms']:6.2f}ms",
                      flush=True)
    except Exception as e:
        rows.append(dict(base, repeat=-1, status="failed",
                         error=f"{type(e).__name__}: {e}"))
    finally:
        if proc is not None:
            srv.stop(proc)
    # 【必须落盘】单轮路径在 run_cell 内部写 raw_cells；
    # 多轮分支漏了这一步 → 控制台有输出、分析却说"没有数据"（2026-11-05 实际发生）。
    _append_jsonl(RAW_CELLS, rows)
    if verbose:
        print(f"    已写入 {len(rows)} 行到 raw_cells.jsonl（每轮一行）", flush=True)

    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("quant")
    ap.add_argument("ctx", type=int)
    ap.add_argument("conc", type=int)
    ap.add_argument("--repeats", type=int, default=1, help="调试时用 1 轮")
    ap.add_argument("--turns", type=int, default=0,
                    help="多轮对话的轮数；>1 时每轮产出一行（repeat=轮次）")
    ap.add_argument("--length-mix", default=None,
                    help="混合请求长度，如 '256:0.7,1024:0.2,2048:0.1'（长度:权重）")
    ap.add_argument("--prompt-tokens", type=int, default=None,
                    help="提示长度（默认 cfg.PROMPT_TOKENS=256）；"
                         "用于补 TTFT 随 prompt 长度的斜率，会自动加 _pNNN 后缀")
    args = ap.parse_args()

    cfg.ensure_dirs()
    print(f"=== 单格实验：{args.model} {args.quant} ctx={args.ctx} 并发={args.conc} ===")
    if args.turns and args.turns > 1:
        rows = run_cell_turns(args.model, args.quant, args.ctx, args.conc,
                              args.turns, prompt_tokens=args.prompt_tokens)
    else:
        rows = run_cell(args.model, args.quant, args.ctx, args.conc,
                        repeats=args.repeats, prompt_tokens=args.prompt_tokens,
                        length_mix=args.length_mix)
    print("\n结果：")
    for r in rows:
        for k, v in r.items():
            if k in ("server_log_tail",):
                continue
            print(f"  {k:<24} {v if isinstance(v, str) else (f'{v:.4f}' if isinstance(v, float) else v)}")


if __name__ == "__main__":
    main()




