"""
tc_smoke_test.py —— 第一次"真模型 + 真测量"的 10 秒体检。

它做的事（每一步都会打印，方便你看到进度）：
  1. 用 3B-Q4 起一个 llama-server（上下文 2048、1 个槽位）
  2. 预热 1 个请求（丢弃）
  3. 正式发 3 个请求，逐 token 打时间戳，算 TTFT / TPOT / 吞吐
  4. 用"权重字节 ÷ TPOT"反推有效带宽，并和标称值比
  5. 关掉服务

为什么要有这个脚本：环境自检只证明"文件在、能启动"。
这里才第一次证明**整条测量链路**（HTTP 流式 → 时间戳 → 指标）在你机器上是通的。

用法：
  & "D:\\llm\\venv\\Scripts\\python.exe" tc_smoke_test.py
  & "D:\\llm\\venv\\Scripts\\python.exe" tc_smoke_test.py --model 7B --quant q4_k_m
"""

import argparse
import asyncio
import sys
import time

import tc_config as cfg
import tc_loadgen as lg
import tc_server as srv

NOMINAL_BW_GBS = 256.0     # RTX 4060 Laptop 标称带宽量级（桌面版 272，笔记本常为 256）


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="3B")
    ap.add_argument("--quant", default="q4_k_m")
    ap.add_argument("--ctx", type=int, default=2048, help="每槽位上下文")
    ap.add_argument("--conc", type=int, default=1)
    ap.add_argument("--requests", type=int, default=3, help="正式测量的请求数")
    args = ap.parse_args()

    key = (args.model, args.quant)
    if key not in cfg.MODEL_SPECS:
        print(f"没有 {key} 这个组合，可选：{list(cfg.MODEL_SPECS)}")
        sys.exit(1)
    spec = cfg.MODEL_SPECS[key]
    if not cfg.model_path(key).exists():
        print(f"模型文件不存在：{cfg.model_path(key)}")
        sys.exit(1)

    w_gb = spec["weight_bytes"] / 1e9
    print("=" * 66)
    print(f"冒烟测试：{args.model} {args.quant}  权重 {w_gb:.3f} GB"
          f"  每槽位 ctx={args.ctx}  并发={args.conc}")
    print(f"模型目录：{cfg.LLAMA_BIN}")
    print("=" * 66)

    proc = None
    t_all = time.time()
    try:
        print("[1/4] 启动 llama-server ...")
        proc, log = srv.start(key, args.ctx, args.conc, tag="smoke")
        print(f"      就绪（{time.time()-t_all:.1f}s），日志：{log.name}")

        url = f"{cfg.BASE_URL}/v1/chat/completions"

        print("[2/4] 预热 1 个请求（结果丢弃）...")
        asyncio.run(lg.run_load(url, args.conc, max(args.conc, 1),
                                cfg.PROMPT_TOKENS, 16, timeout_s=180))

        print(f"[3/4] 正式测量 {args.requests} 个请求 ...")
        t0 = time.time()
        rows = asyncio.run(lg.run_load(url, args.conc, args.requests,
                                       cfg.PROMPT_TOKENS, cfg.GEN_TOKENS,
                                       timeout_s=300, verbose=True))
        agg = lg.aggregate(rows, args.conc)
        print(f"      用时 {time.time()-t0:.1f}s")

        n_ok = agg.get("n_ok", 0)
        if n_ok == 0:
            print("\n[失败] 所有请求都没成功。看下面第一条错误的原文：")
            for e in [r.get("error") for r in rows if r.get("error")][:3]:
                print("   ", e)
            sys.exit(2)

        tpot = agg["tpot_p50_ms"]
        bw = w_gb / (tpot / 1000)
        print("\n[4/4] 结果")
        print("-" * 66)
        print(f"  成功请求            {n_ok} / {len(rows)}")
        print(f"  实际生成 token 中位数 {agg.get('n_out_median'):.0f}")
        print(f"  提示 token 中位数     {agg.get('prompt_tokens_median'):.0f}")
        print(f"  TTFT（首字）P50      {agg['ttft_p50_ms']:.1f} ms   "
              f"P90 {agg['ttft_p90_ms']:.1f} ms")
        print(f"  TPOT（出字间隔）P50   {tpot:.1f} ms   "
              f"P90 {agg['tpot_p90_ms']:.1f} ms")
        print(f"  吞吐                {agg['throughput_tps']:.1f} token/s")
        print(f"  单位成本             {agg['cost_gpu_s_per_1m']:.0f} GPU 秒 / 百万 token")
        if agg.get("client_cpu_pct") is not None:
            print(f"  客户端 CPU           {agg['client_cpu_pct']:.0f}%")
        print("-" * 66)
        print(f"  反推有效带宽         {bw:.0f} GB/s  "
              f"（标称约 {NOMINAL_BW_GBS:.0f} GB/s，占 {bw/NOMINAL_BW_GBS*100:.0f}%）")
        print(f"  等价地：每 token 读 {w_gb:.3f} GB 权重，用了 {tpot:.1f} ms")
        print("-" * 66)

        verdict = []
        # 阈值按 4060 的真实能力设：1.5B 能到 5 ms/token，7B 约 22 ms/token。
        # 原来写 20–200 ms 的下界是错的，会把正常的高速结果误报成异常。
        if not (3.0 <= tpot <= 400):
            verdict.append(f"TPOT {tpot:.1f} ms 明显偏离常见范围（3–400 ms）——先看日志里有没有 CUDA0 buffer")
        if bw < 80:
            verdict.append(f"带宽只有 {bw:.0f} GB/s，偏低——可能有层在 CPU 上跑，或正在降频")
        if bw > NOMINAL_BW_GBS * 1.15:
            verdict.append(f"带宽 {bw:.0f} GB/s 超过标称值，不合物理——检查权重字节数是否记错")
        if agg.get("n_out_median", 0) < 10:
            verdict.append(f"生成 token 只有 {agg['n_out_median']:.0f}，太少，TPOT 会不稳")
        elif abs(agg.get("n_out_median", 0) - cfg.GEN_TOKENS) > 2:
            verdict.append(f"生成 token {agg['n_out_median']:.0f} ≠ 设定值 {cfg.GEN_TOKENS}："
                           "ignore_eos 可能没生效（某些版本字段名不同），TPOT 仍可用但要记录实际长度")

        if verdict:
            print("注意事项：")
            for v in verdict:
                print("  -", v)
        else:
            print("判定：正常。这组数字可以直接作为 S2 的起点。")

        print(f"\n总耗时 {time.time()-t_all:.1f}s")
        print("下一步：按 手册/03_测量基础设施_S2.md 跑单格实验，然后用 tc_probe.py 体检槽位上下文。")

    finally:
        if proc is not None:
            srv.stop(proc)


if __name__ == "__main__":
    main()
