# -*- coding: utf-8 -*-
"""
30 秒上手（`tc.py demo`）：不标定、不装服务，直接看一个场景的成本与延迟

## 为什么需要

现在的路径是"装服务器 → 8 点标定 → 6 参数 → 推荐"，第一次打开项目的人
要先读三份文档才知道价值在哪。这个命令用**已验证的先验参数**跑一个真实场景，
30 秒内给出"这个配置大概多贵、能不能过 SLA、离物理下界多远"。

## 用法

    python code/tc_demo.py                 # 默认：3B fp16 on A10，通用问答
    python code/tc_demo.py --hw A10 --model 3B --quant fp16
    python code/tc_demo.py --sla-ttft 500 --sla-tpot 40
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import tc_config as cfg
import tc_model as tm

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# 已验证的先验（来自 tc_priors 的八份标定；范围的中点）
PRIORS = {
    ("A10", "fp16", "llama.cpp"): dict(bw=471.5, peak=11.33, peak_prefill=39.78,
                                       c_srv=1.529, b_sat=7.17, ovh=13.2),
    ("A10", "fp16", "vllm"): dict(bw=384.9, peak=13.83, peak_prefill=76.69,
                                  c_srv=0.273, b_sat=21.87, ovh=11.87),
    ("V100", "fp16", "llama.cpp"): dict(bw=775.5, peak=15.0, peak_prefill=37.67,
                                        c_srv=3.848, b_sat=7.24, ovh=30.37),
    ("4060", "q4_k_m", "llama.cpp"): dict(bw=249.1, peak=13.16, peak_prefill=31.92,
                                          c_srv=0.766, b_sat=13.2, ovh=15.96),
}
NOMINAL_BW = {"A10": 600.0, "V100": 900.0, "4060": 272.0}
SPECS = {"3B": ("3B", "fp16"), "1.5B": ("1.5B", "q4_k_m")}

# 本地 MODEL_SPECS 里没有 fp16 变体（云端脚本是临时注册的），这里补齐，
# 否则 predict() 会 KeyError —— 第一次跑 demo 就会踩到。
cfg.MODEL_SPECS.setdefault(("3B", "fp16"), dict(
    weight_bytes=6178317312, n_params=3080000000, n_layers=36,
    n_kv_heads=2, head_dim=128))
cfg.MODEL_SPECS.setdefault(("1.5B", "fp16"), dict(
    weight_bytes=3087007744, n_params=1540000000, n_layers=28,
    n_kv_heads=2, head_dim=128))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hw", default="A10", choices=["A10", "V100", "4060"])
    ap.add_argument("--model", default="3B", choices=["3B", "1.5B"])
    ap.add_argument("--backend", default=None,
                    help="llama.cpp / vllm（默认按硬件选已验证的那个）")
    ap.add_argument("--prompt", type=int, default=512, help="prompt 长度")
    ap.add_argument("--gen", type=int, default=128)
    ap.add_argument("--conc", type=int, default=8)
    ap.add_argument("--price", type=float, default=10.5, help="元/小时（默认 A10 按量）")
    ap.add_argument("--sla-ttft", type=float, default=None)
    ap.add_argument("--sla-tpot", type=float, default=None)
    args = ap.parse_args()

    model, quant = SPECS[args.model]
    default_backend = "llama.cpp" if args.hw != "A10" else "vllm"
    backend = args.backend or default_backend
    key = (args.hw, quant, backend)
    if key not in PRIORS:
        # 回退：同硬件同栈用 fp16 的参数
        alt = [(k, v) for k, v in PRIORS.items() if k[0] == args.hw and k[2] == backend]
        if not alt:
            print(f"没有 {args.hw} + {backend} 的先验；可用：")
            for k in PRIORS:
                print(f"  {k[0]} / {k[1]} / {k[2]}")
            return
        key = alt[0][0]
    p = PRIORS[key]

    print("=" * 78)
    print("TokenCost 30 秒演示（用已验证的先验，不标定）")
    print("=" * 78)
    print(f"  硬件 {args.hw}｜栈 {backend}｜模型 {model}-{quant}｜"
          f"prompt {args.prompt} tok｜生成 {args.gen} tok｜并发 {args.conc}")

    row = dict(model=model, quant=quant, ctx=2048, conc=args.conc,
               prompt_tokens=args.prompt,
               ctx_avg_tokens=args.prompt + args.gen / 2)
    params = dict(bw_gbs=p["bw"], peak_tflops=p["peak"],
                  peak_prefill_tflops=p["peak_prefill"], c_srv_ms=p["c_srv"],
                  b_sat=p["b_sat"], ovh_ttft_ms=p["ovh"])
    pr = tm.predict(params, row)

    ttft, tpot = pr["pred_ttft_ms"], pr["pred_tpot_ms"]
    lifetime = ttft + args.gen * tpot
    # 单位成本：1e6 token 需要 1e6/(conc×gen) 个"批次"
    thr = args.conc * args.gen / lifetime * 1e3
    cost_per_1m = 1e6 / thr / 3600 * args.price

    # 物理下界
    spec = cfg.MODEL_SPECS[(model, quant)]
    kv = 2 * 2 * spec["head_dim"] * spec["n_layers"] * 2
    floor_tpot = (spec["weight_bytes"] + kv * row["ctx_avg_tokens"] * args.conc) \
        / (NOMINAL_BW[args.hw] * 1e9) * 1e3

    print(f"\n  预测 TTFT      {ttft:>8.1f} ms")
    print(f"  预测 TPOT      {tpot:>8.1f} ms  （物理下界 {floor_tpot:.1f} ms，"
          f"离下界 {tpot/floor_tpot:.2f}×）")
    print(f"  吞吐           {thr:>8.1f} tok/s")
    print(f"  单位成本       {cost_per_1m:>8.2f} 元/百万 token（按 {args.price} 元/时）")

    if args.sla_ttft or args.sla_tpot:
        ok_ttft = (ttft <= args.sla_ttft) if args.sla_ttft else True
        ok_tpot = (tpot <= args.sla_tpot) if args.sla_tpot else True
        status = "✅ 满足" if (ok_ttft and ok_tpot) else "❌ 不满足"
        print(f"\n  SLA 判定 {status}"
              f"（TTFT ≤ {args.sla_ttft or '—'} / TPOT ≤ {args.sla_tpot or '—'} ms）")
        if not (ok_ttft and ok_tpot):
            print("  想达标：降并发、换更快的栈（vLLM）、或把 `-ub` 调大（高并发 TTFT 可由 -ub 改善 5.6 倍）")

    print("\n  这些数字从哪来：三块 GPU × 两个栈的 8 份实测标定（results/a10_raw/）")
    print("  想更准：在你的机器上跑 8 点标定（约 10 分钟）")
    print("      python code/tc_profile.py --base-url http://127.0.0.1:8000/v1 --spec ...")
    print("  想看完整推荐（含质量-成本权衡）：")
    print("      python code/tc_recommend.py --help")
    print()


if __name__ == "__main__":
    main()
