# -*- coding: utf-8 -*-
"""
前缀复用实验（`tc.py prefixprobe`）：验证"有效 prefill 长度 ≠ prompt_tokens"

## 背景

2026-11-07 在 V100 + llama.cpp 上做多轮测量时发现：TTFT 被系统性高估 **30–47%**。
猜测原因是多轮请求的 prompt 高度重叠，服务端复用了历史前缀的 KV，
而模型假设整个 prompt 都要重新 prefill。

## 实验设计

对**同一个长 prompt** 连发 N 次，逐次记录 TTFT：

| 变体 | 期望 |
|---|---|
| 第 1 次（冷） | 全量 prefill → TTFT 高 |
| 第 2..N 次（重复同一 prompt） | 若服务端复用 KV → TTFT 显著下降 |
| 对照组：每次换一段不重叠的 prompt | TTFT 应当**不下降** |

两者对比即可区分"前缀复用"与"第二次变快是因为其它原因（如 CUDA warmup）"。

## 用法

    # llama.cpp：分别用 cache_prompt=true / false 各跑一次
    python code/tc_measure_prefix.py --base-url http://127.0.0.1:8092/v1 \
        --backend llamacpp --cache-prompt off --length 1600 --repeats 4 \
        --hardware "A10 24GB / llama.cpp" --out results/prefix_a10_llamacpp_off.json

    # vLLM（默认不开前缀缓存，作为对照）
    python code/tc_measure_prefix.py --base-url http://127.0.0.1:8081/v1 \
        --wire-model qwen3b --backend vllm --length 1600 --repeats 4 \
        --hardware "A10 24GB / vLLM" --out results/prefix_a10_vllm.json
"""
import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

import httpx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import tc_config as cfg
import tc_loadgen as lg

sys.stdout.reconfigure(encoding="utf-8", errors="replace")


async def one(client, base_url, model, prompt, gen, backend, cache_prompt):
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": gen, "temperature": 0.0, "stream": True,
        "stream_options": {"include_usage": True}, "ignore_eos": True,
    }
    if backend != "vllm":
        payload["cache_prompt"] = bool(cache_prompt)
    t0 = time.perf_counter()
    first = None
    n_out = 0
    usage = {}
    try:
        async with client.stream("POST", f"{base_url}/chat/completions",
                                 json=payload, timeout=300) as r:
            r.raise_for_status()
            async for line in r.aiter_lines():
                if not line.startswith("data:"):
                    continue
                d = line[5:].strip()
                if not d or d == "[DONE]":
                    continue
                try:
                    obj = json.loads(d)
                except Exception:
                    continue
                if obj.get("usage"):
                    usage = obj["usage"]
                for ch in obj.get("choices") or []:
                    if (ch.get("delta") or {}).get("content"):
                        if first is None:
                            first = time.perf_counter()
                        n_out += 1
    except Exception:
        return None
    if first is None or n_out < 3:
        return None
    return dict(ttft_ms=(first - t0) * 1e3,
                prompt_tokens=int(usage.get("prompt_tokens") or 0),
                n_out=n_out)


def summarize(rows):
    tt = [r["ttft_ms"] for r in rows]
    return dict(n=len(rows),
                ttft_first=round(tt[0], 1),
                ttft_median_rest=round(float(np.median(tt[1:])), 1) if len(tt) > 1 else None,
                ttft_all=[round(x, 1) for x in tt])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--wire-model", default=None)
    ap.add_argument("--backend", choices=["llamacpp", "vllm"], default="llamacpp")
    ap.add_argument("--cache-prompt", choices=["on", "off"], default="off",
                    help="llama.cpp 专有；vLLM 忽略")
    ap.add_argument("--length", type=int, default=1600)
    ap.add_argument("--gen", type=int, default=32, help="只测 TTFT，用短生成即可")
    ap.add_argument("--repeats", type=int, default=4)
    ap.add_argument("--hardware", default="（未填）")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    model = args.wire_model
    if not model:
        d = httpx.get(f"{args.base_url}/models", timeout=10).json()
        e = d.get("data") or d.get("models") or []
        model = e[0].get("id") or e[0].get("model")
    cp = (args.cache_prompt == "on")
    print(f"目标 {args.base_url}｜模型 {model}｜backend {args.backend}｜"
          f"cache_prompt={args.cache_prompt}｜L={args.length}｜重复 {args.repeats}")

    same_prompt = lg.make_prompt(args.length, nonce=12345)

    async def run():
        async with httpx.AsyncClient() as client:
            same, diff = [], []
            for i in range(args.repeats):
                r = await one(client, args.base_url, model, same_prompt,
                              args.gen, args.backend, cp)
                if r:
                    same.append(r)
                print(f"  同一 prompt 第 {i+1} 次：TTFT={r['ttft_ms']:.1f} ms"
                      if r else f"  同一 prompt 第 {i+1} 次：失败")
            for i in range(args.repeats):
                p = lg.make_prompt(args.length, nonce=int(time.time() * 1000) % 10**9 + i * 7919)
                r = await one(client, args.base_url, model, p,
                              args.gen, args.backend, cp)
                if r:
                    diff.append(r)
                print(f"  不同 prompt 第 {i+1} 次：TTFT={r['ttft_ms']:.1f} ms"
                      if r else f"  不同 prompt 第 {i+1} 次：失败")
            return same, diff

    same, diff = asyncio.run(run())
    if len(same) < 2 or len(diff) < 2:
        print("样本不足")
        return

    s_same, s_diff = summarize(same), summarize(diff)
    drop = (1 - s_same["ttft_median_rest"] / s_same["ttft_first"]) * 100
    drop_ctl = (1 - s_diff["ttft_median_rest"] / s_diff["ttft_first"]) * 100

    print("\n" + "=" * 74)
    print("结果")
    print("=" * 74)
    print(f"  同一 prompt：第 1 次 {s_same['ttft_first']:.1f} ms → "
          f"后续中位 {s_same['ttft_median_rest']:.1f} ms（下降 {drop:.0f}%）")
    print(f"  不同 prompt：第 1 次 {s_diff['ttft_first']:.1f} ms → "
          f"后续中位 {s_diff['ttft_median_rest']:.1f} ms（下降 {drop_ctl:.0f}%）")
    if drop - drop_ctl > 20:
        verdict = (f"✅ 存在前缀复用：同 prompt 重复下降 {drop:.0f}%，"
                   f"而对照组只下降 {drop_ctl:.0f}%")
    elif drop > 20:
        verdict = (f"⚠️ 同 prompt 变快（{drop:.0f}%），但对照组也变快（{drop_ctl:.0f}%）"
                   "——不能归因于前缀复用，可能是缓存/预热效应")
    else:
        verdict = f"❌ 未见前缀复用（同 prompt 仅下降 {drop:.0f}%）"
    print("  " + verdict)

    out = dict(hardware=args.hardware, backend=args.backend, model=model,
               cache_prompt=args.cache_prompt, length=args.length,
               same_prompt=s_same, diff_prompt=s_diff,
               drop_same_pct=round(drop, 1), drop_control_pct=round(drop_ctl, 1),
               verdict=verdict,
               note="对照组用不重叠的新 prompt，用来排除'第二次都变快'的干扰")
    fp = Path(args.out) if args.out else (cfg.RESULTS_DIR / "prefix_reuse.json")
    fp.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n→ {fp}")


if __name__ == "__main__":
    main()
