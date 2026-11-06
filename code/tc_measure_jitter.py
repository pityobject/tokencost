# -*- coding: utf-8 -*-
"""
逐请求抖动测量（`tc.py jitterprobe`）：给"prefill 打断 decode 的排队效应"提供独立证据

## 为什么需要它

我们已有四条失效证据（长 prompt×高并发、混合长度、KV 系数 b>1、c_srv 中段鼓包），
F16/F17 猜测根因是**排队效应**：prefill 抢占 decode，使同一格内不同请求的
每字耗时（TPOT）出现抖动。

如果这个猜测成立，应当观察到：
1. **同格内 TPOT 的抖动随并发上升**；
2. 抖动的上升**在两个部署栈上都出现**（否则只是某个栈的实现细节）；
3. 抖动幅度足以解释 TTFT 的系统性偏差。

`tc_profile` 只记录每格的中位数，看不到抖动；本脚本逐请求记录，
输出 CV / p90·p50 比，作为独立检验量。

## 用法（云端或本机均可）

    python code/tc_measure_jitter.py --base-url http://127.0.0.1:8081/v1 \
        --wire-model qwen3b --backend vllm --concs 1,4,8,16,32 \
        --length 1900 --gen 128 --hardware "A10 24GB / vLLM" \
        --out results/jitter_a10_vllm.json

## 产物

每格：n / ttft_p50 / tpot_p50 / tpot_cv / tpot_p90_over_p50 / ttft_p90_over_p50
"""
import argparse
import asyncio
import json
import statistics
import sys
import time
from pathlib import Path

import httpx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import tc_config as cfg
import tc_loadgen as lg

sys.stdout.reconfigure(encoding="utf-8", errors="replace")


async def one_request(client, base_url, model, prompt, gen, backend):
    """返回 dict(ttft_ms, tpot_ms, n_out)；失败返回 None。"""
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": gen,
        "temperature": 0.0,
        "stream": True,
        "stream_options": {"include_usage": True},
        "ignore_eos": True,
    }
    if backend != "vllm":
        payload["cache_prompt"] = False      # llama.cpp 专有；vLLM 会 400
    t0 = time.perf_counter()
    first = None
    last = None
    n_out = 0
    try:
        async with client.stream("POST", f"{base_url}/chat/completions",
                                 json=payload, timeout=300) as r:
            r.raise_for_status()
            async for line in r.aiter_lines():
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if not data or data == "[DONE]":
                    continue
                try:
                    obj = json.loads(data)
                except Exception:
                    continue
                for ch in obj.get("choices") or []:
                    delta = (ch.get("delta") or {}).get("content")
                    if delta:
                        now = time.perf_counter()
                        if first is None:
                            first = now
                        last = now
                        n_out += 1
    except Exception:
        return None
    if first is None or last is None or n_out < 3:
        return None
    ttft_ms = (first - t0) * 1e3
    tpot_ms = (last - first) * 1e3 / (n_out - 1)
    return dict(ttft_ms=ttft_ms, tpot_ms=tpot_ms, n_out=n_out)


async def run_cell(base_url, model, conc, length, gen, n_requests, backend):
    prompt = lg.make_prompt(length, nonce=int(time.time() * 1000) % 10**9)
    sem = asyncio.Semaphore(conc)
    results = []

    async with httpx.AsyncClient() as client:
        async def task(i):
            async with sem:
                r = await one_request(client, base_url, model,
                                      f"{prompt}\n#{i}", gen, backend)
                if r:
                    results.append(r)
        await asyncio.gather(*(task(i) for i in range(n_requests)))
    return results


def summarize(results):
    if len(results) < 5:
        return None
    ttft = np.array([r["ttft_ms"] for r in results], dtype=float)
    tpot = np.array([r["tpot_ms"] for r in results], dtype=float)
    return dict(
        n=len(results),
        ttft_p50=round(float(np.median(ttft)), 2),
        tpot_p50=round(float(np.median(tpot)), 3),
        tpot_cv=round(float(statistics.pstdev(tpot) / np.mean(tpot)), 4),
        tpot_p90_over_p50=round(float(np.percentile(tpot, 90) / np.median(tpot)), 4),
        ttft_p90_over_p50=round(float(np.percentile(ttft, 90) / np.median(ttft)), 4),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", required=True, help="如 http://127.0.0.1:8081/v1")
    ap.add_argument("--wire-model", default=None, help="请求里带的模型名（默认取 /models 第一个）")
    ap.add_argument("--backend", choices=["llamacpp", "vllm"], default="llamacpp")
    ap.add_argument("--concs", default="1,4,8,16,32")
    ap.add_argument("--length", type=int, default=1900, help="提示长度（token）")
    ap.add_argument("--gen", type=int, default=128)
    ap.add_argument("--requests-per-conc", type=int, default=None,
                    help="每格请求数（默认 max(16, 2×并发)）")
    ap.add_argument("--hardware", default="（未填）")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    # 模型名
    model = args.wire_model
    if not model:
        r = httpx.get(f"{args.base_url}/models", timeout=10)
        d = r.json()
        entries = d.get("data") or d.get("models") or []
        model = entries[0].get("id") or entries[0].get("model")
    print(f"目标 {args.base_url}｜模型 {model}｜后端 {args.backend}｜硬件 {args.hardware}")

    cells = {}
    for conc in [int(x) for x in args.concs.split(",")]:
        n_req = args.requests_per_conc or max(16, 2 * conc)
        t0 = time.perf_counter()
        res = asyncio.run(run_cell(args.base_url, model, conc, args.length,
                                   args.gen, n_req, args.backend))
        s = summarize(res)
        dt = time.perf_counter() - t0
        if not s:
            print(f"  c{conc:<4} 失败（成功 {len(res)}/{n_req}）")
            continue
        cells[conc] = s
        print(f"  c{conc:<4} n={s['n']:<3} TTFT p50={s['ttft_p50']:>7.1f} ms  "
              f"TPOT p50={s['tpot_p50']:>6.2f} ms  CV={s['tpot_cv']*100:>5.1f}%  "
              f"TPOTp90/p50={s['tpot_p90_over_p50']:.3f}  ({dt:.0f}s)")

    if not cells:
        print("全部失败——先确认服务可用")
        return

    out = dict(hardware=args.hardware, backend=args.backend, model=model,
               length=args.length, gen=args.gen, cells=cells,
               note=("逐请求抖动；用于检验『prefill 打断 decode 的排队效应』。"
                     "若抖动随并发上升且两栈都有，则排队假说获得独立支持。"))
    fp = Path(args.out) if args.out else (cfg.RESULTS_DIR / "jitter_probe.json")
    fp.parent.mkdir(parents=True, exist_ok=True)
    fp.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n→ {fp}")


if __name__ == "__main__":
    main()
