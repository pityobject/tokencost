# -*- coding: utf-8 -*-
"""
混合长度测量（`tc.py mixedprobe`）：同一格内长短请求混发，检验"均值长度近似"

## 为什么需要

本机的 E2 结论是：**混合长度下 TTFT 不可预测**（均值口径 +45~109%、逐请求 +121%），
而 TPOT 仍可用（17%）。但那是 4060 + llama.cpp 的结果。
问题：这是**模型形式缺陷**还是 **llama.cpp 的实现特性**？只能在 vLLM 上再测一次。

## 做法

每格按给定分布抽请求长度（默认 20% 长 / 80% 短，均值≈1000），并发发出，逐请求记录：

- 实测 TTFT / TPOT
- 服务端返回的 `prompt_tokens`（真实长度）

然后用标定参数算两种预测：
1. **均值口径**：用该格的平均长度预测，与实测中位比较（模型的默认用法）；
2. **逐请求口径**：用每个请求自己的长度预测，与该请求实测比较。

## 用法

    python code/tc_measure_mixed.py --base-url http://127.0.0.1:8081/v1 \
        --wire-model qwen3b --backend vllm \
        --calib results/a10_raw/calib_a10_vllm.json \
        --concs 1,4,8 --long 1800 --short 200 --p_long 0.25 \
        --hardware "A10 24GB / vLLM" --out results/mixed_a10_vllm.json
"""
import argparse
import asyncio
import json
import random
import statistics
import sys
import time
from pathlib import Path

import httpx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import tc_config as cfg
import tc_loadgen as lg
import tc_model as tm

sys.stdout.reconfigure(encoding="utf-8", errors="replace")


async def one_request(client, base_url, model, prompt, gen, backend):
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": gen, "temperature": 0.0, "stream": True,
        "stream_options": {"include_usage": True}, "ignore_eos": True,
    }
    if backend != "vllm":
        payload["cache_prompt"] = False
    t0 = time.perf_counter()
    first = last = None
    n_out = 0
    usage = {}
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
                if obj.get("usage"):
                    usage = obj["usage"]
                for ch in obj.get("choices") or []:
                    if (ch.get("delta") or {}).get("content"):
                        now = time.perf_counter()
                        if first is None:
                            first = now
                        last = now
                        n_out += 1
    except Exception:
        return None
    if first is None or last is None or n_out < 3:
        return None
    return dict(ttft_ms=(first - t0) * 1e3,
                tpot_ms=(last - first) * 1e3 / (n_out - 1),
                n_out=n_out,
                prompt_tokens=int(usage.get("prompt_tokens") or 0))


async def run_cell(base_url, model, conc, plan, gen, backend):
    """plan: [(target_len, count)]；按并发上限并发发出。"""
    items = []
    for L, cnt in plan:
        items += [L] * cnt
    random.shuffle(items)
    sem = asyncio.Semaphore(conc)
    out = []

    async with httpx.AsyncClient() as client:
        async def task(L, i):
            prompt = lg.make_prompt(int(L), nonce=int(time.time() * 1e6) % 10**9 + i)
            async with sem:
                r = await one_request(client, base_url, model, prompt, gen, backend)
            if r:
                r["target_len"] = int(L)
                out.append(r)
        await asyncio.gather(*(task(L, i) for i, L in enumerate(items)))
    return out


def predict(calib, L, conc, spec):
    params = dict(bw_gbs=calib["bw"], peak_tflops=calib["peak"],
                  peak_prefill_tflops=calib["peak_prefill"],
                  c_srv_ms=calib["c_srv"], b_sat=calib["b_sat"],
                  ovh_ttft_ms=calib["ovh_ttft"])
    row = dict(model="3B", quant="fp16", ctx=2304, conc=conc,
               prompt_tokens=L, ctx_avg_tokens=L + cfg.GEN_TOKENS / 2)
    pr = tm.predict(params, row)
    return pr["pred_ttft_ms"], pr["pred_tpot_ms"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--wire-model", default=None)
    ap.add_argument("--backend", choices=["llamacpp", "vllm"], default="vllm")
    ap.add_argument("--calib", required=True, help="用于预测的标定 json")
    ap.add_argument("--concs", default="1,4,8")
    ap.add_argument("--long", type=int, default=1800)
    ap.add_argument("--short", type=int, default=200)
    ap.add_argument("--p-long", type=float, default=0.25)
    ap.add_argument("--gen", type=int, default=128)
    ap.add_argument("--requests-per-conc", type=int, default=None)
    ap.add_argument("--hardware", default="（未填）")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    calib = json.loads(Path(args.calib).read_text(encoding="utf-8"))
    spec = calib["model"]
    cfg.MODEL_SPECS[("3B", "fp16")] = dict(
        weight_bytes=spec["weight_bytes"], n_params=spec["n_params"],
        n_layers=spec.get("n_layers", 36), n_kv_heads=spec.get("n_kv_heads", 2),
        head_dim=spec.get("head_dim", 128))

    model = args.wire_model
    if not model:
        d = httpx.get(f"{args.base_url}/models", timeout=10).json()
        entries = d.get("data") or d.get("models") or []
        model = entries[0].get("id") or entries[0].get("model")
    mean_L = args.p_long * args.long + (1 - args.p_long) * args.short
    print(f"目标 {args.base_url}｜模型 {model}｜后端 {args.backend}")
    print(f"混合分布：{args.p_long:.0%} @ {args.long} + {1-args.p_long:.0%} @ {args.short}"
          f"  → 均值长度 {mean_L:.0f}")

    cells = {}
    for conc in [int(x) for x in args.concs.split(",")]:
        n = args.requests_per_conc or max(16, 4 * conc)
        n_long = max(1, int(round(n * args.p_long)))
        plan = [(args.long, n_long), (args.short, n - n_long)]
        t0 = time.perf_counter()
        res = asyncio.run(run_cell(args.base_url, model, conc, plan, args.gen, args.backend))
        if len(res) < 5:
            print(f"  c{conc:<3} 失败（成功 {len(res)}）")
            continue

        meas_ttft = [r["ttft_ms"] for r in res]
        med_ttft = float(np.median(meas_ttft))
        # 两种口径
        pred_mean_ttft, _ = predict(calib, mean_L, conc, spec)
        e_mean = abs(pred_mean_ttft / med_ttft - 1) * 100
        per_req = []
        for r in res:
            L = r["prompt_tokens"] or r["target_len"]
            p_ttft, _ = predict(calib, L, conc, spec)
            per_req.append(abs(p_ttft / r["ttft_ms"] - 1) * 100)
        e_req = float(np.mean(per_req))

        cells[conc] = dict(n=len(res),
                           mean_target_len=round(mean_L, 1),
                           mean_actual_len=round(float(np.mean(
                               [r["prompt_tokens"] or r["target_len"] for r in res])), 1),
                           ttft_p50=round(med_ttft, 1),
                           tpot_p50=round(float(np.median([r["tpot_ms"] for r in res])), 2),
                           err_mean_len_pct=round(e_mean, 1),
                           err_per_request_pct=round(e_req, 1))
        print(f"  c{conc:<3} n={len(res):<3} L实际均值="
              f"{cells[conc]['mean_actual_len']:>6.0f}  TTFT p50={med_ttft:>8.1f} ms  "
              f"均值口径误差={e_mean:>5.1f}%  逐请求误差={e_req:>5.1f}%  "
              f"({time.perf_counter()-t0:.0f}s)")

    if not cells:
        print("全部失败")
        return
    out = dict(hardware=args.hardware, backend=args.backend, model=model,
               distribution=dict(p_long=args.p_long, long=args.long, short=args.short),
               cells=cells,
               note=("均值口径 = 用该格平均长度预测后与实测中位比；"
                     "逐请求口径 = 每个请求用自己的真实长度预测后与自身实测比"))
    fp = Path(args.out) if args.out else (cfg.RESULTS_DIR / "mixed_probe.json")
    fp.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n→ {fp}")


if __name__ == "__main__":
    main()
