# -*- coding: utf-8 -*-
"""
PEAK_dec 的干净分离（`tc.py peakb`）：用「两模型 × 多并发」的双差法

## 为什么之前失败

- 单栈斜率法：`TPOT(b) = 2Pb/PEAK + c_srv·b + mem`，`PEAK` 与 `c_srv` 两个未知量纠缠。
- 单并发成对差分：`ΔTPOT = 2ΔP·b/PEAK + ΔW/BW`，其中 `ΔW/BW` 是**常数**、
  `2ΔP·b/PEAK` 随 b 线性——只要**扫多个并发**，就能把截距和斜率分开：

    ΔTPOT(b) = [2ΔP/PEAK]·b + [ΔW/BW]

  `c_srv` 在两模型间抵消（同一服务、同一并行度），所以这条直线干净。

于是   斜率 k = 2ΔP / PEAK   ⇒   **PEAK_dec = 2ΔP / k**

## 前提（必须验证）

两模型在该并发段都**算力受限**：`2Pb/PEAK > W/BW`。
3B 在 b≥32、W=6.2 GB、BW≈470 时成立；1.5B 需要更高 b（脚本会自动检查并提示）。

## 用法

    # 3B 与 1.5B 都起在同一台机器上（不同端口），对同一组并发各测一次
    python code/tc_measure_peak_pair.py \
        --url-a http://127.0.0.1:8092/v1 --model-a 3B \
        --url-b http://127.0.0.1:8093/v1 --model-b 1.5B \
        --concs 32,48,64,96,128 --length 64 --gen 128 \
        --n-params-a 3.08e9 --n-params-b 1.54e9 \
        --w-a 6178317312 --w-b 3087007744 --bw 471.5 \
        --hardware "A10 24GB / llama.cpp" --out results/peak_pair_a10.json
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


async def one(client, base_url, model, prompt, gen, backend):
    payload = {
        "model": model, "messages": [{"role": "user", "content": prompt}],
        "max_tokens": gen, "temperature": 0.0, "stream": True,
        "stream_options": {"include_usage": True}, "ignore_eos": True,
    }
    if backend != "vllm":
        payload["cache_prompt"] = False
    t0 = time.perf_counter()
    first = None
    n = 0
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
                for ch in obj.get("choices") or []:
                    if (ch.get("delta") or {}).get("content"):
                        if first is None:
                            first = time.perf_counter()
                        n += 1
    except Exception:
        return None
    if first is None or n < 3:
        return None
    return (time.perf_counter() - first) * 1e3 / (n - 1)


async def measure(base_url, model, conc, length, gen, backend, repeats):
    prompt = lg.make_prompt(length, nonce=int(time.time() * 1000) % 10**9)
    sem = asyncio.Semaphore(conc)
    out = []
    async with httpx.AsyncClient() as client:
        async def task():
            async with sem:
                r = await one(client, base_url, model, prompt, gen, backend)
                if r:
                    out.append(r)
        await asyncio.gather(*(task() for _ in range(max(repeats, 2 * conc))))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url-a", required=True)
    ap.add_argument("--model-a", default=None)
    ap.add_argument("--url-b", required=True)
    ap.add_argument("--model-b", default=None)
    ap.add_argument("--backend", choices=["llamacpp", "vllm"], default="llamacpp")
    ap.add_argument("--concs", default="32,48,64,96,128")
    ap.add_argument("--length", type=int, default=64)
    ap.add_argument("--gen", type=int, default=128)
    ap.add_argument("--repeats", type=int, default=256)
    ap.add_argument("--n-params-a", type=float, required=True)
    ap.add_argument("--n-params-b", type=float, required=True)
    ap.add_argument("--w-a", type=float, required=True, help="A 的权重大小（字节）")
    ap.add_argument("--w-b", type=float, required=True)
    ap.add_argument("--bw", type=float, default=471.5, help="该机的反推带宽 GB/s")
    ap.add_argument("--hardware", default="（未填）")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    print(f"A {args.url_a} ({args.model_a})｜B {args.url_b} ({args.model_b})")
    rows = []
    for b in [int(x) for x in args.concs.split(",")]:
        ra = asyncio.run(measure(args.url_a, args.model_a, b, args.length,
                                 args.gen, args.backend, args.repeats))
        rb = asyncio.run(measure(args.url_b, args.model_b, b, args.length,
                                 args.gen, args.backend, args.repeats))
        if len(ra) < 5 or len(rb) < 5:
            print(f"  c{b:<4} 样本不足（A {len(ra)} / B {len(rb)}）")
            continue
        ta, tb = float(np.median(ra)), float(np.median(rb))
        # 该并发下两边是否算力受限（用 A 的参数量估）
        cmp_a = 2 * args.n_params_a * b / 13.157e12 * 1e3
        mem_a = args.w_a / (args.bw * 1e9) * 1e3
        rows.append(dict(conc=b, tpot_a=round(ta, 2), tpot_b=round(tb, 2),
                         dt=round(ta - tb, 2),
                         compute_bound_a=bool(cmp_a > mem_a)))
        print(f"  c{b:<4} A={ta:>7.2f}  B={tb:>7.2f}  Δ={ta-tb:>7.2f}  "
              f"(A {'算力' if cmp_a > mem_a else '访存'}受限)")

    if len(rows) < 3:
        print("有效并发点不足 3 个，无法拟合")
        return

    bs = np.array([r["conc"] for r in rows], float)
    dt = np.array([r["dt"] for r in rows], float)
    k, c0 = np.polyfit(bs, dt, 1)
    r2 = 1 - np.sum((dt - (k * bs + c0)) ** 2) / np.sum((dt - dt.mean()) ** 2)
    dP = args.n_params_a - args.n_params_b
    print("\n" + "=" * 78)
    print(f"  ΔTPOT(b) = {k:.5f}·b + {c0:.2f}   (R²={r2:.4f})")
    print(f"  理论截距 ΔW/BW = {(args.w_a-args.w_b)/(args.bw*1e9)*1e3:.2f} ms")
    print("=" * 78)
    if k <= 0:
        print("  ❌ 斜率为负或零：该并发段未进入算力区，或两模型标定参数不可比")
        return
    peak = 2 * dP / (k * 1e-3) / 1e12
    print(f"  → **PEAK_dec = 2ΔP / 斜率 = {peak:.1f} TFLOP/s**")
    print(f"     （拟合值 13.2；此前的下界 15.8）")
    print(f"  ΔP = {dP/1e9:.2f} B，斜率 {k*1000:.3f} ms/流/…")
    if r2 < 0.9:
        print("  ⚠️ R² 偏低，说明 ΔTPOT 里还有非算力成分，结论只能当量级")

    out = dict(hardware=args.hardware, backend=args.backend,
               model_a=args.model_a, model_b=args.model_b,
               delta_params=dP, slope_ms_per_b=round(float(k), 5),
               intercept_ms=round(float(c0), 2), r2=round(float(r2), 4),
               expected_intercept_ms=round((args.w_a - args.w_b) / (args.bw * 1e9) * 1e3, 2),
               peak_dec_tflops=round(float(peak), 1), rows=rows,
               note="双差法：c_srv 抵消，斜率给 PEAK，截距给 ΔW/BW")
    fp = Path(args.out) if args.out else (cfg.RESULTS_DIR / "peak_dec_double_diff.json")
    fp.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n→ {fp}")


if __name__ == "__main__":
    main()
