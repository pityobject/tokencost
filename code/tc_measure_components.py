# -*- coding: utf-8 -*-
"""
c_srv 拆解（`tc.py components`）：每步服务开销里，采样 / 序列化 / 推流各占多少

## 思路

在**同一个服务、同一个模型、同一并发**下，逐项关掉一个环节，看 TPOT 掉多少：

| 变体 | 关掉了什么 | 期望 |
|---|---|---|
| `stream=True`（基准） | —— | 完整链路 |
| `stream=False` | SSE 推流 + 逐 token 序列化 | TPOT 应下降（但测法要改成总时长/输出数） |
| `temperature=0` vs `1.0` | 采样（top-p/top-k 随机数） | 温度采样更贵 |
| `logprobs` 关/开 | 每 token 计算并返回 logprobs | 开时更贵 |
| `--n-predict` 短/长 | 每请求固定摊薄 | 长输出 TPOT 更低（固定成本被摊薄） |

## 用法

    python code/tc_measure_components.py --base-url http://127.0.0.1:8092/v1 \
        --backend llamacpp --conc 1 --length 64 --gen 128 \
        --hardware "A10 24GB / llama.cpp" --out results/components_a10_llamacpp.json
"""
import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent))

import tc_config as cfg
import tc_loadgen as lg

sys.stdout.reconfigure(encoding="utf-8", errors="replace")


async def one(client, base_url, model, prompt, gen, backend,
              stream=True, temperature=0.0, logprobs=False):
    payload = {
        "model": model, "messages": [{"role": "user", "content": prompt}],
        "max_tokens": gen, "temperature": temperature,
        "stream": stream, "ignore_eos": True,
    }
    if stream:
        payload["stream_options"] = {"include_usage": True}
    if backend != "vllm":
        payload["cache_prompt"] = False
    if logprobs:
        payload["logprobs"] = True
        payload["top_logprobs"] = 1

    t0 = time.perf_counter()
    first = None
    n_out = 0
    usage_tok = 0
    try:
        if stream:
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
                        usage_tok = int(obj["usage"].get("completion_tokens") or 0)
                    for ch in obj.get("choices") or []:
                        if (ch.get("delta") or {}).get("content"):
                            if first is None:
                                first = time.perf_counter()
                            n_out += 1
        else:
            r = await client.post(f"{base_url}/chat/completions", json=payload, timeout=300)
            r.raise_for_status()
            obj = r.json()
            first = time.perf_counter()
            # 【修正】按空格分词会严重低估中文输出；必须用服务端返回的 completion_tokens
            n_out = int(((obj.get("usage") or {}).get("completion_tokens")) or 0)
    except Exception as e:
        return None
    total = time.perf_counter() - t0
    if n_out < 3:
        return None
    if stream:
        ttft_ms = (first - t0) * 1e3
        # 用 usage 的 token 数更准（比逐 chunk 计数少算最后一个）
        n_eff = max(usage_tok, n_out) - 1
        tpot_ms = (time.perf_counter() - first) * 1e3 / max(n_eff, 1)
    else:
        # 非流式拿不到 TTFT，TPOT 无法从总时长里分离 prefill → 不报，只报总时长
        ttft_ms = None
        tpot_ms = None
    return dict(ttft_ms=ttft_ms, tpot_ms=tpot_ms, n_out=n_out,
                total_ms=total * 1e3)


def med(rows, key):
    vals = sorted(r[key] for r in rows if r.get(key) is not None)
    return round(vals[len(vals) // 2], 3) if vals else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--wire-model", default=None)
    ap.add_argument("--backend", choices=["llamacpp", "vllm"], default="llamacpp")
    ap.add_argument("--conc", type=int, default=1)
    ap.add_argument("--length", type=int, default=64)
    ap.add_argument("--gen", type=int, default=128)
    ap.add_argument("--repeats", type=int, default=8)
    ap.add_argument("--hardware", default="（未填）")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    model = args.wire_model
    if not model:
        d = httpx.get(f"{args.base_url}/models", timeout=10).json()
        e = d.get("data") or d.get("models") or []
        model = e[0].get("id") or e[0].get("model")

    variants = [
        ("baseline（stream, T=0）", dict(stream=True, temperature=0.0, logprobs=False)),
        ("sampling（T=1.2）", dict(stream=True, temperature=1.2, logprobs=False)),
        ("logprobs 开", dict(stream=True, temperature=0.0, logprobs=True)),
        ("非流式", dict(stream=False, temperature=0.0, logprobs=False)),
        ("短输出（gen=32）", dict(stream=True, temperature=0.0, logprobs=False)),
    ]
    print(f"目标 {args.base_url}｜模型 {model}｜后端 {args.backend}｜"
          f"c={args.conc} L={args.length}")

    results = {}
    for name, kw in variants:
        gen = 32 if "短输出" in name else args.gen
        prompt = lg.make_prompt(args.length, nonce=int(time.time() * 1000) % 10**9)

        async def run():
            async with httpx.AsyncClient() as client:
                sem = asyncio.Semaphore(args.conc)
                out = []

                async def task():
                    async with sem:
                        r = await one(client, args.base_url, model, prompt, gen,
                                      args.backend, **kw)
                        if r:
                            out.append(r)
                await asyncio.gather(*(task() for _ in range(max(args.repeats, args.conc * 2))))
                return out

        rows = asyncio.run(run())
        if not rows:
            print(f"  {name:<24} 失败")
            continue
        tp = med(rows, "tpot_ms")
        tt = med(rows, "ttft_ms")
        tot = med(rows, "total_ms")
        results[name] = dict(tpot_ms=tp, ttft_ms=tt, total_ms=tot,
                             n=len(rows), gen=gen)
        if tp is None:
            print(f"  {name:<24} 总时长={tot:>8.1f} ms（非流式，TPOT 不可分）  n={len(rows)}")
        else:
            print(f"  {name:<24} TPOT={tp:>7.2f} ms  "
                  f"TTFT={str(tt or '—'):>8}  总时长={tot:>8.1f}  n={len(rows)}")

    base = (results.get("baseline（stream, T=0）") or {}).get("tpot_ms")
    if base:
        print("\n  相对基准的差值（ms/步）：")
        for name, r in results.items():
            if name == "baseline（stream, T=0）" or r["tpot_ms"] is None:
                continue
            d = r["tpot_ms"] - base
            print(f"    {name:<24} {d:+.2f}  ({(d/base*100):+.0f}%)")
        # 非流式：用总时长的差算 SSE 推流开销
        ns = results.get("非流式") or {}
        bs = results.get("baseline（stream, T=0）") or {}
        if ns.get("total_ms") and bs.get("total_ms"):
            d = bs["total_ms"] - ns["total_ms"]
            ntok = bs.get("gen") or 128
            print(f"\n    SSE 推流开销：流式总时长 − 非流式总时长 = {d:+.1f} ms"
                  f"（每 token {d/ntok:+.3f} ms）")
            print("    （非流式无法分离 prefill，所以只能用总时长相减）")
        print("\n  读法：")
        print("    · sampling 差值 ≈ 采样开销")
        print("    · logprobs 差值 ≈ 每 token 额外计算/传输")
        print("    · 非流式差值 ≈ SSE 推流与逐 token 序列化")
        print("    · 短输出差值（负）≈ 每请求固定成本按更长输出摊薄")

    out = dict(hardware=args.hardware, backend=args.backend, model=model,
               conc=args.conc, length=args.length, variants=results,
               baseline_tpot_ms=base,
               note="逐项关掉一个环节，用 TPOT 差值归因；c=1 时最能分离固定成本")
    fp = Path(args.out) if args.out else (cfg.RESULTS_DIR / "components.json")
    fp.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n→ {fp}")


if __name__ == "__main__":
    main()
