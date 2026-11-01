"""
tc_probe_csrv.py —— 拆解 c_srv：那 0.766 ms/流/步到底花在哪？

## 为什么要做这个实验

现在 c_srv 是**拟合出来的黑箱**：每多一路并发，每个 decode 步多花 0.766 ms。
模型靠它把留出集误差从 33.5% 压到 5.8%，但没人知道这 0.766 ms 是什么。

## 怎么拆

服务端每生成 1 个 token，要经历的路径可以拆成四段，用四种请求方式分别"关掉"其中一段：

| 变体 | 端点 | 流式 | 关掉了什么 |
|---|---|---|---|
| A 基线 | /v1/chat/completions | 是 | ——（全开，就是正式实验用的路径） |
| B 非流式 | /v1/chat/completions | 否 | 关掉 SSE 分块序列化与逐 token 推送 |
| C 原生流式 | /completion | 是 | 关掉 chat 模板渲染（但保留流式推送） |
| D 原生非流式 | /completion | 否 | 两者都关 |

对比 A 与 B 的差值 = SSE/推流成本；A 与 C 的差值 = chat 模板成本。
两者都关（A vs D）应约等于两个差值之和——这本身就是个自洽性检验。

## 判读口径

并发 32 时 c_srv×并发 ≈ 24.5 ms，占 TPOT 的六成，所以差异会很明显。
用"平均单请求总耗时"比较（非流式测不出 TPOT，但总耗时可测）：
    单请求总耗时 ≈ TTFT + 128 × TPOT
total 越小 = 这条路径越省。

用法：
  & "D:\\llm\\venv\\Scripts\\python.exe" tc_probe_csrv.py
  & "D:\\llm\\venv\\Scripts\\python.exe" tc_probe_csrv.py --conc 16,32 --repeats 2
"""

import argparse
import asyncio
import json
import statistics as st
import sys
import time

import httpx

import tc_config as cfg
import tc_server as srv

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

GEN = cfg.GEN_TOKENS
PROMPT_TOKENS = cfg.PROMPT_TOKENS

VARIANTS = [
    ("A 基线 chat+流式",     "/v1/chat/completions", True),
    ("B chat+非流式",        "/v1/chat/completions", False),
    ("C 原生+流式",          "/completion",          True),
    ("D 原生+非流式",        "/completion",          False),
]


def _prompt(nonce):
    """与正式实验用同样的提示词长度与随机前缀（避免 prompt cache）。"""
    filler = ("大模型推理服务在部署时面临一个核心矛盾：用户希望首字延迟尽量低，"
              "而机房希望单位token成本尽量低。")
    chars = int(PROMPT_TOKENS / 0.65)
    body = (filler * (chars // len(filler) + 1))[:chars]
    return f"[探针 {nonce}-{time.time():.6f}] {body}\n\n请用一句话总结上面这段话。"


async def one(client, url, variant_name, chat, stream, max_tokens, nonce):
    prompt = _prompt(nonce)
    if chat:
        payload = dict(model="local",
                       messages=[{"role": "user", "content": prompt}],
                       max_tokens=max_tokens, temperature=0.0,
                       stream=stream, cache_prompt=False, ignore_eos=True)
    else:
        payload = dict(prompt=prompt, n_predict=max_tokens, temperature=0.0,
                       stream=stream, cache_prompt=False, ignore_eos=True)

    t0 = time.perf_counter()
    t_first = None
    n_chunks = 0
    n_out = 0
    if stream:
        async with client.stream("POST", url, json=payload) as r:
            r.raise_for_status()
            async for line in r.aiter_lines():
                if not line:
                    continue
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    obj = json.loads(data)
                except json.JSONDecodeError:
                    continue
                if chat:
                    ch = obj.get("choices") or []
                    piece = ch[0].get("delta", {}).get("content") if ch else None
                else:
                    piece = obj.get("content")
                if piece:
                    if t_first is None:
                        t_first = time.perf_counter()
                    n_chunks += 1
                    n_out += 1
    else:
        r = await client.post(url, json=payload)
        r.raise_for_status()
        obj = r.json()
        t_first = time.perf_counter()          # 非流式：首字与整包同时到达
        if chat:
            usage = obj.get("usage") or {}
            n_out = usage.get("completion_tokens") or 0
        else:
            n_out = (obj.get("timings") or {}).get("predicted_n") or 0
        n_chunks = 1
    t_last = time.perf_counter()
    return dict(variant=variant_name, total_ms=(t_last - t0) * 1e3,
                ttft_ms=((t_first - t0) * 1e3) if t_first else None,
                n_out=n_out, chunks=n_chunks)


async def run_variant(url_base, endpoint, stream, conc, repeats, chat):
    url = f"{url_base}{endpoint}"
    limits = httpx.Limits(max_connections=conc * 2, max_keepalive_connections=conc * 2)
    timeout = httpx.Timeout(600.0, connect=30.0)
    results = []
    async with httpx.AsyncClient(limits=limits, timeout=timeout) as client:
        # 预热
        await asyncio.gather(*[one(client, url, "warm", chat, stream, 16, 10_000 + i)
                               for i in range(min(conc, 4))])
        for rep in range(repeats):
            n_req = max(16, 2 * conc)
            sem = asyncio.Semaphore(conc)

            async def job(i):
                async with sem:
                    return await one(client, url, "run", chat, stream, GEN,
                                     int(time.time() * 1000) % 10_000_000 + i)
            rows = await asyncio.gather(*[job(i) for i in range(n_req)])
            results.append(rows)
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="3B")
    ap.add_argument("--quant", default="q4_k_m")
    ap.add_argument("--ctx", type=int, default=512)
    ap.add_argument("--conc", default="32")
    ap.add_argument("--repeats", type=int, default=2)
    args = ap.parse_args()
    concs = [int(x) for x in args.conc.split(",")]
    key = (args.model, args.quant)

    print("=" * 78)
    print(f"拆解 c_srv：{args.model}-{args.quant} / ctx{args.ctx} / 并发 {concs}")
    print("=" * 78)

    table = []
    for conc in concs:
        proc, log = srv.start(key, args.ctx, conc, tag=f"csrv_c{conc}")
        try:
            time.sleep(2.0)
            for name, endpoint, stream in VARIANTS:
                chat = endpoint.endswith("/chat/completions")
                runs = asyncio.run(run_variant(cfg.BASE_URL, endpoint, stream,
                                               conc, args.repeats, chat))
                flat = [r for run in runs for r in run]
                tot = [r["total_ms"] for r in flat]
                n_out = st.median([r["n_out"] for r in flat]) or GEN
                ttfts = [r["ttft_ms"] for r in flat if r["ttft_ms"]]
                row = dict(conc=conc, variant=name, n_req=len(flat),
                           total_p50=st.median(tot), n_out=n_out,
                           ttft_p50=st.median(ttfts) if ttfts else None)
                # 稳态吞吐：并发 × 生成长度 ÷ 平均单请求总耗时
                row["steady_tps"] = conc * n_out / (st.mean(tot) / 1e3)
                table.append(row)
                print(f"  并发 {conc:>2}  {name:<18} 总耗时 {row['total_p50']:7.1f} ms  "
                      f"TTFT {('%.1f' % row['ttft_p50']) if row['ttft_p50'] else '  n/a':>7}  "
                      f"稳态吞吐 {row['steady_tps']:6.0f} t/s  输出 {n_out:.0f} tok")
        finally:
            srv.stop(proc)
            time.sleep(1.0)

    # ---- 分解 ----
    print("\n" + "=" * 78)
    print("分解（同一并发下，相对基线的差值 = 该路径的成本）")
    print("=" * 78)
    by = {}
    for r in table:
        by.setdefault(r["conc"], {})[r["variant"]] = r
    for conc, v in by.items():
        a = v.get("A 基线 chat+流式")
        if not a:
            continue
        base = a["total_p50"]
        gen = a["n_out"] or GEN
        print(f"\n并发 {conc}（每步开销 ≈ 总耗时差 / 生成 token 数）")
        for name in ("B chat+非流式", "C 原生+流式", "D 原生+非流式"):
            r = v.get(name)
            if not r:
                continue
            d_ms = base - r["total_p50"]
            per_step = d_ms / gen
            print(f"  {name:<18} 省 {d_ms:7.1f} ms/请求  = {per_step*1000:6.0f} µs/token"
                  f"  → 每并发流每步省 {per_step/conc*1000:5.0f} µs")
        sse = base - v["B chat+非流式"]["total_p50"]
        tpl = base - v["C 原生+流式"]["total_p50"]
        both = base - v["D 原生+非流式"]["total_p50"]
        print(f"  自洽性：SSE {sse:.0f} + 模板 {tpl:.0f} = {sse+tpl:.0f} ms，"
              f"实测两者都关 {both:.0f} ms，差 {abs(sse+tpl-both):.0f} ms")

    out = cfg.DATA_DIR / "probe_csrv.csv"
    import csv
    with out.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(table[0].keys()))
        w.writeheader()
        w.writerows(table)
    print(f"\n已写入 {out}")


if __name__ == "__main__":
    main()
