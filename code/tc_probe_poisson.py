"""
tc_probe_poisson.py —— 泊松到达 + 容量曲线（v2：修掉客户端饱和）

## v1 错在哪

旧版每个请求都新建一个 `httpx.Client`（每次都要 TCP 握手），
并用 64 个线程跑 SSE 解析（GIL 互斥）。
结果 λ=4 与 λ=6 都只达成 2.6 qps——**客户端自己成了瓶颈**，
测出来的"TTFT 差 2–8 倍"后两点不可信。

## v2 的三处改动

1. **asyncio + 共享连接池**：一个 Client 复用连接，避免每次握手；
2. **饱和自检**：每个 λ 都核对 `达成 QPS / 目标 λ`，低于 0.85 就直接标注"客户端受限"，
   不再把噪声当结论；
3. **扫 λ 找容量膝点**：从 1 到 12 req/s 扫一遍，画出"QPS → p90 TTFT"曲线。
   膝点 = 这台机器真正的可持续吞吐上限。

## 为什么这条重要

报告里有一句限制："模型只算服务时间，不算排队时间，不能直接用于容量规划。"
容量曲线能把这句限制**量化**：
- 膝点之前的 λ，模型预测与实际接近 → 可用于规划；
- 越过膝点，TTFT 由排队主导 → 必须加排队模型。

用法：
  & "D:\\llm\\venv\\Scripts\\python.exe" tc_probe_poisson.py
  & "D:\\llm\\venv\\Scripts\\python.exe" tc_probe_poisson.py --lambdas 1,2,4,6,8,12 --duration 40
"""

import argparse
import asyncio
import csv
import json
import random
import statistics as st
import sys
import time

import httpx

import tc_config as cfg
import tc_server as srv

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

MODEL, QUANT, CTX, SLOTS = "3B", "q4_k_m", 512, 8
OUT = cfg.DATA_DIR / "probe_poisson.csv"
MAX_INFLIGHT = 128          # 远大于槽位数，保证排队发生在服务端而不是客户端


def make_prompt(nonce):
    filler = ("大模型推理服务在部署时面临一个核心矛盾：用户希望首字延迟尽量低，"
              "而机房希望单位token成本尽量低。")
    chars = int(cfg.PROMPT_TOKENS / 0.65)
    body = (filler * (chars // len(filler) + 1))[:chars]
    return f"[泊松 {nonce}] {body}\n\n请用一句话总结上面这段话。"


async def one(client, url, nonce):
    prompt = make_prompt(nonce)
    payload = dict(model="local", messages=[{"role": "user", "content": prompt}],
                   max_tokens=cfg.GEN_TOKENS, temperature=0.0,
                   stream=True, cache_prompt=False, ignore_eos=True)
    t_send = time.perf_counter()
    t_first = None
    n = 0
    try:
        async with client.stream("POST", url, json=payload) as r:
            r.raise_for_status()
            async for line in r.aiter_lines():
                if not line or not line.startswith("data:"):
                    continue
                d = line[5:].strip()
                if d == "[DONE]":
                    break
                try:
                    o = json.loads(d)
                except json.JSONDecodeError:
                    continue
                ch = o.get("choices") or []
                if ch and ch[0].get("delta", {}).get("content"):
                    if t_first is None:
                        t_first = time.perf_counter()
                    n += 1
        t_end = time.perf_counter()
        return dict(ok=True, send= t_send, first=t_first, end=t_end, n=n,
                    ttft_ms=((t_first - t_send) * 1e3) if t_first else None,
                    total_ms=(t_end - t_send) * 1e3,
                    tpot_ms=(((t_end - t_first) / max(n - 1, 1)) * 1e3
                             if t_first and n > 1 else None))
    except Exception as e:
        return dict(ok=False, send=t_send, first=None, end=time.perf_counter(),
                    n=0, error=str(e)[:100])


async def run_lambda(base_url, lam, duration_s, seed=0):
    """按指数间隔发请求，持续 duration_s 秒。返回 (行结果列表, 达成QPS, 平均并发)。"""
    rng = random.Random(seed)
    limits = httpx.Limits(max_connections=MAX_INFLIGHT + 32,
                          max_keepalive_connections=MAX_INFLIGHT + 32)
    timeout = httpx.Timeout(600.0, connect=30.0)
    url = f"{base_url}/v1/chat/completions"

    # 先把到达时刻排好（指数间隔），再按时刻投递
    arrivals, t = [], 0.0
    while True:
        t += rng.expovariate(lam)
        if t >= duration_s:
            break
        arrivals.append(t)

    sem = asyncio.Semaphore(MAX_INFLIGHT)

    async def job(i):
        async with sem:
            return await one(client, url, i)

    async with httpx.AsyncClient(limits=limits, timeout=timeout) as client:
        # 预热，避免第一波全在等 kernel 编译
        await asyncio.gather(*[one(client, url, f"warm{i}") for i in range(SLOTS)])
        start = time.perf_counter()
        tasks = []
        for i, at in enumerate(arrivals):
            delay = start + at - time.perf_counter()
            if delay > 0:
                await asyncio.sleep(delay)
            tasks.append(asyncio.create_task(job(f"{lam}-{i}")))
        rows = await asyncio.gather(*tasks)
        span = time.perf_counter() - start

    ok = [r for r in rows if r["ok"]]
    achieved = len(ok) / span if span > 0 else 0.0

    # 平均并发：所有请求 [send, end) 区间的平均重叠数
    ev = []
    for r in rows:
        ev.append((r["send"] - start, 1))
        ev.append((r["end"] - start, -1))
    ev.sort()
    cur = 0
    prev = ev[0][0] if ev else 0.0
    area = 0.0
    for ts, d in ev:
        area += cur * (ts - prev)
        prev = ts
        cur += d
    mean_conc = area / span if span > 0 else float("nan")
    return rows, ok, achieved, mean_conc, len(arrivals)


async def main_async(args):
    rows_all = []
    proc, _ = srv.start((MODEL, QUANT), CTX, SLOTS, tag="poisson2", timeout_s=240)
    try:
        print("=" * 92)
        print(f"泊松到达 + 容量曲线：{MODEL}-{QUANT} / ctx{CTX} / 槽位 {SLOTS}")
        print("=" * 92)
        print(f"{'目标λ':>6}{'达成QPS':>9}{'达成率':>8}{'平均并发':>9}{'请求数':>7}"
              f"{'TTFT p50':>10}{'TTFT p90':>10}{'TPOT p50':>10}  备注")
        for lam in args.lambdas:
            rows, ok, achieved, mean_conc, n_arr = await run_lambda(
                cfg.BASE_URL, lam, args.duration, seed=int(lam * 100))
            if not ok:
                print(f"{lam:>6}{0:>9.2f}{'—':>8}")
                continue
            ttft = [r["ttft_ms"] for r in ok if r["ttft_ms"]]
            tpot = [r["tpot_ms"] for r in ok if r["tpot_ms"]]
            p50 = st.median(ttft) if ttft else float("nan")
            p90 = sorted(ttft)[min(len(ttft) - 1, int(len(ttft) * 0.9))] if ttft else float("nan")
            tp = st.median(tpot) if tpot else float("nan")
            ratio = achieved / lam
            note = "" if ratio >= 0.85 else "达成率低（需在分析里区分服务器饱和 / 客户端受限）"
            print(f"{lam:>6}{achieved:>9.2f}{ratio:>8.0%}{mean_conc:>9.2f}{len(ok):>7}"
                  f"{p50:>10.0f}{p90:>10.0f}{tp:>10.1f}  {note}")
            rows_all.append(dict(lam=lam, achieved_qps=achieved, ratio=ratio,
                                 mean_conc=mean_conc, n_ok=len(ok), n_sent=n_arr,
                                 ttft_p50=p50, ttft_p90=p90, tpot_p50=tp,
                                 ratio_low=bool(ratio < 0.85)))
            await asyncio.sleep(2.0)
    finally:
        srv.stop(proc)

    with OUT.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows_all[0].keys()))
        w.writeheader()
        w.writerows(rows_all)
    print(f"\n已写入 {OUT}")
    print("下一步： python tc_analyze_poisson.py   （算膝点 + 与模型对比）")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lambdas", default="1,2,3,4,6,8,12")
    ap.add_argument("--duration", type=float, default=40.0, help="每个 λ 的持续秒数")
    args = ap.parse_args()
    args.lambdas = [float(x) for x in args.lambdas.split(",")]
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
