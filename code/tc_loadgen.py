"""
tc_loadgen.py —— 负载生成器（本项目的核心测量工具）。

它做的事：同时发 N 个请求，给每个请求的每个 token 打时间戳，算出
  TTFT（首字延迟）、TPOT（出字速度）、总耗时、实际生成 token 数。

为什么不用现成的压测工具？
  因为我们需要"逐 token 时间戳"。只有拿到它，才能把 TTFT 和 TPOT 分开，
  而这两者的瓶颈完全不同（prefill 是算力密集，decode 是带宽密集）。

用法（独立跑，用来验证客户端本身是否正常）：
  & "D:\\llm\\venv\\Scripts\\python.exe" tc_loadgen.py --concurrency 4 --requests 24

作为库被调用（run_cell 会用这个）：
  from tc_loadgen import run_load
  rows = asyncio.run(run_load(concurrency=4, n_requests=24, ...))
"""

import argparse
import asyncio
import json
import random
import statistics
import time
from pathlib import Path

import httpx

import tc_config as cfg

# 一段中文垫话，用来把 prompt 撑到目标长度。
_FILLER = (
    "大模型推理服务在部署时面临一个核心矛盾：用户希望首字延迟尽量低，"
    "而机房希望单位token成本尽量低，两者往往不能同时满足。"
    "工程上的做法是通过批处理把多个请求合并计算，从而提高权重读取的摊薄程度。"
)


def make_prompt(target_tokens: int, nonce: int) -> str:
    """造一个长度约等于 target_tokens 的提示词。

    一定要每次都不一样（加 nonce），否则 llama-server 的 prompt cache 会命中，
    prefill 时间会变成 0，你的 TTFT 数据就全废了。这是新手最容易踩的坑。
    """
    # 中文大致 1 个字 ≈ 0.6~0.7 token，这里用字数近似，实际以服务端返回的 usage 为准
    chars_needed = int(target_tokens / 0.65)
    body = (_FILLER * (chars_needed // len(_FILLER) + 1))[:chars_needed]
    return f"[会话编号 {nonce}-{random.random():.8f}] {body}\n\n请用一句话总结上面这段话。"


def _pct(values, q):
    """分位数（线性插值）。样本少时 P99 不可靠，所以主指标用 P90 和 CVaR90。"""
    if not values:
        return float("nan")
    xs = sorted(values)
    if len(xs) == 1:
        return xs[0]
    pos = (len(xs) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (pos - lo)


def cvar(values, q=0.9):
    """CVaR：最差那 (1-q) 部分样本的均值。

    为什么用它而不是 P99？
      ① P99 需要几百个样本才稳，我们每格只有几十个请求；
      ② P99 不可导、不可优化，CVaR 可优化（后面做配置选择时要用）。
    它俩方向一致，CVaR90 可以理解成"尾部延迟的平均水平"。
    """
    if not values:
        return float("nan")
    xs = sorted(values, reverse=True)
    k = max(1, int(round(len(xs) * (1 - q))))
    return statistics.fmean(xs[:k])


async def one_request(client, url, prompt, max_tokens, req_id, t_origin=None):
    """发一个流式请求，返回这个请求的全部指标（dict）。

    t_origin：整轮实验的统一时间原点。每个请求额外记录自己相对原点的起止时刻，
    这样汇总时才能算出**真正的墙钟吞吐**（总 token ÷ 整轮实际耗时），
    而不是"每请求速率 × 并发数"这种会在排队时失真的近似。
    """
    payload = {
        "model": "local",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,                     # 贪心解码，结果可复现
        "stream": True,
        "stream_options": {"include_usage": True},
        "cache_prompt": False,                  # 明确关闭 prompt 复用
        # 强制生成满 max_tokens，不许提前遇到结束符就停。
        # 为什么必须开：不同格生成长度不一样时，TPOT 没法横向比较
        # （短回答里首字/固定开销占比更大）。开了它，每格都稳定 128 个 token。
        # 计时不需要回答内容，所以忽略 EOS 完全无害——这也是业界压测的通行做法。
        "ignore_eos": True,
    }

    t_send = time.perf_counter()
    t_first = None
    t_last = None
    n_content_chunks = 0
    usage = None
    err = None

    try:
        async with client.stream("POST", url, json=payload) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    obj = json.loads(data)
                except json.JSONDecodeError:
                    continue
                if obj.get("usage"):
                    usage = obj["usage"]
                choices = obj.get("choices") or []
                if not choices:
                    continue
                delta = choices[0].get("delta") or {}
                if delta.get("content"):
                    now = time.perf_counter()
                    if t_first is None:
                        t_first = now
                    t_last = now
                    n_content_chunks += 1
    except Exception as e:      # 网络错误、超时、4xx/5xx 都记下来，不中断整轮实验
        err = f"{type(e).__name__}: {e}"

    t_end = time.perf_counter()

    if usage and usage.get("completion_tokens"):
        n_out = usage["completion_tokens"]
    else:
        n_out = n_content_chunks

    row = dict(
        req_id=req_id,
        ok=err is None and t_first is not None,
        error=err,
        n_out=n_out,
        prompt_tokens=(usage or {}).get("prompt_tokens"),
        ttft_ms=(t_first - t_send) * 1e3 if t_first else None,
        total_ms=(t_last - t_send) * 1e3 if t_last else None,
        tpot_ms=((t_last - t_first) / (n_out - 1) * 1e3) if (t_first and n_out and n_out > 1) else None,
        wall_ms=(t_end - t_send) * 1e3,
    )
    if t_origin is not None:
        row["t_start_rel"] = t_send - t_origin
        row["t_end_rel"] = t_end - t_origin
    return row


def parse_length_mix(spec):
    """把 "256:0.7,1024:0.2" 解析成 {256: 0.7, 1024: 0.2}；权重归一化。"""
    if not spec:
        return None
    mix = {}
    for part in str(spec).split(","):
        part = part.strip()
        if not part:
            continue
        if ":" not in part:
            raise SystemExit(f"--length-mix 格式应为 '长度:权重,...'，收到 {part!r}")
        k, v = part.split(":", 1)
        mix[int(float(k))] = float(v)
    total = sum(mix.values())
    if total <= 0:
        raise SystemExit("--length-mix 的权重之和必须为正")
    return {k: v / total for k, v in mix.items()}


def sample_length(mix, rng):
    """按权重抽一个目标长度。"""
    r = rng.random()
    acc = 0.0
    last = None
    for k, w in mix.items():
        acc += w
        last = k
        if r <= acc:
            return int(k)
    return int(last)


async def run_load(url, concurrency, n_requests, prompt_tokens, max_tokens,
                   timeout_s=300, verbose=False, length_mix=None, seed=0):
    """按指定并发发 n_requests 个请求，返回每个请求的指标列表。

    并发怎么实现：开 concurrency 个"工人"，每人循环取任务，直到任务发完。
    这样任意时刻在飞的请求数恰好等于 concurrency，和 llama-server 的槽位数对齐。
    """
    limits = httpx.Limits(max_connections=concurrency + 2,
                          max_keepalive_connections=concurrency + 2)
    sem = asyncio.Semaphore(concurrency)
    results = []
    t_origin = time.perf_counter()

    async with httpx.AsyncClient(timeout=timeout_s, limits=limits) as client:
        import random as _random
        rng = _random.Random(seed)

        async def worker(i):
            async with sem:
                # 【E2】混合长度：每个请求独立抽样，长短混在同一批里
                tgt = sample_length(length_mix, rng) if length_mix else int(prompt_tokens)
                prompt = make_prompt(tgt, nonce=i)
                row = await one_request(client, url, prompt, max_tokens, i, t_origin=t_origin)
                row["prompt_tokens_nominal"] = tgt
                results.append(row)
                if verbose and len(results) % 8 == 0:
                    print(f"    已完成 {len(results)}/{n_requests}")

        await asyncio.gather(*(worker(i) for i in range(n_requests)))

    results.sort(key=lambda r: r["req_id"])
    return results


def aggregate(rows, concurrency):
    """把单请求指标汇总成"这一格"的指标。这是写进 CSV 的那一行。"""
    ok = [r for r in rows if r["ok"] and r["ttft_ms"] and r["tpot_ms"]]
    if not ok:
        return dict(n_ok=0, n_fail=len(rows) - len(ok))

    ttft = [r["ttft_ms"] for r in ok]
    tpot = [r["tpot_ms"] for r in ok]
    n_out = [r["n_out"] for r in ok]

    # 【吞吐的唯一正确算法】总生成 token ÷ 整轮实际墙钟时间。
    #
    # 为什么不能写成 "每请求速率 × 并发数"：那样在请求会排队时必然失真——
    # 每请求耗时里已经包含了排队等待，再乘并发数就把等待重复计了一次。
    # 而吞吐又是成本的倒数（成本 = 1e6/吞吐），所以这里错一点，全项目的
    # 目标值就错一片。宁可多记两个时间戳，也不能图省事。
    mean_total_ms = statistics.fmean(r["total_ms"] for r in ok)
    if all("t_start_rel" in r for r in ok):
        t0 = min(r["t_start_rel"] for r in ok)
        t1 = max(r["t_end_rel"] for r in ok)
        span_s = max(t1 - t0, 1e-9)
        total_tokens = sum(n_out)
        throughput = total_tokens / span_s
        per_slot_tps = throughput / max(min(concurrency, len(ok)), 1)
    else:                                   # 兼容旧数据：退回近似算法
        span_s = float("nan")
        total_tokens = sum(n_out)
        per_slot_tps = statistics.fmean(n / (r["total_ms"] / 1e3) for r, n in zip(ok, n_out))
        throughput = per_slot_tps * min(concurrency, len(ok))

    return dict(
        n_ok=len(ok), n_fail=len(rows) - len(ok),
        n_out_median=statistics.median(n_out),
        prompt_tokens_median=statistics.median(
            [r["prompt_tokens"] for r in ok if r["prompt_tokens"]] or [0]),
        ttft_p50_ms=_pct(ttft, 0.50),
        ttft_p90_ms=_pct(ttft, 0.90),
        ttft_cvar90_ms=cvar(ttft, 0.90),
        ttft_max_ms=max(ttft),
        tpot_p50_ms=_pct(tpot, 0.50),
        tpot_p90_ms=_pct(tpot, 0.90),
        tpot_cvar90_ms=cvar(tpot, 0.90),
        total_p50_ms=_pct([r["total_ms"] for r in ok], 0.50),
        measure_span_s=span_s,     # 测量窗口（最早开始→最晚结束），注意不要叫 wall_s：
                                   # tc_run_cell 里另有一个 wall_s（含客户端开销的整轮耗时），
                                   # 同名会在 dict(base, ..., **agg) 里撞车：
                                   # TypeError: dict() got multiple values for keyword argument
        total_tokens=total_tokens,
        per_slot_tps=per_slot_tps,
        throughput_tps=throughput,
        cost_gpu_s_per_1m=1e6 / throughput if throughput > 0 else float("nan"),
        mean_total_ms=mean_total_ms,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default=f"{cfg.BASE_URL}/v1/chat/completions")
    ap.add_argument("--concurrency", type=int, default=1)
    ap.add_argument("--requests", type=int, default=8)
    ap.add_argument("--prompt-tokens", type=int, default=cfg.PROMPT_TOKENS)
    ap.add_argument("--max-tokens", type=int, default=cfg.GEN_TOKENS)
    ap.add_argument("--out", default=None, help="把逐请求明细写到这个 jsonl")
    args = ap.parse_args()

    t0 = time.time()
    rows = asyncio.run(run_load(args.url, args.concurrency, args.requests,
                                args.prompt_tokens, args.max_tokens, verbose=True))
    agg = aggregate(rows, args.concurrency)
    print(f"\n总耗时 {time.time()-t0:.1f}s")
    for k, v in agg.items():
        print(f"  {k:<20} {v if isinstance(v, (int, str)) else f'{v:.3f}'}")

    if args.out:
        p = Path(args.out)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"\n逐请求明细已写入 {p}")


if __name__ == "__main__":
    main()
