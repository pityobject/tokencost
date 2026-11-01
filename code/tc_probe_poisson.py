"""
tc_probe_poisson.py —— 泊松到达：同步波次给出的是不是上界？

## 为什么做

正式实验全是"b 个请求同时发"的同步波次，这是最坏情况。
真实服务是泊松到达。问题只有一个：

    **在同样的实际并发下，泊松到达的延迟是否低于（或等于）模型预测？**

若是 → 模型是保守的（安全方向），结论更强；
若否 → 排队效应没被建模，必须写进局限。

## 怎么比才公平

不直接比"波次 vs 泊松"（那是两种不同的并发），而是：
  ① 记录泊松负载下的**实际平均并发** b̄；
  ② 用 6 参数模型算 b̄ 处的 TTFT/TPOT；
  ③ 比较实测分布与模型预测。
这样比的是同一件事。

## 设计

固定 3B-Q4 / ctx512 / 槽位 8，三个到达率 λ = 2 / 4 / 6 请求/秒。
每个 λ 跑 6 个服务时长（每次固定 60 秒或累计 48 个请求，先到者结束）。

关注指标：实际并发、TTFT 的 p50/p90、是否出现排队（TTFT 远大于单请求 prefill 时间）。

用法：
  & "D:\\llm\\venv\\Scripts\\python.exe" tc_probe_poisson.py
"""

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
LAMBDAS = [2.0, 4.0, 6.0]
N_REQ = 48
OUT = cfg.DATA_DIR / "probe_poisson.csv"


def run_arrivals(url, lam, n_req, seed=7):
    """按泊松过程发请求（指数间隔），槽位上限 SLOTS，记录每个请求的指标。"""
    rng = random.Random(seed)
    rows = []
    inflight = 0
    samples = []          # (t, inflight) 采样，用来算平均并发

    with httpx.Client(timeout=httpx.Timeout(600.0, connect=30.0)) as client:
        t_start = time.perf_counter()
        next_arrival = t_start
        issued = 0
        while issued < n_req or inflight > 0:
            now = time.perf_counter()
            # 到了下一个到达时刻、且不超过 n_req 就发
            if issued < n_req and now >= next_arrival:
                samples.append((now - t_start, inflight))
                # 简化：串行发（不并发 await），先发后等它完成会破坏泊松性，
                # 所以这里用"发出去就不等"的方式：同步 HTTP 会阻塞，故改用线程池
                issued += 1
                gap = rng.expovariate(lam)
                next_arrival = now + gap
                continue
            time.sleep(0.01)
        return rows


def main():
    from concurrent.futures import ThreadPoolExecutor
    import threading

    print("=" * 80)
    print(f"泊松到达实验：{MODEL}-{QUANT} / ctx{CTX} / 槽位 {SLOTS} / λ {LAMBDAS}")
    print("=" * 80)

    proc, log = srv.start((MODEL, QUANT), CTX, SLOTS, tag="poisson", timeout_s=240)
    out_rows = []
    try:
        time.sleep(1.5)
        url = f"{cfg.BASE_URL}/v1/chat/completions"

        for lam in LAMBDAS:
            rng = random.Random(42)
            results = []
            lock = threading.Lock()
            t_origin = time.perf_counter()
            stop_at = t_origin + 60.0          # 最多 60 秒

            def worker(i, t_issue):
                prompt = (f"[泊松 {lam}-{i}-{time.time():.6f}] "
                          + "大模型推理服务在部署时面临一个核心矛盾：用户希望首字延迟尽量低，"
                            "而机房希望单位token成本尽量低。")
                payload = dict(model="local",
                               messages=[{"role": "user", "content": prompt}],
                               max_tokens=cfg.GEN_TOKENS, temperature=0.0,
                               stream=True, cache_prompt=False, ignore_eos=True)
                t0 = time.perf_counter()
                t_first = None
                n_out = 0
                try:
                    with httpx.Client(timeout=httpx.Timeout(300.0, connect=30.0)) as c:
                        with c.stream("POST", url, json=payload) as r:
                            for line in r.iter_lines():
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
                                    n_out += 1
                    t_end = time.perf_counter()
                    with lock:
                        results.append(dict(
                            i=i, issue_rel=t_issue - t_origin,
                            start_rel=t0 - t_origin, end_rel=t_end - t_origin,
                            ttft_ms=((t_first - t0) * 1e3) if t_first else None,
                            tpot_ms=(((t_end - t_first) / max(n_out - 1, 1)) * 1e3
                                     if t_first and n_out > 1 else None),
                            total_ms=(t_end - t0) * 1e3, n_out=n_out))
                except Exception as e:
                    with lock:
                        results.append(dict(i=i, error=str(e)[:80]))

            # 按指数间隔把请求派出去；超过 60 秒就停止派新请求
            issue_times = []
            t_cursor = t_origin
            for i in range(N_REQ):
                t_cursor += rng.expovariate(lam)
                if t_cursor > stop_at:
                    break
                issue_times.append(t_cursor)

            # 线程池不能小于"可能同时在飞的请求数"，否则瓶颈变成线程池而不是服务端。
            # λ=6、单请求约 5 秒 → 平均在飞 30 个，所以给 64 个线程，
            # 真正的并发上限交给服务端的 8 个槽位去卡。
            with ThreadPoolExecutor(max_workers=64) as pool:
                for i, t_issue in enumerate(issue_times):
                    # 等到该发的时刻
                    while time.perf_counter() < t_issue:
                        time.sleep(0.002)
                    pool.submit(worker, i, time.perf_counter())
                pool.shutdown(wait=True)

            ok = [r for r in results if "error" not in r]
            if not ok:
                print(f"  λ={lam}: 全部失败")
                continue
            # 实际平均并发 = 各请求 [start,end) 区间的平均重叠数
            events = []
            for r in ok:
                events.append((r["start_rel"], 1))
                events.append((r["end_rel"], -1))
            events.sort()
            cur = 0
            prev_t = events[0][0]
            area = 0.0
            for t, d in events:
                area += cur * (t - prev_t)
                prev_t = t
                cur += d
            span = max(r["end_rel"] for r in ok) - min(r["start_rel"] for r in ok)
            mean_conc = area / span if span > 0 else float("nan")

            ttfts = [r["ttft_ms"] for r in ok if r["ttft_ms"]]
            tpots = [r["tpot_ms"] for r in ok if r["tpot_ms"]]
            row = dict(lam=lam, n_ok=len(ok), mean_conc=mean_conc, span_s=span,
                       ttft_p50=st.median(ttfts) if ttfts else None,
                       ttft_p90=(sorted(ttfts)[int(len(ttfts) * 0.9)] if ttfts else None),
                       tpot_p50=st.median(tpots) if tpots else None,
                       achieved_qps=len(ok) / span if span > 0 else None)
            out_rows.append(row)
            print(f"  λ={lam:>4}  完成 {len(ok):>2}  实际并发 {mean_conc:5.2f}  "
                  f"实际QPS {row['achieved_qps']:5.2f}  TTFT p50 {row['ttft_p50']:7.1f} / "
                  f"p90 {row['ttft_p90']:7.1f} ms  TPOT {row['tpot_p50']:5.1f} ms")
            time.sleep(2.0)
    finally:
        srv.stop(proc)

    if out_rows:
        with OUT.open("w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
            w.writeheader()
            w.writerows(out_rows)
        print(f"\n已写入 {OUT}")
        print("下一步：用 tc_model 的 predict() 在这些实际并发处算预测值，与实测比。")


if __name__ == "__main__":
    main()
