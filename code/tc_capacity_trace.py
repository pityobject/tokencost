# -*- coding: utf-8 -*-
"""
容量折减：用真实 trace 把"×0.7"这个拍出来的常数换成模拟值（`tc.py capacity`）

## 问题

§5.14 的"容量上界 × 0.7"是经验：Poisson 探针给出 2.47 请求/秒的理论上限，
我们保守地乘 0.7 作为可用容量。但 M1 发现真实到达的突发度是泊松的 **3 倍**——
那 0.7 到底够不够？没人算过。

## 做法

单服务台 FCFS 队列、**同一套服务时间分布**，只换到达序列：

1. **真实序列**：BurstGPT 的时间戳
2. **泊松序列**：同平均到达率、同长度

对每条序列，按"目标 p95 延迟"求最大可持续到达率，两者之比就是
**突发性导致的容量折减**。这比"×0.7"有依据得多。

## 服务时间

用本项目模型算单流服务时间：
`s_i = TTFT(L_i, b=1) + gen_i × TPOT(b=1)`
（固定 `gen`，只用 prompt 长度分布；这是保守的"单槽串行"基线）

## 用法

    python code/tc_capacity_trace.py --n 120000 --gen 36
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import tc_config as cfg
import tc_model as tm

sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def wait_times(arrivals, service):
    """FCFS 单台：返回每个请求的完成时间与等待时间（向量化不够，用累积法）。"""
    n = len(arrivals)
    wait = np.empty(n)
    w = 0.0
    prev = arrivals[0]
    for i in range(n):
        if i > 0:
            gap = arrivals[i] - prev
            w = max(0.0, w + service[i - 1] - gap)
            prev = arrivals[i]
        wait[i] = w
    return wait


def simulate(arrivals, service, sla_ms, step=2000):
    """按到达序列前缀逐步放宽到达率，找满足 p95 延迟的最大缩放因子。"""
    best = 0.0
    lo, hi = 0.5, 500.0   # 允许把时间轴压缩（提高到达率）
    for _ in range(22):
        mid = (lo + hi) / 2
        a = arrivals[:step] / mid if mid > 0 else arrivals[:step]
        s = service[:step]
        w = wait_times(a, s)
        p95 = np.percentile(w + s, 95) * 1e3
        if p95 <= sla_ms:
            best = mid
            lo = mid
        else:
            hi = mid
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=120000)
    ap.add_argument("--gen", type=int, default=36, help="生成长度（真实中位 36）")
    ap.add_argument("--sla", type=float, default=2000.0, help="p95 端到端延迟上限（ms）")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    trace = Path(cfg.__file__).resolve().parent.parent / "data" / "traces" / "BurstGPT_1.csv"
    if not trace.exists():
        print(f"❌ 缺 {trace}（M1 的 trace 数据不进仓库）")
        return

    df = pd.read_csv(trace, nrows=args.n)
    df["Timestamp"] = pd.to_numeric(df["Timestamp"], errors="coerce")
    df["Request tokens"] = pd.to_numeric(df["Request tokens"], errors="coerce")
    df = df.dropna(subset=["Timestamp", "Request tokens"]).sort_values("Timestamp")
    L = df["Request tokens"].to_numpy(float)
    t = df["Timestamp"].to_numpy(float)
    span = t[-1] - t[0]
    print("=" * 84)
    print(f"数据：{min(len(t), args.n):,} 条真实请求｜跨度 {span/3600:.1f} 小时｜"
          f"平均到达率 {len(t)/span:.3f} 请求/秒")
    print(f"服务时间：模型单流 TTFT(L,1) + {args.gen}×TPOT(1)，SLA = p95 {args.sla:.0f} ms")
    print("=" * 84)

    # 用本机拟合参数
    fit = json.loads((cfg.RESULTS_DIR / "fit.json").read_text(encoding="utf-8"))
    params = dict(bw_gbs=fit["bw"], peak_tflops=fit["peak"],
                  peak_prefill_tflops=fit["peak_prefill"],
                  c_srv_ms=fit["c_srv"], b_sat=fit["b_sat"],
                  ovh_ttft_ms=fit["ovh_ttft"])
    cfg.MODEL_SPECS[("3B", "q4_k_m")] = cfg.MODEL_SPECS.get(
        ("3B", "q4_k_m"), dict(weight_bytes=2.1e9, n_params=3.08e9, n_layers=36,
                               n_kv_heads=2, head_dim=128))
    svc = np.empty(len(L))
    for i in range(len(L)):
        row = dict(model="3B", quant="q4_k_m", ctx=2048, conc=1,
                   prompt_tokens=float(L[i]), ctx_avg_tokens=float(L[i]) + args.gen / 2)
        pr = tm.predict(params, row)
        svc[i] = (pr["pred_ttft_ms"] + args.gen * pr["pred_tpot_ms"]) / 1e3
    print(f"  服务时间：均值 {svc.mean()*1e3:.0f} ms｜p95 {np.percentile(svc,95)*1e3:.0f} ms"
          f"｜理论单流上限 {1/svc.mean():.2f} 请求/秒")

    # 真实序列的最大缩放
    scale_real = simulate(t, svc, args.sla)
    cap_real = len(t) / span * scale_real
    print(f"\n  真实序列：可行到达率 = {cap_real:.3f} 请求/秒"
          f"（= 原始 {len(t)/span:.3f} × {scale_real:.2f}）")

    # 泊松序列：同率、同长度
    rng = np.random.default_rng(42)
    rate = len(t) / span
    gaps_pois = rng.exponential(1 / rate, size=args.n)
    t_pois = np.cumsum(gaps_pois)
    scale_pois = simulate(t_pois, svc, args.sla)
    cap_pois = rate * scale_pois
    print(f"  泊松序列：可行到达率 = {cap_pois:.3f} 请求/秒"
          f"（= 原始 {rate:.3f} × {scale_pois:.2f}）")

    print("\n" + "=" * 84)
    print("结论")
    print("=" * 84)
    ratio = cap_pois / cap_real if cap_real > 0 else float("nan")
    print(f"  突发性导致的容量折减：**{ratio:.2f}×**")
    print(f"  即：同样的 SLA 下，真实到达只能跑到泊松假设的 "
          f"**{cap_real/cap_pois*100:.0f}%**")
    print(f"\n  对照我们原来的经验折减 ×0.7：")
    print(f"    实测需要的是 **×{cap_real/cap_pois:.2f}**"
          f" {'（比 0.7 更保守，符合预期）' if cap_real/cap_pois < 0.7 else '（比 0.7 宽松）'}")

    out = dict(n=int(len(t)), span_h=round(float(span / 3600), 2),
               rate_rps=round(float(rate), 4), gen=args.gen, sla_ms=args.sla,
               service_ms=dict(mean=round(float(svc.mean() * 1e3), 1),
                               p95=round(float(np.percentile(svc, 95) * 1e3), 1),
                               single_stream_ceiling_rps=round(float(1 / svc.mean()), 3)),
               capacity_real_rps=round(float(cap_real), 4),
               capacity_poisson_rps=round(float(cap_pois), 4),
               burstiness_derating=round(float(ratio), 3),
               heuristic_derating=0.7,
               note=("FCFS 单台、同服务时间分布，只换到达序列；"
                     "服务时间用本机模型单流预测。用来检验 ×0.7 这条经验"))
    fp = Path(args.out) if args.out else (cfg.RESULTS_DIR / "capacity_trace.json")
    fp.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n→ {fp}")


if __name__ == "__main__":
    main()
