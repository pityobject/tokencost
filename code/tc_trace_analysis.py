# -*- coding: utf-8 -*-
"""
真实 trace 回放分析（`tc.py trace`，M1）

## 数据

BurstGPT（真实 ChatGPT / GPT-4 生产负载，经 GitHub API 获取，48.5 MB）：
`Timestamp, Model, Request tokens, Response tokens, Total tokens, Log Type`

## 要回答四个问题

1. **到达过程是泊松吗？** 我们排队层用的是指数分布假设。
   判据：到达间隔的 CV。泊松 → CV≈1；突发 → CV>1；规整 → CV<1。
2. **长度分布形状对吗？** 我们假设 prompt 对数正态。实测偏度/尾部是什么样？
3. **突发有多强？** 用 10 秒窗口的到达率离散度衡量，看真实负载是不是"平稳泊松"。
4. **对"容量上界 × 0.7"意味着什么？** 如果真实是突发，那 0.7 可能不够保守。

## 用法

    python code/tc_trace_analysis.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import tc_config as cfg

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
TRACE = Path(cfg.__file__).resolve().parent.parent / "data" / "traces" / "BurstGPT_1.csv"


def main():
    if not TRACE.exists():
        print(f"❌ 找不到 {TRACE}")
        return
    print("=" * 86)
    print("一、读取真实 trace")
    print("=" * 86)
    df = pd.read_csv(TRACE)
    print(f"  文件: {TRACE.name}（{TRACE.stat().st_size/1024/1024:.1f} MB）")
    print(f"  行数: {len(df):,}｜列: {list(df.columns)}")
    df["Timestamp"] = pd.to_numeric(df["Timestamp"], errors="coerce")
    df = df.dropna(subset=["Timestamp"]).sort_values("Timestamp").reset_index(drop=True)
    span = df["Timestamp"].iloc[-1] - df["Timestamp"].iloc[0]
    print(f"  时间跨度: {span/3600:.2f} 小时（{span:.0f} 秒）")
    print(f"  平均到达率: {len(df)/span*60:.1f} 请求/分钟 "
          f"（{len(df)/span:.3f} 请求/秒）")
    print(f"  模型分布: {df['Model'].value_counts().to_dict()}")

    # ---------------- 到达过程 ----------------
    print("\n" + "=" * 86)
    print("二、到达过程：是泊松吗？")
    print("=" * 86)
    dt = np.diff(df["Timestamp"].to_numpy(dtype=float))
    dt = dt[dt > 0]
    cv = float(np.std(dt) / np.mean(dt))
    # 指数分布的理论 CV = 1
    print(f"  到达间隔：均值 {dt.mean():.2f} s｜中位 {np.median(dt):.2f} s｜"
          f"CV = **{cv:.3f}**")
    verdict_arrival = ("接近泊松（CV≈1）" if 0.8 <= cv <= 1.25 else
                       ("突发性强（CV>1.25，重尾间隔）" if cv > 1.25 else "比泊松更规整（CV<0.8）"))
    print(f"  → {verdict_arrival}")
    p_exp = 1 - np.exp(-dt / dt.mean())
    # KS 检验：与指数分布比较
    from scipy import stats as st
    ks = st.kstest(np.sort(p_exp), "uniform")
    print(f"  与指数分布的一致性（KS）：D={ks.statistic:.4f}，p={ks.pvalue:.2e}"
          f" → {'不能拒绝泊松' if ks.pvalue > 0.05 else '显著偏离泊松'}")

    # 窗口突发性：10 秒窗口的到达数离散度
    win = 10.0
    bins = np.floor((df["Timestamp"] - df["Timestamp"].iloc[0]) / win).astype(int)
    counts = bins.value_counts().reindex(range(bins.max() + 1), fill_value=0).to_numpy()
    idx = np.std(counts) / np.mean(counts)
    print(f"\n  {win:.0f} 秒窗口：平均 {counts.mean():.2f} 个/窗口，"
          f"离散指数 σ/μ = **{idx:.2f}**（泊松应为 1.0）")
    verdict_burst = ("平稳泊松" if 0.85 <= idx <= 1.2 else
                     ("存在突发（超泊松）" if idx > 1.2 else "比泊松更均匀"))

    # ---------------- 长度分布 ----------------
    print("\n" + "=" * 86)
    print("三、长度分布：与我们假设的对数正态比")
    print("=" * 86)
    for col in ("Request tokens", "Response tokens"):
        x = pd.to_numeric(df[col], errors="coerce").dropna()
        x = x[x > 0]
        if x.empty:
            continue
        mu, sd = float(x.mean()), float(x.std())
        cvx = sd / mu
        # 对数正态拟合
        ln = np.log(x)
        shape = float(ln.std())
        theo = np.sqrt(np.exp(shape ** 2) - 1)
        p95 = float(x.quantile(0.95))
        p99 = float(x.quantile(0.99))
        print(f"  {col}：n={len(x):,}｜均值 {mu:.0f}｜中位 {x.median():.0f}｜"
              f"CV {cvx:.2f}｜p95 {p95:.0f}｜p99 {p99:.0f}")
        print(f"      对数正态拟合 σ={shape:.2f} → 理论 CV={theo:.2f}"
              f"（{'吻合' if abs(theo-cvx)/max(cvx,1e-9) < 0.25 else '偏离'}）")
        if col == "Request tokens":
            req_stats = dict(n=int(len(x)), mean=round(mu, 1), cv=round(cvx, 3),
                             p95=round(p95, 1), p99=round(p99, 1),
                             lognormal_sigma=round(shape, 3),
                             lognormal_cv=round(float(theo), 3))

    # ---------------- 对容量结论的影响 ----------------
    print("\n" + "=" * 86)
    print("四、对『容量上界 × 0.7』意味着什么")
    print("=" * 86)
    print(f"  我们排队层假设：泊松到达（指数间隔）。真实 trace 的间隔 CV = {cv:.2f}。")
    if cv > 1.25 or idx > 1.2:
        print("  → **真实负载比泊松更突发**。在突发下，尾延迟（p90/p99）会比泊松模型更差，")
        print("     所以『×0.7』这个折减**方向是对的，但可能不够**——")
        print("     突发越强，需要的折减越深。")
    else:
        print("  → 真实负载接近泊松，我们排队层的假设基本站得住。")
    print(f"  另外：真实 prompt 的 CV = {req_stats['cv']:.2f}"
          f"（我们仿真时用的是对数正态 σ≈0.5–1.0，对应 CV 0.5–1.3）")

    out = dict(trace="BurstGPT_1.csv", rows=int(len(df)), span_s=round(float(span), 1),
               mean_rate_rps=round(float(len(df) / span), 4),
               interarrival=dict(mean_s=round(float(dt.mean()), 3),
                                 median_s=round(float(np.median(dt)), 3),
                                 cv=round(cv, 3), ks_D=round(float(ks.statistic), 4),
                                 ks_p=float(ks.pvalue), verdict=verdict_arrival),
               burstiness_10s=dict(mean=round(float(counts.mean()), 3),
                                   sigma_over_mu=round(float(idx), 3),
                                   verdict=verdict_burst),
               request_tokens=req_stats,
               note=("真实生产 trace（ChatGPT/GPT-4）；由 GitHub API 获取，"
                     "仅存于 data/traces/（已 gitignore），仓库只放统计结果"))
    (cfg.RESULTS_DIR / "trace_stats.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n→ results/trace_stats.json")


if __name__ == "__main__":
    main()
