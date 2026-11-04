# -*- coding: utf-8 -*-
"""
A-1 分位数模型：从"预测 p50"到"预测 p90 / p99"

## 为什么做这个

推荐器的 SLA 判定用的是 `CVaR90`，而尾部是**按对数正态 1.645σ 假设出来的**——
从来没有用实测数据验证过。`raw_requests.jsonl` 里有**每一个请求**的延迟，
所以这件事完全可以在本机算清楚。

## 做什么

1. 每格算 TTFT / TPOT 的 p50、p90、p99；
2. 看 `p90/p50`、`p99/p50` 随**并发**与**模型**怎么变；
3. 判断：现有的"尾部 = 1.645σ"假设，是偏乐观还是偏保守。

## 产出

- `results/quantile_tails.csv`：每格的 p50/p90/p99 与比值
- `results/quantile_tails.json`：汇总（各并发的比值中位数）
- `figures/fig11_tail_ratio.png`：p90/p50 与 p99/p50 随并发的变化
"""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import tc_config as cfg

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

TAIL_Q = (50, 90, 99)


def load_requests():
    rows = []
    with open(cfg.DATA_DIR / "raw_requests.jsonl", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    df = pd.DataFrame(rows)
    # 只保留有效请求（探测格 _p 也算，它们有独立的长度信息）
    df = df[(df["ok"] == True) & df["ttft_ms"].notna() & df["tpot_ms"].notna()]
    return df


def per_cell(df):
    out = []
    for cid, g in df.groupby("cell_id"):
        if len(g) < 8:            # 少于 8 个请求算不出稳定的 p99
            continue
        ttft = np.asarray(g["ttft_ms"], dtype=float)
        tpot = np.asarray(g["tpot_ms"], dtype=float)
        row = dict(
            cell_id=cid, model=g["model"].iloc[0], quant=g["quant"].iloc[0],
            ctx=int(g["ctx"].iloc[0]), conc=int(g["conc"].iloc[0]),
            prompt_tokens=float(g["prompt_tokens"].median()) if "prompt_tokens" in g else np.nan,
            n_requests=len(g),
        )
        for q in TAIL_Q:
            row[f"ttft_p{q}"] = float(np.percentile(ttft, q))
            row[f"tpot_p{q}"] = float(np.percentile(tpot, q))
        row["ttft_p90_over_p50"] = row["ttft_p90"] / row["ttft_p50"]
        row["ttft_p99_over_p50"] = row["ttft_p99"] / row["ttft_p50"]
        row["tpot_p90_over_p50"] = row["tpot_p90"] / row["tpot_p50"]
        row["tpot_p99_over_p50"] = row["tpot_p99"] / row["tpot_p50"]
        # 变异系数（对数空间），用来和"1.645σ"的假设对照
        lg = np.log(ttft)
        row["ttft_sigma_log"] = float(lg.std(ddof=1))
        row["ttft_z90_implied"] = row["ttft_p90_over_p50"]   # 对数正态下 p90/p50 = exp(1.2816σ)
        row["ttft_sigma_implied"] = (
            float(np.log(row["ttft_p90_over_p50"]) / 1.2816)
            if row["ttft_p90_over_p50"] > 1 else 0.0
        )
        out.append(row)
    return pd.DataFrame(out)


def main():
    df = load_requests()
    print(f"请求总数 {len(df)}，格数 {df['cell_id'].nunique()}")

    t = per_cell(df)
    t = t.sort_values(["model", "conc", "prompt_tokens"])
    t.to_csv(cfg.RESULTS_DIR / "quantile_tails.csv", index=False, encoding="utf-8")
    print(f"逐格结果 {len(t)} 行 → results/quantile_tails.csv")

    # ---- 按并发汇总 ----
    summ = {}
    for metric in ("ttft", "tpot"):
        for q in (90, 99):
            col = f"{metric}_p{q}_over_p50"
            by = t.groupby("conc")[col].median()
            summ[f"{col}_median_by_conc"] = {int(k): round(float(v), 4) for k, v in by.items()}
    # 与"对数正态 1.645σ 尾部"的对照：现有 cvar90 用的是 exp(1.645σ)
    sig = t["ttft_sigma_log"].median()
    implied = float(np.exp(1.645 * sig))
    summ["ttft_sigma_log_median"] = round(float(sig), 4)
    summ["cvar90_implied_from_sigma"] = round(implied, 4)
    summ["cvar90_actual_p95_over_p50"] = round(
        float((t["ttft_p90_over_p50"]).median()), 4)

    (cfg.RESULTS_DIR / "quantile_tails.json").write_text(
        json.dumps(summ, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n=== TTFT 尾部随并发（各格中位数）===")
    print(f"  {'并发':>4}{'p90/p50':>10}{'p99/p50':>10}{'格数':>6}")
    for c, g in t.groupby("conc"):
        print(f"  {int(c):>4}{g['ttft_p90_over_p50'].median():>10.2f}"
              f"{g['ttft_p99_over_p50'].median():>10.2f}{len(g):>6}")

    print("\n=== TPOT 尾部随并发（各格中位数）===")
    print(f"  {'并发':>4}{'p90/p50':>10}{'p99/p50':>10}")
    for c, g in t.groupby("conc"):
        print(f"  {int(c):>4}{g['tpot_p90_over_p50'].median():>10.2f}"
              f"{g['tpot_p99_over_p50'].median():>10.2f}")

    print(f"\n=== 与现有假设的对照 ===")
    print(f"  TTFT 对数标准差中位数 σ = {sig:.3f}")
    print(f"  现有 CVaR90 假设（exp(1.645σ)）= {implied:.3f} × p50")
    print(f"  实测 TTFT p90/p50 中位数      = {t['ttft_p90_over_p50'].median():.3f} × p50")
    print(f"  实测 TTFT p99/p50 中位数      = {t['ttft_p99_over_p50'].median():.3f} × p50")
    # 【关键】尾部不是常数——它随并发增长。所以判定必须逐并发做，
    # 用全体中位数会掩盖"低并发布尾部准、高并发布尾部严重偏窄"这件事。
    print(f"\n  {'并发':>4}{'实测p90/p50':>13}{'假设':>8}{'偏差':>9}  判定")
    by_conc = {}
    for c, g in t.groupby("conc"):
        m90 = float(g["ttft_p90_over_p50"].median())
        ratio = m90 / implied
        tag = ("假设偏窄" if ratio > 1.15 else
               "接近" if ratio > 0.87 else "假设偏宽")
        by_conc[int(c)] = dict(measured_p90_over_p50=round(m90, 3),
                               assumed=round(implied, 3),
                               ratio=round(ratio, 3), tag=tag)
        print(f"  {int(c):>4}{m90:>13.2f}{implied:>8.2f}{ratio:>8.2f}×  {tag}")
    bad = [c for c, v in by_conc.items() if v["tag"] == "假设偏窄"]
    if bad:
        verdict = (f"⚠️ 现有尾部假设只在低并发成立；并发 ≥{min(bad)} 时"
                   f"实测 p90 比假设宽 {(max(v['ratio'] for v in by_conc.values())):.1f}× ——"
                   f"SLA 判定偏乐观")
    else:
        verdict = "尾部假设在各并发档都够用"
    print(f"\n  → 判定：{verdict}")
    summ["per_conc"] = by_conc
    summ["verdict"] = verdict
    (cfg.RESULTS_DIR / "quantile_tails.json").write_text(
        json.dumps(summ, ensure_ascii=False, indent=2), encoding="utf-8")

    # ---- 图 ----
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    models = sorted(t["model"].unique())
    cmap = plt.get_cmap("tab10")
    for i, m in enumerate(models):
        g = t[t["model"] == m].groupby("conc")[["ttft_p90_over_p50", "ttft_p99_over_p50"]].median()
        if g.empty:
            continue
        axes[0].plot(g.index, g["ttft_p90_over_p50"], "o-", color=cmap(i % 10), label=f"{m} p90/p50")
        axes[0].plot(g.index, g["ttft_p99_over_p50"], "s--", color=cmap(i % 10), alpha=0.55,
                     label=f"{m} p99/p50")
    axes[0].axhline(implied, color="crimson", lw=1.2, ls=":",
                    label=f"现有 CVaR90 假设 {implied:.2f}×")
    axes[0].set_xscale("log", base=2)
    axes[0].set_xlabel("并发")
    axes[0].set_ylabel("TTFT 尾部 / 中位数")
    axes[0].set_title("TTFT 尾部倍数（实测 vs 假设）")
    axes[0].legend(fontsize=7, ncol=2)
    axes[0].grid(alpha=0.3)

    for i, m in enumerate(models):
        g = t[t["model"] == m].groupby("conc")[["tpot_p90_over_p50", "tpot_p99_over_p50"]].median()
        if g.empty:
            continue
        axes[1].plot(g.index, g["tpot_p90_over_p50"], "o-", color=cmap(i % 10), label=f"{m} p90/p50")
        axes[1].plot(g.index, g["tpot_p99_over_p50"], "s--", color=cmap(i % 10), alpha=0.55,
                     label=f"{m} p99/p50")
    axes[1].set_xscale("log", base=2)
    axes[1].set_xlabel("并发")
    axes[1].set_ylabel("TPOT 尾部 / 中位数")
    axes[1].set_title("TPOT 尾部倍数")
    axes[1].legend(fontsize=7, ncol=2)
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(cfg.FIG_DIR / "fig11_tail_ratio.png", dpi=150)
    print("\n图 → figures/fig11_tail_ratio.png")
    return summ


if __name__ == "__main__":
    main()

