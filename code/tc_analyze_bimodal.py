# -*- coding: utf-8 -*-
"""
E1：c16 的双峰从哪来——热状态，还是批次状态？

## 背景

`3B/ctx512/c16` 这个格跑过两批：
- 第一批三行 TPOT ≈ 23.7–24.1 ms
- 第二批五行 TPOT ≈ 21.8–22.9 ms

**时钟都是 2475 MHz、功耗都在 62–65 W**，但差 8.6%。所以不是简单的降频。

## 两种假设

| 假设 | 预期特征 |
|---|---|
| **H1 热状态** | 同一批**内部**逐轮变慢；TPOT 与温度正相关、与时钟负相关 |
| **H2 批次状态** | 批**内部**平稳；批与批之间跳变（显存碎片、驱动状态、机器负载历史） |

## 判据（写在脚本里，先定后算）

对每一批做线性回归 `TPOT ~ 轮次`：
- 若**多数批**的斜率显著为正（每轮 >+0.5%）→ **H1**
- 若各批内部斜率≈0、但批均值之间差 >3% → **H2**
- 两者都成立 → 混合（先热后批次）

## 用法

    python tc_run_cell.py 3B q4_k_m 512 16 --repeats 4
    （等 2 分钟）
    python tc_run_cell.py 3B q4_k_m 512 16 --repeats 4
    （等 2 分钟）
    python tc_run_cell.py 3B q4_k_m 512 16 --repeats 4
    python tc_analyze_bimodal.py --cell 3B_q4_k_m_ctx512_c16

## 产出

- `results/bimodal_<cell>.json`
- `figures/fig14_bimodal.png`
"""
import argparse
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

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def split_batches(rows):
    """按 repeat 回退切批（文件顺序）。"""
    batches, cur, prev = [], [], None
    for r in rows:
        rep = int(r["repeat"])
        if prev is not None and rep <= prev and cur:
            batches.append(cur)
            cur = []
        cur.append(r)
        prev = rep
    if cur:
        batches.append(cur)
    return batches


def slope_pct(xs, ys):
    """相对斜率：每轮变化占均值的百分比。"""
    if len(xs) < 2:
        return float("nan")
    k = np.polyfit(xs, ys, 1)[0]
    m = float(np.mean(ys))
    return float(k / m * 100) if m else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", default="3B_q4_k_m_ctx512_c16")
    args = ap.parse_args()

    rows = [json.loads(l) for l in (cfg.DATA_DIR / "raw_cells.jsonl").open(encoding="utf-8") if l.strip()]
    mine = [r for r in rows if r.get("cell_id") == args.cell and r.get("status") == "ok"]
    if len(mine) < 4:
        print(f"{args.cell} 只有 {len(mine)} 行，先跑三批（每批 ≥3 轮）。")
        return

    batches = split_batches(mine)
    print(f"{args.cell}：共 {len(mine)} 行，切出 {len(batches)} 批\n")

    recs = []
    for i, b in enumerate(batches, 1):
        t = np.array([r["tpot_p50_ms"] for r in b], dtype=float)
        temp = np.array([r.get("temp_c_median") or np.nan for r in b], dtype=float)
        clk = np.array([r.get("clock_sm_median") or np.nan for r in b], dtype=float)
        sl = slope_pct(list(range(len(b))), t)
        recs.append(dict(batch=i, n=len(b), tpot_mean=float(t.mean()),
                         tpot_std=float(t.std(ddof=1)) if len(t) > 1 else 0.0,
                         slope_pct=sl,
                         temp_mean=float(np.nanmean(temp)) if np.isfinite(temp).any() else None,
                         clock_mean=float(np.nanmean(clk)) if np.isfinite(clk).any() else None,
                         tpots=[float(x) for x in t]))
        print(f"  批{i}: n={len(b)}  TPOT 均值 {t.mean():6.2f}  "
              f"批内斜率 {sl:+.2f}%/轮  温度 {recs[-1]['temp_mean']}  "
              f"时钟 {recs[-1]['clock_mean']}")

    means = [r["tpot_mean"] for r in recs]
    batch_spread = (max(means) / min(means) - 1) * 100 if len(means) > 1 else 0.0
    slopes = [r["slope_pct"] for r in recs if np.isfinite(r["slope_pct"])]
    positive = [s for s in slopes if s > 0.5]

    ms = float(np.mean(slopes)) if slopes else float("nan")
    if len(recs) < 2:
        verdict = "只有一批，无法区分热/批次；请再跑两批（中间等 2 分钟）"
    elif batch_spread < 1.5:
        verdict = f"本次无批次差异（批间 {batch_spread:.1f}% < 1.5%）"
    elif ms > 0.5:
        verdict = "H1 热状态：批内逐轮明显变慢（越跑越热）"
    elif batch_spread > 3 and abs(ms) < 0.5:
        verdict = "H2 批次状态：批内平稳、批间跳变"
    elif ms < -0.5:
        verdict = ("批间有差异，但批内是**变快**的（斜率 %+.2f%%/轮）——"
                   "不符合'越跑越热'，更像批次状态或温度已饱和；建议再跑第三批确认" % ms)
    else:
        verdict = f"混合：批间差 {batch_spread:.1f}%，批内斜率 {ms:+.2f}%/轮"

    # 温度与延迟的关系（跨批）
    pairs = [(r["temp_mean"], r["tpot_mean"]) for r in recs if r["temp_mean"]]
    if len(pairs) >= 2:
        tt = np.array([p[0] for p in pairs]); yy = np.array([p[1] for p in pairs])
        if tt.std() > 0:
            rr = float(np.corrcoef(tt, yy)[0, 1])
            print(f"\n  跨批：温度 {tt.min():.0f}–{tt.max():.0f}°C，"
                  f"与 TPOT 的相关 r={rr:+.2f}"
                  + ("（温度越高越慢 → 支持热状态）" if rr > 0.5
                     else "（温度与延迟无正相关 → 不支持单纯散热解释）"))
    print(f"\n  批间最大差 {batch_spread:.2f}%；批内斜率均值 "
          f"{np.mean(slopes) if slopes else float('nan'):+.2f}%/轮")
    print(f"  → 判定：{verdict}")

    out = dict(cell=args.cell, n_rows=len(mine), n_batches=len(batches),
               batch_spread_pct=round(batch_spread, 3),
               slope_pct_mean=round(float(np.mean(slopes)), 3) if slopes else None,
               verdict=verdict, batches=recs)
    p = cfg.RESULTS_DIR / f"bimodal_{args.cell}.json"
    p.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n→ {p}")

    # ---- 图：TPOT vs 全局轮次，按批分色 ----
    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    cmap = plt.get_cmap("tab10")
    k = 0
    for i, b in enumerate(batches):
        ys = [r["tpot_p50_ms"] for r in b]
        xs = list(range(k, k + len(ys)))
        ax.plot(xs, ys, "o-", color=cmap(i % 10), label=f"批{i+1}（温度 {recs[i]['temp_mean']}°C）")
        k += len(ys)
    ax.set_xlabel("全局轮次（按运行顺序）")
    ax.set_ylabel("TPOT (ms)")
    ax.set_title(f"{args.cell}：批内趋势 vs 批间跳变")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(cfg.FIG_DIR / "fig14_bimodal.png", dpi=150)
    print("图 → figures/fig14_bimodal.png")


if __name__ == "__main__":
    main()



