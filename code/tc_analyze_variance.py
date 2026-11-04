# -*- coding: utf-8 -*-
"""
B-1 重复性方差：把 `8.70%` 从点估计变成区间

## 问题

现在每格重复 3 轮取中位数，但**从来没报过"重复测量本身有多大波动"**。
于是"MAPE 8.70%"这个数字的置信度是未知的——它可能是 8.70% ± 0.2%，
也可能是 ± 2%。

## 做法

对每个有多次重复的格，算 TTFT/TPOT 在**重复之间**的变异：

    sd_within = 各轮中位数的样本标准差
    cv_within = sd_within / 均值

再把每格的 cv 汇总，用**误差传播**估计 MAPE 本身的不确定度：

    MAPE 的近似方差 ≈ (Σ cv_i² ) / N  （相对误差的均值的方差）

## 产出

- `results/variance_by_cell.csv`：逐格的重复波动
- `results/variance_summary.json`：汇总 + MAPE 的不确定度
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import tc_config as cfg

sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def main():
    rows = []
    with open(cfg.DATA_DIR / "raw_cells.jsonl", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    df = pd.DataFrame(rows)
    df = df[(df["status"] == "ok") & (df["repeat"] >= 0)]
    # 【与 tc_analyze 保持一致】同一格被多次运行时，repeat 编号会从 0 重新开始，
    # 出现 (cell_id, repeat) 重复。tc_analyze 用 .last() 去重，这里必须同样处理，
    # 否则会把同一轮算两次，CV 被低估。
    df = df.groupby(["cell_id", "repeat"], as_index=False).last()
    if df.empty:
        print("没有可用数据")
        return

    multi = df.groupby("cell_id").filter(lambda g: len(g) >= 2)
    if multi.empty:
        print("⚠️ 没有任何格有 ≥2 次重复。先跑：")
        print("   python tc_run_cell.py 3B q4_k_m 8192 1 --repeats 5")
        return

    out = []
    for cid, g in multi.groupby("cell_id"):
        ttft = g["ttft_p50_ms"].astype(float).to_numpy()
        tpot = g["tpot_p50_ms"].astype(float).to_numpy()
        n = len(g)
        rec = dict(cell_id=cid, n_repeats=n,
                   ttft_mean=float(ttft.mean()), ttft_sd=float(ttft.std(ddof=1)),
                   tpot_mean=float(tpot.mean()), tpot_sd=float(tpot.std(ddof=1)))
        rec["ttft_cv"] = rec["ttft_sd"] / rec["ttft_mean"] if rec["ttft_mean"] else np.nan
        rec["tpot_cv"] = rec["tpot_sd"] / rec["tpot_mean"] if rec["tpot_mean"] else np.nan
        rec["conc"] = int(g["conc"].iloc[0])
        rec["model"] = g["model"].iloc[0]
        out.append(rec)

    v = pd.DataFrame(out).sort_values(["model", "conc"])
    v.to_csv(cfg.RESULTS_DIR / "variance_by_cell.csv", index=False, encoding="utf-8")

    print(f"有重复的格：{len(v)} 个（重复次数 {sorted(v['n_repeats'].unique())}）\n")
    print(f"  {'格':<26}{'重复':>4}{'TTFT均值':>10}{'CV':>8}{'TPOT均值':>10}{'CV':>8}")
    for _, r in v.iterrows():
        print(f"  {r['cell_id']:<26}{int(r['n_repeats']):>4}"
              f"{r['ttft_mean']:>9.1f}ms{r['ttft_cv']*100:>7.1f}%"
              f"{r['tpot_mean']:>9.2f}ms{r['tpot_cv']*100:>7.1f}%")

    cv_t = float(v["ttft_cv"].median())
    cv_p = float(v["tpot_cv"].median())
    print(f"\n  TTFT 重复 CV 中位 = {cv_t*100:.2f}%")
    print(f"  TPOT 重复 CV 中位 = {cv_p*100:.2f}%")

    # 误差传播：MAPE 是 N 个格相对误差的均值 → 其标准误 ≈ cv / sqrt(N)
    n_cells = len(v)
    se = cv_p / np.sqrt(max(n_cells, 1)) * 100
    try:
        fj = json.loads((cfg.RESULTS_DIR / "fit.json").read_text(encoding="utf-8"))
        base = float(fj["mape_cost"])
    except Exception:
        base = float("nan")
    summ = dict(n_cells_with_repeats=n_cells,
                ttft_cv_median=round(cv_t, 5), tpot_cv_median=round(cv_p, 5),
                mape_cost_reported=round(base, 4),
                mape_cost_se_est=round(se, 4),
                note="se 用 TPOT 的重复 CV 与格数做误差传播，是量级估计而非严格区间")
    (cfg.RESULTS_DIR / "variance_summary.json").write_text(
        json.dumps(summ, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n  报告值 MAPE = {base:.2f}%，重复性带来的标准误 ≈ ±{se:.2f} pp")
    print("  → 也就是说：MAPE 的差异小于这个数量级时，不要解读为真实差异")
    print("\n→ results/variance_by_cell.csv / variance_summary.json")


if __name__ == "__main__":
    main()

