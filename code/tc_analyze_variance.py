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

    # 原始顺序很重要：同一批的 repeat 从 0 递增，换批会重新从 0 开始
    raw = [json.loads(l) for l in (cfg.DATA_DIR / "raw_cells.jsonl").open(encoding="utf-8")
           if l.strip()]
    order = {id(r): i for i, r in enumerate(raw)}   # 仅用于排序（不依赖对象相等）

    multi = df.groupby("cell_id").filter(lambda g: len(g) >= 2)
    if multi.empty:
        print("⚠️ 没有任何格有 ≥2 次重复。先跑：")
        print("   python tc_run_cell.py 3B q4_k_m 8192 1 --repeats 5")
        return

    # 【必须用未去重的原始行】前面的 (cell_id, repeat) 去重会把不同批次的编号合并，
    # 批次信息就没了。所以批次划分必须在 raw 上做。
    raw_by_cell = {}
    for r in raw:
        if r.get("status") == "ok" and int(r.get("repeat", -1)) >= 0:
            raw_by_cell.setdefault(r["cell_id"], []).append(r)

    def split_sessions(rows_of_cell):
        """按 repeat 是否回退切批；输入是原始 dict 列表（文件顺序）。"""
        sessions, cur, prev = [], [], None
        for r in rows_of_cell:
            rep = int(r["repeat"])
            if prev is not None and rep <= prev and cur:
                sessions.append(pd.DataFrame(cur))
                cur = []
            cur.append(r)
            prev = rep
        if cur:
            sessions.append(pd.DataFrame(cur))
        return sessions

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

        # ---- 同批次 vs 跨批次 ----
        sess = split_sessions(raw_by_cell.get(cid, []))
        rec["n_sessions"] = len(sess)
        within_t, within_p = [], []
        for one in sess:
            if len(one) < 2:
                continue
            t = one["ttft_p50_ms"].astype(float).to_numpy()
            pp = one["tpot_p50_ms"].astype(float).to_numpy()
            if t.mean():
                within_t.append(float(t.std(ddof=1) / t.mean()))
            if pp.mean():
                within_p.append(float(pp.std(ddof=1) / pp.mean()))
        rec["ttft_cv_within"] = float(np.median(within_t)) if within_t else np.nan
        rec["tpot_cv_within"] = float(np.median(within_p)) if within_p else np.nan
        if len(sess) >= 2:
            med_t = [float(one["ttft_p50_ms"].median()) for one in sess]
            med_p = [float(one["tpot_p50_ms"].median()) for one in sess]
            rec["ttft_cv_between"] = float(np.std(med_t, ddof=1) / np.mean(med_t)) if np.mean(med_t) else np.nan
            rec["tpot_cv_between"] = float(np.std(med_p, ddof=1) / np.mean(med_p)) if np.mean(med_p) else np.nan
        else:
            rec["ttft_cv_between"] = np.nan
            rec["tpot_cv_between"] = np.nan
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
    # 跨批次：只统计真正有 ≥2 批的格
    b = v[v["n_sessions"] >= 2]
    tpot_between_max = float(b["tpot_cv_between"].max()) if len(b) else float("nan")
    tpot_between_med = float(b["tpot_cv_between"].median()) if len(b) else float("nan")
    ttft_between_max = float(b["ttft_cv_between"].max()) if len(b) else float("nan")
    within_p = float(v["tpot_cv_within"].median())
    se_between = (tpot_between_max / np.sqrt(max(n_cells, 1)) * 100
                  if np.isfinite(tpot_between_max) else float("nan"))

    print(f"\n  同批次 TPOT CV 中位 = {within_p*100:.2f}%")
    if len(b):
        print(f"  跨批次（{len(b)} 格有两批以上）：TPOT CV 中位 {tpot_between_med*100:.2f}%、"
              f"最大 {tpot_between_max*100:.2f}%")
        print(f"  跨批次最坏的格：{b.loc[b['tpot_cv_between'].idxmax(), 'cell_id']}")

    summ = dict(n_cells_with_repeats=n_cells,
                ttft_cv_median=round(cv_t, 5), tpot_cv_median=round(cv_p, 5),
                n_cells_multi_session=int(len(b)) if 'b' in dir() else 0,
                tpot_cv_within_median=round(within_p if 'within_p' in dir() else cv_p, 5),
                tpot_cv_between_median=round(tpot_between_med if 'tpot_between_med' in dir() else float('nan'), 5),
                tpot_cv_between_max=round(tpot_between_max if 'tpot_between_max' in dir() else float('nan'), 5),
                ttft_cv_between_max=round(ttft_between_max if 'ttft_between_max' in dir() else float('nan'), 5),
                mape_cost_se_between_est=round(se_between if 'se_between' in dir() else float('nan'), 4),
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



