# -*- coding: utf-8 -*-
"""
A2：交叉验证——`8.59%` 是"一种划分"的结果，随机重划分能移动多少？

## 为什么做

现在的 12 个标定点是**设计出来的**（`cfg.CALIB_CELLS`），留出 68 格。
所有对外数字都建立在**这一个划分**上。但换一组 12 个点，MAPE 会变多少？
如果不做这一项，"8.59%" 就缺一个最根本的不确定性来源：**划分本身**。

## 做法

1. 在 core 口径的格子里，**随机抽 12 格**当标定集（与现行预算相同）；
2. 用同样的 `fit()` 拟合，在**其余所有格**上算成本 MAPE；
3. 重复 N 次 → 得到"划分不确定性"的分布；
4. 与现行"设计过的 12 格"对比：设计到底值不值？

## 产出

- `results/cv_splits.json`（每次划分的 MAPE 与选中的格）
- `results/cv_summary.json`
- `figures/fig15_cv.png`
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
import tc_model as tm

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def load_cells(population="core"):
    """与 tc_model 完全一致的口径过滤，否则 MAPE 不可比。

    教训：第一版把 wave/knee 探针和跨族格都当留出集，算出 11.98%，
    而官方是 8.59%——**同样的代码、不同的口径，数字差 40%**。
    """
    df = pd.read_csv(cfg.DATA_DIR / "summary_steady.csv")
    cid = df["cell_id"].astype(str)
    # 与 tc_model 同一套判定：白名单（设计空间四元组）+ 排除探针/新实验
    has_suffix = cid.str.contains(r"_(?:w\d+|n\d+|p\d+|mt\d+)$", regex=True)
    is_mix = cid.str.contains("_mix", regex=False)
    is_xf = cid.str.endswith("_xf")
    design = cfg.design_space_cells()
    keys = list(zip(df["model"], df["quant"],
                    df["ctx"].astype(int), df["conc"].astype(int)))
    in_design = pd.Series([k in design for k in keys], index=df.index)
    probe_like = has_suffix | is_mix
    if population == "core":
        df = df[in_design & ~probe_like & ~is_xf]
    elif population == "valid":
        df = df[~probe_like]
    df = df.copy()
    df["prompt_tokens"] = df["prompt_tokens_median"]
    df["ctx_avg_tokens"] = df["prompt_tokens"] + cfg.GEN_TOKENS / 2

    # 【必须以稳态成本为目标】tc_model 用的就是 steady_cost_gpu_s_per_1m；
    # 用原始 cost_gpu_s_per_1m（墙钟）会得到 11.28%——那正是"墙钟口径"的数字。
    if "steady_cost_gpu_s_per_1m" in df.columns:
        df["cost_gpu_s_per_1m"] = df["steady_cost_gpu_s_per_1m"]
    return df


def mape_cost(pred, hold_ids):
    h = pred[pred["cell_id"].isin(hold_ids)]
    if not len(h):
        return float("nan")
    return float((h["ratio_cost"] - 1).abs().mean() * 100)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=40)
    ap.add_argument("--k", type=int, default=12, help="每次抽多少个标定点（与现行预算相同）")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    df = load_cells()   # core 口径：与官方 8.59% 可比
    designed = {cfg.cell_id(*c) for c in cfg.CALIB_CELLS}
    pool = [c for c in df["cell_id"] if c not in designed]
    print(f"可抽样的候选格 {len(pool)} 个（已排除设计过的 12 个）；每次抽 {args.k} 个，"
          f"重复 {args.trials} 次\n")

    rng = np.random.default_rng(args.seed)
    trials = []
    for t in range(args.trials):
        pick = set(rng.choice(pool, size=min(args.k, len(pool)), replace=False))
        calib = df[df["cell_id"].isin(pick)]
        try:
            params = tm.fit(calib)
        except Exception as e:
            print(f"  第{t+1}次拟合失败：{type(e).__name__}")
            continue
        pred = tm.evaluate(df, params, pick)
        hold = set(df["cell_id"]) - pick
        trials.append(dict(trial=t, calib=sorted(pick),
                           mape_cost=mape_cost(pred, hold),
                           bw=float(params["bw_gbs"]), c_srv=float(params["c_srv_ms"]),
                           b_sat=float(params["b_sat"])))
        if (t + 1) % 10 == 0:
            print(f"  已完成 {t+1}/{args.trials}", flush=True)

    if not trials:
        print("所有划分都失败了")
        return

    t = pd.DataFrame(trials)
    t.to_csv(cfg.RESULTS_DIR / "cv_splits.csv", index=False, encoding="utf-8")

    # 现行（设计过的 12 格）作为参照
    calib0 = df[df["cell_id"].isin(designed)]
    p0 = tm.fit(calib0)
    pred0 = tm.evaluate(df, p0, designed)
    base = mape_cost(pred0, set(df["cell_id"]) - designed)

    m = t["mape_cost"]
    summ = dict(trials=int(len(t)), k=args.k,
                designed_mape=round(base, 2),
                random_median=round(float(m.median()), 2),
                random_p5=round(float(np.percentile(m, 5)), 2),
                random_p95=round(float(np.percentile(m, 95)), 2),
                random_min=round(float(m.min()), 2),
                random_max=round(float(m.max()), 2),
                shift_vs_designed=round(float(m.median()) - base, 2),
                note=("随机重划分（同样 12 个点）对 MAPE 的影响；"
                      "反映的是'选点'这一层不确定度，而不是测量噪声"))
    (cfg.RESULTS_DIR / "cv_summary.json").write_text(
        json.dumps(summ, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n  设计过的 12 格：MAPE {base:.2f}%")
    print(f"  随机 12 格（{len(t)} 次）：中位 {m.median():.2f}%，"
          f"5–95% 区间 [{np.percentile(m, 5):.2f}, {np.percentile(m, 95):.2f}]，"
          f"最差 {m.max():.2f}%")
    print(f"  中位相对设计值：{m.median() - base:+.2f} pp")

    # ---- 图 ----
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    ax.hist(m, bins=min(20, max(6, len(t) // 2)), color="#4c78a8", alpha=0.85)
    ax.axvline(base, color="crimson", ls="--", lw=1.6,
               label=f"现行设计 12 格 {base:.2f}%")
    ax.axvline(m.median(), color="black", ls=":", lw=1.2,
               label=f"随机 12 格中位 {m.median():.2f}%")
    ax.set_xlabel("留出集成本 MAPE (%)")
    ax.set_ylabel("次数")
    ax.set_title("换一组 12 个标定点，MAPE 会怎么动")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(cfg.FIG_DIR / "fig15_cv.png", dpi=150)
    print("  图 → figures/fig15_cv.png")


if __name__ == "__main__":
    main()
