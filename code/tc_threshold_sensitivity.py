# -*- coding: utf-8 -*-
"""
tc_threshold_sensitivity.py —— 质量阈值敏感性图（fig6）。

## 这张图在讲什么

报告说"推荐 7B"，而"6pp 还是 7pp"这个人为约定能让推荐在 7B 与 3B 之间翻转、
成本差数倍。与其把它藏起来，不如**画出翻转本身**：
横轴 = 允许的质量损失，纵轴 = 最便宜合规配置的成本。

## 第三轮复核的教训（为什么要改成现在这样）

本脚本原来自己实现了一遍"候选集 + SLA 过滤"，结果与推荐器给出两套答案
（预测区间半宽差 36.5 倍、候选集也不同：一个用 cfg.all_cells()、
一个用实测汇总表）。现已改为**全部调用 `tc_common.rank()`**，
保证与推荐器是同一个实现。

用法：
    python tc_threshold_sensitivity.py
"""

import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import tc_common as tc
import tc_config as cfg

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

TTFT_CAP, TPOT_CAP, PMEAN, SIGMA = 600.0, 60.0, 256.0, 0.5


def main():
    params, fit = tc.load_params()

    rows = []
    for d in np.arange(0, 15.01, 0.5):
        pick, _ = tc.rank(params, fit, quality_drop_pp=float(d),
                          prompt_mean=PMEAN, prompt_sigma=SIGMA,
                          ttft_cap=TTFT_CAP, tpot_cap=TPOT_CAP)
        if pick is None:
            continue
        rows.append(dict(drop=float(d), cell_id=pick["cell_id"], model=pick["model"],
                         cost=float(pick["cost"]), acc=float(pick["acc"])))
    t = pd.DataFrame(rows)
    if t.empty:
        raise SystemExit("没有可行配置")

    fig, ax = plt.subplots(figsize=(8.6, 5.4))
    colors = {"0.5B": "#999999", "1.5B": "#4C72B0", "3B": "#DD8452", "7B": "#C44E52"}
    for m, sub in t.groupby("model"):
        ax.step(sub["drop"], sub["cost"], where="post", lw=2.2,
                color=colors.get(m, "#333333"), label=f"{m} 档")

    flips = t[t["model"] != t["model"].shift(1)]
    for _, r in flips.iterrows():
        ax.axvline(r["drop"], color="#bbbbbb", ls=":", lw=1, zorder=0)
        ax.annotate(f"{r['model']}\n{r['cost']:.0f}",
                    (r["drop"], r["cost"]), textcoords="offset points",
                    xytext=(4, 10), fontsize=7.5, color=colors.get(r["model"], "k"))

    ax.set_yscale("log")
    ax.set_xlabel("允许的质量损失上限（百分点，相对最优组合）")
    ax.set_ylabel("满足 SLA 的最低成本（GPU 秒 / 百万 token，对数轴）")
    ax.set_title("质量约束每放宽一档，推荐配置与成本就跳一级")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=9)
    ax.margins(x=0.03, y=0.12)
    fig.tight_layout()
    out = cfg.FIG_DIR / "fig6_quality_threshold_sensitivity.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)

    print(f"{'允许损失':>9}{'推荐':<28}{'成本':>10}{'质量':>8}")
    for _, r in flips.iterrows():
        print(f"{r['drop']:>8.1f}pp  {r['cell_id']:<28}{r['cost']:>10.0f}{r['acc']*100:>7.1f}%")
    print(f"\n已保存 {out}")
    t.to_csv(cfg.RESULTS_DIR / "threshold_sensitivity.csv", index=False,
             encoding="utf-8-sig")
    print(f"已写入 {cfg.RESULTS_DIR / 'threshold_sensitivity.csv'}")


if __name__ == "__main__":
    main()
