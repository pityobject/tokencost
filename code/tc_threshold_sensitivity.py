# -*- coding: utf-8 -*-
"""
tc_threshold_sensitivity.py —— 质量阈值敏感性图（fig6）。

## 为什么这张图是"把破绽变素材"

第二轮复核查出的最大矛盾是：
报告说推荐 3B-Q4，但同一份报告的质量章节证明 7B 显著更好；
而"6pp 还是 7pp"这个人为约定，能直接让推荐在 3B 与 7B 之间翻转、成本差数倍。

与其把它藏起来，不如**画出翻转本身**：
横轴 = 允许的质量损失（0→14 pp），纵轴 = 满足 SLA 的最便宜配置成本，
曲线上的台阶就是"质量汇率"。

用法：
    python tc_threshold_sensitivity.py
"""

import json
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import tc_config as cfg
from tc_recommend import simulate

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

TTFT_CAP, TPOT_CAP, PMEAN, SIGMA = 600.0, 60.0, 256.0, 0.5


def main():
    summ = pd.read_csv(cfg.DATA_DIR / "summary_steady.csv")
    fit = json.loads((cfg.RESULTS_DIR / "fit.json").read_text(encoding="utf-8"))
    params = dict(bw_gbs=fit["bw"], peak_tflops=fit["peak"],
                  peak_prefill_tflops=fit.get("peak_prefill", fit["peak"]),
                  c_srv_ms=fit["c_srv"], b_sat=fit["b_sat"], ovh_ttft_ms=fit["ovh_ttft"])
    half_log = 0.5 * np.log(1 + (fit.get("sigma_log") or 0.1) ** 2)

    q = pd.read_csv(cfg.DATA_DIR / "quality.csv").dropna(subset=["acc"])
    acc = {(r["model"], r["quant"]): float(r["acc"]) for _, r in q.iterrows()}
    lo = {(r["model"], r["quant"]): float(r["acc_lo"]) for _, r in q.iterrows()
          if "acc_lo" in q.columns and r["acc_lo"] == r["acc_lo"]}
    best_lo = max(lo.values())

    need_ctx = int(PMEAN * np.exp(1.645 * SIGMA) + cfg.GEN_TOKENS)

    # 先筛出满足上下文与 SLA 的候选（质量后面按阈值动态筛）
    base = []
    for _, r in summ.iterrows():
        key = (r["model"], r["quant"])
        if key not in lo or int(r["ctx"]) < need_ctx:
            continue
        sim = simulate(params, r["model"], r["quant"], int(r["conc"]),
                       PMEAN, SIGMA, ctx=int(r["ctx"]))
        if (sim["ttft_cvar90"] * np.exp(half_log) > TTFT_CAP or
                sim["tpot_cvar90"] * np.exp(half_log) > TPOT_CAP):
            continue
        base.append(dict(cell_id=r["cell_id"], key=key,
                         cost=sim["cost_gpu_s_per_1m"], lo=lo[key], acc=acc[key]))
    if not base:
        raise SystemExit("没有满足 SLA 的候选，检查参数")
    bdf = pd.DataFrame(base)

    drops = np.arange(0, 15.0, 0.5)
    rows = []
    for d in drops:
        th = best_lo - d / 100.0
        sub = bdf[bdf["lo"] >= th]
        if sub.empty:
            continue
        pick = sub.sort_values("cost").iloc[0]
        rows.append(dict(drop=d, th=th, cost=pick["cost"], model=pick["key"][0],
                         acc=pick["acc"], cell_id=pick["cell_id"]))
    t = pd.DataFrame(rows)

    # ---- 作图 ----
    fig, ax = plt.subplots(figsize=(8.6, 5.4))
    colors = {"0.5B": "#999999", "1.5B": "#4C72B0", "3B": "#DD8452", "7B": "#C44E52"}
    for m, sub in t.groupby("model"):
        ax.step(sub["drop"], sub["cost"], where="post", lw=2.2,
                color=colors.get(m, "#333333"), label=f"{m} 档")

    # 标注翻转点
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

    print(f"{'允许损失':>8}{'质量门槛':>10}{'推荐':<28}{'成本':>10}{'质量':>8}")
    for _, r in flips.iterrows():
        print(f"{r['drop']:>7.1f}pp{r['th']*100:>9.1f}%  {r['cell_id']:<28}"
              f"{r['cost']:>10.0f}{r['acc']*100:>7.1f}%")
    print(f"\n已保存 {out}")

    t.to_csv(cfg.RESULTS_DIR / "threshold_sensitivity.csv", index=False, encoding="utf-8-sig")
    print(f"已写入 {cfg.RESULTS_DIR / 'threshold_sensitivity.csv'}")


if __name__ == "__main__":
    main()
