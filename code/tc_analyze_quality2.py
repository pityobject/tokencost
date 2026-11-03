# -*- coding: utf-8 -*-
"""
tc_analyze_quality2.py —— 能力画像：同一个模型在三个任务族上强在哪、弱在哪

## 为什么需要

原来的质量轴只有一种任务族（算术/字符串操作），于是"质量"其实是"算术质量"。
`tc_eval.py --kinds format,word` 补了两族：
  · format  格式遵循（JSON / 长度上限 / 全大写 / 要点列表 / 不许多说）
  · word    多步应用题（两到三步推理）

本脚本把三族放在一起看，回答一个线上部署真会遇到的问题：
**小模型最先坏在哪——是算错，还是不听格式？**

用法：
    python tc_analyze_quality2.py
"""

import json
import sys
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import tc_config as cfg

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ARITH = {"add", "mul", "cmp", "sort", "count", "reverse"}
FAMILY = {"arith": "算术/字符串", "format": "格式遵循", "word": "多步应用题"}


def load():
    # 优先读第二任务族的独立明细（tc_eval --out 现在会写 *_details.jsonl）；
    # 否则会把旧版有 bug 的 word 题记录混进来。
    cand = [cfg.DATA_DIR / "quality2_details.jsonl",
            cfg.DATA_DIR / "quality_details.jsonl"]
    path = next((p for p in cand if p.exists()), None)
    if path is None:
        raise SystemExit("找不到质量明细文件")
    print(f"  数据源：{path.name}")
    rows = [json.loads(l) for l in path.open(encoding="utf-8")
            if l.strip()]
    agg = defaultdict(lambda: [0, 0])       # (combo, family) -> [ok, n]
    for r in rows:
        kind = r["kind"]
        fam = "arith" if kind in ARITH else kind
        if fam not in FAMILY:
            continue
        key = (f"{r['model']}-{r['quant']}", fam)
        agg[key][1] += 1
        agg[key][0] += 1 if r.get("ok") else 0
    combos = sorted({k[0] for k in agg})
    return agg, combos


def main():
    agg, combos = load()
    fams = ["arith", "format", "word"]

    print("=" * 88)
    print("能力画像：同一模型在三个任务族上的准确率")
    print("=" * 88)
    print(f"{'组合':<16}{'算术/字符串':>12}{'格式遵循':>11}{'多步应用题':>12}{'算术−格式':>11}")
    table = []
    for c in combos:
        vals = {}
        for f in fams:
            ok, n = agg.get((c, f), [0, 0])
            vals[f] = (ok / n * 100) if n else float("nan")
        gap = vals["arith"] - vals["format"]
        table.append(dict(combo=c, **{f: vals[f] for f in fams}, gap_arith_format=gap))
        print(f"{c:<16}{vals['arith']:>11.1f}%{vals['format']:>10.1f}%"
              f"{vals['word']:>11.1f}%{gap:>10.1f}pp")

    t = pd.DataFrame(table)
    t.to_csv(cfg.RESULTS_DIR / "quality_profile.csv", index=False, encoding="utf-8-sig")

    # ---- 判读 ----
    print()
    worst_fam = t[["arith", "format", "word"]].mean().idxmin()
    best_fam = t[["arith", "format", "word"]].mean().idxmax()
    print(f"  各族平均：算术 {t['arith'].mean():.1f}% · "
          f"格式 {t['format'].mean():.1f}% · 应用题 {t['word'].mean():.1f}%")
    print(f"  最弱的是 **{FAMILY[worst_fam]}**（{t[worst_fam].mean():.1f}%），"
          f"最强的是 **{FAMILY[best_fam]}**（{t[best_fam].mean():.1f}%）")
    print(f"  算术−格式 的差距：中位 {t['gap_arith_format'].median():.1f} pp，"
          f"最大 {t['gap_arith_format'].max():.1f} pp")
    if t["gap_arith_format"].median() > 10:
        print("  → 判读：**格式遵循普遍弱于算术**。线上部署小模型时，"
              "最先坏的不是正确率，而是输出格式。")
    else:
        print("  → 判读：两族差距不大，格式不是主要短板。")

    # ---- 排序是否变化（模型规模 vs 质量）----
    rank_arith = t.sort_values("arith", ascending=False)["combo"].tolist()
    rank_word = t.sort_values("word", ascending=False)["combo"].tolist()
    print(f"\n  算术排序前三 : {rank_arith[:3]}")
    print(f"  应用题排序前三: {rank_word[:3]}")
    same = rank_arith[:3] == rank_word[:3]
    print(f"  前三名是否一致: {'是' if same else '否（说明单一任务族不足以代表质量）'}")

    # ---- 图 ----
    fig, ax = plt.subplots(figsize=(9.0, 5.0))
    x = np.arange(len(t))
    w = 0.26
    colors = {"arith": "#4C72B0", "format": "#DD8452", "word": "#C44E52"}
    for i, f in enumerate(fams):
        ax.bar(x + (i - 1) * w, t[f], w, label=FAMILY[f], color=colors[f])
    ax.set_xticks(x)
    ax.set_xticklabels(t["combo"], rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("准确率（%）")
    ax.set_title("能力画像：算术好 ≠ 格式也不差")
    ax.grid(alpha=0.3, axis="y")
    ax.legend(fontsize=9)
    fig.tight_layout()
    out = cfg.FIG_DIR / "fig9_ability_profile.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"\n已保存 {out}")
    print(f"已写入 {cfg.RESULTS_DIR / 'quality_profile.csv'}")


if __name__ == "__main__":
    main()

