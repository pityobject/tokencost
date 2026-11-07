# -*- coding: utf-8 -*-
"""
质量—成本 Pareto 前沿（`tc.py pareto`）：把两个"魔法阈值"换成一条曲线

## 现状

推荐器用硬阈值剪枝：质量损失 0–6 pp → 7B，8 pp → 3B-Q8，≥10 pp → 1.5B-Q4。
阈值 7.5 pp / 10 pp 是**约定**，不是算出来的。答辩时最难解释的就是这两个数。

## 做法

1. 质量：`data/quality.csv`（400 题，10 个组合，含 Wilson 区间）
2. 成本：`data/summary_steady.csv` 的稳态 `cost_gpu_s_per_1m`（同并发同上下文下取中位）
3. 画 Pareto 前沿（质量↑、成本↓ 的不可支配点集）
4. 对每个"允许的质量损失"，给出**前沿上的最优配置**，并与硬阈值规则比较
5. 算"边际交换率"：每多 1 个百分点的质量要点多少成本

## 用法

    python code/tc_pareto_quality.py            # 默认参考并发 8、ctx 512
    python code/tc_pareto_quality.py --conc 16
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import tc_config as cfg

sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def load():
    q = pd.read_csv(Path(cfg.__file__).resolve().parent.parent / "data" / "quality.csv")
    s = pd.read_csv(Path(cfg.__file__).resolve().parent.parent / "data" / "summary_steady.csv")
    return q, s


def reference_cost(s, model, quant, conc, ctx):
    """在给定并发与上下文下取该组合的稳态单位成本（中位）。"""
    d = s[(s["model"] == model) & (s["quant"] == quant) &
          (s["conc"] == conc) & (s["ctx"] == ctx)]
    if d.empty:
        d = s[(s["model"] == model) & (s["quant"] == quant) & (s["conc"] == conc)]
    if d.empty:
        return float("nan")
    return float(d["cost_gpu_s_per_1m"].median())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--conc", type=int, default=8)
    ap.add_argument("--ctx", type=int, default=512)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    q, s = load()
    rows = []
    for _, r in q.iterrows():
        c = reference_cost(s, r["model"], r["quant"], args.conc, args.ctx)
        if not np.isfinite(c):
            continue
        rows.append(dict(model=r["model"], quant=r["quant"],
                         acc=float(r["acc"]), lo=float(r["acc_lo"]),
                         hi=float(r["acc_hi"]), cost=c,
                         label=f"{r['model']}-{r['quant']}"))
    if len(rows) < 3:
        print("有效组合不足")
        return
    df = pd.DataFrame(rows).sort_values("cost").reset_index(drop=True)

    print("=" * 86)
    print(f"一、10 个组合的质量与成本（参考：并发 {args.conc}、ctx {args.ctx}）")
    print("=" * 86)
    print("  （成本 = 稳态 cost_gpu_s_per_1m ÷ 1000，即 ms/token）")
    print(f"  {'组合':<14}{'质量':>8}{'95% 区间':>20}{'ms/token':>12}{'支配?':>8}")
    # 标出被支配的点
    best_q = -1
    dom = []
    for _, r in df.iterrows():
        is_dom = r["acc"] <= best_q + 1e-12
        dom.append(is_dom)
        best_q = max(best_q, r["acc"])
    df["dominated"] = dom
    for _, r in df.iterrows():
        ci = f"[{r['lo']*100:.0f}%, {r['hi']*100:.0f}%]"
        print(f"  {r['label']:<14}{r['acc']*100:>7.1f}%{ci:>20}"
              f"{r['cost']/1000:>15.1f}{'受支配' if r['dominated'] else '前沿':>8}")

    front = df[~df["dominated"]].reset_index(drop=True)
    print("\n" + "=" * 86)
    print("二、Pareto 前沿（不可支配点）")
    print("=" * 86)
    print(f"  {'组合':<14}{'质量':>8}{'成本 ms/token':>16}{'相对最低成本':>14}")
    base = front["cost"].iloc[0]
    for _, r in front.iterrows():
        print(f"  {r['label']:<14}{r['acc']*100:>7.1f}%{r['cost']/1000:>15.1f}"
              f"{r['cost']/base:>13.2f}×")

    print("\n" + "=" * 86)
    print("三、边际交换率：每多 1 pp 质量，成本涨多少")
    print("=" * 86)
    rates = []
    for i in range(1, len(front)):
        a, b = front.iloc[i - 1], front.iloc[i]
        dq = (b["acc"] - a["acc"]) * 100
        dc = b["cost"] / a["cost"]
        if dq <= 0:
            continue
        extra = (dc - 1) * 100 / dq
        rates.append(dict(from_=a["label"], to=b["label"], dq=round(dq, 1),
                          cost_ratio=round(dc, 2), pct_per_pp=round(extra, 1)))
        print(f"  {a['label']:<14}→ {b['label']:<14} +{dq:>4.1f} pp  "
              f"成本 ×{dc:.2f}   ⇒ 每 pp +{extra:.1f}%")
    med_rate = float(np.median([r["pct_per_pp"] for r in rates])) if rates else float("nan")
    print(f"\n  中位边际成本：**每 1 pp 质量 ≈ +{med_rate:.0f}% 成本**")
    print("  （这个数比'7.5 pp / 10 pp 两个阈值'好讲得多——它是一条连续的价格）")

    # 四、按"允许质量损失"选最优配置，与硬阈值规则比较
    print("\n" + "=" * 86)
    print("四、把两个魔法阈值换成曲线：给定质量损失预算，前沿上最便宜的配置")
    print("=" * 86)
    qmax = float(front["acc"].max())
    HARD = {0.0: "7B-q4_k_m", 7.5: "3B-q8_0", 10.0: "1.5B-q4_k_m"}
    hdr = f"  {'loss':>6}{'Pareto pick':<16}{'ms/token':>10}  {'hard rule':<16}{'ms/token':>10}{'waste':>8}"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    picks = []
    for loss in (0, 2, 4, 6, 8, 10, 12):
        keep = front[front["acc"] >= qmax - loss / 100]
        if keep.empty:
            continue
        pick = keep.iloc[0]      # 前沿已按成本升序，第一个就是最便宜的可行点
        # 硬阈值：找最近的阈值
        thr = max([k for k in HARD if k <= loss], default=0.0)
        hard_label = HARD[thr]
        hard_row = df[df["label"] == hard_label]
        hard_cost = float(hard_row["cost"].iloc[0]) if not hard_row.empty else float("nan")
        diff = (hard_cost / pick["cost"] - 1) * 100 if pick["cost"] else float("nan")
        picks.append(dict(loss_pct=loss, pick=pick["label"],
                          pick_cost=pick["cost"], hard=hard_label,
                          hard_cost=hard_cost, hard_over_pick_pct=round(diff, 1)))
        print(f"  {loss:>9.0f}pp{pick['label']:<16}{pick['cost']/1000:>12.1f}"
              f"{hard_label:<16}{hard_cost/1000:>12.2f}{diff:>7.0f}%")

    print("\n  读法：")
    print("    · 硬阈值规则在多数损失预算下**不是最便宜的**（它跳到更大模型）")
    print("    · 『差』为正表示硬阈值规则比 Pareto 最优贵多少")

    out = dict(conc=args.conc, ctx=args.ctx,
               points=df[["label", "acc", "cost", "dominated"]].to_dict("records"),
               frontier=front[["label", "acc", "cost"]].to_dict("records"),
               marginal=sorted(rates, key=lambda r: -r["pct_per_pp"]),
               median_pct_cost_per_pp=round(med_rate, 1) if rates else None,
               picks=picks,
               note="成本用稳态 cost_gpu_s_per_1m；质量用 400 题准确率")
    fp = Path(args.out) if args.out else (cfg.RESULTS_DIR / "pareto_quality.json")
    fp.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n→ {fp}")


if __name__ == "__main__":
    main()
