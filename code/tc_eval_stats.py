# -*- coding: utf-8 -*-
"""P1-7 第二步：配对统计量（McNemar）+ Wilson 下界 + 分层。

第三轮审计的 C1 条说得对：五个模型的 95% 置信区间全部重叠，
用独立 Wilson 区间去比较"两个模型谁更好"是浪费信息——
**它们回答的是同一批题**，必须用配对检验。

## 为什么 McNemar 比独立区间灵敏

设两个模型在 140 题上：
  b = A 对 B 错的题数，c = A 错 B 对的题数
若两个模型真的等价，那么"只有一方答对"的题应当是 b ≈ c。
McNemar 检验的就是 b 与 c 是否显著不等（H0: p = 0.5）。
"两模型都答对/都答错"的题不提供信息，所以配对检验的方差比独立检验小得多。

小样本用**精确二项检验**（b+c 通常只有几十，卡方近似不可靠）。

用法：
    python tc_eval_stats.py                    # 读 data/quality_details.jsonl
    python tc_eval_stats.py --details 路径
"""

import argparse
import itertools
import json
import math
import pathlib
import sys

import pandas as pd
from scipy import stats

import tc_config as cfg


def wilson(k, n, z=1.96):
    """Wilson 区间：小样本比例的标准做法（比正态近似稳，不会超出 [0,1]）。"""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    r = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - r) / d, (c + r) / d)


def mcnemar_exact(b, c):
    """配对 McNemar 的精确二项检验（双侧）。b/c 是两类不一致的题数。"""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    # 双侧：2 × P(X <= k)，截断到 1
    p = 2 * stats.binom.cdf(k, n, 0.5)
    return min(1.0, float(p))


def holm(pvals):
    """Holm–Bonferroni：多重比较校正。不做校正时报"显著"是自欺。"""
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    adj = [0.0] * m
    running = 0.0
    for rank, i in enumerate(order):
        val = min(1.0, (m - rank) * pvals[i])
        running = max(running, val)
        adj[i] = running
    return adj


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("--details", default=str(cfg.DATA_DIR / "quality_details.jsonl"))
    args = ap.parse_args()

    path = pathlib.Path(args.details)
    if not path.exists():
        raise SystemExit(f"没有 {path}，先跑 tc_eval.py")

    rows = [json.loads(l) for l in path.open(encoding="utf-8") if l.strip()]
    df = pd.DataFrame(rows)
    df["combo"] = df["model"] + "-" + df["quant"]

    # 每组合取最近一次评测（重复跑时以最后一次为准）
    piv = df.pivot_table(index="iid", columns="combo", values="ok", aggfunc="last")
    combos = list(piv.columns)
    n_items = len(piv)
    print(f"题数 {n_items}，模型组合 {len(combos)}：{combos}\n")

    # ---------- 1) 准确率 + Wilson 区间 + 下界 ----------
    print("=" * 78)
    print("① 准确率与 Wilson 区间（下界用于质量约束）")
    print("=" * 78)
    print(f"{'组合':<16}{'对/总':>9}{'准确率':>9}{'95%CI':>18}{'下界':>8}")
    stats_rows = []
    for c in combos:
        k = int(piv[c].fillna(0).sum())
        lo, hi = wilson(k, n_items)
        stats_rows.append(dict(combo=c, correct=k, n=n_items, acc=k / n_items,
                               lo=lo, hi=hi))
        print(f"{c:<16}{k:>4}/{n_items:<4}{k/n_items*100:>8.1f}%"
              f"   [{lo*100:5.1f}%, {hi*100:5.1f}%] {lo*100:>7.1f}%")

    # ---------- 2) 配对 McNemar ----------
    print("\n" + "=" * 78)
    print("② 配对 McNemar（H0：两模型等价）")
    print("=" * 78)
    pairs = list(itertools.combinations(combos, 2))
    raw_p = []
    detail = []
    for a, b in pairs:
        x = piv[a].fillna(0).astype(int)
        y = piv[b].fillna(0).astype(int)
        n01 = int(((x == 1) & (y == 0)).sum())     # a 对 b 错
        n10 = int(((x == 0) & (y == 1)).sum())     # a 错 b 对
        p = mcnemar_exact(n01, n10)
        raw_p.append(p)
        detail.append(dict(a=a, b=b, a_only=n01, b_only=n10, p_raw=p,
                           diff=(x.mean() - y.mean())))
    adj = holm(raw_p)
    print(f"{'对比':<34}{'仅前者对':>9}{'仅后者对':>9}{'p':>9}{'Holm校正':>10}{'结论':>10}")
    n_sig = 0
    for d, pa in zip(detail, adj):
        d["p_holm"] = pa
        sig = "显著" if pa < 0.05 else "无差异"
        n_sig += pa < 0.05
        print(f"{d['a']+' vs '+d['b']:<34}{d['a_only']:>9}{d['b_only']:>9}"
              f"{d['p_raw']:>9.4f}{pa:>10.4f}{sig:>10}")
    print(f"\n  显著对数：{n_sig}/{len(pairs)}（Holm 校正后 p<0.05）")
    print("  说明：若显著对数为 0，则'换大模型能提升质量'在这套题上**不被支持**，")
    print("        这本身就是一条有效结论，应照实写进报告，而不是含糊说'差异不大'。")

    # ---------- 3) 按题型分层 ----------
    print("\n" + "=" * 78)
    print("③ 按题型分层（看差异来自哪种任务）")
    print("=" * 78)
    kinds = sorted(df["kind"].unique())
    hdr = "".join(f"{k:>12}" for k in kinds)
    print(f"{'组合':<16}{hdr}")
    for c in combos:
        sub = df[df["combo"] == c]
        vals = []
        for k in kinds:
            s = sub[sub["kind"] == k]
            vals.append(f"{(s['ok'].mean()*100 if len(s) else float('nan')):>11.1f}%")
        print(f"{c:<16}{''.join(vals)}")

    # ---------- 4) 阈值敏感性 ----------
    print("\n" + "=" * 78)
    print("④ 质量阈值敏感性：底线定在哪，可选集就变多少")
    print("=" * 78)
    best = max(r["acc"] for r in stats_rows)
    best_lo = max(r["lo"] for r in stats_rows)
    print(f"{'底线口径':<26}{'阈值':>9}{'通过的组合':>34}")
    for drop in (0, 2, 4, 6, 8, 10):
        th = best - drop / 100
        passed = [r["combo"] for r in stats_rows if r["acc"] >= th]
        print(f"{'点估计 − '+str(drop)+'pp':<26}{th*100:>8.1f}%{'、'.join(passed):>34}")
    for drop in (0, 2, 4, 6, 8, 10):
        th = best_lo - drop / 100
        passed = [r["combo"] for r in stats_rows if r["lo"] >= th]
        print(f"{'Wilson 下界 − '+str(drop)+'pp':<26}{th*100:>8.1f}%{'、'.join(passed):>34}")
    print("\n  点估计口径对 0.4 个百分点的抖动敏感；下界口径更保守，")
    print("  两者给出的可选集不同，报告里应把这张表贴出来。")

    out = cfg.RESULTS_DIR / "quality_stats.json"
    out.write_text(json.dumps(dict(n_items=n_items, combos=combos,
                                   stats=stats_rows,
                                   mcnemar=detail,
                                   n_significant=n_sig),
                              ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n已写入 {out}")


if __name__ == "__main__":
    main()
