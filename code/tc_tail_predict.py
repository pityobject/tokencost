# -*- coding: utf-8 -*-
"""
尾部律驱动的 SLA 翻转预测（`tc.py tailpredict`）

## 问题

§5.12 的结论是"尾部口径会改变这一格能不能过 SLA"（假设通过 126 → 实测通过 98，28 格翻转），
但那 28 格是**用实测尾部**算出来的——等于先知道答案。真正的问题是：

> 能不能在**只做少量测量**的情况下，提前预测哪些格会翻转？

## 做法

1. 先从逐请求数据拟合**两条尾部律**（TTFT 与 TPOT 各一条）：
   - TPOT：`CV(b,L)` 已拟（§5.29），`p90/p50 ≈ exp(1.2816·CV)`
   - TTFT：直接对实测 `p90/p50` 拟合 `1 + a·b^p·(L/1000)^q`
2. 用这两条律替换"常数 1.09"假设，重算 132 格的 SLA 判定；
3. 与实测判定对比：**抓住多少真翻转、误报多少**。

## 输出

- 混淆矩阵（预测 vs 实测通过/不通过）
- 与"常数假设"基线对比

    python code/tc_tail_predict.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import tc_config as cfg

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
Z90 = 1.2816

# 拟合 TTFT 尾部的数据：(硬件/栈, 文件)
JITTER_SETS = [
    ("vLLM", "jitter_a10_vllm_L64.json"),
    ("vLLM", "jitter_a10_vllm_L256.json"),
    ("vLLM", "jitter_a10_vllm.json"),
    ("llama.cpp", "jitter_a10_llamacpp.json"),
    ("vLLM", "jitter_a10_vllm_high.json"),
    ("llama.cpp", "jitter_a10_llamacpp_high2.json"),
]


def collect():
    rows = []
    for stack, name in JITTER_SETS:
        fp = cfg.RESULTS_DIR / "a10_raw" / name
        if not fp.exists():
            continue
        d = json.loads(fp.read_text(encoding="utf-8"))
        L = float(d.get("length") or 0)
        for k, v in d["cells"].items():
            rows.append(dict(stack=stack, b=int(k), L=L,
                             ttft_tail=float(v.get("ttft_p90_over_p50") or 0),
                             tpot_cv=float(v.get("tpot_cv") or 0)))
    return [r for r in rows if r["L"] > 0 and r["ttft_tail"] > 0]


def fit_ttft_law(rows):
    """TTFT 的 p90/p50 ≈ 1 + a·b^p·(L/1000)^q（对 (tail-1) 取对数线性化）。"""
    b = np.array([r["b"] for r in rows], float)
    L = np.array([r["L"] for r in rows], float) / 1000.0
    y = np.array([r["ttft_tail"] for r in rows], float) - 1.0
    ok = (y > 1e-6) & (b > 0) & (L > 0)
    if ok.sum() < 4:
        return None
    A = np.column_stack([np.ones(ok.sum()), np.log(b[ok]), np.log(L[ok])])
    coef, *_ = np.linalg.lstsq(A, np.log(y[ok]), rcond=None)
    la, p, q = coef
    pred = 1.0 + np.exp(la) * b[ok] ** p * L[ok] ** q
    r2 = 1 - np.sum((np.array([r["ttft_tail"] for r in rows], float)[ok] - pred) ** 2) / \
        np.sum((np.array([r["ttft_tail"] for r in rows], float)[ok] -
                np.array([r["ttft_tail"] for r in rows], float)[ok].mean()) ** 2)
    return dict(a=float(np.exp(la)), exp_b=float(p), exp_L=float(q),
                r2=float(r2), n=int(ok.sum()))


def main():
    rows = collect()
    if not rows:
        print("没有逐请求数据")
        return
    print("=" * 82)
    print("一、TTFT 尾部律（逐请求实测，两栈两并发段）")
    print("=" * 82)
    law = fit_ttft_law(rows)
    if not law:
        print("  拟合失败")
        return
    print(f"  p90/p50 ≈ 1 + {law['a']:.3f}·b^{law['exp_b']:.2f}·(L/1000)^{law['exp_L']:.2f}"
          f"   R²={law['r2']:.3f}  (n={law['n']})")
    print(f"  {'栈':>10}{'L':>6}{'b':>5}{'实测':>8}{'律预测':>9}")
    for r in sorted(rows, key=lambda x: (x["stack"], x["L"], x["b"])):
        pr = 1 + law["a"] * r["b"] ** law["exp_b"] * (r["L"] / 1000) ** law["exp_L"]
        print(f"  {r['stack']:>10}{r['L']:>6.0f}{r['b']:>5}"
              f"{r['ttft_tail']:>8.2f}{pr:>9.2f}")

    # 二、用两条律重算 132 格的 SLA 判定
    print("\n" + "=" * 82)
    print("二、用尾部律预测 SLA 翻转（TTFT≤2000 ms、TPOT≤100 ms 的 p90）")
    print("=" * 82)
    flip = pd.read_csv(cfg.RESULTS_DIR / "tail_flip.csv")
    summ = pd.read_csv(Path(cfg.__file__).resolve().parent.parent / "data" / "summary_steady.csv")
    Lmap = dict(zip(summ["cell_id"], summ["prompt_tokens_median"]))
    tv = json.loads((cfg.RESULTS_DIR / "tail_variance.json").read_text(encoding="utf-8"))

    def cv_law(b, L):
        return tv["c0"] + tv["a"] * max(b - 1, 1e-9) ** tv["exp_b"] * L ** tv["exp_L"]

    res = []
    for _, r in flip.iterrows():
        L = float(Lmap.get(r["cell_id"], 300.0)) + 64.0
        b = int(r["conc"])
        # 律预测的 p90 因子
        f_ttft = 1 + law["a"] * b ** law["exp_b"] * (L / 1000) ** law["exp_L"]
        f_tpot = float(np.exp(Z90 * cv_law(b, L)))
        ttft_p90 = float(r["ttft_assumed"]) / float(r["factor_ttft"]) * f_ttft
        tpot_p90 = float(r["tpot_assumed"]) / float(r["factor_tpot"]) * f_tpot
        pred_pass = (ttft_p90 <= 2000) and (tpot_p90 <= 100)
        res.append(dict(cell=r["cell_id"], conc=b,
                        pred=pred_pass, meas=bool(r["pass_measured"]),
                        const=bool(r["pass_assumed"])))
    df = pd.DataFrame(res)

    def conf(col):
        tp = int(((df[col] == False) & (df["meas"] == False)).sum())   # 预测不过、实际也不过
        fp = int(((df[col] == False) & (df["meas"] == True)).sum())    # 误报
        fn = int(((df[col] == True) & (df["meas"] == False)).sum())    # 漏报
        tn = int(((df[col] == True) & (df["meas"] == True)).sum())
        rec = tp / (tp + fn) * 100 if (tp + fn) else float("nan")
        prec = tp / (tp + fp) * 100 if (tp + fp) else float("nan")
        return tp, fp, fn, tn, rec, prec

    for col, label in (("const", "常数 1.09 假设"), ("pred", "两条尾部律")):
        tp, fp, fn, tn, rec, prec = conf(col)
        print(f"  {label:<16} 抓住翻转 {tp}/{(tp+fn)}（召回 {rec:.0f}%），"
              f"误报 {fp}，漏 {fn}，通过格 {tn}｜精确率 {prec:.0f}%")

    print(f"\n  实测：{int((df['meas']==False).sum())} 格不通过 / {len(df)} 格")
    print(f"  常数假设说：{int((df['const']==False).sum())} 格不通过")
    print(f"  尾部律说：  {int((df['pred']==False).sum())} 格不通过")

    out = dict(ttft_law=law, variance_law=dict(c0=tv["c0"], a=tv["a"],
                                              exp_b=tv["exp_b"], exp_L=tv["exp_L"]),
               confusion={c: dict(zip(("tp", "fp", "fn", "tn", "recall_pct", "precision_pct"),
                                     conf(c))) for c in ("const", "pred")},
               measured_fail=int((df["meas"] == False).sum()),
               n_cells=len(df),
               note="TTFT 律由逐请求 p90/p50 拟合；TPOT 律来自 tail_variance.json")
    (cfg.RESULTS_DIR / "tail_predict.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n→ results/tail_predict.json")


if __name__ == "__main__":
    main()
