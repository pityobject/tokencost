# -*- coding: utf-8 -*-
"""
tc_calib_curve.py —— 标定点数的敏感性分析（回答"12 个点够吗"）

## 为什么需要它

简历/答辩里必被问一句：「12 个标定点够吗？」

现在的答案是「我挑了 12 个 DOE 点，覆盖各维度极端与中心」——那是**理由**，不是**证据**。
这个脚本把它变成证据：用 4/5/…/12 个标定点分别重新拟合，看留出集误差怎么降。

## 方法

对每个标定点数 k（4…12），随机抽 k 个标定格（抽样是分层约束的：尽量每个
模型×量化组合至少 1 格），重新拟合 6 参数，在**同一批留出格**上算成本 MAPE。
每个 k 重复 N 次，报中位数与 5–95% 分位。

判据：
  · 曲线在 k≈8–10 变平 → 说明 12 个点已经够，再加收益小（诚实结论）
  · 曲线仍在陡降       → 说明 12 个点不够（也要诚实写）

用法：
    python tc_calib_curve.py                 # 默认每个 k 抽 30 次
    python tc_calib_curve.py --draws 60
"""

import argparse
import json
import sys
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import tc_config as cfg
import tc_model as tm

sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def load_data():
    """与 tc_model.main() 完全一致的预处理：稳态成本 + 实测 prompt 长度。"""
    steady = cfg.DATA_DIR / "summary_steady.csv"
    path = steady if steady.exists() else cfg.DATA_DIR / "summary.csv"
    df = pd.read_csv(path)
    if "prompt_tokens_median" in df.columns:
        df["prompt_tokens"] = df["prompt_tokens_median"].fillna(cfg.PROMPT_TOKENS)
    else:
        df["prompt_tokens"] = float(cfg.PROMPT_TOKENS)
    df["ctx_avg_tokens"] = df["prompt_tokens"] + cfg.GEN_TOKENS / 2
    if "steady_cost_gpu_s_per_1m" in df.columns:
        df["cost_gpu_s_per_1m"] = df["steady_cost_gpu_s_per_1m"]
    df = df.dropna(subset=["tpot_p50_ms", "ttft_p50_ms", "cost_gpu_s_per_1m"])

    # 【口径必须与对外一致】core = 设计空间（剔除探针格 _w*/_n* 与跨族格 _xf），
    # 否则这里算出的 MAPE 会是 10.21%（all 口径），与 README 的 8.70% 对不上。
    cid = df["cell_id"].astype(str)
    # 【唯一实现】口径判定统一来自 tc_config
    _marks = cfg.mark_population(df)
    has_suffix = _marks["is_probe"]
    is_xf = cid.str.endswith("_xf")
    df = df[~has_suffix & ~is_xf].copy()

    calib_ids = {cfg.cell_id(m, q, c, n) for (m, q, c, n) in cfg.CALIB_CELLS}
    hold = df[~df["cell_id"].isin(calib_ids)].copy()
    calib = df[df["cell_id"].isin(calib_ids)].copy()
    return df, calib, hold, calib_ids


def mape_cost(params, rows):
    """成本 MAPE（与 tc_model 的口径一致：预测/实测−1 的绝对值均值）。"""
    errs = []
    for _, r in rows.iterrows():
        pr = tm.predict(params, r)
        pred_c = pr["pred_cost_gpu_s_per_1m"]
        meas_c = r["cost_gpu_s_per_1m"]
        if meas_c > 0 and np.isfinite(pred_c):
            errs.append(abs(pred_c / meas_c - 1))
    return float(np.mean(errs) * 100) if errs else float("nan")


def stratified_sample(calib, k, rng):
    """分层抽样：尽量让每个 (model, quant) 组合至少 1 格，再随机补齐。"""
    groups = defaultdict(list)
    for _, r in calib.iterrows():
        groups[(r["model"], r["quant"])].append(r["cell_id"])
    picked = []
    keys = list(groups)
    order = rng.permutation(len(keys))
    # 第一轮：每个组合抽 1 个（如果 k 够）
    for idx in order:
        if len(picked) >= k:
            break
        key = keys[int(idx)]
        picked.append(str(rng.choice(groups[key])))
    # 第二轮：从剩余里随机补
    rest = [c for c in calib["cell_id"] if c not in picked]
    if len(picked) < k and rest:
        extra = rng.choice(rest, size=min(k - len(picked), len(rest)), replace=False)
        picked.extend(str(x) for x in extra)
    return picked


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--draws", type=int, default=30, help="每个 k 抽多少次")
    ap.add_argument("--seed", type=int, default=20261103)
    args = ap.parse_args()

    df, calib, hold, calib_ids = load_data()
    n_calib = len(calib)
    print("=" * 76)
    print("标定点数敏感性：k 个标定格 → 同一批留出格上的成本 MAPE")
    print("=" * 76)
    print(f"  标定池 {n_calib} 格 · 留出池 {len(hold)} 格 · 每个 k 抽 {args.draws} 次")

    # 基准：全部 12 格
    p_full = tm.fit(calib)
    base = mape_cost(p_full, hold)
    print(f"  基准（12 格全用）：MAPE {base:.2f}%\n")

    rng = np.random.default_rng(args.seed)
    rows = []
    print(f"{'k':>3}{'MAPE 中位':>12}{'5–95%':>18}{'相对 12 格':>12}")
    for k in range(4, n_calib + 1):
        vals = []
        for _ in range(args.draws):
            ids = stratified_sample(calib, k, rng)
            sub = calib[calib["cell_id"].isin(ids)]
            try:
                p = tm.fit(sub)
                v = mape_cost(p, hold)
            except Exception:
                continue
            if np.isfinite(v):
                vals.append(v)
        if not vals:
            continue
        med = float(np.median(vals))
        lo, hi = np.percentile(vals, [5, 95])
        rows.append(dict(k=k, mape_med=med, mape_lo=float(lo), mape_hi=float(hi),
                         delta_vs_full=med - base, n_draws=len(vals)))
        print(f"{k:>3}{med:>11.2f}%   [{lo:.2f}, {hi:.2f}]{med - base:>11.2f}pp")

    t = pd.DataFrame(rows)
    out_csv = cfg.RESULTS_DIR / "calib_curve.csv"
    t.to_csv(out_csv, index=False, encoding="utf-8-sig")

    # ---- 作图 ----
    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    ax.plot(t["k"], t["mape_med"], "o-", color="#C44E52", lw=1.8, label="MAPE 中位")
    ax.fill_between(t["k"], t["mape_lo"], t["mape_hi"], color="#C44E52", alpha=0.15,
                    label="5–95% 分位")
    ax.axhline(base, color="#555555", ls="--", lw=1, label=f"12 格全用 = {base:.2f}%")
    ax.set_xlabel("标定点数 k")
    ax.set_ylabel("留出集成本 MAPE（%）")
    ax.set_title("标定点数够不够？—— 曲线变平的地方就是答案")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    out_png = cfg.FIG_DIR / "fig7_calib_curve.png"
    fig.savefig(out_png, dpi=150)
    plt.close(fig)

    # ---- 判读 ----
    print()
    if len(t) >= 3:
        last = t.iloc[-1]
        prev = t.iloc[-2]
        plateau = abs(prev["mape_med"] - last["mape_med"]) < 0.5
        if plateau:
            print(f"  判读：从 k={int(prev['k'])} 到 {int(last['k'])} 只降 "
                  f"{prev['mape_med'] - last['mape_med']:.2f} pp —— **曲线已变平**，")
            print("        说明再多标定点收益很小，12 个点够用（这是诚实结论，不是自夸）。")
        else:
            print(f"  判读：最后一步还降了 {prev['mape_med'] - last['mape_med']:.2f} pp —— "
                  "**曲线未平**，说明 12 个点偏少，应如实写进局限。")
    print(f"\n已写入 {out_csv}")
    print(f"已保存 {out_png}")


if __name__ == "__main__":
    main()


