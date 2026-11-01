"""
_diag_regimes.py —— 诊断脚本：验证"服务端开销与并发成正比"这个假设。

只做分析，不改任何产物。回答三个问题：
  Q1 物理模型（3 参数）在 b≤4 的区间是否本来就准？（也许根本不需要第 4 个参数）
  Q2 加一个 c_srv·b 项（第 4 个参数）能改善多少？
  Q3 c_srv 是不是常数（跨模型稳定）？

用法：& "D:\\llm\\venv\\Scripts\\python.exe" _diag_regimes.py
"""

import json
import sys

import numpy as np
import pandas as pd
from scipy.optimize import minimize

import tc_config as cfg


def load():
    df = pd.read_csv(cfg.DATA_DIR / "summary.csv")
    if "prompt_tokens_median" in df.columns:
        df["prompt_tokens"] = df["prompt_tokens_median"].fillna(cfg.PROMPT_TOKENS)
    else:
        df["prompt_tokens"] = float(cfg.PROMPT_TOKENS)
    df["ctx_avg_tokens"] = df["prompt_tokens"] + cfg.GEN_TOKENS / 2
    return df.dropna(subset=["tpot_p50_ms", "ttft_p50_ms", "throughput_tps"])


def spec_of(row):
    return cfg.MODEL_SPECS[(row["model"], row["quant"])]


def predict3(p, row):
    s = spec_of(row)
    w_eff = s["weight_bytes"] + cfg.kv_bytes_per_token(s) * row["ctx_avg_tokens"] * row["conc"]
    t_mem = w_eff / (p["bw"] * 1e9)
    t_cmp = 2 * s["n_params"] * row["conc"] / (p["peak"] * 1e12)
    return 1e3 * max(t_mem, t_cmp) + p["ovh"]


def predict4(p, row):
    """在 3 参数基础上，加一个与并发成正比的每步开销。"""
    return predict3(p, row) + p["c_srv"] * row["conc"]


def fit(df, use_srv):
    def unpack(x):
        d = dict(bw=float(np.exp(x[0])), peak=float(np.exp(x[1])), ovh=float(x[2]))
        if use_srv:
            d["c_srv"] = float(x[3])
        return d

    def loss(x):
        p = unpack(x)
        if p["ovh"] < 0 or (use_srv and p["c_srv"] < 0):
            return 1e9
        s = 0.0
        for _, r in df.iterrows():
            pred = predict4(p, r) if use_srv else predict3(p, r)
            if pred <= 0:
                return 1e9
            s += np.log(pred / r["tpot_p50_ms"]) ** 2
        return s

    x0 = [np.log(240.0), np.log(30.0), 1.0] + ([0.5] if use_srv else [])
    bounds = [(np.log(50), np.log(1500)), (np.log(2), np.log(500)), (0, 50)]
    if use_srv:
        bounds.append((0, 20))
    res = minimize(loss, x0, method="L-BFGS-B", bounds=bounds)
    return unpack(res.x)


def mape(df, p, fn):
    errs = []
    for _, r in df.iterrows():
        pred = fn(p, r)
        errs.append(abs(pred / r["tpot_p50_ms"] - 1))
    return float(np.mean(errs) * 100), float(np.max(errs) * 100)


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    df = load()
    calib_ids = {cfg.cell_id(*c) for c in cfg.CALIB_CELLS}
    calib = df[df["cell_id"].isin(calib_ids)]
    hold = df[~df["cell_id"].isin(calib_ids)]

    print("=" * 74)
    print("Q1：物理模型（3 参数）分并发区间看 TPOT 误差")
    print("=" * 74)
    p3 = fit(calib, use_srv=False)
    print(f"  拟合（全部标定格）：BW {p3['bw']:.1f} GB/s  PEAK {p3['peak']:.1f}  ovh {p3['ovh']:.2f} ms")
    for lo, hi in [(1, 1), (2, 8), (9, 16), (17, 64)]:
        sub = hold[(hold["conc"] >= lo) & (hold["conc"] <= hi)]
        if len(sub):
            m, mx = mape(sub, p3, predict3)
            print(f"  留出集 并发 {lo:>2}-{hi:<2}  n={len(sub):>2}  MAPE {m:6.1f}%  最大 {mx:6.1f}%")

    print()
    print("=" * 74)
    print("Q2：只用低并发标定格（b≤4）重新拟合 3 参数，看低并发区间能多准")
    print("=" * 74)
    c_low = calib[calib["conc"] <= 4]
    print(f"  低并发标定格数：{len(c_low)}")
    if len(c_low) >= 4:
        p3l = fit(c_low, use_srv=False)
        print(f"  拟合：BW {p3l['bw']:.1f} GB/s  PEAK {p3l['peak']:.1f}  ovh {p3l['ovh']:.2f} ms")
        for lo, hi in [(1, 1), (2, 4), (5, 64)]:
            sub = hold[(hold["conc"] >= lo) & (hold["conc"] <= hi)]
            if len(sub):
                m, mx = mape(sub, p3l, predict3)
                print(f"  留出集 并发 {lo:>2}-{hi:<2}  n={len(sub):>2}  MAPE {m:6.1f}%  最大 {mx:6.1f}%")

    print()
    print("=" * 74)
    print("Q3：加 c_srv·b（第 4 个参数），全区间重新拟合")
    print("=" * 74)
    p4 = fit(calib, use_srv=True)
    print(f"  BW {p4['bw']:.1f} GB/s   PEAK {p4['peak']:.1f}   ovh {p4['ovh']:.2f} ms   "
          f"c_srv {p4['c_srv']:.3f} ms/流")
    for lo, hi in [(1, 1), (2, 8), (9, 16), (17, 64)]:
        sub = hold[(hold["conc"] >= lo) & (hold["conc"] <= hi)]
        if len(sub):
            m, mx = mape(sub, p4, predict4)
            print(f"  留出集 并发 {lo:>2}-{hi:<2}  n={len(sub):>2}  MAPE {m:6.1f}%  最大 {mx:6.1f}%")
    m_all, mx_all = mape(hold, p4, predict4)
    print(f"  → 全留出集：MAPE {m_all:.1f}%  最大 {mx_all:.1f}%")

    print()
    print("=" * 74)
    print("Q4：c_srv 逐模型单独解（看它是不是常数）")
    print("=" * 74)
    for (m, q), sub in df.groupby(["model", "quant"]):
        # 用 c1 定 BW/ovh，再用高并发解 c_srv
        r1 = sub[sub["conc"] == 1]
        if len(r1) == 0:
            continue
        s = cfg.MODEL_SPECS[(m, q)]
        t1 = float(r1["tpot_p50_ms"].median())
        # 假设 BW≈物理模型值，先由 t1 反解 ovh（用 p4 的 BW）
        p = dict(p4)
        t_mem1 = (s["weight_bytes"] + cfg.kv_bytes_per_token(s) * r1["ctx_avg_tokens"].iloc[0]) / (p["bw"] * 1e9) * 1e3
        ovh = t1 - t_mem1
        vals = []
        for c in [16, 32]:
            rr = sub[sub["conc"] == c]
            if len(rr) == 0:
                continue
            t = float(rr["tpot_p50_ms"].median())
            tm = (s["weight_bytes"] + cfg.kv_bytes_per_token(s) * rr["ctx_avg_tokens"].iloc[0] * c) / (p["bw"] * 1e9) * 1e3
            vals.append((t - tm - ovh) / c)
        if vals:
            print(f"  {m:>4} {q:<7} ovh={ovh:5.2f} ms   c_srv≈{np.mean(vals):.3f} ms/流  "
                  f"（{', '.join(f'{v:.2f}' for v in vals)}）")


if __name__ == "__main__":
    main()
