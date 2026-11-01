"""
_diag_v5.py —— 验证 5 参数模型（TPOT 与 TTFT 都建模），并给出成本误差。

模型：
    TPOT = max( W_eff(b)/BW , 2·P·b/PEAK ) + c_srv·b
    TTFT = max( W/BW , 2·P·L·min(b, B_SAT)/PEAK ) + c_srv·b + ovh_ttft
    吞吐 = b·GEN / (TTFT + GEN·TPOT)
    成本 = 1e6/吞吐

其中 W_eff(b) = W + kv_per_token × 上下文 × b（KV 也是每条流各读各的）。

5 个参数：BW、PEAK、c_srv、B_SAT、ovh_ttft
物理含义：
    c_srv  每条并发流每步的服务端开销（采样 + JSON + SSE），ms
    B_SAT  prefill 的并行放大饱和点（llama.cpp 按 ubatch 分块，不是无限并行）
    ovh_ttft 首字路径的固定开销（HTTP + 分词 + 调度），ms

用法：& "D:\\llm\\venv\\Scripts\\python.exe" _diag_v5.py
"""

import sys

import numpy as np
import pandas as pd
from scipy.optimize import minimize

import tc_config as cfg


def load():
    df = pd.read_csv(cfg.DATA_DIR / "summary.csv")
    df["prompt_tokens"] = (df["prompt_tokens_median"].fillna(cfg.PROMPT_TOKENS)
                           if "prompt_tokens_median" in df.columns else float(cfg.PROMPT_TOKENS))
    df["ctx_avg_tokens"] = df["prompt_tokens"] + cfg.GEN_TOKENS / 2
    return df.dropna(subset=["tpot_p50_ms", "ttft_p50_ms", "throughput_tps"])


def predict(p, row):
    s = cfg.MODEL_SPECS[(row["model"], row["quant"])]
    b = row["conc"]
    kv = cfg.kv_bytes_per_token(s)
    w_eff = s["weight_bytes"] + kv * row["ctx_avg_tokens"] * b
    t_mem = w_eff / (p["bw"] * 1e9)
    t_cmp = 2 * s["n_params"] * b / (p["peak"] * 1e12)
    tpot_ms = 1e3 * max(t_mem, t_cmp) + p["c_srv"] * b

    b_eff = min(b, p["b_sat"])
    t_pf_mem = s["weight_bytes"] / (p["bw"] * 1e9)
    t_pf_cmp = 2 * s["n_params"] * row["prompt_tokens"] * b_eff / (p["peak"] * 1e12)
    ttft_ms = 1e3 * max(t_pf_mem, t_pf_cmp) + p["c_srv"] * b + p["ovh_ttft"]

    per_req_s = (ttft_ms + cfg.GEN_TOKENS * tpot_ms) / 1e3
    thr = b * cfg.GEN_TOKENS / per_req_s
    return dict(tpot=tpot_ms, ttft=ttft_ms, thr=thr, cost=1e6 / thr)


def fit(df):
    def unpack(x):
        return dict(bw=float(np.exp(x[0])), peak=float(np.exp(x[1])),
                    c_srv=float(x[2]), b_sat=float(np.exp(x[3])), ovh_ttft=float(x[4]))

    def loss(x):
        p = unpack(x)
        if min(p["c_srv"], p["ovh_ttft"]) < 0:
            return 1e9
        s = 0.0
        for _, r in df.iterrows():
            pr = predict(p, r)
            if pr["tpot"] <= 0 or pr["ttft"] <= 0 or pr["thr"] <= 0:
                return 1e9
            s += np.log(pr["tpot"] / r["tpot_p50_ms"]) ** 2
            s += np.log(pr["ttft"] / r["ttft_p50_ms"]) ** 2
        return s

    x0 = [np.log(250.0), np.log(30.0), 0.8, np.log(8.0), 10.0]
    bnd = [(np.log(60), np.log(1200)), (np.log(2), np.log(500)),
           (0, 20), (np.log(1.0), np.log(64)), (0, 200)]
    res = minimize(loss, x0, method="L-BFGS-B", bounds=bnd)
    return unpack(res.x)


def report(df, p, name):
    errs = {"tpot": [], "ttft": [], "cost": []}
    for _, r in df.iterrows():
        pr = predict(p, r)
        errs["tpot"].append(abs(pr["tpot"] / r["tpot_p50_ms"] - 1))
        errs["ttft"].append(abs(pr["ttft"] / r["ttft_p50_ms"] - 1))
        errs["cost"].append(abs(pr["cost"] / r["cost_gpu_s_per_1m"] - 1))
    print(f"  {name}:  TPOT MAPE {np.mean(errs['tpot'])*100:5.1f}% (max {np.max(errs['tpot'])*100:5.1f}%) | "
          f"TTFT {np.mean(errs['ttft'])*100:5.1f}% (max {np.max(errs['ttft'])*100:5.1f}%) | "
          f"成本 {np.mean(errs['cost'])*100:5.1f}% (max {np.max(errs['cost'])*100:5.1f}%)")
    return np.mean(errs["cost"]) * 100


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    df = load()
    calib_ids = {cfg.cell_id(*c) for c in cfg.CALIB_CELLS}
    calib = df[df["cell_id"].isin(calib_ids)]
    hold = df[~df["cell_id"].isin(calib_ids)]

    print("=" * 78)
    print("5 参数模型（BW / PEAK / c_srv / B_SAT / ovh_ttft）")
    print("=" * 78)
    p = fit(calib)
    print(f"  BW = {p['bw']:.1f} GB/s    PEAK = {p['peak']:.1f} TFLOP/s")
    print(f"  c_srv = {p['c_srv']:.3f} ms/流     B_SAT = {p['b_sat']:.1f}     "
          f"ovh_ttft = {p['ovh_ttft']:.1f} ms")
    print()
    report(calib, p, "标定集")
    report(hold, p, "留出集")

    print()
    print("分并发区间（留出集，看误差是否还有系统趋势）：")
    for lo, hi in [(1, 1), (2, 8), (9, 16), (17, 64)]:
        sub = hold[(hold["conc"] >= lo) & (hold["conc"] <= hi)]
        if len(sub):
            report(sub, p, f"  并发 {lo:>2}-{hi:<2} (n={len(sub):>2})")

    # 对比：只用 TPOT 拟合时成本误差主要来自 TTFT 还是 TPOT
    print()
    print("误差归因（留出集）：")
    for _, r in hold.sort_values("conc").iterrows():
        pr = predict(p, r)
        if abs(pr["cost"] / r["cost_gpu_s_per_1m"] - 1) > 0.15:
            print(f"  {r['cell_id']:<26} 成本偏差 {(pr['cost']/r['cost_gpu_s_per_1m']-1)*100:+6.1f}%  "
                  f"TPOT {(pr['tpot']/r['tpot_p50_ms']-1)*100:+6.1f}%  "
                  f"TTFT {(pr['ttft']/r['ttft_p50_ms']-1)*100:+7.1f}%")


if __name__ == "__main__":
    main()
