# -*- coding: utf-8 -*-
"""验证：给 decode 单独一个有效算力（PEAK_decode）能不能改善拟合。

依据：llama-batched-bench 反推 decode 有效算力 ≈ 9 TFLOP/s，
而 prefill（TTFT）反推的有效算力 ≈ 31.6 TFLOP/s，相差 3.5 倍。
"""
import sys

import numpy as np
import pandas as pd
from scipy.optimize import minimize

import tc_config as cfg

sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def load():
    df = pd.read_csv(cfg.DATA_DIR / "summary.csv")
    df["prompt_tokens"] = (df["prompt_tokens_median"].fillna(cfg.PROMPT_TOKENS)
                           if "prompt_tokens_median" in df.columns else float(cfg.PROMPT_TOKENS))
    df["ctx_avg_tokens"] = df["prompt_tokens"] + cfg.GEN_TOKENS / 2
    df = df.dropna(subset=["tpot_p50_ms", "ttft_p50_ms", "cost_gpu_s_per_1m"])
    fits = [cfg.kv_fits_vram((r["model"], r["quant"]), int(r["ctx"]), int(r["conc"]))
            for _, r in df.iterrows()]
    return df[fits].copy()


def predict(p, row):
    s = cfg.MODEL_SPECS[(row["model"], row["quant"])]
    b = row["conc"]
    w_eff = s["weight_bytes"] + cfg.kv_bytes_per_token(s) * row["ctx_avg_tokens"] * b
    t_mem = w_eff / (p["bw"] * 1e9)
    t_cmp = 2 * s["n_params"] * b / (p["peak_dec"] * 1e12)
    tpot = 1e3 * max(t_mem, t_cmp) + p["c_srv"] * b

    b_eff = min(b, p["b_sat"])
    t_pf = 2 * s["n_params"] * row["prompt_tokens"] * b_eff / (p["peak_pre"] * 1e12)
    ttft = 1e3 * max(s["weight_bytes"] / (p["bw"] * 1e9), t_pf) + p["c_srv"] * b + p["ovh_ttft"]

    per = (ttft + cfg.GEN_TOKENS * tpot) / 1e3
    thr = b * cfg.GEN_TOKENS / per
    return dict(tpot=tpot, ttft=ttft, cost=1e6 / thr)


def unpack(x, six):
    p = dict(bw=float(np.exp(x[0])), peak_pre=float(np.exp(x[1])),
             c_srv=float(x[2]), b_sat=float(np.exp(x[3])), ovh_ttft=float(x[4]))
    p["peak_dec"] = float(np.exp(x[5])) if six else p["peak_pre"]
    return p


def fit(calib, six):
    def loss(x):
        p = unpack(x, six)
        if min(p["c_srv"], p["ovh_ttft"]) < 0:
            return 1e9
        s = 0.0
        for _, r in calib.iterrows():
            pr = predict(p, r)
            for a, b in (("tpot", "tpot_p50_ms"), ("ttft", "ttft_p50_ms")):
                v = pr[a] / r[b]
                if not np.isfinite(v) or v <= 0:
                    return 1e9
                s += np.log(v) ** 2
        return s

    x0 = [np.log(250), np.log(31), 0.85, np.log(14), 16] + ([np.log(9)] if six else [])
    bnd = [(np.log(60), np.log(1200)), (np.log(3), np.log(300)), (0, 20),
           (np.log(1), np.log(64)), (0, 200)]
    if six:
        bnd.append((np.log(2), np.log(300)))
    res = minimize(loss, x0, method="L-BFGS-B", bounds=bnd)
    return unpack(res.x, six)


def mape(df, p, key, meas):
    e = [abs(predict(p, r)[key] / r[meas] - 1) for _, r in df.iterrows()]
    return np.mean(e) * 100, np.max(e) * 100


def main():
    df = load()
    calib_ids = {cfg.cell_id(*c) for c in cfg.CALIB_CELLS}
    calib = df[df["cell_id"].isin(calib_ids)]
    hold = df[~df["cell_id"].isin(calib_ids)]
    print(f"数据 {len(df)} 格（标定 {len(calib)} / 留出 {len(hold)}）\n")

    for six, name in [(False, "5 参数（decode 与 prefill 共用一个 PEAK）"),
                      (True, "6 参数（decode / prefill 分开）")]:
        p = fit(calib, six)
        pc, pm = mape(hold, p, "cost", "cost_gpu_s_per_1m")
        tc, tm = mape(hold, p, "tpot", "tpot_p50_ms")
        print(f"=== {name} ===")
        print(f"  BW {p['bw']:.1f} GB/s   PEAK_pre {p['peak_pre']:.1f}"
              + (f"   PEAK_dec {p['peak_dec']:.1f}" if six else "")
              + f"   c_srv {p['c_srv']:.3f}   B_SAT {p['b_sat']:.1f}   ovh_ttft {p['ovh_ttft']:.1f}")
        print(f"  留出集：成本 MAPE {pc:.2f}%（max {pm:.1f}%）  TPOT MAPE {tc:.2f}%（max {tm:.1f}%）\n")


if __name__ == "__main__":
    main()
