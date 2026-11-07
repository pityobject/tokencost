# -*- coding: utf-8 -*-
"""
模型比较（`tc.py modelcompare`，M9）：把"为什么是 6 个参数"从叙事变成统计

## 三个嵌套模型

| 模型 | 参数 | 形式 |
|---|---|---|
| **M4** | 4 | `TPOT = W·b/BW + c·b`；`TTFT = 2·P·L·b/PEAK_pre + c·b + ovh`（无 KV、无饱和、无 max） |
| **M6** | 6 | 现行六参数：`TPOT = max((W+KV·L·b)/BW, 2·P·b/PEAK_dec) + c·b`；`TTFT = max(W/BW, 2·P·L·min(b,B_SAT)/PEAK_pre) + c·b + ovh` |
| **M7** | 7 | M6 + 抖动项：`TPOT × (1 + k·CV)`（CV 来自逐请求抖动，即 F24 的假说） |

三者都在**同一批标定格**上拟合，用**同一套留出格**评估。

## 判据

- `AIC = 2k − 2lnL`，`BIC = k·ln(n) − 2lnL`
- 高斯误差、log 空间（与项目一贯的拟合口径一致）
- **只有 AIC/BIC 同时更小，才说明多出来的参数值得**

## 用法

    python code/tc_model_compare.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

sys.path.insert(0, str(Path(__file__).resolve().parent))

import tc_config as cfg

sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def spec_of(model, quant):
    return cfg.MODEL_SPECS[(model, quant)]


def kv_bytes(spec):
    return 2 * 2 * spec["head_dim"] * spec["n_layers"] * 2  # K,V × kv_heads × head_dim × fp16


def predict_m4(p, rows):
    BW, PEAK_pre, c, ovh = p
    out = []
    for r in rows:
        s = spec_of(r["model"], r["quant"])
        b = float(r["conc"])
        tpot = s["weight_bytes"] * b / (BW * 1e9) * 1e3 + c * b
        ttft = 2 * s["n_params"] * r["prompt_tokens"] * b / (PEAK_pre * 1e12) * 1e3 + c * b + ovh
        out.append((ttft, tpot))
    return np.array(out)


def predict_m6(p, rows):
    BW, PEAK_dec, PEAK_pre, c, B_SAT, ovh = p
    out = []
    for r in rows:
        s = spec_of(r["model"], r["quant"])
        b = float(r["conc"])
        kv = kv_bytes(s) * r["ctx_avg_tokens"] * b
        t_mem = (s["weight_bytes"] + kv) / (BW * 1e9) * 1e3
        t_cmp = 2 * s["n_params"] * b / (PEAK_dec * 1e12) * 1e3
        tpot = max(t_mem, t_cmp) + c * b
        ttft = max(s["weight_bytes"] / (BW * 1e9) * 1e3,
                   2 * s["n_params"] * r["prompt_tokens"] * min(b, B_SAT) / (PEAK_pre * 1e12) * 1e3) \
            + c * b + ovh
        out.append((ttft, tpot))
    return np.array(out)


def predict_m7(p, rows, cv):
    base = predict_m6(p[:6], rows)
    k = p[6]
    base[:, 1] *= (1 + k * cv)
    return base


def residuals(p, rows, cv, kind):
    if kind == "m4":
        pr = predict_m4(p, rows)
    elif kind == "m6":
        pr = predict_m6(p, rows)
    else:
        pr = predict_m7(p, rows, cv)
    # log 空间残差（避免大值主导），对 TTFT 与 TPOT 同等对待
    r = []
    for i, row in enumerate(rows):
        for j, key in ((0, "ttft_p50_ms"), (1, "tpot_p50_ms")):
            m = max(float(row[key]), 1e-6)
            pv = max(float(pr[i, j]), 1e-6)
            r.append(np.log(pv) - np.log(m))
    return np.array(r)


def aic_bic(sse, n, k):
    sigma2 = sse / n
    logL = -0.5 * n * (np.log(2 * np.pi * sigma2) + 1)
    return 2 * k - 2 * logL, k * np.log(n) - 2 * logL, logL


def main():
    df = cfg.load_population("core").reset_index(drop=True)
    jit = pd.read_csv(cfg.RESULTS_DIR / "jitter_by_cell.csv")
    cvmap = dict(zip(jit["cell_id"], jit["cv"]))
    df["cv"] = df["cell_id"].map(cvmap).fillna(df["cell_id"].map(cvmap).median())

    # 标定 12 格：设计空间里的 6 维角点（与项目一贯做法一致：每个模型若干并发）
    calib_ids = set()
    for (m, q) in df[["model", "quant"]].drop_duplicates().itertuples(index=False):
        sub = df[(df["model"] == m) & (df["quant"] == q)]
        for c in (1, 4, 16, 32):
            s2 = sub[sub["conc"] == c]
            if not s2.empty:
                calib_ids.add(s2.iloc[0]["cell_id"])
    calib = df[df["cell_id"].isin(calib_ids)]
    hold = df[~df["cell_id"].isin(calib_ids)]
    print("=" * 88)
    print(f"模型比较｜标定 {len(calib)} 格，留出 {len(hold)} 格，总 {len(df)} 格")
    print("=" * 88)

    rows_c = calib.to_dict("records")
    cv_c = calib["cv"].to_numpy()

    # 起点：项目已发布的六参数
    fit = json.loads((cfg.RESULTS_DIR / "fit.json").read_text(encoding="utf-8"))
    x0_m6 = [fit["bw"], fit["peak"], fit["peak_prefill"], fit["c_srv"],
             fit["b_sat"], fit["ovh_ttft"]]
    x0_m4 = [fit["bw"], fit["peak_prefill"], fit["c_srv"], fit["ovh_ttft"]]
    x0_m7 = x0_m6 + [1.0]

    models = [
        ("M4（4 参数）", "m4", x0_m4,
         dict(bounds=([50, 5, 0.05, 0.5], [1500, 500, 5, 200]))),
        ("M6（6 参数，现行）", "m6", x0_m6,
         dict(bounds=([50, 1, 5, 0.05, 0.5, 0.5], [1500, 500, 500, 5, 32, 200]))),
        ("M7（6 + 抖动项）", "m7", x0_m7,
         dict(bounds=([50, 1, 5, 0.05, 0.5, 0.5, -5], [1500, 500, 500, 5, 32, 200, 100]))),
    ]

    res_rows = []
    for name, kind, x0, opts in models:
        r = least_squares(residuals, x0, args=(rows_c, cv_c, kind),
                          method="trf", max_nfev=4000, **opts)
        sse = float(np.sum(r.fun ** 2))
        n = len(r.fun)
        k = len(x0)
        aic, bic, logL = aic_bic(sse, n, k)
        # 留出评估
        rows_h = hold.to_dict("records")
        cv_h = hold["cv"].to_numpy()
        rr = residuals(r.x, rows_h, cv_h, kind)
        mape = float(np.expm1(np.abs(rr)).mean() * 100)
        res_rows.append(dict(name=name, kind=kind, k=k, params=[round(float(v), 3) for v in r.x],
                             sse=round(sse, 4), aic=round(aic, 1), bic=round(bic, 1),
                             logL=round(logL, 2), holdout_mape=round(mape, 2)))
        print(f"\n  【{name}】")
        print(f"    参数: " + ", ".join(f"{v:.3f}" for v in r.x))
        print(f"    n={n}｜SSE(log)={sse:.3f}｜logL={logL:.1f}")
        print(f"    AIC={aic:.1f}   BIC={bic:.1f}   留出 MAPE={mape:.2f}%")

    print("\n" + "=" * 88)
    print("对比表（AIC/BIC 越小越好）")
    print("=" * 88)
    print(f"  {'模型':<20}{'k':>3}{'AIC':>10}{'BIC':>11}{'留出 MAPE':>12}")
    for r in res_rows:
        print(f"  {r['name']:<20}{r['k']:>3}{r['aic']:>10.1f}{r['bic']:>11.1f}"
              f"{r['holdout_mape']:>11.2f}%")

    base = next(r for r in res_rows if r["kind"] == "m6")
    print(f"\n  以 M6 为基准：")
    for r in res_rows:
        if r["kind"] == "m6":
            continue
        print(f"    {r['name']:<20} ΔAIC={r['aic']-base['aic']:+.1f}  "
              f"ΔBIC={r['bic']-base['bic']:+.1f}  "
              f"Δ留出MAPE={r['holdout_mape']-base['holdout_mape']:+.2f}pp")
    print("\n  读法：ΔAIC/ΔBIC 为正 = 比 M6 差（多出的参数不值得）。")
    print("  注意：TTFT 与 TPOT 一起拟合，n = 2×格数；所以 k 每多 1，")
    print(f"  BIC 罚项 = ln(n) = {np.log(len(res_rows)*0+2*len(calib)):.2f}。")

    out = dict(n_calib=len(calib), n_hold=len(hold), n_obs=2 * len(calib),
               models=res_rows,
               note=("三个嵌套模型同数据同优化器同口径；AIC/BIC 用高斯 log 似然。"
                     "M7 的抖动项来自 F24 的假说（TPOT×(1+k·CV)）"))
    (cfg.RESULTS_DIR / "model_compare.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n→ results/model_compare.json")


if __name__ == "__main__":
    main()
