# -*- coding: utf-8 -*-
"""
A4：`c_srv × 并发` 这个线性假设到底成不成立？

## 模型里的假设

    TPOT(b) = max(t_mem(b), t_cmp(b)) + c_srv · b

`c_srv` 被当作**常数**。如果实际是超线性（并发越高、每流每步越贵），
那这个线性项在高并发就会系统性低估——而我们已经观察到过"中间并发 TPOT 偏低"。

## 做法

1. 用拟合参数算出硬件下限 `floor(b) = max(t_mem, t_cmp)`（可逐格算）；
2. **服务时间** `s(b) = TPOT实测 − floor(b)`；
3. **隐含 c_srv(b) = s(b)/b** —— 若线性，它应与 b 无关；
4. 对 `s(b) = a + c·b` 做回归，看截距 a 是否≈0、R² 是否高；
5. 比较低并发（b≤4）与高并发（b≥16）的平均隐含 c_srv，给出偏离倍数。

## 产出

- `results/csrv_linearity.csv` / `csrv_linearity.json`
- `figures/fig16_csrv_linearity.png`
"""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import tc_config as cfg

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def load():
    df = pd.read_csv(cfg.DATA_DIR / "summary_steady.csv")
    cid = df["cell_id"].astype(str)
    df = df[~cid.str.contains(r"_(?:w\d+|n\d+|p\d+|mt\d+)$", regex=True)
            & ~cid.str.endswith("_xf")].copy()
    if "steady_cost_gpu_s_per_1m" in df.columns:
        df["cost_gpu_s_per_1m"] = df["steady_cost_gpu_s_per_1m"]
    df["prompt_tokens"] = df["prompt_tokens_median"]
    df["ctx_avg_tokens"] = df["prompt_tokens"] + cfg.GEN_TOKENS / 2
    return df


def main():
    fit = json.loads((cfg.RESULTS_DIR / "fit.json").read_text(encoding="utf-8"))
    bw = float(fit["bw"]) * 1e9
    peak = float(fit["peak"]) * 1e12
    df = load()

    rows = []
    for _, r in df.iterrows():
        spec = cfg.MODEL_SPECS.get((r["model"], r["quant"]))
        if not spec:
            continue
        b = float(r["conc"])
        kv = cfg.kv_bytes_per_token(spec) * float(r["ctx_avg_tokens"]) * b
        t_mem = (spec["weight_bytes"] + kv) / bw * 1e3
        t_cmp = 2 * spec["n_params"] * b / peak * 1e3
        floor = max(t_mem, t_cmp)
        tpot = float(r["tpot_p50_ms"])
        s = tpot - floor
        if s <= 0:
            continue
        rows.append(dict(model=r["model"], quant=r["quant"], ctx=int(r["ctx"]),
                         conc=int(b), tpot=tpot, floor=floor, service=s,
                         implied_csrv=s / b))

    t = pd.DataFrame(rows)
    t.to_csv(cfg.RESULTS_DIR / "csrv_linearity.csv", index=False, encoding="utf-8")

    print("=== 每个模型：隐含 c_srv 随并发的走向 ===")
    out = {}
    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    cmap = plt.get_cmap("tab10")
    best = []
    for i, (m, g) in enumerate(t.groupby("model")):
        by = g.groupby("conc")["implied_csrv"].median()
        ax.plot(by.index, by.values, "o-", color=cmap(i % 10), label=m)
        low = g[g["conc"] <= 4]["implied_csrv"].median()
        high = g[g["conc"] >= 16]["implied_csrv"].median()
        ratio = high / low if low else float("nan")

        # 线性回归 s = a + c·b
        x = g["conc"].to_numpy(float)
        y = g["service"].to_numpy(float)
        c, a = np.polyfit(x, y, 1)
        pred = a + c * x
        ss_res = float(((y - pred) ** 2).sum())
        ss_tot = float(((y - y.mean()) ** 2).sum())
        r2 = 1 - ss_res / ss_tot if ss_tot else float("nan")

        shr = ", ".join(f"{int(k)}:{v:.2f}" for k, v in by.items())
        print(f"  {m:<8} 隐含 c_srv 扫描 {shr}")
        print(f"           回归 s=a+c·b → a={a:6.2f} ms  c={c:.3f} ms  R²={r2:.3f}")
        print(f"           低并发(≤4) {low:.2f} vs 高并发(≥16) {high:.2f} → {ratio:.2f}×")
        out[m] = dict(implied_by_conc={int(k): round(float(v), 3) for k, v in by.items()},
                      intercept_ms=round(float(a), 3), slope_ms=round(float(c), 3),
                      r2=round(r2, 3),
                      low_conc_csrv=round(float(low), 3) if np.isfinite(low) else None,
                      high_conc_csrv=round(float(high), 3) if np.isfinite(high) else None,
                      high_over_low=round(float(ratio), 3) if np.isfinite(ratio) else None)
        best.append(ratio if np.isfinite(ratio) else 1.0)

    ratios = [v for v in best if np.isfinite(v)]
    mean_ratio = float(np.mean(ratios)) if ratios else float("nan")
    verdict = ("线性成立（高低并发比值 ≈1）" if mean_ratio < 1.15
               else "**超线性**：高并发的每步开销更大，线性项会低估")
    print(f"\n  平均比值 {mean_ratio:.2f}× → {verdict}")

    ax.axhline(float(fit["c_srv"]), color="crimson", ls="--", lw=1.2,
               label=f"拟合常数 c_srv={fit['c_srv']:.2f} ms")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("并发 b")
    ax.set_ylabel("隐含 c_srv = 服务时间 / b  (ms)")
    ax.set_title("每步服务开销是否与并发无关（线性假设）")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(cfg.FIG_DIR / "fig16_csrv_linearity.png", dpi=150)

    summ = dict(fitted_csrv=round(float(fit["c_srv"]), 3),
                mean_high_over_low=round(mean_ratio, 3),
                verdict=verdict, models=out)
    (cfg.RESULTS_DIR / "csrv_linearity.json").write_text(
        json.dumps(summ, ensure_ascii=False, indent=2), encoding="utf-8")
    print("  图 → figures/fig16_csrv_linearity.png")
    print("→ results/csrv_linearity.csv / csrv_linearity.json")


if __name__ == "__main__":
    main()
