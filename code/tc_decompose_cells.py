# -*- coding: utf-8 -*-
"""
A-4 逐格拆解：每一格的 TPOT 里，有多少是"服务开销"而不是硬件下限

## 背景

`tc_decompose_csrv.py` 是**一次性分析**（只在几个格上做过）。
这里把它做成**对全部 108 格都算一遍**的产物，让"高并发下 TPOT 的 58% 不是访存"
这句话变成**可以逐格查的表**。

## 每格输出

| 列 | 含义 |
|---|---|
| `t_mem_ms` | 读权重 + KV 的时间（硬件下限，访存受限那一支） |
| `t_cmp_ms` | decode 算力时间（算力受限那一支） |
| `t_floor_ms` | `max(t_mem, t_cmp)`，硬件真正能给出的下限 |
| `t_srv_ms` | `c_srv × b`，服务路径开销 |
| `t_pred_ms` | 模型预测的 TPOT（应为 floor + srv） |
| `srv_share` | `t_srv / t_pred`，服务开销占比 |

## 产出

- `results/decompose_cells.csv`
- `figures/fig13_srv_share.png`：服务开销占比 vs 并发（按模型分色）
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
import tc_model as tm

sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def main():
    fj = json.loads((cfg.RESULTS_DIR / "fit.json").read_text(encoding="utf-8"))
    base = dict(bw_gbs=fj["bw"], peak_tflops=fj["peak"],
                peak_prefill_tflops=fj.get("peak_prefill", fj["peak"]),
                c_srv_ms=fj["c_srv"], b_sat=fj["b_sat"],
                ovh_ttft_ms=fj["ovh_ttft"])

    df = pd.read_csv(cfg.DATA_DIR / "summary_steady.csv")
    df = df[df["cell_id"].str.contains("_p", regex=False) == False].copy()
    df["prompt_tokens"] = df["prompt_tokens_median"]
    df["ctx_avg_tokens"] = df["prompt_tokens"] + cfg.GEN_TOKENS / 2

    rows = []
    for _, r in df.iterrows():
        spec = cfg.MODEL_SPECS.get((r["model"], r["quant"]))
        if spec is None:
            continue
        b = float(r["conc"])
        kv = cfg.kv_bytes_per_token(spec) * float(r["ctx_avg_tokens"]) * b
        t_mem = (spec["weight_bytes"] + kv) / (base["bw_gbs"] * 1e9) * 1e3
        t_cmp = 2 * spec["n_params"] * b / (base["peak_tflops"] * 1e12) * 1e3
        t_floor = max(t_mem, t_cmp)
        t_srv = base["c_srv_ms"] * b
        t_pred = t_floor + t_srv
        pr = tm.predict(base, r)
        rows.append(dict(
            cell_id=r["cell_id"], model=r["model"], quant=r["quant"],
            ctx=int(r["ctx"]), conc=int(b),
            prompt_tokens=float(r["prompt_tokens"]),
            tpot_meas=float(r["tpot_p50_ms"]),
            tpot_pred=float(pr["pred_tpot_ms"]),
            t_mem_ms=t_mem, t_cmp_ms=t_cmp, t_floor_ms=t_floor,
            t_srv_ms=t_srv, t_pred_ms=t_pred,
            srv_share=t_srv / t_pred if t_pred > 0 else np.nan,
            compute_bound=(t_cmp > t_mem),
        ))

    out = pd.DataFrame(rows).sort_values(["model", "conc"])
    out.to_csv(cfg.RESULTS_DIR / "decompose_cells.csv", index=False, encoding="utf-8")
    print(f"逐格拆解 {len(out)} 行 → results/decompose_cells.csv\n")

    print(f"  {'并发':>4}{'格数':>6}{'服务开销占比':>14}{'硬件下限(中位)':>16}")
    for c, g in out.groupby("conc"):
        print(f"  {int(c):>4}{len(g):>6}{g['srv_share'].median():>13.1%}"
              f"{g['t_floor_ms'].median():>14.2f} ms")

    hi = out[out["conc"] >= 16]
    if not hi.empty:
        print(f"\n  并发 ≥16 的服务开销占比：中位 {hi['srv_share'].median():.1%}、"
              f"最大 {hi['srv_share'].max():.1%}")
    lo = out[out["conc"] <= 4]
    if not lo.empty:
        print(f"  并发 ≤4 的服务开销占比：中位 {lo['srv_share'].median():.1%}")
    n_bound = int(out["compute_bound"].sum())
    print(f"  算力受限的格：{n_bound}/{len(out)}"
          + ("（0 = decode 全程访存受限，所以 peak_tflops 不可辨识）" if n_bound == 0 else ""))

    # ---- 图 ----
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    cmap = plt.get_cmap("tab10")
    for i, m in enumerate(sorted(out["model"].unique())):
        g = out[out["model"] == m].groupby("conc")["srv_share"].median()
        ax.plot(g.index, g.values * 100, "o-", color=cmap(i % 10), label=m)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("并发")
    ax.set_ylabel("服务开销占 TPOT 的比例 (%)")
    ax.set_title("TPOT 里有多少不是硬件下限（108 格逐格拆解）")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(cfg.FIG_DIR / "fig13_srv_share.png", dpi=150)
    print("\n图 → figures/fig13_srv_share.png")


if __name__ == "__main__":
    main()
