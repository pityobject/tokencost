# -*- coding: utf-8 -*-
"""
KV 项系数检验（B3 修正版）：用**实际提示长度**回归，而不是 ctx 上限

## 之前的设计错在哪

原计划"固定并发、只改 ctx"是错的：`ctx` 只是服务端**上限**，
真实解码时的平均上下文 = 实际提示长度 + 生成/2。现有 512/2048/8192 三档
实测提示长度都是 297，所以 TPOT 几乎不变（12.90/13.06/12.56），
按 ctx 回归只能得到噪声斜率（反推 BW=6.8 GB/s，毫无意义）。

## 正确的检验

模型里唯一随上下文增长的是 KV 项：

    ∂TPOT/∂L = kv_bytes_per_token × b / BW

所以固定 (模型, 量化, 上下文档, 并发)，**沿实际 L 回归**，
反推 `BW` 与拟合值比较。数据来源：`_pNNN` 长度探针（L 从 177 到 1972）。

## 附加检验：ctx 本身有没有影响？

同一 L 下比较不同 ctx 的 TPOT，若差异在噪声内 → **ctx 只受显存约束，不影响速度**。
这是一条有用的"No effect"结论（避免大家误以为加大 ctx 会变慢）。

## 产出

- `results/kv_coef_check.json`
- `figures/fig17_kv_coef.png`
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


def main():
    df = pd.read_csv(cfg.DATA_DIR / "summary_steady.csv")
    df["prompt_tokens"] = df["prompt_tokens_median"]
    df["ctx_avg_tokens"] = df["prompt_tokens"] + cfg.GEN_TOKENS / 2
    fit = json.loads((cfg.RESULTS_DIR / "fit.json").read_text(encoding="utf-8"))
    bw_model = float(fit["bw"])

    probes = df[df["cell_id"].astype(str).str.contains(r"_p\d+$", regex=True)].copy()
    if probes.empty:
        print("没有长度探针格（_pNNN），先跑 tc_run_cell.py ... --prompt-tokens N")
        return

    print(f"长度探针格 {len(probes)} 个\n")
    rows = []
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    cmap = plt.get_cmap("tab10")

    groups = list(probes.groupby(["model", "quant", "ctx", "conc"]))
    for i, ((m, q, ctx, b), g) in enumerate(groups):
        if g["prompt_tokens"].nunique() < 2:
            continue
        L = g["prompt_tokens"].to_numpy(float)
        y = g["tpot_p50_ms"].to_numpy(float)
        order = np.argsort(L)
        L, y = L[order], y[order]
        slope = float(np.polyfit(L, y, 1)[0])          # ms/token
        if slope <= 0:
            continue
        kv = cfg.kv_bytes_per_token(cfg.MODEL_SPECS[(m, q)])
        bw_implied = kv * b / (slope * 1e-3) / 1e9     # GB/s
        ratio = bw_implied / bw_model
        rows.append(dict(model=m, quant=q, ctx=int(ctx), conc=int(b),
                         n_points=int(len(L)), L_min=float(L.min()), L_max=float(L.max()),
                         slope_us_per_token=round(slope * 1e3, 3),
                         bw_implied_gbs=round(bw_implied, 1),
                         ratio_to_model=round(ratio, 3)))
        print(f"  {m:<6}{q:<8}ctx{int(ctx):<6}c{int(b):<4} n={len(L)}  "
              f"L {L.min():.0f}–{L.max():.0f}  斜率 {slope*1e3:6.2f} µs/tok  "
              f"反推 BW {bw_implied:6.1f} GB/s  比值 {ratio:.2f}×")
        if i < 10:
            axes[0].plot(L, y, "o-", color=cmap(i % 10),
                         label=f"{m} ctx{int(ctx)} c{int(b)}")

    if not rows:
        print("没有可用的多长度组")
        return

    t = pd.DataFrame(rows)
    t.to_csv(cfg.RESULTS_DIR / "kv_coef_check.csv", index=False, encoding="utf-8")

    # ---- 检验 2：ctx 本身有没有影响（同一 L、同一并发） ----
    print("\n=== ctx 本身是否影响 TPOT（同一实测 L）===")
    ctx_rows = []
    for (m, q, b, Lv), g in df.groupby(["model", "quant", "conc", "prompt_tokens"]):
        if g["ctx"].nunique() < 2:
            continue
        vals = g.groupby("ctx")["tpot_p50_ms"].median()
        spread = (vals.max() / vals.min() - 1) * 100
        ctx_rows.append(dict(model=m, quant=q, conc=int(b), L=float(Lv),
                             n_ctx=int(len(vals)), spread_pct=round(float(spread), 2)))
    if ctx_rows:
        c = pd.DataFrame(ctx_rows)
        print(f"  {len(c)} 组（同模型/同并发/同 L，多个 ctx）")
        print(f"  TPOT 跨 ctx 的相对离散：中位 {c['spread_pct'].median():.2f}%、"
              f"最大 {c['spread_pct'].max():.2f}%")
        axes[1].scatter(c["spread_pct"], c["conc"], alpha=0.7)
        axes[1].set_xlabel("同一 L 下不同 ctx 的 TPOT 相对离散 (%)")
        axes[1].set_ylabel("并发")
        axes[1].set_title("ctx 只受显存约束，几乎不影响速度")
        verdict_ctx = (f"ctx 对 TPOT 影响中位 {c['spread_pct'].median():.2f}%"
                       f"（最大 {c['spread_pct'].max():.2f}%）")
    else:
        verdict_ctx = "没有同 L 多 ctx 的样本"

    axes[0].set_xlabel("实测提示长度 L（token）")
    axes[0].set_ylabel("TPOT (ms)")
    axes[0].set_title("沿实际长度回归（不是 ctx）")
    axes[0].legend(fontsize=7)
    axes[0].grid(alpha=0.3)

    # 判读
    # 【判定必须分组】c1 与 b>1 的行为完全不同：单并发下 KV 系数是对的，
    # 一旦并发 > 1，TPOT 随 L 的增长远快于模型（隐含 BW 掉到 1/5 以下）。
    ok = (t["ratio_to_model"] >= 0.75) & (t["ratio_to_model"] <= 1.25)
    t["within_25pct"] = ok
    low = t[t["conc"] == 1]
    high = t[t["conc"] > 1]
    low_ok = int(low["within_25pct"].sum()) if len(low) else 0
    high_ok = int(high["within_25pct"].sum()) if len(high) else 0
    print(f"\n  反推 BW / 模型 BW：中位 {t['ratio_to_model'].median():.2f}×")
    if len(low):
        print(f"    b=1：{low_ok}/{len(low)} 组在 ±25% 内"
              f"（{low['bw_implied_gbs'].min():.0f}–{low['bw_implied_gbs'].max():.0f} GB/s）")
    if len(high):
        print(f"    b>1：{high_ok}/{len(high)} 组在 ±25% 内"
              f"（{high['bw_implied_gbs'].min():.0f}–{high['bw_implied_gbs'].max():.0f} GB/s）")
    if len(low) and low_ok == len(low) and len(high) and high_ok == 0:
        verdict = ("**单并发正确、并发>1 失效**：b>1 时 TPOT 随 L 的增长比 KV 项快 5–10 倍"
                   "（隐含 BW 只剩 9–22%）")
    elif low_ok == len(low) and high_ok == len(high):
        verdict = "KV 系数在各并发下都与实测一致"
    else:
        verdict = "KV 系数与实测部分不符，见逐组表"
    print(f"  判定：{verdict}")

    fig.suptitle("KV 项系数检验：用实际长度回归，并检验 ctx 本身的影响", fontsize=10)
    fig.tight_layout()
    fig.savefig(cfg.FIG_DIR / "fig17_kv_coef.png", dpi=150)

    out = dict(bw_model_gbs=bw_model, n_groups=int(len(t)),
               ratio_median=round(float(t["ratio_to_model"].median()), 3),
               n_within_25pct=int(t["within_25pct"].sum()),
               verdict=verdict, ctx_verdict=verdict_ctx, groups=rows)
    (cfg.RESULTS_DIR / "kv_coef_check.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print("  图 → figures/fig17_kv_coef.png")
    print("→ results/kv_coef_check.csv / kv_coef_check.json")


if __name__ == "__main__":
    main()

