# -*- coding: utf-8 -*-
"""
tc_analyze_prompt_sweep.py —— 用模型验证"TTFT 随 prompt 长度的斜率"

## 为什么需要

原数据集里所有格的 prompt 都是 ~297 token，而模型 TTFT 里有一项
`2·P·L·min(b,B_SAT)/PEAK_pre` 随 L 线性增长，推荐器又在 512/1024/2048 的
prompt 均值上做决策 —— 这条斜率此前**从未被实测验证**。

`tc_run_cell.py --prompt-tokens` 补了这个维度。本脚本做三件事：
  1. 把实测 TTFT 与模型预测并排，算误差；
  2. 拟合"实测 TTFT vs L"的斜率，与模型给出的斜率对比；
  3. 出图（fig8）：点 = 实测，线 = 模型。

用法：
    python tc_analyze_prompt_sweep.py
"""

import json
import re
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import tc_config as cfg
import tc_model as tm

sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def load_sweep():
    """从 raw_cells.jsonl 里取带 _pNNN 后缀的格（每次重复取中位）。"""
    rows = []
    p = cfg.DATA_DIR / "raw_cells.jsonl"
    with p.open(encoding="utf-8") as fh:
        for line in fh:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            cid = str(r.get("cell_id", ""))
            if not re.search(r"_p\d+$", cid):
                continue
            if r.get("status") != "ok":
                continue
            rows.append(r)
    if not rows:
        raise SystemExit("没有 _pNNN 格的数据；先跑 tc_run_cell.py --prompt-tokens")
    df = pd.DataFrame(rows)
    # 每格多轮取中位
    agg = df.groupby("cell_id").agg(
        model=("model", "first"), quant=("quant", "first"),
        ctx=("ctx", "first"), conc=("conc", "first"),
        L=("prompt_tokens_median", "median"),
        ttft=("ttft_p50_ms", "median"),
        tpot=("tpot_p50_ms", "median"),
        total=("total_p50_ms", "median"),
        weight_bytes=("weight_bytes", "first"),
        n_params=("n_params", "first"),
    ).reset_index()
    return agg.sort_values("L")


def main():
    params, fit = tm.load_params() if hasattr(tm, "load_params") else (None, None)
    if params is None:
        fit = json.loads((cfg.RESULTS_DIR / "fit.json").read_text(encoding="utf-8"))
        params = dict(bw_gbs=fit["bw"], peak_tflops=fit["peak"],
                      peak_prefill_tflops=fit.get("peak_prefill", fit["peak"]),
                      c_srv_ms=fit["c_srv"], b_sat=fit["b_sat"],
                      ovh_ttft_ms=fit["ovh_ttft"])

    d = load_sweep()
    print("=" * 84)
    print("TTFT 随 prompt 长度的斜率验证")
    print("=" * 84)

    # 用模型的 predict（需要 prompt_tokens / ctx_avg_tokens 字段）
    preds = []
    for _, r in d.iterrows():
        row = dict(model=r["model"], quant=r["quant"], ctx=int(r["ctx"]),
                   conc=int(r["conc"]), prompt_tokens=float(r["L"]),
                   ctx_avg_tokens=float(r["L"]) + cfg.GEN_TOKENS / 2,
                   tpot_p50_ms=r["tpot"], ttft_p50_ms=r["ttft"])
        pr = tm.predict(params, row)
        preds.append(pr)
    d["pred_ttft"] = [p["pred_ttft_ms"] for p in preds]
    d["pred_tpot"] = [p["pred_tpot_ms"] for p in preds]
    d["err_ttft"] = d["pred_ttft"] / d["ttft"] - 1
    d["err_tpot"] = d["pred_tpot"] / d["tpot"] - 1

    print(f"{'L(实测)':>9}{'TTFT实测':>10}{'TTFT预测':>10}{'误差':>8}"
          f"{'TPOT实测':>10}{'TPOT预测':>10}{'误差':>8}")
    for _, r in d.iterrows():
        print(f"{r['L']:>9.0f}{r['ttft']:>9.1f}ms{r['pred_ttft']:>9.1f}ms"
              f"{r['err_ttft']*100:>7.0f}%{r['tpot']:>9.2f}ms{r['pred_tpot']:>9.2f}ms"
              f"{r['err_tpot']*100:>7.0f}%")

    per_model = {f"{m}-{q}": float(np.mean(np.abs(sub["err_ttft"])) * 100)
                 for (m, q), sub in d.groupby(["model", "quant"])}
    mape_ttft = float(np.mean(np.abs(d["err_ttft"])) * 100)
    max_ttft = float(np.max(np.abs(d["err_ttft"])) * 100)
    print(f"\n  TTFT：MAPE {mape_ttft:.1f}%，最大误差 {max_ttft:.1f}%"
          f"（L 跨度 {d['L'].min():.0f}–{d['L'].max():.0f} token，"
          f"{d['L'].max()/d['L'].min():.1f} 倍）")

    # ---- 斜率对比：**必须按模型分别拟合** ----
    # 不同模型的斜率不同（权重项 W/BW 不同、参数量不同），混在一起拟合没有意义。
    slopes = {}
    print()
    for (m, q), sub in d.groupby(["model", "quant"]):
        sub = sub.sort_values("L")
        if len(sub) >= 2:
            k_meas = float(np.polyfit(sub["L"], sub["ttft"], 1)[0])
            k_pred = float(np.polyfit(sub["L"], sub["pred_ttft"], 1)[0])
            slopes[f"{m}-{q}"] = dict(k_meas=k_meas, k_pred=k_pred,
                                      ratio=k_meas / k_pred, n=len(sub))
            print(f"  {m}-{q:<8} {len(sub)} 点  实测斜率 {k_meas:.4f}  "
                  f"模型斜率 {k_pred:.4f}  ms/token  比值 {k_meas/k_pred:.2f}")
        else:
            print(f"  {m}-{q:<8} 只有 {len(sub)} 点，**不足以拟合斜率**"
                  f"（只能看幅值误差）")
    if slopes:
        print("  说明：斜率 = 2·P·min(b,B_SAT)/PEAK_pre 的等效值，"
              "是 prefill 算力项的直接检验。")
        # ---- 从实测斜率反推 PEAK_pre，看它是不是"硬件常数" ----
        print("\n  从实测斜率反推 prefill 有效算力（b=1，PEAK_pre = 2·P / slope）：")
        peak_implied = {}
        for (m, q), sub in d.groupby(["model", "quant"]):
            if len(sub) < 2:
                continue
            k = float(np.polyfit(sub["L"], sub["ttft"], 1)[0])   # ms/token
            P = float(sub["n_params"].iloc[0])
            peak = 2.0 * P / (k * 1e-3) / 1e12                    # TFLOP/s
            peak_implied[f"{m}-{q}"] = peak
            print(f"    {m}-{q:<10} 参数量 {P/1e9:.2f}B  反推 PEAK_pre = {peak:.1f} TFLOP/s")
        if len(peak_implied) >= 2:
            vals = list(peak_implied.values())
            spread = (max(vals) - min(vals)) / (sum(vals) / len(vals)) * 100
            print(f"    → 两个模型的差 {spread:.0f}%，模型里用的是同一个 "
                  f"{fit.get('peak_prefill', float('nan')):.1f} TFLOP/s")
            print("      结论：PEAK_pre 近似是**硬件属性**（跨模型差 {:.0f}%，同一量级），"
                  "但 n_params 不足以完全刻画架构差异（注意力/MLP 配比不同）。".format(spread))

    # ---- 出图 ----
    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    ax.plot(d["L"], d["pred_ttft"], "-", color="#4C72B0", lw=1.8, label="模型预测")
    ax.plot(d["L"], d["ttft"], "o", color="#C44E52", ms=7, label="实测")
    for _, r in d.iterrows():
        ax.annotate(f"{r['err_ttft']*100:+.0f}%", (r["L"], r["ttft"]),
                    textcoords="offset points", xytext=(6, -12), fontsize=8,
                    color="#666666")
    ax.set_xlabel("提示长度 L（token，实测中位）")
    ax.set_ylabel("TTFT（ms）")
    ax.set_title(f"TTFT 随 prompt 长度的斜率：模型 vs 实测（MAPE {mape_ttft:.1f}%）")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    out = cfg.FIG_DIR / "fig8_prompt_length_slope.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"\n已保存 {out}")

    d.to_csv(cfg.RESULTS_DIR / "prompt_sweep.csv", index=False, encoding="utf-8-sig")
    (cfg.RESULTS_DIR / "prompt_sweep_summary.json").write_text(
        json.dumps(dict(
            n_cells=len(d), mape_ttft=mape_ttft, max_err_ttft=max_ttft,
            L_range=[float(d["L"].min()), float(d["L"].max())],
            slopes=slopes,                  # 只含 ≥2 个长度点的模型
            per_model_mape=per_model,      # 单模型口径的 TTFT MAPE
        ), ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"已写入 {cfg.RESULTS_DIR / 'prompt_sweep_summary.json'}")


if __name__ == "__main__":
    main()



