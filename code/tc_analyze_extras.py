# -*- coding: utf-8 -*-
"""其余三项探针的对照分析：KV 量化、泊松到达、跨模型族。

三项共用一个原则：**不重新拟合任何参数**。
  · KV 量化：同参数，比较"同样并发下吞吐是否回升"
  · 泊松到达：用模型在实测平均并发处的预测，与实测分布比
  · 跨模型族：用 Qwen 上标定的 6 参数直接预测 Llama
"""
import json
import sys

import numpy as np
import pandas as pd

import tc_config as cfg
import tc_model as tm

sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def load_params():
    f = json.loads((cfg.RESULTS_DIR / "fit.json").read_text(encoding="utf-8"))
    return dict(bw_gbs=f["bw"], peak_tflops=f["peak"],
                peak_prefill_tflops=f.get("peak_prefill", f["peak"]),
                c_srv_ms=f["c_srv"], b_sat=f["b_sat"], ovh_ttft_ms=f["ovh_ttft"])


def kv_quant():
    print("=" * 80)
    print("① KV 量化（--cache-type-k/v q8_0）：悬崖被推远了多远")
    print("=" * 80)
    p = cfg.DATA_DIR / "probe_kvquant.csv"
    if not p.exists():
        print("  没跑（跳过）\n")
        return
    kv = pd.read_csv(p)
    # 同一格同时有 f16 与 q8_0 两种 kv_type，按 (cell_id, kv_type) 先聚合再对照
    agg = kv.groupby(["cell_id", "kv_type"], as_index=False).median(numeric_only=True)
    print(f"{'格':<26}{'f16 吞吐':>10}{'q8 吞吐':>10}{'变化':>8}"
          f"{'f16 TPOT':>10}{'q8 TPOT':>9}{'变化':>8}")
    for cid, sub in agg.groupby("cell_id"):
        row = {r["kv_type"]: r for _, r in sub.iterrows()}
        if "f16" not in row or "q8_0" not in row:
            continue
        a, b = row["f16"], row["q8_0"]
        d_thr = b["throughput_tps"] / a["throughput_tps"] - 1
        d_tpot = b["tpot_p50_ms"] / a["tpot_p50_ms"] - 1
        print(f"{cid:<26}{a['throughput_tps']:>10.0f}{b['throughput_tps']:>10.0f}"
              f"{d_thr*100:>7.0f}%{a['tpot_p50_ms']:>10.1f}{b['tpot_p50_ms']:>9.1f}{d_tpot*100:>7.0f}%")
    print("  判读：吞吐大幅回升 = 原本在外溢，KV 减半后装回显存；")
    print("        几乎没变   = 权重本身就占满（或本来就装得下），KV 量化救不回来。\n")


def poisson():
    print("=" * 80)
    print("② 泊松到达：同步波次是不是上界")
    print("=" * 80)
    p = cfg.DATA_DIR / "probe_poisson.csv"
    if not p.exists():
        print("  没跑（跳过）\n")
        return
    ps = pd.read_csv(p)
    params = load_params()
    print(f"{'λ':>5}{'实际并发':>9}{'实测TTFT p50':>13}{'模型预测':>10}{'比值':>7}{'实测TPOT':>10}{'模型预测':>10}{'比值':>7}")
    for _, r in ps.iterrows():
        row = dict(model="3B", quant="q4_k_m", ctx=512, conc=float(r["mean_conc"]),
                   prompt_tokens=float(cfg.PROMPT_TOKENS),
                   ctx_avg_tokens=cfg.PROMPT_TOKENS + cfg.GEN_TOKENS / 2)
        pr = tm.predict(params, row)
        ratio_t = r["ttft_p50"] / pr["pred_ttft_ms"] if r["ttft_p50"] else float("nan")
        ratio_p = r["tpot_p50"] / pr["pred_tpot_ms"] if r["tpot_p50"] else float("nan")
        print(f"{r['lam']:>5.0f}{r['mean_conc']:>9.2f}{r['ttft_p50']:>13.1f}{pr['pred_ttft_ms']:>10.1f}"
              f"{ratio_t:>7.2f}{r['tpot_p50']:>10.1f}{pr['pred_tpot_ms']:>10.1f}{ratio_p:>7.2f}")
    print("  比值 < 1 表示实测比模型预测更快 → 模型偏保守（安全方向）。\n")


def xfamily():
    print("=" * 80)
    print("③ 跨模型族：用 Qwen 标定的参数直接预测 Llama-3.2-1B")
    print("=" * 80)
    p = cfg.DATA_DIR / "summary_steady.csv"
    if not p.exists():
        print("  缺汇总表（跳过）\n")
        return
    df = pd.read_csv(p)
    xf = df[df["cell_id"].astype(str).str.endswith("_xf")]
    if not len(xf):
        print("  没跑（跳过）\n")
        return
    xf = xf.copy()
    xf["prompt_tokens"] = xf.get("prompt_tokens_median", cfg.PROMPT_TOKENS).fillna(cfg.PROMPT_TOKENS)
    xf["ctx_avg_tokens"] = xf["prompt_tokens"] + cfg.GEN_TOKENS / 2
    params = load_params()
    # 【口径统一】主指标是"单位成本 MAPE"，跨族检验也必须用同一口径，
    # 否则会出现"跨族 12.7%"（TPOT/TTFT 平均误差）与"core 8.70%"（成本 MAPE）
    # 混着引用的问题（第三轮复核指出）。
    print(f"{'格':<32}{'实测成本':>10}{'预测成本':>10}{'成本误差':>9}"
          f"{'TPOT误差':>9}{'TTFT误差':>9}")
    cost_err, other_err = [], []
    for _, r in xf.iterrows():
        pr = tm.predict(params, r)
        # 【P1-3】主指标是稳态口径；这里原来读的是墙钟列，导致"与主指标同口径"的说法不成立。
        ec = pr["pred_cost_gpu_s_per_1m"] / r["steady_cost_gpu_s_per_1m"] - 1
        e1 = pr["pred_tpot_ms"] / r["tpot_p50_ms"] - 1
        e2 = pr["pred_ttft_ms"] / r["ttft_p50_ms"] - 1
        cost_err.append(abs(ec))
        other_err += [abs(e1), abs(e2)]
        print(f"{r['cell_id']:<32}{r['cost_gpu_s_per_1m']:>10.0f}"
              f"{pr['pred_cost_gpu_s_per_1m']:>10.0f}{ec*100:>8.0f}%"
              f"{e1*100:>8.0f}%{e2*100:>8.0f}%")
    print(f"\n  成本 MAPE            {np.mean(cost_err)*100:.1f}%（与主指标同口径，零重标定）")
    print(f"  TPOT/TTFT 平均误差   {np.mean(other_err)*100:.1f}%（旧口径，仅作对照）")
    print("  判读：<15% 说明模型刻画的是硬件与部署栈，与模型族无关；>30% 说明词表/注意力实现对 c_srv 有影响。\n")


def main():
    print()
    kv_quant()
    poisson()
    xfamily()


if __name__ == "__main__":
    main()
