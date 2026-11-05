# -*- coding: utf-8 -*-
r"""E3 分析：多轮对话下 TPOT 随轮数的增长 vs 模型预测

## 要回答什么

模型的 TPOT 里，KV 项是 `kv_per_token × ctx_avg × 并发`——**随上下文线性增长**。
多轮对话正好是"上下文逐步变长"的自然实验：
如果实测 TPOT 的增长率与模型一致，说明 KV 项的系数是对的；
如果不一致，说明模型低估/高估了"长上下文对解码的拖累"。

## 判据

对每一轮：`ctx_avg = 前缀长度 + 生成长度/2`，用模型预测 TPOT，与实测比。
再对"轮次 → TPOT"做线性回归，得到实测的每轮增长率（%）。

## 产出

- `results/multiturn.csv` / `multiturn.json`
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import tc_config as cfg
import tc_model as tm

sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def main():
    rows = [json.loads(l) for l in (cfg.DATA_DIR / "raw_cells.jsonl").open(encoding="utf-8")
            if l.strip()]
    mt = [r for r in rows if "_mt" in str(r.get("cell_id", "")) and r.get("status") == "ok"]
    if not mt:
        print("还没有多轮数据。先跑：")
        print("  python tc_run_cell.py 3B q4_k_m 4096 1 --turns 6")
        return

    fit = json.loads((cfg.RESULTS_DIR / "fit.json").read_text(encoding="utf-8"))
    params = dict(bw_gbs=fit["bw"], peak_tflops=fit["peak"],
                  peak_prefill_tflops=fit.get("peak_prefill", fit["peak"]),
                  c_srv_ms=fit["c_srv"], b_sat=fit["b_sat"], ovh_ttft_ms=fit["ovh_ttft"])

    out = []
    for cid, g in pd.DataFrame(mt).groupby("cell_id"):
        g = g.sort_values("repeat")
        g = g.groupby("repeat", as_index=False).last()
        for _, r in g.iterrows():
            pre = float(r.get("prompt_tokens_median") or 0)
            if pre <= 0:
                continue
            spec = dict(model=r["model"], quant=r["quant"], ctx=int(r["ctx"]),
                        conc=int(r["conc"]), prompt_tokens=pre,
                        ctx_avg_tokens=pre + cfg.GEN_TOKENS / 2)
            pr = tm.predict(params, spec)
            out.append(dict(cell_id=cid, turn=int(r["repeat"]),
                            prefix_tokens=pre,
                            ttft_meas=float(r["ttft_p50_ms"]),
                            ttft_pred=float(pr["pred_ttft_ms"]),
                            tpot_meas=float(r["tpot_p50_ms"]),
                            tpot_pred=float(pr["pred_tpot_ms"])))
    if not out:
        print("多轮数据缺少前缀长度，无法分析")
        return

    t = pd.DataFrame(out).sort_values(["cell_id", "turn"])
    t["tpot_err_pct"] = (t["tpot_pred"] / t["tpot_meas"] - 1) * 100
    t["ttft_err_pct"] = (t["ttft_pred"] / t["ttft_meas"] - 1) * 100
    t.to_csv(cfg.RESULTS_DIR / "multiturn.csv", index=False, encoding="utf-8")

    print(f"多轮格 {t['cell_id'].nunique()} 个，共 {len(t)} 轮观测\n")
    for cid, g in t.groupby("cell_id"):
        print(f"  {cid}")
        print(f"    {'轮':>3}{'前缀':>7}{'TTFT实测':>10}{'预测':>8}{'TPOT实测':>10}{'预测':>8}{'误差':>7}")
        for _, r in g.iterrows():
            print(f"    {r['turn']:>3}{r['prefix_tokens']:>7.0f}{r['ttft_meas']:>10.0f}"
                  f"{r['ttft_pred']:>8.0f}{r['tpot_meas']:>10.2f}{r['tpot_pred']:>8.2f}"
                  f"{r['tpot_err_pct']:>6.0f}%")
        if len(g) >= 2:
            slope = np.polyfit(g["turn"], g["tpot_meas"], 1)[0]
            growth = slope / float(g["tpot_meas"].mean()) * 100
            print(f"    → 实测每轮 TPOT 增长 {growth:+.2f}%")

    summ = dict(n_cells=int(t["cell_id"].nunique()), n_turns=int(len(t)),
                tpot_mape=round(float(t["tpot_err_pct"].abs().mean()), 2),
                ttft_mape=round(float(t["ttft_err_pct"].abs().mean()), 2),
                note="多轮下 TPOT 随前缀增长；模型用 KV 线性项近似")
    (cfg.RESULTS_DIR / "multiturn.json").write_text(
        json.dumps(summ, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n  TPOT MAPE {summ['tpot_mape']:.1f}% · TTFT MAPE {summ['ttft_mape']:.1f}%")
    print("→ results/multiturn.csv / multiturn.json")


if __name__ == "__main__":
    main()
