# -*- coding: utf-8 -*-
"""
E2 分析：混合请求长度 vs 模型的"平均长度近似"

## 问题

模型的 prefill 项用**单一长度 L**。真实负载是长短混合的。
如果长度对延迟的影响是线性的，用加权平均 L 就该差不多；
但批处理里"长请求占着槽位更久"，混合可能比平均更糟。

## 判据

对每个 `_mix` 格：
1. 从 `raw_requests.jsonl` 取该格**每个请求**的真实 prompt_tokens 分布；
2. 用**加权平均长度**喂模型，得到预测 TTFT / TPOT / 吞吐；
3. 与实测中位数对比。
4. 同时算"均匀长度假设下"的分布宽度（p90−p10），看混合把方差放大了多少。

## 产出

- `results/mixed_lengths.csv`
- `results/mixed_lengths.json`
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
    cell_p = cfg.DATA_DIR / "raw_cells.jsonl"
    req_p = cfg.DATA_DIR / "raw_requests.jsonl"
    if not cell_p.exists():
        print("没有 raw_cells.jsonl")
        return

    cells = [json.loads(l) for l in cell_p.open(encoding="utf-8") if l.strip()]
    mix = [c for c in cells if str(c.get("cell_id", "")).endswith("_mix") and c.get("status") == "ok"]
    if not mix:
        print("还没有混合长度的实验数据。先跑：")
        print("  python tc_run_cell.py 3B q4_k_m 2048 4 --length-mix 256:0.7,1024:0.2,2048:0.1")
        return

    reqs = [json.loads(l) for l in req_p.open(encoding="utf-8") if l.strip()]
    by_cell = {}
    for r in reqs:
        by_cell.setdefault(r["cell_id"], []).append(r)

    fit = json.loads((cfg.RESULTS_DIR / "fit.json").read_text(encoding="utf-8"))
    params = dict(bw_gbs=fit["bw"], peak_tflops=fit["peak"],
                  peak_prefill_tflops=fit.get("peak_prefill", fit["peak"]),
                  c_srv_ms=fit["c_srv"], b_sat=fit["b_sat"], ovh_ttft_ms=fit["ovh_ttft"])

    rows = []
    for c in mix:
        cid = c["cell_id"]
        rr = [r for r in by_cell.get(cid, []) if r.get("ok")]
        if len(rr) < 8:
            continue
        L = np.array([r.get("prompt_tokens") or 0 for r in rr], dtype=float)
        L = L[L > 0]
        if not len(L):
            continue
        ttft = np.array([r["ttft_ms"] for r in rr if r.get("ttft_ms")], dtype=float)
        Lmean = float(L.mean())
        spec = dict(model=c["model"], quant=c["quant"], ctx=int(c["ctx"]),
                    conc=int(c["conc"]), prompt_tokens=Lmean,
                    ctx_avg_tokens=Lmean + cfg.GEN_TOKENS / 2)
        pr = tm.predict(params, spec)
        rows.append(dict(
            cell_id=cid, n_requests=len(rr),
            L_mean=round(Lmean, 1), L_p10=round(float(np.percentile(L, 10)), 1),
            L_p90=round(float(np.percentile(L, 90)), 1),
            L_cv=round(float(L.std() / L.mean()) if L.mean() else np.nan, 3),
            ttft_meas_p50=round(float(np.median(ttft)), 1),
            ttft_meas_p90=round(float(np.percentile(ttft, 90)), 1),
            ttft_pred=round(float(pr["pred_ttft_ms"]), 1),
            ttft_err_pct=round((pr["pred_ttft_ms"] / float(np.median(ttft)) - 1) * 100, 1),
            tpot_meas=round(float(c["tpot_p50_ms"]), 2),
            tpot_pred=round(float(pr["pred_tpot_ms"]), 2),
            tpot_err_pct=round((pr["pred_tpot_ms"] / c["tpot_p50_ms"] - 1) * 100, 1),
        ))

    if not rows:
        print("_mix 格还没有足够的单请求数据，先重跑（需 --save-requests，默认开启）")
        return

    t = pd.DataFrame(rows).sort_values(["L_cv"], ascending=False)
    t.to_csv(cfg.RESULTS_DIR / "mixed_lengths.csv", index=False, encoding="utf-8")

    print(f"混合长度格：{len(t)} 个\n")
    print(f"  {'格':<34}{'L均值':>7}{'L CV':>6}{'TTFT实测':>9}{'预测':>8}{'误差':>7}"
          f"{'TPOT实测':>9}{'预测':>7}{'误差':>7}")
    for _, r in t.iterrows():
        print(f"  {r['cell_id']:<34}{r['L_mean']:>7.0f}{r['L_cv']:>6.2f}"
              f"{r['ttft_meas_p50']:>9.0f}{r['ttft_pred']:>8.0f}{r['ttft_err_pct']:>6.0f}%"
              f"{r['tpot_meas']:>9.2f}{r['tpot_pred']:>7.2f}{r['tpot_err_pct']:>6.0f}%")

    summ = dict(n_cells=int(len(t)),
                ttft_mape=round(float(t["ttft_err_pct"].abs().mean()), 2),
                tpot_mape=round(float(t["tpot_err_pct"].abs().mean()), 2),
                max_L_cv=round(float(t["L_cv"].max()), 3),
                note=("模型的『平均长度近似』在混合负载上的误差；"
                      "L_cv 越大说明混合越极端"))
    (cfg.RESULTS_DIR / "mixed_lengths.json").write_text(
        json.dumps(summ, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n  TTFT MAPE {summ['ttft_mape']:.1f}% · TPOT MAPE {summ['tpot_mape']:.1f}%"
          f" · 最大长度 CV {summ['max_L_cv']:.2f}")
    print("→ results/mixed_lengths.csv / mixed_lengths.json")


if __name__ == "__main__":
    main()
