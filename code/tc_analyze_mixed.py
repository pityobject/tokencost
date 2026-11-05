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
    # 【匹配规则】带指纹的 `_mixXXXX` 才是有效格；
    # 早期没有指纹的 `_mix` 是 id 冲突事故留下的（见 失败与修复记录 F13），排除。
    def _is_mix(cid):
        i = cid.find("_mix")
        return i >= 0 and len(cid) > i + 4

    mix = [c for c in cells if _is_mix(str(c.get("cell_id", ""))) and c.get("status") == "ok"]
    legacy = [c for c in cells if str(c.get("cell_id", "")).endswith("_mix") and c.get("status") == "ok"]
    if legacy:
        print(f"（跳过 {len({c['cell_id'] for c in legacy})} 个旧的冲突格："
              f"无指纹的 `_mix`，见 F13）\n")
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

        # 【口径 B】逐请求预测再取中位：检验"模型是否只在均值近似上出错"
        per_req = []
        for Li in L:
            sp = dict(spec)
            sp["prompt_tokens"] = float(Li)
            sp["ctx_avg_tokens"] = float(Li) + cfg.GEN_TOKENS / 2
            pr_i = tm.predict(params, sp)
            if np.isfinite(pr_i["pred_ttft_ms"]):
                per_req.append(pr_i["pred_ttft_ms"])
        ttft_pred_perreq = float(np.median(per_req)) if per_req else float("nan")

        rows.append(dict(
            cell_id=cid, n_requests=len(rr),
            L_mean=round(Lmean, 1), L_p10=round(float(np.percentile(L, 10)), 1),
            L_p90=round(float(np.percentile(L, 90)), 1),
            L_cv=round(float(L.std() / L.mean()) if L.mean() else np.nan, 3),
            ttft_meas_p50=round(float(np.median(ttft)), 1),
            ttft_meas_p90=round(float(np.percentile(ttft, 90)), 1),
            ttft_pred=round(float(pr["pred_ttft_ms"]), 1),
            ttft_pred_perreq=round(ttft_pred_perreq, 1),
            ttft_err_perreq_pct=round((ttft_pred_perreq / float(np.median(ttft)) - 1) * 100, 1),
            ttft_err_pct=round((pr["pred_ttft_ms"] / float(np.median(ttft)) - 1) * 100, 1),
            tpot_meas=round(float(c["tpot_p50_ms"]), 2),
            tpot_pred=round(float(pr["pred_tpot_ms"]), 2),
            tpot_err_pct=round((pr["pred_tpot_ms"] / c["tpot_p50_ms"] - 1) * 100, 1),
        ))

    if not rows:
        print("_mix 格还没有足够的单请求数据，先重跑（需 --save-requests，默认开启）")
        return

    # 【固定长度参照】找均值最接近的 `_pNNN` 格（同一模型/ctx/并发），
    # 这是本实验的核心对照：均值相同、只有"是否混合"不同。
    ref = []
    fixed = [c for c in cells if str(c.get("cell_id")).find("_p") > 0
             and str(c.get("cell_id")).split("_p")[-1].isdigit()
             and c.get("status") == "ok"]
    for c in fixed:
        cid = c["cell_id"]
        base = cid.split("_p")[0]
        for r in rows:
            if not r["cell_id"].startswith(base):
                continue
            try:
                Lfix = float(c.get("prompt_tokens_median") or 0)
            except Exception:
                continue
            if Lfix and abs(Lfix / r["L_mean"] - 1) <= 0.06:
                ref.append(dict(mix_cell=r["cell_id"], fixed_cell=cid,
                                L_fixed=round(Lfix, 0), L_mix=r["L_mean"],
                                ttft_fixed=round(float(c["ttft_p50_ms"]), 1),
                                ttft_mix=r["ttft_meas_p50"],
                                tpot_fixed=round(float(c["tpot_p50_ms"]), 2),
                                tpot_mix=r["tpot_meas"]))

    t = pd.DataFrame(rows).sort_values(["L_cv"], ascending=False)
    t.to_csv(cfg.RESULTS_DIR / "mixed_lengths.csv", index=False, encoding="utf-8")

    print(f"混合长度格：{len(t)} 个\n")
    print(f"  {'格':<32}{'L均值':>7}{'L CV':>6}{'实测TTFT':>9}"
          f"{'均值口径':>9}{'误差':>7}{'逐请求':>9}{'误差':>7}")
    for _, r in t.iterrows():
        print(f"  {r['cell_id']:<32}{r['L_mean']:>7.0f}{r['L_cv']:>6.2f}"
              f"{r['ttft_meas_p50']:>9.0f}{r['ttft_pred']:>9.0f}{r['ttft_err_pct']:>6.0f}%"
              f"{r['ttft_pred_perreq']:>9.0f}{r['ttft_err_perreq_pct']:>6.0f}%")

    if ref:
        print("\n=== 核心对照：均值相同，只差「是否混合」 ===")
        print(f"  {'混合格':<34}{'L(混/固)':>12}{'TTFT(混/固)':>16}{'TPOT(混/固)':>16}")
        for r0 in ref:
            print(f"  {r0['mix_cell']:<34}"
                  f"{r0['L_mix']:>6.0f}/{r0['L_fixed']:<5.0f}"
                  f"{r0['ttft_mix']:>8.0f}/{r0['ttft_fixed']:<7.0f}"
                  f"{r0['tpot_mix']:>8.2f}/{r0['tpot_fixed']:<7.2f}")
        d_t = [(r0["ttft_mix"] / r0["ttft_fixed"] - 1) * 100 for r0 in ref]
        d_p = [(r0["tpot_mix"] / r0["tpot_fixed"] - 1) * 100 for r0 in ref]
        print(f"\n  混合 vs 固定：TTFT {sum(d_t)/len(d_t):+.1f}% · TPOT {sum(d_p)/len(d_p):+.1f}%")

    summ = dict(n_cells=int(len(t)),
                ttft_mape_mean_approx=round(float(t["ttft_err_pct"].abs().mean()), 2),
                ttft_mape_per_request=round(float(t["ttft_err_perreq_pct"].abs().mean()), 2),
                tpot_mape=round(float(t["tpot_err_pct"].abs().mean()), 2),
                max_L_cv=round(float(t["L_cv"].max()), 3),
                n_ref_pairs=len(ref),
                note=("模型的『平均长度近似』在混合负载上的误差；"
                      "L_cv 越大说明混合越极端"))
    (cfg.RESULTS_DIR / "mixed_lengths.json").write_text(
        json.dumps(summ, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n  TTFT MAPE：均值口径 {summ['ttft_mape_mean_approx']:.1f}%"
          f" vs 逐请求口径 {summ['ttft_mape_per_request']:.1f}%")
    print(f"  TPOT MAPE {summ['tpot_mape']:.1f}%"
          f" · 最大长度 CV {summ['max_L_cv']:.2f}")
    print("→ results/mixed_lengths.csv / mixed_lengths.json")


if __name__ == "__main__":
    main()
