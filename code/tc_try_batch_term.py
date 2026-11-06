# -*- coding: utf-8 -*-
r"""P0-1 修正实验：把失效区放进标定集，再看第 7 参数 Q 是否可辨识。

第一版失败的原因不是模型形式，而是**设计**：
- 标定集 12 格全是 L≈297、并发 1/4/16/32 → 对"长 prompt × 高并发"零信息，
  于是 Q 撞到上界（= 不要修正），结论其实是"数据不足以判定"。
- 诊断指标还 bug 了（探针格被 core 口径过滤掉 → nan）。

这一版：
1. 标定集 = 设计 12 格 + 失效区 6 格（3B 的 c4/c8 长 prompt 探针 + 两种混合长度）；
2. 只搜 Q（其余 6 个固定为现有拟合值），避免高维过拟合；
3. 评价集分两类：**同族留出**（别的长度探针 / 另一种混合）与 **core 留出**；
4. 判据先定：Q 落在网格内部（不撞界）且失效区误差改善 >20%，同时 core MAPE 不劣化 >0.3pp。
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


def predict7(params, row):
    Q = float(params.get("q_batch", 0.0) or 0.0)
    if Q <= 0:
        return tm.predict(params, row)
    spec = tm._spec(row)
    kv = cfg.kv_bytes_per_token(spec)
    b = float(row["conc"])
    L = float(row["prompt_tokens"])

    w_eff = spec["weight_bytes"] + kv * row["ctx_avg_tokens"] * b
    t_mem_dec = w_eff / (params["bw_gbs"] * 1e9)
    t_cmp_dec = 2 * spec["n_params"] * b / (params["peak_tflops"] * 1e12)
    tpot_ms = 1e3 * max(t_mem_dec, t_cmp_dec) + params["c_srv_ms"] * b

    L_eff = L / (1.0 + (b - 1.0) * L / Q)
    b_eff = min(b, params["b_sat"])
    t_mem_pre = spec["weight_bytes"] / (params["bw_gbs"] * 1e9)
    t_cmp_pre = 2 * spec["n_params"] * L_eff * b_eff / (params["peak_prefill_tflops"] * 1e12)
    ttft_ms = 1e3 * max(t_mem_pre, t_cmp_pre) + params["c_srv_ms"] * b + params["ovh_ttft_ms"]

    per_req_s = (ttft_ms + cfg.GEN_TOKENS * tpot_ms) / 1e3
    thr = b * cfg.GEN_TOKENS / per_req_s
    return dict(pred_tpot_ms=tpot_ms, pred_ttft_ms=ttft_ms,
                pred_req_lifetime_ms=per_req_s * 1e3,
                pred_throughput_tps=thr, pred_cost_gpu_s_per_1m=1e6 / thr)


def load_all():
    """返回 (行列表, 索引)：每行含 model/quant/ctx/conc/L/实测 ttft/tpot/cost/族标签。"""
    rows = []
    raw = [json.loads(l) for l in (cfg.DATA_DIR / "raw_cells.jsonl").open(encoding="utf-8")
           if l.strip()]
    s = pd.read_csv(cfg.DATA_DIR / "summary_steady.csv")
    if "steady_cost_gpu_s_per_1m" in s.columns:
        s["cost_gpu_s_per_1m"] = s["steady_cost_gpu_s_per_1m"]
    med = s.set_index("cell_id")
    for r in raw:
        if r.get("status") != "ok":
            continue
        cid = r["cell_id"]
        if cid not in med.index:
            continue
        m = med.loc[cid]
        if float(m.get("conc", -1)) != float(r.get("conc", -2)):
            continue
        L = m.get("prompt_tokens_median")
        if not L or not np.isfinite(float(L)):
            continue
        kind = ("probe" if "_p" in cid else
                "mix" if "_mix" in cid else
                "mt" if "_mt" in cid else
                "design" if not any(x in cid for x in ("_w", "_n", "_xf")) else "probe")
        rows.append(dict(cell_id=cid, kind=kind, model=r["model"], quant=r["quant"],
                         ctx=int(r["ctx"]), conc=int(r["conc"]),
                         prompt_tokens=float(L),
                         ctx_avg_tokens=float(L) + cfg.GEN_TOKENS / 2,
                         ttft=float(m["ttft_p50_ms"]), tpot=float(m["tpot_p50_ms"]),
                         cost=float(m["cost_gpu_s_per_1m"])))
    return pd.DataFrame(rows)


def main():
    base = json.loads((cfg.RESULTS_DIR / "fit.json").read_text(encoding="utf-8"))
    params = dict(bw_gbs=base["bw"], peak_tflops=base["peak"],
                  peak_prefill_tflops=base.get("peak_prefill", base["peak"]),
                  c_srv_ms=base["c_srv"], b_sat=base["b_sat"],
                  ovh_ttft_ms=base["ovh_ttft"], q_batch=0.0)
    df = load_all()

    design_ids = {cfg.cell_id(*c) for c in cfg.CALIB_CELLS}
    # 失效区：3B 的 c4/c8 长 prompt 探针 + 两种混合长度
    fail_ids = {cid for cid in df["cell_id"]
                if ("3B" in cid and "ctx8192" in cid and
                    (("_c4_p" in cid) or ("_c8_p" in cid)))
                or "_mix" in cid}
    calib_ids = design_ids | fail_ids
    calib = df[df["cell_id"].isin(calib_ids)]

    # 评价：同族留出的长度探针（c1/c2）+ core 留出
    loo_probe = set(df[(df["kind"] == "probe") & ~df["cell_id"].isin(calib_ids)]["cell_id"])
    core_hold = set(df[(df["kind"] == "design") & ~df["cell_id"].isin(design_ids)]["cell_id"])

    print("=" * 74)
    print("P0-1（修正版）：标定集含失效区，只搜 Q")
    print("=" * 74)
    print(f"标定 {len(calib)} 格（设计 12 + 失效区 {len(fail_ids)}）"
          f" | 同族留出探针 {len(loo_probe)} | core 留出 {len(core_hold)}")

    def err(p, ids, metric="ttft"):
        e = []
        for _, r in df[df["cell_id"].isin(ids)].iterrows():
            pr = predict7(p, r)
            e.append(abs(pr["pred_ttft_ms"] / r["ttft"] - 1) * 100)
        return float(np.mean(e)) if e else float("nan")

    def cost_mape(p, ids):
        e = []
        for _, r in df[df["cell_id"].isin(ids)].iterrows():
            pr = predict7(p, r)
            e.append(abs(pr["pred_cost_gpu_s_per_1m"] / r["cost"] - 1) * 100)
        return float(np.mean(e)) if e else float("nan")

    # 在标定集上搜 Q（只看 TTFT，因为 Q 只影响 prefill）
    best = (None, np.inf)
    grid = np.geomspace(300.0, 300000.0, 60)
    for Q in grid:
        p = dict(params); p["q_batch"] = float(Q)
        e = []
        for _, r in calib.iterrows():
            pr = predict7(p, r)
            e.append(abs(pr["pred_ttft_ms"] / r["ttft"] - 1))
        m = float(np.mean(e))
        if m < best[1]:
            best = (float(Q), m)
    Q, _ = best
    p7 = dict(params); p7["q_batch"] = Q

    at_bound = (Q <= grid[0] * 1.05) or (Q >= grid[-1] * 0.95)
    print(f"\n拟合得 Q = {Q:.0f} token（{'⚠️ 撞界 → 不可辨识' if at_bound else '落在网格内部'}）")

    print(f"\n  {'指标':<26}{'基线':>9}{'7参数':>9}{'变化':>9}")
    rows = []
    for label, ids in (("失效区 TTFT（标定内）", fail_ids),
                       ("同族留出探针 TTFT", loo_probe),
                       ("core 留出 成本 MAPE", core_hold)):
        f = cost_mape if "成本" in label else err
        a, b = f(params, ids), f(p7, ids)
        chg = (b / a - 1) * 100 if a else float("nan")
        rows.append(dict(metric=label, base=round(a, 2), q7=round(b, 2),
                         change_pct=round(chg, 1)))
        print(f"  {label:<26}{a:>8.1f}%{b:>8.1f}%{chg:>+8.0f}%")

    # 也看 mixed 单列
    mix_ids = {c for c in df["cell_id"] if "_mix" in c}
    a, b = err(params, mix_ids), err(p7, mix_ids)
    print(f"  {'（其中混合长度 TTFT）':<26}{a:>8.1f}%{b:>8.1f}%{(b/a-1)*100:>+8.0f}%")
    rows.append(dict(metric="混合长度 TTFT", base=round(a, 2), q7=round(b, 2),
                     change_pct=round((b / a - 1) * 100, 1)))

    base_core = cost_mape(params, core_hold)
    q7_core = cost_mape(p7, core_hold)
    ok_mape = q7_core <= base_core + 0.3
    improved = sum(1 for r in rows if "core" not in r["metric"] and r["change_pct"] < -20)
    if at_bound:
        verdict = "❌ 不可辨识：Q 撞到边界（等价于不修正）——数据还不足以定它"
    elif ok_mape and improved >= 2:
        verdict = "✅ 成立"
    elif ok_mape and improved >= 1:
        verdict = "部分成立"
    else:
        verdict = "❌ 不成立"
    print(f"\n  判据：core MAPE 不劣化({'✓' if ok_mape else '✗'}) · 改善项 {improved}")
    print(f"  结论：{verdict}")

    out = dict(q_batch=round(Q, 1), at_bound=bool(at_bound),
               rows=rows, core_base=round(base_core, 3), core_q7=round(q7_core, 3),
               n_improved=improved, verdict=verdict)
    (cfg.RESULTS_DIR / "batch_term_trial.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n→ results/batch_term_trial.json")


if __name__ == "__main__":
    main()
