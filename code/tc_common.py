# -*- coding: utf-8 -*-
"""
tc_common.py —— 决策链路的**唯一实现**（第三轮复核 §6 的治理方案）

## 为什么会有这个文件

三轮复核里出现了三次同一类错误，全都是"两套实现、两套数字"：

| 轮次 | 现象 |
|---|---|
| v1→v2 | 质量轴 140 题说"无差异" vs 400 题说"显著" |
| v2 | 稳态 8.70% vs 墙钟 11.28% 混用 |
| **v3** | fig6 的 1605/7.5pp vs 推荐器的 2182/8pp（半宽差 36.5 倍 + 候选集不同） |

每次的根因都一样：**新脚本为了省事，自己又实现了一遍**。

所以这里把决策链路上所有容易各写一遍的东西集中到一处：
    load_params()        读 fit.json（唯一参数来源）
    half_log()           90% 预测区间半宽（唯一实现）
    quality_table()      准确率与 Wilson 下界（唯一来源）
    quality_threshold()  质量门槛（唯一定义）
    candidates()         候选集（唯一定义）
    evaluate()           单格的预测成本 + SLA 判定（唯一实现）
    rank()               给定约束下选最优（唯一实现）

**验收标准：同一个问题，仓库里只有一个函数能回答它。**

现状（第四轮复核后）：`load_params` / `half_log` / `quality_table` 已被
`tc_recommend`、`tc_threshold_sensitivity`、`tc_verify`、`tc_regret_controls` 共用；
`rank()` 目前由 `tc_recommend --quality-sweep` 与 `tc_threshold_sensitivity` 调用，
`tc_verify` 与 `tc_regret_controls` 因为要遍历自己的场景集合，只复用了前两样。
"""

import json

import numpy as np
import pandas as pd

import tc_config as cfg
from tc_recommend import simulate

Z_90 = 1.2816          # 单侧 90%
RTOL = 0.0


def load_params():
    """从 results/fit.json 读六参数（唯一参数来源）。"""
    fit = json.loads((cfg.RESULTS_DIR / "fit.json").read_text(encoding="utf-8"))
    params = dict(bw_gbs=fit["bw"], peak_tflops=fit["peak"],
                  peak_prefill_tflops=fit.get("peak_prefill", fit["peak"]),
                  c_srv_ms=fit["c_srv"], b_sat=fit["b_sat"], ovh_ttft_ms=fit["ovh_ttft"])
    return params, fit


def half_log(fit):
    """90% 预测区间的对数半宽 —— **唯一实现**。

    优先读 fit.json 里的 half_width_log（由 tc_model 算好，含 t 分布与自由度校正）；
    没有才退回正态近似。**不要**在别处用 0.5*log(1+σ²)——那是 lognormal 的
    均值/中位数换算项，不是区间，两者差过 36 倍。
    """
    if "half_width_log" in fit:
        return float(fit["half_width_log"])
    return Z_90 * float(fit.get("sigma_log", 0.1))


def quality_table():
    """返回 ((model,quant) -> 准确率, (model,quant) -> Wilson 下界)。"""
    p = cfg.DATA_DIR / "quality.csv"
    if not p.exists():
        return {}, {}
    q = pd.read_csv(p).dropna(subset=["acc"])
    acc = {(r["model"], r["quant"]): float(r["acc"]) for _, r in q.iterrows()}
    lo = {(r["model"], r["quant"]): float(r["acc_lo"]) for _, r in q.iterrows()
          if "acc_lo" in q.columns and r["acc_lo"] == r["acc_lo"]}
    return acc, lo


def quality_threshold(lo, drop_pp, basis="lcb"):
    """质量门槛 = 最优组合 − drop_pp。默认按 Wilson 下界（更保守）。"""
    best = max(lo.values())
    return best - drop_pp / 100.0


def candidates():
    """候选集 —— **唯一定义**：设计空间 cfg.all_cells()。

    注意：不要用"实测汇总表"当候选集（那会多出 2/8/12/24/48/64 这些
    不在设计空间里的并发档，导致同一个问题两个脚本给出不同答案）。
    """
    return cfg.all_cells()


def evaluate(params, fit, cell, prompt_mean=256.0, prompt_sigma=0.5,
             ttft_cap=600.0, tpot_cap=60.0):
    """单格评估：预测成本 + 90% 上界 + 是否满足显存/上下文/SLA。"""
    hl = half_log(fit)
    need_ctx = int(prompt_mean * np.exp(1.645 * prompt_sigma) + cfg.GEN_TOKENS)
    sim = simulate(params, cell["model"], cell["quant"], int(cell["conc"]),
                   prompt_mean, prompt_sigma, ctx=int(cell["ctx"]))
    ttft_hi = sim["ttft_cvar90"] * np.exp(hl)
    tpot_hi = sim["tpot_cvar90"] * np.exp(hl)
    return dict(
        cell_id=cfg.cell_id(cell["model"], cell["quant"], cell["ctx"], cell["conc"],
                            cell.get("suffix", "")),
        model=cell["model"], quant=cell["quant"], ctx=int(cell["ctx"]),
        conc=int(cell["conc"]),
        cost=float(sim["cost_gpu_s_per_1m"]),
        ttft_hi=float(ttft_hi), tpot_hi=float(tpot_hi),
        feasible=bool(cell["feasible"]),
        ctx_fits=bool(int(cell["ctx"]) >= need_ctx),
        sla_ok=bool(ttft_hi <= ttft_cap and tpot_hi <= tpot_cap),
    )


def rank(params, fit, quality_drop_pp=6.0, prompt_mean=256.0, prompt_sigma=0.5,
         ttft_cap=600.0, tpot_cap=60.0, basis="lcb", cells=None):
    """给定约束，返回 (选中格, 全部候选 DataFrame)。

    **唯一实现**：推荐器、阈值扫描、后悔值、图 6 全都调它。
    """
    acc, lo = quality_table()
    q = lo if (basis == "lcb" and lo) else acc
    th = quality_threshold(q, quality_drop_pp) if q else None

    rows = []
    for c in (cells if cells is not None else candidates()):
        e = evaluate(params, fit, c, prompt_mean, prompt_sigma, ttft_cap, tpot_cap)
        key = (c["model"], c["quant"])
        e["acc"] = q.get(key, float("nan"))
        e["meets_quality"] = True if th is None else (e["acc"] == e["acc"] and e["acc"] >= th)
        rows.append(e)
    df = pd.DataFrame(rows)
    ok = df[df["feasible"] & df["ctx_fits"] & df["sla_ok"] & df["meets_quality"]]
    pick = ok.sort_values("cost").iloc[0] if len(ok) else None
    return pick, df
