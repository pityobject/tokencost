# -*- coding: utf-8 -*-
"""tc_verify.py —— 推荐器的决策后悔值（第三轮审计 P1-8 重写版）

## 旧版的三个问题（都来自审计）

1. **只有 4 个场景**，其中 1 个无解、2 个结果完全相同 → 真正独立的最多 2 个；
2. **基线是稻草人**：`3B-Q8/c4` 本身又贵又不是最优；
3. **最优解总落在 SLA 边界**（"满足 SLA 的最大并发"），成本面很平 → 后悔值天然小，
   证明不了模型的排序能力；而且只有 4 个点，**没有置信区间**。

## 新版的四个改动

① **16 个场景**：TTFT 上限 × TPOT 上限 × 提示长度的网格，刻意包含"内点最优"
   （cap 宽松到并发不必顶格）；
② **基线**：默认配置（合规时才计），并把不降反升也如实显示；
③ **bootstrap 区间**：每格有 3 轮实测成本，重采样得到后悔值的 90% 区间；
④ **按质量档位分开算**：质量损失 6/8/10/15/20 pp 各跑一遍——
   质量门槛一收紧，候选集就只剩 7B，后悔值退化为 0，那个 0 没有信息量。

用法：
    python tc_verify.py
    python tc_verify.py --boot 500
"""

import argparse
import json
import random
import sys

import numpy as np
import pandas as pd

import tc_config as cfg
from tc_model import predict          # noqa: F401  （保留导入，便于扩展）
from tc_recommend import simulate

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# (名称, TTFT 上限 ms, TPOT 上限 ms, 提示均值 token)
SCENARIOS = [
    ("实时对话-严",   300,  25,  256),
    ("实时对话-中",   500,  40,  256),
    ("实时对话-松",   800,  60,  256),
    ("通用问答-严",   600,  40,  512),
    ("通用问答-中",  1000,  60,  512),
    ("通用问答-松",  2000, 120,  512),
    ("长文档-严",    1500,  60, 1024),
    ("长文档-松",    3000, 120, 1024),
    ("批处理-严",    1500,  60,  256),
    ("批处理-中",    2500,  80,  256),
    ("批处理-松",    4000, 120,  256),
    ("高并发-严",    2000,  80,  512),
    ("高并发-中",    3000, 100,  512),
    ("高并发-松",    5000, 150,  512),
    ("宽松-低延迟",   900,  30,  256),
    ("宽松-高吞吐",  5000, 200,  256),
]
DEFAULT_BASE = "3B_q8_0_ctx2048_c4"
QUALITY_TIERS = [6, 8, 10, 15, 20]


def load_meas():
    """每格每轮的**稳态成本**（用于 bootstrap）。

    注意：`raw_cells.jsonl` 里存的是墙钟成本，稳态成本是后算出来的，
    所以这里必须从逐请求明细 `raw_requests.jsonl` 重算：
        每轮稳态吞吐 = 并发 × 生成长度 ÷ 该轮所有请求的平均总耗时
        每轮稳态成本 = 1e6 ÷ 稳态吞吐
    直接读 raw_cells 的旧字段会全部拿到 NaN，导致所有场景被静默跳过。
    """
    path = cfg.DATA_DIR / "raw_requests.jsonl"
    per = {}
    acc = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not r.get("ok") or not r.get("total_ms"):
                continue
            key = (r["cell_id"], r["repeat"])
            acc.setdefault(key, []).append(float(r["total_ms"]))
            per.setdefault(r["cell_id"], {"conc": int(r["conc"]), "reps": {}})
            per[r["cell_id"]]["reps"][r["repeat"]] = 0.0      # 占位，下面填

    gen = cfg.GEN_TOKENS
    med = {}
    for (cid, rep), tot in acc.items():
        conc = per[cid]["conc"]
        mean_ms = sum(tot) / len(tot)
        thr = conc * gen / (mean_ms / 1e3)          # 稳态吞吐（与波次无关）
        per[cid]["reps"][rep] = 1e6 / thr
    for cid, d in per.items():
        vals = [v for v in d["reps"].values() if v > 0]
        if vals:
            med[cid] = float(np.mean(vals))
    # 只保留成本列表，供 bootstrap 抽样
    costs = {cid: [v for v in d["reps"].values() if v > 0] for cid, d in per.items()}
    return costs, med


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--boot", type=int, default=500)
    ap.add_argument("--sigma", type=float, default=0.5,
                    help="合成请求长度分布的对数标准差（与推荐器一致）")
    args = ap.parse_args()

    summ = pd.read_csv(cfg.DATA_DIR / "summary_steady.csv")
    fit = json.loads((cfg.RESULTS_DIR / "fit.json").read_text(encoding="utf-8"))
    params = dict(bw_gbs=fit["bw"], peak_tflops=fit["peak"],
                  peak_prefill_tflops=fit.get("peak_prefill", fit["peak"]),
                  c_srv_ms=fit["c_srv"], b_sat=fit["b_sat"], ovh_ttft_ms=fit["ovh_ttft"])
    half_log = 0.5 * np.log(1 + (fit.get("sigma_log") or 0.1) ** 2)

    qp = cfg.DATA_DIR / "quality.csv"
    qdf = pd.read_csv(qp).dropna(subset=["acc"]) if qp.exists() else pd.DataFrame()
    acc = {(r["model"], r["quant"]): float(r["acc"]) for _, r in qdf.iterrows()}
    lo = {(r["model"], r["quant"]): float(r["acc_lo"]) for _, r in qdf.iterrows()
          if "acc_lo" in qdf.columns and r["acc_lo"] == r["acc_lo"]}
    best_lo = max(lo.values()) if lo else (max(acc.values()) if acc else None)

    per, med = load_meas()
    rng = random.Random(20261101)
    all_rows = []

    print("=" * 100)
    print("推荐器决策后悔值（16 场景 × 5 质量档位 × bootstrap 区间）")
    print("=" * 100)

    for tier in QUALITY_TIERS:
        th = (best_lo - tier / 100.0) if best_lo is not None else None
        eligible_q = {k for k, v in lo.items() if v >= th} if th is not None else set(acc)
        print(f"\n【允许质量损失 {tier} pp（Wilson 下界 ≥ {th*100:.1f}%）】")
        print(f"{'场景':<12}{'推荐':<24}{'推荐实测':>9}{'事后最优':<24}{'最优实测':>9}"
              f"{'后悔值':>8}{'90%区间':>16}{'较默认':>9}")
        for name, ttft_cap, tpot_cap, pmean in SCENARIOS:
            need_ctx = int(pmean * np.exp(1.645 * args.sigma) + cfg.GEN_TOKENS)
            cands = []
            for _, r in summ.iterrows():
                if (r["model"], r["quant"]) not in eligible_q:
                    continue
                if int(r["ctx"]) < need_ctx:
                    continue
                sim = simulate(params, r["model"], r["quant"], int(r["conc"]),
                               pmean, args.sigma, ctx=int(r["ctx"]))
                if (sim["ttft_cvar90"] * np.exp(half_log) > ttft_cap or
                        sim["tpot_cvar90"] * np.exp(half_log) > tpot_cap):
                    continue
                cands.append(dict(cell_id=r["cell_id"], pred=sim["cost_gpu_s_per_1m"],
                                  meas=med.get(r["cell_id"], np.nan)))
            if not cands:
                print(f"{name:<12}{'（无可行配置）'}")
                continue
            df = pd.DataFrame(cands).dropna(subset=["meas"])
            if df.empty:
                continue
            pick = df.sort_values("pred").iloc[0]
            oracle = df.sort_values("meas").iloc[0]
            regret = (pick["meas"] / oracle["meas"] - 1) * 100

            # 【配对 bootstrap】旧实现把 pick 与 oracle 的实测成本**独立**重采样，
            # 于是 oracle 可能被抽成比 pick 更贵 → 后悔值出现负值（如 [-8%, +9%]），
            # 而按定义 pick_meas ≥ oracle_meas，下限不可能为负。那种区间不能当 CI 用。
            # 正确做法：每轮抽一个**共同的重复序号**，在重采样后的成本集里**重新选 oracle**，
            # 这样 oracle 永远是重采样集里的最小值，后悔值恒 ≥ 0。
            boots = []
            all_ids = list(df["cell_id"])
            n_rep = max((len(v) for v in per.values()), default=0)
            for _ in range(args.boot):
                k = rng.randrange(max(n_rep, 1))
                def cost_at(cid):
                    v = per.get(cid, [])
                    return v[k % len(v)] if v else np.nan
                om = min(cost_at(c) for c in all_ids if cost_at(c) == cost_at(c))
                pm = cost_at(pick["cell_id"])
                if pm == pm and om == om and om > 0:
                    boots.append(max(0.0, (pm / om - 1) * 100))
            lo_ci, hi_ci = (np.percentile(boots, [5, 95]) if len(boots) > 20 else (np.nan, np.nan))

            base = df[df["cell_id"] == DEFAULT_BASE]
            drop = ((1 - pick["meas"] / base.iloc[0]["meas"]) * 100 if len(base) else np.nan)

            print(f"{name:<12}{pick['cell_id']:<24}{pick['meas']:>9.0f}"
                  f"{oracle['cell_id']:<24}{oracle['meas']:>9.0f}"
                  f"{regret:>7.1f}%  [{lo_ci:>5.1f}%, {hi_ci:>5.1f}%]"
                  f"{('—' if drop != drop else f'{drop:>8.1f}%')}")

            all_rows.append(dict(tier=tier, scenario=name, ttft_cap=ttft_cap,
                                 tpot_cap=tpot_cap, prompt_mean=pmean,
                                 pick=pick["cell_id"], pick_meas=pick["meas"],
                                 oracle=oracle["cell_id"], oracle_meas=oracle["meas"],
                                 regret_pct=regret, regret_lo=lo_ci, regret_hi=hi_ci,
                                 default_drop_pct=drop, n_cands=len(df)))

    out = cfg.RESULTS_DIR / "verify_recommendation.csv"
    d = pd.DataFrame(all_rows)
    d.to_csv(out, index=False, encoding="utf-8-sig")

    if len(d):
        print("\n" + "=" * 100)
        print("汇总")
        print("=" * 100)
        print(f"  有解的场景 {d['scenario'].nunique()} / {len(SCENARIOS)}；"
              f"共 {len(d)} 条（场景 × 质量档位）")
        print(f"  后悔值：中位 {d['regret_pct'].median():.1f}%，"
              f"均值 {d['regret_pct'].mean():.1f}%，最大 {d['regret_pct'].max():.1f}%")
        w = (d["regret_hi"] - d["regret_lo"]).dropna()
        print(f"  后悔值 90% 区间宽度：中位 {w.median():.1f} 个百分点")
        dd = d["default_drop_pct"].dropna()
        print(f"  较默认配置降幅：中位 {dd.median():.1f}%（{len(dd)} 条基线合规）")
        print("\n  判读：后悔值小不代表模型完美，只代表这套候选集的**成本面比较平**。")
        print("        真正说明排序能力的是：区间宽度 + 跨 16 个场景的稳定性。")
    print(f"\n已写入 {out}")


if __name__ == "__main__":
    main()
