# -*- coding: utf-8 -*-
"""
tc_regret_controls.py —— 后悔值的负对照：这个指标到底能不能分辨"会选"和"乱选"？

## 为什么必须做

报告里现在的后悔值几乎是 0（中位 0.0%）。但"0"可能有两种含义：
  ① 我的模型排序很准 → 值得说；
  ② 候选集太粗、怎么选都对 → 这个指标没有分辨力，说了等于没说。

**只有加负对照才能区分**：让"乱选"的后悔值涨起来，说明指标有分辨力；
如果乱选也是 0，那这个指标就是坏的，应当从卖点里删掉。

## 四组对照

| 选择策略 | 含义 |
|---|---|
| `model`   | 六参数模型按预测成本选（现役） |
| `random`  | 在合规候选里随机选（下界基准） |
| `largest` | 选最大的模型（"越大越好"的直觉） |
| `perturb` | 模型参数各乘 exp(±0.5σ) 后重选（模拟模型误差） |

`perturb` 用 fit.json 里的 sigma_log 做扰动幅度。

用法：
    python tc_regret_controls.py
"""

import json
import random
import sys

import numpy as np
import pandas as pd

import tc_common as tc
import tc_config as cfg
from tc_model import predict
from tc_recommend import simulate

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

SCENARIOS = [
    ("实时对话-严",   300,  25,  256),
    ("实时对话-中",   500,  40,  256),
    ("实时对话-松",   800,  60,  256),
    ("通用问答-中",  1000,  60,  512),
    ("通用问答-松",  2000, 120,  512),
    ("长文档-严",    1500,  60, 1024),
    ("长文档-松",    3000, 120, 1024),
    ("批处理-中",    2500,  80,  256),
    ("批处理-松",    4000, 120,  256),
    ("高并发-中",    3000, 100,  512),
    ("高并发-松",    5000, 150,  512),
    ("宽松-高吞吐",  5000, 200,  256),
]
TIERS = [6, 8, 10, 15]
N_RANDOM = 40


def load_meas():
    """每格实测稳态成本（均值 + 每轮列表）。"""
    per = {}
    with (cfg.DATA_DIR / "raw_requests.jsonl").open(encoding="utf-8") as f:
        acc = {}
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not r.get("ok") or not r.get("total_ms"):
                continue
            acc.setdefault((r["cell_id"], r["repeat"]), []).append(float(r["total_ms"]))
            acc.setdefault(("conc", r["cell_id"]), [])
            if not isinstance(acc[("conc", r["cell_id"])], int):
                acc[("conc", r["cell_id"])] = int(r["conc"])
    reps = {}
    for (cid, rep), tot in acc.items():
        if cid == "conc":
            continue
        conc = acc.get(("conc", cid), 1)
        mean_ms = sum(tot) / len(tot)
        cost = 1e6 / (conc * cfg.GEN_TOKENS / (mean_ms / 1e3))
        reps.setdefault(cid, []).append(cost)
    return reps, {k: float(np.mean(v)) for k, v in reps.items()}


def main():
    summ = pd.read_csv(cfg.DATA_DIR / "summary_steady.csv")
    fit = json.loads((cfg.RESULTS_DIR / "fit.json").read_text(encoding="utf-8"))
    params = dict(bw_gbs=fit["bw"], peak_tflops=fit["peak"],
                  peak_prefill_tflops=fit.get("peak_prefill", fit["peak"]),
                  c_srv_ms=fit["c_srv"], b_sat=fit["b_sat"], ovh_ttft_ms=fit["ovh_ttft"])
    sigma = float(fit.get("sigma_log") or 0.1)
    # 同上：与推荐器共用同一个半宽实现（第四轮复核 5.1-1）
    from tc_common import half_log as _common_half_log
    half_log = _common_half_log(fit)

    qdf = pd.read_csv(cfg.DATA_DIR / "quality.csv").dropna(subset=["acc"])
    lo = {(r["model"], r["quant"]): float(r["acc_lo"]) for _, r in qdf.iterrows()
          if "acc_lo" in qdf.columns and r["acc_lo"] == r["acc_lo"]}
    best_lo = max(lo.values())

    reps, med = load_meas()
    rng = random.Random(20261102)
    rows = []

    print("=" * 96)
    print("后悔值负对照：模型 / 随机 / 最大模型 / 参数扰动")
    print("=" * 96)
    print(f"{'档位':>5}{'场景':<12}{'候选':>5}{'model':>9}{'random':>9}{'largest':>9}{'perturb':>9}")

    for tier in TIERS:
        th = best_lo - tier / 100.0
        elig = {k for k, v in lo.items() if v >= th}
        for name, ttft_cap, tpot_cap, pmean in SCENARIOS:
            # 【唯一实现】候选集交给 tc_common.rank()，与推荐器/验算/图 6 同一路径。
            # 第五轮复核的核心教训：每处自己写一遍过滤，就会给出不同答案。
            _pick, cdf = tc.rank(params, fit, quality_drop_pp=tier,
                                 prompt_mean=pmean, prompt_sigma=0.5,
                                 ttft_cap=ttft_cap, tpot_cap=tpot_cap)
            ok_df = cdf[cdf["feasible"] & cdf["ctx_fits"]
                        & cdf["sla_ok"] & cdf["meets_quality"]]
            cands = []
            for _, r in ok_df.iterrows():
                if r["cell_id"] not in med:
                    continue
                cands.append(dict(cell_id=r["cell_id"], model=r["model"],
                                  pred=float(r["cost"]), meas=med[r["cell_id"]]))
            if len(cands) < 2:
                continue
            oracle = min(c["meas"] for c in cands)

            def regret_of(c):
                return max(0.0, (c["meas"] / oracle - 1) * 100)

            pick_model = min(cands, key=lambda c: c["pred"])
            r_model = regret_of(pick_model)

            r_rand = np.mean([regret_of(rng.choice(cands)) for _ in range(N_RANDOM)])

            order = {"0.5B": 0, "1.5B": 1, "3B": 2, "7B": 3}
            pick_large = max(cands, key=lambda c: order.get(c["model"], 0))
            r_large = regret_of(pick_large)

            pert = []
            for _ in range(20):
                p2 = dict(params,
                          bw_gbs=params["bw_gbs"] * np.exp(rng.gauss(0, 0.5 * sigma)),
                          c_srv_ms=params["c_srv_ms"] * np.exp(rng.gauss(0, 0.5 * sigma)),
                          ovh_ttft_ms=params["ovh_ttft_ms"] * np.exp(rng.gauss(0, 0.5 * sigma)))
                best, bestc = None, None
                for c in cands:
                    sim = simulate(p2, c["model"], c["model"] and
                                   summ.loc[summ["cell_id"] == c["cell_id"], "quant"].iloc[0],
                                   int(summ.loc[summ["cell_id"] == c["cell_id"], "conc"].iloc[0]),
                                   pmean, 0.5,
                                   ctx=int(summ.loc[summ["cell_id"] == c["cell_id"], "ctx"].iloc[0]))
                    if best is None or sim["cost_gpu_s_per_1m"] < best:
                        best, bestc = sim["cost_gpu_s_per_1m"], c
                pert.append(regret_of(bestc))
            r_pert = float(np.mean(pert))

            print(f"{tier:>5}{name:<12}{len(cands):>5}{r_model:>8.1f}%{r_rand:>8.1f}%"
                  f"{r_large:>8.1f}%{r_pert:>8.1f}%")
            rows.append(dict(tier=tier, scenario=name, n_cands=len(cands),
                             regret_model=r_model, regret_random=r_rand,
                             regret_largest=r_large, regret_perturb=r_pert))

    d = pd.DataFrame(rows)
    print("\n" + "=" * 96)
    print("汇总（%）")
    print("=" * 96)
    print(f"{'策略':<10}{'中位':>9}{'均值':>9}{'最大':>9}")
    for col, label in (("regret_model", "model"), ("regret_random", "random"),
                       ("regret_largest", "largest"), ("regret_perturb", "perturb")):
        v = d[col]
        print(f"{label:<10}{v.median():>8.1f}%{v.mean():>8.1f}%{v.max():>8.1f}%")

    spread = d["regret_random"].mean() - d["regret_model"].mean()
    print(f"\n  随机选 比 模型选 平均多付 {spread:.1f} 个百分点。")
    if spread > 5:
        print("  判读：**指标有分辨力**——乱选确实更贵，后悔值不是一个恒为 0 的空指标。")
    else:
        print("  判读：**指标几乎没有分辨力**——乱选也不贵，说明候选集太粗，")
        print("        后悔值不该作为卖点，应如实标注或删除。")

    out = cfg.RESULTS_DIR / "regret_controls.csv"
    d.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"\n已写入 {out}")


if __name__ == "__main__":
    main()

