# -*- coding: utf-8 -*-
"""
决策鲁棒性（`tc.py robust`）：参数不确定时，推荐会不会变？

## 为什么

我们报了参数的 5–95% 区间（bootstrap），但**从没检验它对决策的影响**。
"后悔值 0.0%"是**点估计参数**下算的；真实的说法应该是：
"在参数区间内，推荐有 X% 的情况下不变"。

## 做法

1. 从 `results/CLAIMS.md` 的参数区间解析出各参数的上下界；
2. 在区间内抽 N 组参数（对数均匀，符合"参数是正量、误差是比例"的事实）；
3. 对每组参数，重算每个候选配置的预测 TTFT/TPOT/成本；
4. 在每个 (场景, 质量档) 下选"满足 SLA 的最便宜配置"；
5. 统计**首选配置的稳定率**（众数占比）与"落在同族栈内"的比例。

## 用法

    python code/tc_decision_robust.py --draws 200
"""
import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import tc_config as cfg
import tc_model as tm
from tc_regret_controls import SCENARIOS, TIERS

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PARAM_KEYS = ["bw", "peak", "peak_prefill", "c_srv", "b_sat", "ovh"]


def parse_intervals():
    """从 CLAIMS.md 的 bootstrap 表解析 5–95% 区间。"""
    fp = cfg.RESULTS_DIR / "CLAIMS.md"
    txt = fp.read_text(encoding="utf-8")
    out = {}
    m = re.search(r"\|\s*带宽 GB/s\s*\|\s*([\d.]+)\s*\|\s*\[([\d.]+),\s*([\d.]+)\]", txt)
    if m:
        out["bw"] = (float(m.group(2)), float(m.group(3)))
    m = re.search(r"\|\s*decode 算力 TFLOP/s\s*\|\s*([\d.]+)\s*\|\s*\[([\d.]+),\s*([\d.]+)\]", txt)
    if m:
        out["peak"] = (float(m.group(2)), float(m.group(3)))
    m = re.search(r"\|\s*prefill 算力 TFLOP/s\s*\|\s*([\d.]+)\s*\|\s*\[([\d.]+),\s*([\d.]+)\]", txt)
    if m:
        out["peak_prefill"] = (float(m.group(2)), float(m.group(3)))
    m = re.search(r"\|\s*c_srv ms/流/步\s*\|\s*([\d.]+)\s*\|\s*\[([\d.]+),\s*([\d.]+)\]", txt)
    if m:
        out["c_srv"] = (float(m.group(2)), float(m.group(3)))
    m = re.search(r"\|\s*B_SAT ms\s*\|\s*([\d.]+)\s*\|\s*\[([\d.]+),\s*([\d.]+)\]", txt)
    if m:
        out["b_sat"] = (float(m.group(2)), float(m.group(3)))
    m = re.search(r"\|\s*ovh ms\s*\|\s*([\d.]+)\s*\|\s*\[([\d.]+),\s*([\d.]+)\]", txt)
    if m:
        out["ovh"] = (float(m.group(2)), float(m.group(3)))
    return out


def load_candidates():
    """候选 = 设计空间里实测过的格；用它确保成本口径与项目一致。"""
    df = cfg.load_population("core").reset_index(drop=True)
    df = df[(df["prompt_tokens"] > 0) & (df["tpot_p50_ms"] > 0)]
    q = pd.read_csv(Path(cfg.__file__).resolve().parent.parent / "data" / "quality.csv")
    acc = {(r["model"], r["quant"]): float(r["acc"]) for _, r in q.iterrows()}
    cands = []
    for _, r in df.iterrows():
        a = acc.get((r["model"], r["quant"]))
        if a is None:
            continue
        cands.append(dict(label=f"{r['model']}-{r['quant']}@{r['conc']}",
                          model=r["model"], quant=r["quant"], conc=int(r["conc"]),
                          ctx=int(r["ctx"]), prompt=float(r["prompt_tokens"]),
                          acc=a))
    return cands


def predict_all(draw, cands, gen=128):
    # tc_model.predict 用的是关键字名，抽样用的是短名，这里做一次映射
    params = dict(bw_gbs=draw["bw"], peak_tflops=draw["peak"],
                  peak_prefill_tflops=draw["peak_prefill"],
                  c_srv_ms=draw["c_srv"], b_sat=draw["b_sat"],
                  ovh_ttft_ms=draw["ovh"])
    out = []
    for c in cands:
        row = dict(model=c["model"], quant=c["quant"], ctx=c["ctx"], conc=c["conc"],
                   prompt_tokens=c["prompt"], ctx_avg_tokens=c["prompt"] + gen / 2)
        pr = tm.predict(params, row)
        ttft, tpot = pr["pred_ttft_ms"], pr["pred_tpot_ms"]
        life = ttft + gen * tpot
        thr = c["conc"] * gen / life * 1e3          # tok/s
        cost = 1e6 / thr                            # 每 1e6 token 的 GPU 秒
        out.append((c, ttft, tpot, cost))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--draws", type=int, default=200)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    iv = parse_intervals()
    miss = [k for k in PARAM_KEYS if k not in iv]
    if miss:
        print("缺参数区间：", miss)
        return
    cands = load_candidates()
    print("=" * 88)
    print(f"决策鲁棒性：{args.draws} 组参数 × {len(cands)} 个候选 × "
          f"{len(SCENARIOS)} 场景 × {len(TIERS)} 质量档")
    print("=" * 88)
    print("  参数区间（5–95%）：" + "｜".join(f"{k} {v[0]:.3g}–{v[1]:.3g}" for k, v in iv.items()))

    rng = np.random.default_rng(42)
    draws = []
    for _ in range(args.draws):
        d = {}
        for k in PARAM_KEYS:
            lo, hi = iv[k]
            # 对数均匀：参数是正量，误差是比例
            d[k] = float(np.exp(rng.uniform(np.log(lo), np.log(hi))))
        draws.append(d)

    preds = [predict_all(d, cands) for d in draws]

    rows = []
    print(f"\n  {'场景':<14}{'档':>4}{'首选配置':<22}{'稳定率':>8}{'次选':>18}")
    for (name, ttft_cap, tpot_cap, pmean) in SCENARIOS:
        for tier in TIERS:
            amax = max(c["acc"] for c in cands)
            picks = []
            for plist in preds:
                feas = []
                for (c, ttft, tpot, cost) in plist:
                    if c["acc"] < amax - tier / 100:
                        continue
                    # 该场景下的 TTFT：按 prompt 长度缩放（最直接用该格的 prompt）
                    if ttft <= ttft_cap and tpot <= tpot_cap:
                        feas.append((cost, c["label"]))
                picks.append(min(feas)[1] if feas else "无可行")
            cnt = Counter(picks)
            top, n = cnt.most_common(1)[0]
            second = cnt.most_common(2)[1][0] if len(cnt) > 1 else "—"
            rows.append(dict(scenario=name, tier=tier, pick=top,
                             stability=round(n / args.draws * 100, 1),
                             second=second))
            if tier == TIERS[0]:
                print(f"  {name:<14}{tier:>4}{top:<22}{n/args.draws*100:>7.0f}%{second:>18}")

    st = np.array([r["stability"] for r in rows])
    print("\n" + "=" * 88)
    print("汇总")
    print("=" * 88)
    print(f"  {len(rows)} 个（场景×档）组合：首选配置稳定率 **中位 {np.median(st):.0f}%**，"
          f"范围 {st.min():.0f}–{st.max():.0f}%")
    low = [r for r in rows if r["stability"] < 70]
    print(f"  稳定率 <70% 的组合：**{len(low)}/{len(rows)}**"
          + (f"（例如 {low[0]['scenario']} tier={low[0]['tier']}："
             f"{low[0]['pick']} vs {low[0]['second']}）" if low else ""))
    print("\n  读法：")
    print("    · 稳定率 = 在参数 5–95% 区间内，200 组抽样里同一配置被选中的比例")
    print("    · 稳定率低 ⇒ 这个场景的推荐**依赖参数精度**，对外应给区间而不是单一答案")

    out = dict(draws=args.draws, n_candidates=len(cands),
               n_combos=len(rows),
               intervals={k: list(v) for k, v in iv.items()},
               median_stability_pct=round(float(np.median(st)), 1),
               min_stability_pct=round(float(st.min()), 1),
               max_stability_pct=round(float(st.max()), 1),
               unstable_combos=len(low),
               rows=rows,
               note=("参数在 CLAIMS 的 5–95% 区间内对数均匀抽样；"
                     "每个 (场景, 质量档) 选满足 SLA 的最便宜配置"))
    fp = Path(args.out) if args.out else (cfg.RESULTS_DIR / "decision_robust.json")
    fp.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n→ {fp}")


if __name__ == "__main__":
    main()
