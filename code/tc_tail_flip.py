# -*- coding: utf-8 -*-
r"""A3 补充：尾部口径真正影响的是"某个配置能不能过 SLA"，而不是"选哪个"。

## 为什么

推荐器只在**通过 SLA 的候选**里挑最便宜的。高并发配置在 TTFT 上本来就过不了
常见 SLA，所以尾部口径改变不了推荐结果（扫描 10 组 SLA，0 组不同）。

但如果你拿模型去**检验一个已有配置**（"我这台 c16 能不能扛 2000ms"），
尾部口径就会改变结论。这里逐格算两种口径下的 90% 上界并统计翻转数。

## 产出

- `results/tail_flip.csv`
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import tc_config as cfg
import tc_recommend as rc

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

TTFT_SLA = 2000.0
TPOT_SLA = 100.0
PROMPT_MEAN, PROMPT_SIGMA = cfg.PROMPT_TOKENS, 0.5


def main():
    fit = json.loads((cfg.RESULTS_DIR / "fit.json").read_text(encoding="utf-8"))
    params = dict(bw_gbs=fit["bw"], peak_tflops=fit["peak"],
                  peak_prefill_tflops=fit.get("peak_prefill", fit["peak"]),
                  c_srv_ms=fit["c_srv"], b_sat=fit["b_sat"], ovh_ttft_ms=fit["ovh_ttft"])
    half = float(fit.get("half_width_log", 0.163))
    tails = rc.load_measured_tails()
    if not tails:
        raise SystemExit("缺 results/quantile_tails.csv，先跑 tc.py tails")

    rows = []
    for c in cfg.all_cells():
        sim = rc.simulate(params, c["model"], c["quant"], c["conc"],
                          PROMPT_MEAN, PROMPT_SIGMA, ctx=c["ctx"])
        f_t = rc.measured_tail_factor(tails, "ttft", c["conc"])
        f_p = rc.measured_tail_factor(tails, "tpot", c["conc"])
        a_ttft = sim["ttft_cvar90"] * np.exp(half)
        m_ttft = a_ttft * f_t
        a_tpot = sim["tpot_cvar90"] * np.exp(half)
        m_tpot = a_tpot * f_p
        rows.append(dict(cell_id=cfg.cell_id(c["model"], c["quant"], c["ctx"], c["conc"]),
                         conc=c["conc"], factor_ttft=round(f_t, 3), factor_tpot=round(f_p, 3),
                         ttft_assumed=round(a_ttft, 1), ttft_measured=round(m_ttft, 1),
                         tpot_assumed=round(a_tpot, 2), tpot_measured=round(m_tpot, 2),
                         pass_assumed=(a_ttft <= TTFT_SLA and a_tpot <= TPOT_SLA),
                         pass_measured=(m_ttft <= TTFT_SLA and m_tpot <= TPOT_SLA)))

    t = pd.DataFrame(rows)
    t.to_csv(cfg.RESULTS_DIR / "tail_flip.csv", index=False, encoding="utf-8")

    flips = t[t["pass_assumed"] != t["pass_measured"]]
    print(f"SLA 取 TTFT≤{TTFT_SLA:.0f}ms、TPOT≤{TPOT_SLA:.0f}ms；共 {len(t)} 格")
    print(f"  假设口径通过 {int(t['pass_assumed'].sum())} 格；"
          f"实测口径通过 {int(t['pass_measured'].sum())} 格")
    print(f"  **结论翻转 {len(flips)} 格**")
    if len(flips):
        print(f"  {'格':<32}{'并发':>4}{'TTFT假设':>10}{'TTFT实测':>10}{'方向':>8}")
        for _, r in flips.head(10).iterrows():
            print(f"  {r['cell_id']:<32}{r['conc']:>4}{r['ttft_assumed']:>10.0f}"
                  f"{r['ttft_measured']:>10.0f}{'假设→否' if r['pass_assumed'] else '假设→是':>8}")
    up = t["ttft_measured"] / t["ttft_assumed"]
    print(f"\n  TTFT 上界被放大的倍数：中位 {up.median():.2f}×、最大 {up.max():.2f}×"
          f"（并发 {int(t.loc[up.idxmax(), 'conc'])}）")


if __name__ == "__main__":
    main()
