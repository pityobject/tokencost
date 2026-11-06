# -*- coding: utf-8 -*-
"""
A3 敏感性扫描：尾部口径（假设 vs 实测）在哪些 SLA 下会改变推荐？

## 为什么需要

单点 SLA 下两种口径可能选同一个配置（比如推荐的是 c1，尾部因子≈1）。
只有扫描一组 SLA，才能看清"尾部假设在什么条件下开始起作用"。

## 做法

对若干 (TTFT上限, TPOT上限) 组合，分别用 `--tail assumed` 与 `--tail measured`
跑推荐器，比较：
- 推荐配置是否不同
- 若不同，成本差多少（实测口径通常更保守 → 成本更高）

## 产出

- `results/tail_sensitivity.csv` / `tail_sensitivity.json`
"""
import json
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import tc_config as cfg

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PY = sys.executable
RC = str(Path(__file__).resolve().parent / "tc_recommend.py")

GRID = [
    (600, 45), (600, 60), (800, 50), (800, 60), (1000, 60),
    (1200, 70), (1500, 80), (2000, 100), (2500, 150), (3000, 200),
]


def run(ttft, tpot, tail):
    r = subprocess.run([PY, RC, "--ttft", str(ttft), "--tpot", str(tpot),
                        "--tail", tail, "--top", "1"],
                       capture_output=True, text=True, encoding="utf-8",
                       errors="replace", timeout=300)
    out = (r.stdout or "") + (r.stderr or "")
    m = re.search(r"推荐配置：(\S+)", out)
    c = re.search(r"预测成本\s+([\d.]+) GPU 秒", out)
    if m:
        return m.group(1), float(c.group(1)) if c else float("nan")
    if "没有任何配置" in out:
        return None, float("nan")
    return "ERROR", float("nan")


def main():
    rows = []
    print(f"  {'TTFT':>6}{'TPOT':>6}  {'assumed':<26}{'measured':<26}{'变化'}")
    for ttft, tpot in GRID:
        a, ca = run(ttft, tpot, "assumed")
        b, cb = run(ttft, tpot, "measured")
        same = (a == b)
        delta = (cb / ca - 1) * 100 if (ca == ca and cb == cb and ca) else float("nan")
        rows.append(dict(ttft=ttft, tpot=tpot,
                         assumed=a, cost_assumed=round(ca, 1) if ca == ca else None,
                         measured=b, cost_measured=round(cb, 1) if cb == cb else None,
                         same=bool(same),
                         cost_delta_pct=round(delta, 2) if delta == delta else None))
        print(f"  {ttft:>6}{tpot:>6}  {str(a):<26}{str(b):<26}"
              f"{'相同' if same else f'不同（成本 {delta:+.1f}%）'}")

    n_diff = sum(1 for r in rows if not r["same"])
    worst = max((abs(r["cost_delta_pct"] or 0) for r in rows), default=0.0)
    summ = dict(n_cases=len(rows), n_different=n_diff,
                max_cost_delta_pct=round(worst, 2),
                note=("尾部口径（对数正态假设 vs 实测 p90/p50）对推荐的影响；"
                      "不同 SLA 下触发的程度不同"))
    (cfg.RESULTS_DIR / "tail_sensitivity.json").write_text(
        json.dumps(dict(summary=summ, cases=rows), ensure_ascii=False, indent=2),
        encoding="utf-8")
    import pandas as pd
    pd.DataFrame(rows).to_csv(cfg.RESULTS_DIR / "tail_sensitivity.csv",
                              index=False, encoding="utf-8")

    print(f"\n  {n_diff}/{len(rows)} 组 SLA 下推荐不同；最大成本差 {worst:.1f}%")
    print("→ results/tail_sensitivity.csv / tail_sensitivity.json")


if __name__ == "__main__":
    main()
