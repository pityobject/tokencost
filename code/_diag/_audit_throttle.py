# -*- coding: utf-8 -*-
"""审计：哪些格是在降频/低功耗状态下测的？这些数据必须作废重跑。"""
import json
import pathlib
import statistics as st
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = pathlib.Path(__file__).resolve().parents[2]
rows = [json.loads(l) for l in (ROOT / "data/raw_cells.jsonl").open(encoding="utf-8") if l.strip()]
ok = [r for r in rows if r.get("status") == "ok"]
g = {}
for r in ok:
    g.setdefault(r["cell_id"], []).append(r)

CLK_OK = 1800      # MHz：正常满血应 > 1800
PWR_OK = 50        # W：正常应 > 50

bad, good = [], []
for cid, rs in g.items():
    clk = st.median(x.get("clock_sm_median") or 0 for x in rs)
    pwr = st.median(x.get("power_w_median") or 0 for x in rs)
    (bad if (clk < CLK_OK or pwr < PWR_OK) else good).append((cid, clk, pwr))

print(f"总格数 {len(g)}：正常 {len(good)}，疑似降频 {len(bad)}\n")
if bad:
    print(f"{'cell':<28}{'SM MHz':>8}{'功耗W':>7}")
    for cid, clk, pwr in sorted(bad, key=lambda x: x[1]):
        print(f"{cid:<28}{clk:>8.0f}{pwr:>7.1f}")
    print(f"\n这些格必须重跑（当前环境：插电 + 最佳性能 + 不熄屏）。")
    print("重跑方式（逐格删掉旧数据后单独跑）：")
    print('  python tc_rerun_cells.py          # 见下方脚本')
else:
    print("全部正常。")

# 写出待重跑清单
out = ROOT / "data" / "cells_to_rerun.txt"
out.write_text("\n".join(cid for cid, _, _ in sorted(bad, key=lambda x: x[1])), encoding="utf-8")
print(f"\n清单已写入 {out}（{len(bad)} 格）")
