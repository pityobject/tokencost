# -*- coding: utf-8 -*-
"""跑 dense 之前：列出这 35 格，并标出显存接近悬崖的。"""
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import tc_config as cfg  # noqa: E402

cells = cfg.preset_cells("dense")
print(f"dense 预设：{len(cells)} 格（ctx 512/2048 × 并发 2/8/12/24 × 5 个模型组合）\n")
print(f"{'模型':>6} {'量化':<7} {'ctx':>5} {'并发':>4} {'显存估计':>9} {'KV':>7}  风险")
for c in sorted(cells, key=lambda x: (x["model"], x["quant"], x["ctx"], x["conc"])):
    spec = cfg.MODEL_SPECS[(c["model"], c["quant"])]
    kv = cfg.kv_gb(spec, c["ctx"], c["conc"])
    vram = c["vram_est_gb"]
    risk = ""
    if vram > 7.0:
        risk = "接近悬崖（可能 KV 外溢）"
    elif vram > 6.0:
        risk = "偏高"
    print(f"{c['model']:>6} {c['quant']:<7} {c['ctx']:>5} {c['conc']:>4} "
          f"{vram:>8.2f}G {kv:>6.2f}G  {risk}")

n_risky = sum(1 for c in cells if c["vram_est_gb"] > 7.0)
print(f"\n35 格里 {n_risky} 格显存估计 > 7 GB（跑完如果吞吐断崖，就是撞了悬崖，正常记录即可）")
print(f"预计耗时：35 格 × 约 45 秒 ≈ 26 分钟")
