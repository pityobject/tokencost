# -*- coding: utf-8 -*-
"""确认新预设的格列表（含请求数与后缀），并核对核心集仍是 45 格。"""
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import tc_config as c  # noqa: E402

print("核心集格数：", len([x for x in c.all_cells(specs=c.CORE_SPECS) if x["feasible"]]), "(应为 45)")
print()
for name in ("step", "wave", "knee"):
    cells = c.preset_cells(name)
    print(f"=== {name}（{len(cells)} 格）===")
    for x in cells:
        cid = c.cell_id(x["model"], x["quant"], x["ctx"], x["conc"], x.get("suffix", ""))
        print(f"  {cid:<30} 请求 {x.get('requests'):>3}  显存 {x['vram_est_gb']:.2f}G")
    print()
