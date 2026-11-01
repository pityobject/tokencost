# -*- coding: utf-8 -*-
"""确认那 14 格是不是把 KV 放到了主机内存（kv-offload），以及它们能否与核心网格比较。"""
import json
import pathlib
import statistics as st
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = pathlib.Path(r"C:\Users\王钟垚\Desktop\find_career\tokencost")
rows = [json.loads(l) for l in (ROOT / "data/raw_cells.jsonl").open(encoding="utf-8") if l.strip()]
ok = [r for r in rows if r.get("status") == "ok"]

g = {}
for r in ok:
    g.setdefault(r["cell_id"], []).append(r)


def med(cid, f):
    v = [x[f] for x in g.get(cid, []) if x.get(f) is not None]
    return st.median(v) if v else float("nan")


print("=== 同模型同并发：不同 ctx 下的显存占用与吞吐 ===")
print(f"{'cell':<28}{'显存MB':>8}{'吞吐':>8}{'TPOT':>8}{'备注':>22}")
for m, q in [("3B", "q4_k_m"), ("3B", "q8_0"), ("1.5B", "q4_k_m")]:
    for conc in [16, 32]:
        for ctx in [512, 2048, 8192]:
            cid = f"{m}_{q}_ctx{ctx}_c{conc}"
            if cid not in g:
                continue
            mm = med(cid, "mem_used_mb_median")
            spec_bytes = None
            import tc_config as cfg
            wgb = cfg.MODEL_SPECS[(m, q)]["weight_bytes"] / 1e6
            note = ""
            if mm > 0:
                # 权重 GPU 占用通常约为文件大小的 85–95%
                if mm < wgb * 0.95:
                    note = "显存≈仅权重→KV在主机"
                else:
                    note = "显存含KV"
            print(f"{cid:<28}{mm:>8.0f}{med(cid,'throughput_tps'):>8.0f}"
                  f"{med(cid,'tpot_p50_ms'):>8.1f}{note:>22}")
    print()

print("=== 结论怎么用 ===")
print("如果高 ctx 格的显存占用只有权重大小、吞吐明显下降 → 那批数据是 kv-offload 状态，")
print("不能和核心网格（KV 全在显存）放进同一个模型拟合。")
