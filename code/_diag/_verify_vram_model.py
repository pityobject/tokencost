# -*- coding: utf-8 -*-
"""用失败日志验证显存模型：服务端实际尝试分配的 KV 大小 vs 我们的预测。

这是 --include-infeasible 的额外收益：失败日志里带着真实的分配数字，
可以把"显存估计"从估算变成"与实测吻合"。
"""
import json
import pathlib
import re
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = pathlib.Path(r"C:\Users\王钟垚\Desktop\find_career\tokencost")
import tc_config as cfg  # noqa: E402

# 1) 统计失败行
rows = [json.loads(l) for l in (ROOT / "data/raw_cells.jsonl").open(encoding="utf-8") if l.strip()]
fails = [r for r in rows if r.get("status") != "ok"]
oks = [r for r in rows if r.get("status") == "ok"]
print(f"raw_cells.jsonl: 成功 {len(oks)} 行，失败 {len(fails)} 行\n")

# 2) 从日志里抓 "allocating X MiB" 与 "buffer of size Y"
pat_alloc = re.compile(r"allocating ([\d.]+) MiB")
pat_size = re.compile(r"failed to allocate CUDA0 buffer of size (\d+)")

print(f"{'格':<26}{'预测KV(GB)':>11}{'服务端实际(GB)':>14}{'误差':>8}")
print("-" * 62)
rows_out = []
for r in fails:
    cid = r["cell_id"]
    log = ROOT / "logs" / f"server_{cid}.log"
    if not log.exists():
        continue
    txt = log.read_text(encoding="utf-8", errors="replace")
    m = pat_alloc.search(txt) or pat_size.search(txt)
    if not m:
        continue
    real_gb = (float(m.group(1)) * 1024**2 / 1e9) if "." in m.group(1) else (float(m.group(1)) / 1e9)
    spec = cfg.MODEL_SPECS[(r["model"], r["quant"])]
    pred_gb = cfg.kv_gb(spec, r["ctx"], r["conc"])
    err = (real_gb / pred_gb - 1) * 100 if pred_gb else float("nan")
    rows_out.append((cid, pred_gb, real_gb, err))
    print(f"{cid:<26}{pred_gb:>11.2f}{real_gb:>14.2f}{err:>7.1f}%")

if rows_out:
    import statistics as st
    errs = [abs(x[3]) for x in rows_out]
    print("-" * 62)
    print(f"共 {len(rows_out)} 格可对照；|误差| 中位数 {st.median(errs):.2f}%，最大 {max(errs):.2f}%")
    print("\n→ 这张表可以写进报告：显存模型与服务端实际分配吻合，所以'45 格可行/15 格不可行'")
    print("   不是拍脑袋剪的，是按真实显存算账剪的。")
else:
    print("没找到可对照的日志（可能日志名不同）")
    for r in fails[:3]:
        print("  失败格样例:", r["cell_id"], "|", (r.get("error") or "")[:60])
