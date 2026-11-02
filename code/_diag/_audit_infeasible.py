# -*- coding: utf-8 -*-
"""核对：--include-infeasible 到底跑成了什么，以及显存模型的验证点。"""
import json
import pathlib
import re
import statistics as st
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = pathlib.Path(__file__).resolve().parents[2]
import tc_config as cfg  # noqa: E402

rows = [json.loads(l) for l in (ROOT / "data/raw_cells.jsonl").open(encoding="utf-8") if l.strip()]
by = {}
for r in rows:
    by.setdefault(r["cell_id"], []).append(r)

print(f"总行数 {len(rows)}，去重后 {len(by)} 个格\n")

# 哪些格是"原判不可行"的
infeasible_ids = set()
for c in cfg.all_cells(specs=cfg.CORE_SPECS):
    if not c["feasible"]:
        infeasible_ids.add(cfg.cell_id(c["model"], c["quant"], c["ctx"], c["conc"]))

print("=== 原判 15 个不可行格的真实结局 ===")
print(f"{'cell':<26}{'总ctx':>7}{'显存估计':>9}{'行数':>5}{'状态':>8}")
for cid in sorted(infeasible_ids):
    rs = by.get(cid, [])
    c = next(x for x in cfg.all_cells(specs=cfg.CORE_SPECS)
             if cfg.cell_id(x["model"], x["quant"], x["ctx"], x["conc"]) == cid)
    st_ = "ok" if any(r.get("status") == "ok" for r in rs) else ("failed" if rs else "未跑")
    print(f"{cid:<26}{c['ctx_total']:>7}{c['vram_est_gb']:>8.2f}G{len(rs):>5}{st_:>8}")

# 显存模型验证：从成功格的日志里读 KV buffer size
print("\n=== 显存模型验证（成功格的启动日志）===")
pat_kv = re.compile(r"KV buffer size\s*=\s*([\d.]+)\s*MiB", re.I)
pat_kv2 = re.compile(r"kv_cache.*?size\s*=\s*([\d.]+)\s*MiB", re.I)
print(f"{'cell':<26}{'配置ctx':>8}{'预测KV GB':>10}{'日志KV GB':>10}{'误差':>8}")
n_ok = 0
errs = []
for cid, rs in sorted(by.items()):
    if not any(r.get("status") == "ok" for r in rs):
        continue
    log = ROOT / "logs" / f"server_{cid}.log"
    if not log.exists():
        continue
    txt = log.read_text(encoding="utf-8", errors="replace")
    m = pat_kv.search(txt) or pat_kv2.search(txt)
    if not m:
        continue
    r = [x for x in rs if x.get("status") == "ok"][0]
    spec = cfg.MODEL_SPECS[(r["model"], r["quant"])]
    pred = cfg.kv_gb(spec, r["ctx"], r["conc"])
    real = float(m.group(1)) * 1024**2 / 1e9
    if pred <= 0:
        continue
    err = (real / pred - 1) * 100
    errs.append(abs(err))
    n_ok += 1
    if n_ok <= 12:
        print(f"{cid:<26}{r['ctx']:>8}{pred:>10.2f}{real:>10.2f}{err:>7.1f}%")

if errs:
    print(f"\n共 {len(errs)} 格可对照：|误差| 中位数 {st.median(errs):.2f}%，最大 {max(errs):.2f}%")
else:
    print("\n日志里没找到 KV buffer size 行，看看样例日志的显存相关行：")
    sample = sorted(ROOT.glob("logs/server_*.log"))
    if sample:
        t = sample[-1].read_text(encoding="utf-8", errors="replace")
        for line in t.splitlines():
            if any(k in line.lower() for k in ("buffer", "kv cache", "kv_cache")):
                print("   ", line[:130])
