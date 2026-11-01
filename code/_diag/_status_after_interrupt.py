# -*- coding: utf-8 -*-
"""中断后的状态检查：数据完整吗？有没有残留进程？从哪继续？
"""
import json
import pathlib
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = pathlib.Path(r"C:\Users\王钟垚\Desktop\find_career\tokencost")
import tc_config as cfg  # noqa: E402

path = ROOT / "data" / "raw_cells.jsonl"
lines = path.read_text(encoding="utf-8").splitlines() if path.exists() else []

good, bad_json, bad_utf8 = [], [], []
for i, ln in enumerate(lines, 1):
    if not ln.strip():
        continue
    try:
        good.append(json.loads(ln))
    except json.JSONDecodeError:
        bad_json.append(i)

print(f"文件：{path}")
print(f"总行数 {len(lines)}；可解析 {len(good)}；坏行 {len(bad_json)} {bad_json[:5]}")
print()

# 每格几轮
cnt = {}
for r in good:
    if r.get("status") == "ok":
        cnt[r["cell_id"]] = cnt.get(r["cell_id"], 0) + 1
full = [c for c, n in cnt.items() if n >= cfg.REPEATS]
part = [c for c, n in cnt.items() if 0 < n < cfg.REPEATS]
print(f"完整格（{cfg.REPEATS} 轮）：{len(full)}    半截格（1–2 轮）：{len(part)}")
if part:
    for c in part:
        print(f"   半截：{c}（{cnt[c]} 轮，重跑会自动补齐）")
print()

# ctx 预设进度
try:
    ctx_cells = [cfg.cell_id(c["model"], c["quant"], c["ctx"], c["conc"])
                 for c in cfg.preset_cells("ctx")]
    done = [c for c in ctx_cells if cnt.get(c, 0) >= cfg.REPEATS]
    print(f"preset=ctx 进度：{len(done)}/{len(ctx_cells)} 格完成")
    remain = [c for c in ctx_cells if cnt.get(c, 0) < cfg.REPEATS]
    if remain:
        print(f"  还差 {len(remain)} 格：{remain[:6]}{' ...' if len(remain) > 6 else ''}")
except Exception as e:
    print("读取 ctx 预设失败：", e)
print()

# 残留进程 / 端口
print("=== 残留进程检查 ===")
try:
    import psutil
    alive = [p for p in psutil.process_iter(["name", "pid"])
             if p.info["name"] and "llama" in p.info["name"].lower()]
    if alive:
        print("  [!!] 发现残留进程（会占显存和端口）：")
        for p in alive:
            print(f"      PID {p.info['pid']}  {p.info['name']}")
        print("  处理：Get-Process llama-server | Stop-Process -Force")
    else:
        print("  [OK] 没有残留的 llama-server")
except Exception as e:
    print("  （psutil 不可用，跳过）", e)

# 端口
import socket
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    busy = s.connect_ex((cfg.HOST, cfg.PORT)) == 0
print(f"  端口 {cfg.PORT}：{'被占用 [!!]' if busy else '空闲 [OK]'}")

# 总格数
print()
print(f"当前可用于分析的数据：{len(good)} 行，{len(cnt)} 个格")
print("下一步：直接重跑同一条命令即可，已完成的格会自动跳过。")
