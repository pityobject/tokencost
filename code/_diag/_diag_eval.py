# -*- coding: utf-8 -*-
"""诊断 tc_eval：为什么大模型反而更差。看真实回答长什么样。"""
import json
import pathlib
import sys
from collections import Counter, defaultdict

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = pathlib.Path(r"C:\Users\王钟垚\Desktop\find_career\tokencost")
rows = [json.loads(l) for l in (ROOT / "data/quality_details.jsonl").open(encoding="utf-8") if l.strip()]
print(f"明细共 {len(rows)} 条\n")

by = defaultdict(list)
for r in rows:
    by[(r["model"], r["quant"])].append(r)

for k, rs in sorted(by.items()):
    ok = sum(1 for r in rs if r["ok"])
    empty = sum(1 for r in rs if not (r.get("answer") or "").strip())
    lens = [len((r.get("answer") or "").strip()) for r in rs]
    print(f"{k[0]:>5} {k[1]:<7} ok {ok:>2}/{len(rs)}  空回答 {empty:>2}  "
          f"答案字符数 中位 {sorted(lens)[len(lens)//2]:>3} 最大 {max(lens):>3}")

print("\n=== 各题型的正确率（看是不是某一类题挂了）===")
kinds = sorted({r["kind"] for r in rows})
print(f"{'模型':>6} {'量化':<7} " + " ".join(f"{k:>8}" for k in kinds))
for k, rs in sorted(by.items()):
    line = f"{k[0]:>6} {k[1]:<7} "
    for kind in kinds:
        sub = [r for r in rs if r["kind"] == kind]
        acc = sum(1 for r in sub if r["ok"]) / len(sub) if sub else 0
        line += f"{acc*100:>7.0f}% "
    print(line)

print("\n=== 失败样例（每个模型挑 3 条）===")
for k, rs in sorted(by.items()):
    bad = [r for r in rs if not r["ok"]][:3]
    print(f"\n--- {k[0]} {k[1]} ---")
    for r in bad:
        ans = (r.get("answer") or "").replace("\n", "\\n")
        print(f"  [{r['kind']}] 期望={r['expected']!r}")
        print(f"      实际={ans[:90]!r}")

print("\n=== 回答里能不能找到期望值（判断是判分问题还是模型真错）===")
for k, rs in sorted(by.items()):
    miss = 0
    for r in rs:
        if r["ok"]:
            continue
        a = (r.get("answer") or "")
        if r["expected"].strip() and r["expected"].strip() in a:
            miss += 1
    print(f"  {k[0]:>5} {k[1]:<7} 失败 {len(rs)-sum(1 for r in rs if r['ok']):>2} 条中，"
          f"回答里其实含正确答案的有 {miss:>2} 条  ← 这些是判分问题")
