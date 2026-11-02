# -*- coding: utf-8 -*-
"""把评测原始记录摊开：确认到底是模型答错，还是请求/判分错位。"""
import json
import pathlib
import sys
from collections import defaultdict

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = pathlib.Path(__file__).resolve().parents[2]
rows = [json.loads(l) for l in (ROOT / "data/quality_details.jsonl").open(encoding="utf-8") if l.strip()]

by = defaultdict(list)
for r in rows:
    by[(r["model"], r["quant"])].append(r)

for key in [("1.5B", "q4_k_m"), ("7B", "q4_k_m")]:
    print("=" * 78)
    print(f"模型 {key[0]} {key[1]} —— add 类前 4 条完整记录")
    print("=" * 78)
    adds = [r for r in by[key] if r["kind"] == "add"][:4]
    for r in adds:
        print(f"  prompt  : {r['prompt']!r}")
        print(f"  expected: {r['expected']!r}")
        print(f"  answer  : {r['answer']!r}")
        print(f"  ok      : {r['ok']}")
        print()
    print(f"  {key[0]} 的 count 类前 2 条：")
    for r in [x for x in by[key] if x["kind"] == "count"][:2]:
        print(f"    prompt={r['prompt'][:40]!r} expected={r['expected']!r} answer={r['answer']!r} ok={r['ok']}")
    print()

print("=" * 78)
print("每类题目的『期望答案分布』（看是不是都能被数字判分器处理）")
print("=" * 78)
bykind = defaultdict(list)
for r in rows:
    bykind[r["kind"]].append(r)
for kind, rs in bykind.items():
    exp = {r["expected"] for r in rs}
    print(f"  {kind:<8} 期望值样例 {sorted(exp)[:6]}")
