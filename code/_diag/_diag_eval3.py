# -*- coding: utf-8 -*-
"""看 reverse / count 两类题的真实回答。"""
import json
import pathlib
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
p = pathlib.Path(__file__).resolve().parents[2] / "data" / "quality_details.jsonl"
rows = [json.loads(l) for l in p.open(encoding="utf-8") if l.strip()]

for kind in ["reverse", "count", "sort"]:
    print("=" * 70)
    print(kind)
    print("=" * 70)
    for r in [x for x in rows if x["kind"] == kind][:8]:
        print(f"  {r['model']:>5} {r['quant']:<7} exp={r['expected']!r:22} "
              f"ans={r['answer']!r:28} ok={r['ok']}")
    print()
