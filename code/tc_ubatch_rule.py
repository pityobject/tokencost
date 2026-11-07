# -*- coding: utf-8 -*-
"""
ubatch 选择规则（`tc.py ubatchrule`）：给定 (SLA, 并发)，选哪个 `-ub`

## 数据

四档 `-ub`（64/128/256/512）各一次 6 点标定，共 24 个测量点（A10，3B fp16，32 槽）。

## 从数据里读出来的两条

1. **`TTFT` 随并发线性增长，斜率 ∝ `ub^-0.83`**：
   c32 的 TTFT 从 2464 ms（ub=64）降到 438 ms（ub=512），5.6 倍。
2. **`ub` 大不只对 TTFT 好**：`c_srv` 也从 3.04 降到 2.25（−26%）。
   **两个指标同向** —— 所以没有"用带宽换延迟"的权衡，能开多大就开多大。

## 规则

    TTFT(b) ≈ TTFT₃₂ · (b/32) · (64/ub)^0.83

反解"要满足 TTFT ≤ target"，得到最小 ub：

    ub ≥ 64 · ( 2464·b / (32·target) )^(1/0.83)

## 用法

    python code/tc_ubatch_rule.py
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import tc_config as cfg

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
UBS = (64, 128, 256, 512)


def load():
    out = {}
    for ub in UBS:
        fp = cfg.RESULTS_DIR / "a10_raw" / f"calib_ub{ub}.json"
        if not fp.exists():
            continue
        d = json.loads(fp.read_text(encoding="utf-8"))
        # L≈307 的 c1/c4/c16/c32 + L=1978 的 c1
        short = [m for m in d["measurements"] if 300 <= m["L"] <= 310]
        long1 = [m for m in d["measurements"] if m["L"] > 1900]
        out[ub] = dict(c_srv=d["c_srv"], b_sat=d["b_sat"],
                       ttft={m["conc"]: m["ttft"] for m in short},
                       ttft_long=long1[0]["ttft"] if long1 else None)
    return out


def main():
    d = load()
    if len(d) < 3:
        print("ubatch 数据不足")
        return

    print("=" * 84)
    print("一、四档 ubatch 的实测（L≈307，c1–c32）")
    print("=" * 84)
    print(f"  {'ub':>5}{'c1':>9}{'c4':>9}{'c16':>10}{'c32':>10}{'c32/c1':>9}"
          f"{'c_srv':>9}{'b_sat':>8}")
    for ub in sorted(d):
        r = d[ub]
        t = r["ttft"]
        print(f"  {ub:>5}{t.get(1, float('nan')):>9.1f}{t.get(4, float('nan')):>9.1f}"
              f"{t.get(16, float('nan')):>10.1f}{t.get(32, float('nan')):>10.1f}"
              f"{t.get(32, float('nan'))/t.get(1, 1):>9.2f}×{r['c_srv']:>9.3f}"
              f"{r['b_sat']:>8.2f}")

    # 拟合 c32 的幂律 + 并发的线性
    ubs = np.array(sorted(d), float)
    c32 = np.array([d[int(u)]["ttft"][32] for u in ubs], float)
    slope, intercept = np.polyfit(np.log(64.0 / ubs), np.log(c32), 1)
    c32_at64 = float(np.exp(intercept))
    alpha = float(slope)
    print(f"\n  拟合：TTFT_c32(ub) ≈ {c32_at64:.0f} ms × (64/ub)^{alpha:.2f}")
    pred = c32_at64 * (64 / ubs) ** alpha
    r2 = 1 - np.sum((c32 - pred) ** 2) / np.sum((c32 - c32.mean()) ** 2)
    print(f"  R² = {r2:.4f}")

    # 并发的线性性
    print(f"\n  并发线性性（各 ub 下 c32/c1 的斜率）：")
    for ub in sorted(d):
        t = d[ub]["ttft"]
        k = (t[32] - t[1]) / 31
        print(f"    ub={ub:<4} 每增 1 并发 +{k:.1f} ms")

    # 决策表
    print("\n" + "=" * 84)
    print("二、决策表：给定「并发」和「TTFT 上限」，最小该用多少 ub")
    print("=" * 84)
    targets = (3000, 2000, 1500, 1000, 700, 500)
    print(f"  {'并发':>5}" + "".join(f"{('≤'+str(t)+'ms'):>11}" for t in targets))
    for b in (8, 16, 32, 48, 64):
        row = f"  {b:>5}"
        for tgt in targets:
            need = 64 * (c32_at64 * b / (32 * tgt)) ** (1 / alpha)
            pick = next((u for u in sorted(d) if u >= need), None)
            row += f"{(str(pick) if pick else '—'):>11}"
        print(row)
    print("\n  （单元格 = 满足该延迟上限的最小 `-ub`；`—` = 512 也不够，需要降并发或换卡）")

    print("\n" + "=" * 84)
    print("三、结论：没有权衡，能开多大开多大")
    print("=" * 84)
    lo, hi = d[min(d)], d[max(d)]
    print(f"  ub {min(d)} → {max(d)}：c32 的 TTFT {lo['ttft'][32]:.0f} → "
          f"{hi['ttft'][32]:.0f} ms（**{lo['ttft'][32]/hi['ttft'][32]:.1f}×** 更好）")
    print(f"  同时 c_srv {lo['c_srv']:.2f} → {hi['c_srv']:.2f} ms（也变好，−{(1-hi['c_srv']/lo['c_srv'])*100:.0f}%）")
    print("  → 两个指标同向，所以 `-ub` 应当取内存允许的最大值")
    print("  → 唯一代价是批处理缓冲占显存；超长 prompt 时 ub > L 无额外收益")

    out = dict(alpha=round(alpha, 3), c32_at_ub64_ms=round(c32_at64, 1),
               r2=round(float(r2), 4),
               formula=f"TTFT(b) ≈ {c32_at64:.0f}·(b/32)·(64/ub)^{alpha:.2f} ms",
               rule=f"ub ≥ 64·( {c32_at64:.0f}·b / (32·target) )^(1/{alpha:.2f})",
               table={ub: dict(c_srv=d[ub]["c_srv"], ttft=d[ub]["ttft"],
                               ttft_long=d[ub]["ttft_long"]) for ub in sorted(d)},
               verdict="没有权衡：ub 越大，TTFT 与 c_srv 同时变好；取内存允许的最大值",
               note="L≈307 拟合；L=1978 时趋势相同（c1 的 TTFT 也随 ub 下降约 39%）")
    (cfg.RESULTS_DIR / "ubatch_rule.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n→ results/ubatch_rule.json")


if __name__ == "__main__":
    main()
