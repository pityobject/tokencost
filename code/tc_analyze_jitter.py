# -*- coding: utf-8 -*-
"""
排队假说的第一步：量「同一格内 TPOT 的抖动」与并发 / 提示长度的关系

## 为什么量这个

F16 的结论是：失效区更像 **prefill 打断 decode 的排队效应**，而不是"批内摊薄"，
所以一个标量修正因子救不了（它修好失效区、弄坏正常区）。

如果排队假说成立，应当能看到：
1. **抖动随并发上升**（批内调度冲突增多）；
2. **抖动随提示长度上升**（prefill 占用更久，decode 被挤）；
3. 抖动的上升幅度足以解释"隐性误差"——即模型预测的是中位数/均值，
   而实测里被拖慢的那部分请求把 TPOT 中位数/吞吐推高了。

## 数据

`data/raw_requests.jsonl`（每请求一行：ttft_ms / total_ms / n_out / t_start_rel）。
逐请求 TPOT = (total_ms − ttft_ms) / (n_out − 1)。

## 产出

- `results/jitter_by_cell.csv`：每格的 CV、p90/p50、以及 b、L、b·L
- `results/jitter_summary.json`
- `figures/fig18_jitter.png`
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import tc_config as cfg

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def main():
    raw = [json.loads(l) for l in (cfg.DATA_DIR / "raw_requests.jsonl").open(encoding="utf-8")
           if l.strip()]
    by_cell = defaultdict(list)
    for r in raw:
        if not r.get("ok", True):
            continue
        n_out = int(r.get("n_out") or 0)
        ttft = float(r.get("ttft_ms") or 0)
        total = float(r.get("total_ms") or 0)
        if n_out < 3 or total <= 0 or ttft <= 0:
            continue
        tp = (total - ttft) / max(n_out - 1, 1)
        if tp <= 0 or tp > 10000:
            continue
        by_cell[r["cell_id"]].append(tp)

    s = pd.read_csv(cfg.DATA_DIR / "summary_steady.csv")
    meta = s.drop_duplicates("cell_id").set_index("cell_id")

    rows = []
    for cid, vals in by_cell.items():
        if len(vals) < 8 or cid not in meta.index:
            continue
        v = np.array(vals, dtype=float)
        m = meta.loc[cid]
        L = float(m.get("prompt_tokens_median") or 0)
        if L <= 0:
            continue
        rows.append(dict(
            cell_id=cid, conc=int(m["conc"]), ctx=int(m["ctx"]),
            L=L, bL=int(m["conc"]) * L,
            n=len(v), tp_med=float(np.median(v)),
            cv=float(v.std(ddof=1) / v.mean()) if v.mean() else np.nan,
            p90_p50=float(np.percentile(v, 90) / np.median(v)),
            p99_p50=float(np.percentile(v, 99) / np.median(v)),
        ))

    t = pd.DataFrame(rows)
    if t.empty:
        print("没有足够逐请求数据")
        return
    t = t.sort_values("conc")
    t.to_csv(cfg.RESULTS_DIR / "jitter_by_cell.csv", index=False, encoding="utf-8")

    print(f"逐格抖动统计：{len(t)} 格\n")
    print(f"  {'并发':>4}{'格数':>5}{'CV 中位':>9}{'p90/p50':>9}{'p99/p50':>9}")
    for b, g in t.groupby("conc"):
        print(f"  {b:>4}{len(g):>5}{g['cv'].median():>8.1%}{g['p90_p50'].median():>9.3f}"
              f"{g['p99_p50'].median():>9.3f}")

    # 相关性：抖动 vs 并发、L、b·L
    def corr(x, y):
        m = np.isfinite(t[x]) & np.isfinite(t[y])
        if m.sum() < 4:
            return float("nan")
        return float(np.corrcoef(t.loc[m, x], t.loc[m, y])[0, 1])

    c_b = corr("conc", "p90_p50")
    c_L = corr("L", "p90_p50")
    c_bL = corr("bL", "p90_p50")
    c_b_cv = corr("conc", "cv")
    print(f"\n  相关系数（p90/p50 对 …）：并发 {c_b:+.2f} · 提示长度 {c_L:+.2f} · "
          f"b·L {c_bL:+.2f}")
    print(f"  相关系数（CV 对并发）：{c_b_cv:+.2f}")

    # 检验"抖动足以解释误差"：同一 b 内，抖动随 L 的斜率
    sub = t[t["conc"] >= 4]
    slope_L = float(np.polyfit(np.log(sub["L"].clip(lower=1)), sub["p90_p50"], 1)[0]) if len(sub) > 3 else float("nan")
    print(f"  b≥4 时 log L 的斜率：{slope_L:+.3f}（每 e 倍长度，p90/p50 变化）")

    verdict = []
    if c_b > 0.3:
        verdict.append("抖动随并发上升 → 支持排队假说")
    if c_L > 0.2:
        verdict.append("抖动随提示长度上升 → 支持 prefill 挤占 decode")
    if not verdict:
        verdict.append("抖动与并发/长度关系不明显 → 排队假说证据不足")
    print(f"\n  判定：{'；'.join(verdict)}")

    # 图：p90/p50 vs 并发（每格一点）
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    for b, g in t.groupby("conc"):
        axes[0].scatter([b] * len(g), g["p90_p50"], s=18, alpha=0.6)
    axes[0].set_xlabel("并发 b")
    axes[0].set_ylabel("同格内 TPOT 的 p90/p50")
    axes[0].set_title("抖动 vs 并发")
    axes[0].grid(alpha=0.3)

    sc = axes[1].scatter(t["L"], t["p90_p50"], c=t["conc"], s=22, alpha=0.75)
    axes[1].set_xlabel("实测提示长度 L（token）")
    axes[1].set_ylabel("p90/p50")
    axes[1].set_title("抖动 vs 提示长度（颜色=并发）")
    plt.colorbar(sc, ax=axes[1], label="并发")
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(cfg.FIG_DIR / "fig18_jitter.png", dpi=150)

    out = dict(n_cells=int(len(t)), corr_conc=c_b, corr_L=c_L, corr_bL=c_bL,
               corr_cv_conc=c_b_cv, slope_logL_ge4=slope_L,
               by_conc={int(b): dict(n=int(len(g)), cv=round(float(g['cv'].median()), 4),
                                     p90_p50=round(float(g['p90_p50'].median()), 4))
                        for b, g in t.groupby("conc")},
               verdict="；".join(verdict))
    (cfg.RESULTS_DIR / "jitter_summary.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print("  图 → figures/fig18_jitter.png")
    print("→ results/jitter_by_cell.csv / jitter_summary.json")


if __name__ == "__main__":
    main()

