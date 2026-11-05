# -*- coding: utf-8 -*-
r"""把时钟相对判据应用到历史数据：输出派生表（绝不改写 raw_cells）。

产物：
  results/clock_audit.csv   每格：clock_sm / 参考值 / ratio / gate
  results/clock_audit.json  汇总：各档格数、弹性估计

同时给 tc_check_contamination 提供统一的"低时钟清单"。
"""
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import tc_config as cfg
import tc_gpu

sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def main():
    ref = tc_gpu.clock_reference()
    rows = [json.loads(l) for l in (cfg.DATA_DIR / "raw_cells.jsonl").open(encoding="utf-8") if l.strip()]
    ok = [r for r in rows if r.get("status") == "ok"]

    # 同一个格可能有多行（重复轮次）；先按 (cell_id, repeat) 取最后一次，再取格内中位
    df = pd.DataFrame(ok)
    df = df.groupby(["cell_id", "repeat"], as_index=False).last()
    out = []
    for cid, g in df.groupby("cell_id"):
        clk = pd.to_numeric(g["clock_sm_median"], errors="coerce").median()
        if pd.isna(clk):
            continue
        model = g["model"].iloc[0]
        ratio, level, note = tc_gpu.clock_gate(clk, model, ref)
        out.append(dict(cell_id=cid, model=model, conc=int(g["conc"].iloc[0]),
                        ctx=int(g["ctx"].iloc[0]), clock_mhz=float(clk),
                        clock_ref_mhz=ref.get(model), clock_ratio=ratio,
                        clock_gate=level, note=note))

    t = pd.DataFrame(out).sort_values("clock_ratio")   # 全局最差的排前面
    t.to_csv(cfg.RESULTS_DIR / "clock_audit.csv", index=False, encoding="utf-8")

    dist = t["clock_gate"].value_counts().to_dict()
    print(f"共 {len(t)} 格；参考值 {ref}")
    print(f"分布：{dist}\n")

    print("最低的 8 格：")
    print(f"  {'格':<32}{'并发':>4}{'时钟':>7}{'参考':>7}{'比值':>7}  判定")
    for _, r in t.head(8).iterrows():
        print(f"  {r['cell_id']:<32}{r['conc']:>4}{r['clock_mhz']:>7.0f}"
              f"{r['clock_ref_mhz']:>7.0f}{r['clock_ratio']:>7.3f}  {r['clock_gate']}")

    # ---- 弹性估计：用 c8 的两批对照（+5.2% 时钟 / -1.5% TPOT）----
    elasticity = 1.5 / 5.2
    bad = t[t["clock_gate"] == "bad"]
    summ = dict(clock_ref=ref, n_cells=int(len(t)), distribution=dist,
                elasticity_est=round(elasticity, 3),
                elasticity_source="c8 两批对照：时钟 +5.2% → TPOT −1.5%",
                n_bad=int(len(bad)),
                bad_cells=bad["cell_id"].tolist()[:20],
                note=("低时钟不足以否定数据：弹性仅约 0.3，"
                      "6% 的时钟差只对应 ~2% 的 TPOT 差；"
                      "但跨批次 6% 的实测差提示还有别的因素（见方差分析 §5.7）"))
    (cfg.RESULTS_DIR / "clock_audit.json").write_text(
        json.dumps(summ, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n弹性估计 ≈ {elasticity:.2f}（时钟变化 1% → TPOT 变化 {elasticity:.2f}%）")
    print(f"被标 bad 的格：{len(bad)} 个")
    print("→ results/clock_audit.csv / clock_audit.json")


if __name__ == "__main__":
    main()

