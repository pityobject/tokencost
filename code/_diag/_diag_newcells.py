# -*- coding: utf-8 -*-
"""诊断：新增的 110 个留出格里，误差集中在哪个维度。"""
import pathlib
import sys

import pandas as pd

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = pathlib.Path(r"C:\Users\王钟垚\Desktop\find_career\tokencost")
df = pd.read_csv(ROOT / "results/predictions.csv")

print(f"共 {len(df)} 格（标定 {(df['group']=='标定').sum()} / 留出 {(df['group']=='留出').sum()}）\n")

d = df.copy()
d["err_pct"] = (d["ratio_cost"] - 1) * 100

print("=== 误差最大的 12 格 ===")
print(f"{'cell':<26}{'实测成本':>9}{'预测成本':>9}{'误差%':>7}")
for _, r in d.reindex(d["err_pct"].abs().sort_values(ascending=False).index).head(12).iterrows():
    print(f"{r['cell_id']:<26}{r['meas_cost']:>9.0f}{r['pred_cost_gpu_s_per_1m']:>9.0f}{r['err_pct']:>7.0f}")

print("\n=== 按并发看误差 ===")
for c, sub in d.groupby("conc"):
    print(f"  并发 {c:>2}  n={len(sub):>3}  中位误差 {sub['err_pct'].median():+7.1f}%  "
          f"|误差| 中位 {sub['err_pct'].abs().median():6.1f}%  最大 {sub['err_pct'].abs().max():6.1f}%")

print("\n=== 按配置的 ctx 看误差（注意：模型只用了实际上下文 320，没用配置的 ctx）===")
for c, sub in d.groupby("ctx"):
    print(f"  ctx {c:>5}  n={len(sub):>3}  中位误差 {sub['err_pct'].median():+7.1f}%  "
          f"|误差| 中位 {sub['err_pct'].abs().median():6.1f}%")

print("\n=== 按模型看误差 ===")
for m, sub in d.groupby("model"):
    print(f"  {m:>5}  n={len(sub):>3}  中位误差 {sub['err_pct'].median():+7.1f}%  "
          f"|误差| 中位 {sub['err_pct'].abs().median():6.1f}%")

print("\n=== 交叉：并发 × ctx 的 |误差| 中位数 ===")
piv = d.pivot_table(index="conc", columns="ctx", values="err_pct",
                    aggfunc=lambda s: s.abs().median())
print(piv.round(1).to_string())
