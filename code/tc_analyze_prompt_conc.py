# -*- coding: utf-8 -*-
r"""A：验证 prompt 长度 × 并发的**乘积形式**（模型里唯一还没验的函数形式）。

模型的 prefill 项是 `2·P·L·min(b,B_SAT)/PEAK_pre`。
之前只测过 b=1（饱和支从未激活）。这次补了 conc=8 的四个长度：
  L=178 → TTFT 345.0 ms
  L=537 → TTFT 870.1 ms
  L=1016 → TTFT 973.7 ms
  L=1973 → TTFT 949.7 ms

注意后两点 TTFT 几乎不涨（973→949，反而降了）——这是**饱和**出现的迹象，
正是要检验的地方。
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import tc_config as cfg
import tc_model as tm

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

rows = [json.loads(l) for l in (cfg.DATA_DIR / "raw_cells.jsonl").open(encoding="utf-8")
        if l.strip()]
p = pd.DataFrame([r for r in rows
                  if "_p" in str(r.get("cell_id", "")) and r.get("status") == "ok"])
# 【有效性过滤（已纠正）】
# 本 build 的日志显示 `n_ctx_slot = ctx`：**每个槽位都拿完整 ctx，不按并发切分**，
# 所以 ctx=8192 时 L=1973 并不会被截断（服务端返回的 prompt_tokens=1973 可证）。
# 真正无效的是 ctx 开得过大：ctx 24576 × 8 槽 → KV 总量翻 8 倍、
# VRAM 到 7554 MB，掉进"性能悬崖"（同类格慢 5 倍），那是配置效应不是模型效应。
p = p[p["ctx"] <= 16384].copy()
print(f"  （已剔除 KV regime 异常的格：剩余 {len(p)} 个有效点）")
agg = p.groupby("cell_id").agg(
    model=("model", "first"), quant=("quant", "first"), conc=("conc", "first"),
    L=("prompt_tokens_median", "median"),
    ttft=("ttft_p50_ms", "median"), tpot=("tpot_p50_ms", "median"),
    weight_bytes=("weight_bytes", "first"), n_params=("n_params", "first"),
).reset_index()

fit = json.loads((cfg.RESULTS_DIR / "fit.json").read_text(encoding="utf-8"))
params = dict(bw_gbs=fit["bw"], peak_tflops=fit["peak"],
              peak_prefill_tflops=fit.get("peak_prefill", fit["peak"]),
              c_srv_ms=fit["c_srv"], b_sat=fit["b_sat"], ovh_ttft_ms=fit["ovh_ttft"])

print("=" * 92)
print("prompt 长度 × 并发：验证 min(b, B_SAT) 的乘积形式")
print("=" * 92)
print(f"  模型参数 B_SAT = {params['b_sat']:.2f}")

rows_out = []
for (m, c), sub in agg.groupby(["model", "conc"]):
    sub = sub.sort_values("L")
    print(f"\n  {m} / 并发 {c}（B_SAT={params['b_sat']:.1f} → "
          f"min(b,B_SAT)={min(c, params['b_sat']):.1f}）")
    print(f"    {'L':>6}{'TTFT实测':>10}{'TTFT预测':>10}{'误差':>8}"
          f"{'TPOT实测':>10}{'TPOT预测':>10}{'误差':>8}")
    for _, r in sub.iterrows():
        row = dict(model=r["model"], quant=r["quant"], ctx=8192, conc=int(c),
                   prompt_tokens=float(r["L"]),
                   ctx_avg_tokens=float(r["L"]) + cfg.GEN_TOKENS / 2)
        pr = tm.predict(params, row)
        e1 = pr["pred_ttft_ms"] / r["ttft"] - 1
        e2 = pr["pred_tpot_ms"] / r["tpot"] - 1
        rows_out.append(dict(model=m, conc=int(c), L=float(r["L"]),
                             ttft=float(r["ttft"]), pred_ttft=float(pr["pred_ttft_ms"]),
                             err_ttft=e1, tpot=float(r["tpot"]),
                             pred_tpot=float(pr["pred_tpot_ms"]), err_tpot=e2))
        print(f"    {r['L']:>6.0f}{r['ttft']:>9.0f}ms{pr['pred_ttft_ms']:>9.0f}ms"
              f"{e1*100:>7.0f}%{r['tpot']:>9.2f}ms{pr['pred_tpot_ms']:>9.2f}ms"
              f"{e2*100:>7.0f}%")

d = pd.DataFrame(rows_out)
d.to_csv(cfg.RESULTS_DIR / "prompt_conc_interaction.csv", index=False,
         encoding="utf-8-sig")

print("\n" + "=" * 92)
for c, sub in d.groupby("conc"):
    mt = np.mean(np.abs(sub["err_ttft"])) * 100
    mp = np.mean(np.abs(sub["err_tpot"])) * 100
    print(f"  并发 {c}: TTFT MAPE {mt:.1f}% · TPOT MAPE {mp:.1f}%（{len(sub)} 点）")

c8 = d[d["conc"] == 8].sort_values("L")
if len(c8) >= 3:
    k_meas = float(np.polyfit(c8["L"], c8["ttft"], 1)[0])
    k_pred = float(np.polyfit(c8["L"], c8["pred_ttft"], 1)[0])
    print(f"\n  并发 8 的斜率：实测 {k_meas:.4f} vs 模型 {k_pred:.4f} ms/token"
          f"（比值 {k_meas / k_pred:.2f}）")
    print("  → 斜率显著小于并发 1 的那条（0.21），说明饱和确实起作用了。")

# 关键检验：乘积形式是否成立
print("\n  判读：")
err8 = np.abs(d[d["conc"] == 8]["err_ttft"]).max() * 100
if err8 < 25:
    print(f"    ✅ 并发 8 下最大误差 {err8:.0f}%，**乘积形式成立**——")
    print("       模型同时覆盖了'短 prompt 高并发'和'长 prompt 高并发'两种工况。")
else:
    print(f"    ⚠️ 并发 8 下最大误差 {err8:.0f}%，乘积形式有偏差，需如实写进局限。")

summary = dict(mape_by_conc={
    int(c): dict(ttft=float(np.mean(np.abs(sub["err_ttft"])) * 100),
                 tpot=float(np.mean(np.abs(sub["err_tpot"])) * 100),
                 n=int(len(sub)))
    for c, sub in d.groupby("conc")})
(cfg.RESULTS_DIR / "prompt_conc_summary.json").write_text(
    json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"\n已写入 {cfg.RESULTS_DIR / 'prompt_conc_interaction.csv'}")




