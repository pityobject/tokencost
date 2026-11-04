# -*- coding: utf-8 -*-
"""
回答："背景程序会不会污染数据、要不要重测？"

用**已记录的环境指纹**做量化检验：
  若真的被后台程序干扰，应出现两种可观测信号——
  ① SM 时钟 / 功耗偏低（被抢 SMs）
  ② 残差与这些指纹相关（越热的格子误差越大）

如果两者都不存在，说明"没有污染的可观测痕迹"，不需要重测。
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import tc_config as cfg

sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def main():
    # ---- 最近的背景记录 ----
    rows = [json.loads(l) for l in (cfg.DATA_DIR / "raw_cells.jsonl").open(encoding="utf-8") if l.strip()]
    has = [r for r in rows if r.get("status") == "ok" and r.get("bg_level")]
    print("=== 最近的背景记录 ===")
    for r in has[-3:]:
        print(f"  {r['cell_id']:<28} bg={r['bg_level']:<7} "
              f"mem={r.get('bg_mem_mb')} util={r.get('bg_util_p95')} "
              f"| {str(r.get('bg_notes'))[:40]}")

    # ---- 环境指纹 vs 残差 ----
    s = pd.read_csv(cfg.DATA_DIR / "summary_steady.csv")
    s = s[s["cell_id"].str.contains("_p", regex=False) == False]
    fj = json.loads((cfg.RESULTS_DIR / "fit.json").read_text(encoding="utf-8"))
    params = dict(bw_gbs=fj["bw"], peak_tflops=fj["peak"],
                  peak_prefill_tflops=fj.get("peak_prefill", fj["peak"]),
                  c_srv_ms=fj["c_srv"], b_sat=fj["b_sat"], ovh_ttft_ms=fj["ovh_ttft"])

    import tc_model as tm
    s = s.copy()
    s["prompt_tokens"] = s["prompt_tokens_median"]
    s["ctx_avg_tokens"] = s["prompt_tokens"] + cfg.GEN_TOKENS / 2
    errs = []
    for _, r in s.iterrows():
        pr = tm.predict(params, r)
        errs.append(abs(np.log(pr["pred_tpot_ms"] / r["tpot_p50_ms"])))
    s["abs_err_tpot"] = errs

    print("\n=== 环境指纹 vs 误差（若被后台干扰，两者应相关）===")
    for col in ("clock_sm_median", "power_w_median", "temp_c_median"):
        if col not in s.columns:
            continue
        x = pd.to_numeric(s[col], errors="coerce")
        y = s["abs_err_tpot"]
        m = x.notna() & y.notna()
        if m.sum() < 10:
            continue
        r = float(np.corrcoef(x[m], y[m])[0, 1])
        print(f"  {col:<18} 覆盖 {m.sum():>3} 格  与 |TPOT 残差| 的相关系数 r = {r:+.3f}")

    # 【必须分模型看】时钟与误差的相关性是混淆的：7B 本身跑得更慢（时钟略低）
    # 且模型对 7B 的高并发拟合更差——那是模型局限，不是污染。
    print("\n=== 分模型核对（去掉模型间的混淆）===")
    for m, g in s.groupby("model"):
        x = pd.to_numeric(g["clock_sm_median"], errors="coerce")
        y = g["abs_err_tpot"]
        k = x.notna() & y.notna()
        if k.sum() < 6:
            continue
        r = float(np.corrcoef(x[k], y[k])[0, 1])
        print(f"  {m:<10} {k.sum():>3} 格  clock {x[k].min():.0f}–{x[k].max():.0f} MHz  "
              f"|TPOT 残差| 中位 {y[k].median()*100:>4.1f}%  与时钟相关 r={r:+.2f}")

    # 【交替解释】时钟与"并发"混淆：高并发时 GPU 更忙、更容易撞功耗墙而略降频，
    # 而模型在高并发下本来就偏得多。所以相关 ≠ 污染，必须把这层拆开看。
    print("\n=== 时钟 vs 并发（3B，检验上面那层混淆）===")
    g3 = s[s["model"] == "3B"]
    by = g3.groupby("conc").agg(clock=("clock_sm_median", "median"),
                                err=("abs_err_tpot", "median"), n=("cell_id", "size"))
    for c, r in by.iterrows():
        print(f"  并发 {int(c):>3}  {int(r['n']):>2} 格  clock {r['clock']:.0f} MHz  "
              f"|残差| {r['err']*100:>4.1f}%")

    print("\n=== 时钟最低的 5 格（降频嫌疑）===")
    if "clock_sm_median" in s.columns:
        t = s.sort_values("clock_sm_median").head(5)
        for _, r in t.iterrows():
            print(f"  {r['cell_id']:<32} clock={r['clock_sm_median']:.0f} MHz "
                  f"power={r['power_w_median']:.1f} W  |TPOT 残差|={r['abs_err_tpot']*100:.1f}%")

    # ---- 降频闸门是否触发过 ----
    low = s[(pd.to_numeric(s["clock_sm_median"], errors="coerce") < cfg.MIN_SM_CLOCK_MHZ)
            & (pd.to_numeric(s["power_w_median"], errors="coerce") < cfg.MIN_POWER_W)]
    print(f"\n=== 触发『时钟低 + 功耗低』闸门的格：{len(low)} 个"
          f"（阈值 {cfg.MIN_SM_CLOCK_MHZ} MHz / {cfg.MIN_POWER_W} W）===")


if __name__ == "__main__":
    main()



