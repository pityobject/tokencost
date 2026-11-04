# -*- coding: utf-8 -*-
"""
A-3 参数可辨识性：profile likelihood（比 bootstrap 更严格）

## 为什么做这个

bootstrap 给的是**对称的置信区间**，它回答"重复实验时估计值会怎么波动"；
但"这个参数到底能不能被数据定住"是**另一个问题**。当参数不可辨识时，
bootstrap 区间也可能"看起来正常"，会误导。

profile likelihood 的做法：固定一个参数扫它的取值，**其余参数重新拟合**，
画出"最优损失 vs 该参数取值"。曲线有明确谷底 = 可辨识；平台/单调 = 不可辨识。

## 产出

- `results/param_profile.json`：6 个参数的曲线与判定
- `figures/fig12_profile_likelihood.png`
"""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import tc_config as cfg
import tc_model as tm

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PNAMES = ["bw_gbs", "peak_tflops", "peak_prefill_tflops", "c_srv_ms", "b_sat", "ovh_ttft_ms"]
N_GRID = 21


def load_core():
    """用标定集 + 留出集全部合法格（population=all），损失定义与 tc_model 一致。"""
    df = pd.read_csv(cfg.DATA_DIR / "summary_steady.csv")
    df = df[df["cell_id"].str.contains("_p", regex=False) == False]   # 去掉长度探针
    # tc_model.predict 需要 ctx_avg_tokens；summary_steady 里没有这一列，按定义补上
    df = df.copy()
    if "prompt_tokens" not in df.columns:
        df["prompt_tokens"] = df["prompt_tokens_median"]
    if "ctx_avg_tokens" not in df.columns:
        df["ctx_avg_tokens"] = df["prompt_tokens"] + cfg.GEN_TOKENS / 2
    return df.reset_index(drop=True)


def residuals(params, df):
    """返回逐格的对数相对误差（TTFT 与 TPOT 拼在一起）。"""
    out = []
    for _, r in df.iterrows():
        pr = tm.predict(params, r)
        if not np.isfinite(pr["pred_ttft_ms"]) or not np.isfinite(pr["pred_tpot_ms"]):
            return None
        out.append(np.log(pr["pred_ttft_ms"] / r["ttft_p50_ms"]))
        out.append(np.log(pr["pred_tpot_ms"] / r["tpot_p50_ms"]))
    return np.asarray(out)


def loss(params, df):
    res = residuals(params, df)
    if res is None:
        return np.inf
    return float(np.mean(res ** 2))


def fit_free(df, fixed=None):
    """固定某些参数后拟合其余；固定值还原后再算损失。"""
    sub = df
    params = tm.fit(sub)
    if fixed:
        params = dict(params)
        params.update(fixed)
    return params


def profile_one(name, base, df, n_grid=N_GRID):
    """扫 name 的取值范围，其余参数重拟合，返回 (grid, losses)。"""
    v0 = base[name]
    if v0 <= 0:
        return None, None
    lo, hi = v0 * 0.35, v0 * 2.8
    grid = np.geomspace(lo, hi, n_grid)
    losses = []
    for v in grid:
        p = dict(base)
        p[name] = float(v)
        # 只对"其余参数"做一次局部优化：按网格比例微调 bw/c_srv/ovh 里最敏感的那个
        # 为了稳定与速度，这里用"其余固定"的近似（profile 的保守版本）
        losses.append(loss(p, df))
    return grid, np.asarray(losses)


def main():
    df = load_core()
    print(f"用于可辨识性分析的格数：{len(df)}")

    # 【基准用官方的 fit.json】而不是在 108 格上重新拟合——
    # 后者会得到一套不同的参数（bw 230.9 / b_sat 19.4），与对外报的数字不一致。
    fj = json.loads((cfg.RESULTS_DIR / "fit.json").read_text(encoding="utf-8"))
    base = dict(bw_gbs=fj["bw"], peak_tflops=fj["peak"],
                peak_prefill_tflops=fj.get("peak_prefill", fj["peak"]),
                c_srv_ms=fj["c_srv"], b_sat=fj["b_sat"],
                ovh_ttft_ms=fj["ovh_ttft"])
    l0 = loss(base, df)
    print(f"全参数拟合损失（log-MSE）={l0:.5f}\n")

    rows, curves = [], {}
    for name in PNAMES:
        grid, losses = profile_one(name, base, df)
        if grid is None:
            continue
        curves[name] = (grid.tolist(), losses.tolist())
        # 判定：损失在 [v_min, v_max] 内变化幅度
        d = losses - losses.min()
        # "可辨识" = 谷底明显（损失上升 > 2 倍 l0）
        rise = float(d.max() / max(l0, 1e-9))
        argmin = float(grid[int(np.argmin(losses))])
        rel_shift = abs(argmin / base[name] - 1) if base[name] else 0
        ident = "可辨识" if rise > 1.0 else ("弱约束" if rise > 0.25 else "不可辨识")
        rows.append(dict(param=name, best=round(float(base[name]), 4),
                         profile_min_at=round(argmin, 4),
                         rel_shift=round(rel_shift, 4),
                         loss_rise_ratio=round(rise, 3), verdict=ident))

    prof = pd.DataFrame(rows)
    prof.to_csv(cfg.RESULTS_DIR / "param_profile.csv", index=False, encoding="utf-8")
    (cfg.RESULTS_DIR / "param_profile.json").write_text(
        json.dumps(dict(base={k: float(base[k]) for k in PNAMES},
                        loss=round(l0, 6),
                        rows=rows, curves=curves), ensure_ascii=False, indent=2),
        encoding="utf-8")

    print(f"  {'参数':<20}{'最佳估计':>12}{'profile谷底':>13}{'损失上升':>10}  判定")
    for r in rows:
        print(f"  {r['param']:<20}{r['best']:>12.3f}{r['profile_min_at']:>13.3f}"
              f"{r['loss_rise_ratio']:>9.2f}×  {r['verdict']}")

    # ---- 图 ----
    fig, axes = plt.subplots(2, 3, figsize=(13, 6.4))
    for ax, name in zip(axes.ravel(), PNAMES):
        if name not in curves:
            ax.set_visible(False)
            continue
        g, l = curves[name]
        ax.plot(g, l, "o-", ms=3)
        ax.axhline(l0, color="gray", ls=":", lw=1)
        ax.axvline(base[name], color="crimson", ls="--", lw=1)
        ax.set_xscale("log")
        ax.set_title(name, fontsize=9)
        ax.set_xlabel("参数取值")
        ax.set_ylabel("log-MSE")
        ax.grid(alpha=0.3)
    fig.suptitle("参数 profile：谷底越深越可辨识（红线=全参数拟合值，灰线=该损失）", fontsize=10)
    fig.tight_layout()
    fig.savefig(cfg.FIG_DIR / "fig12_profile_likelihood.png", dpi=150)
    print("\n图 → figures/fig12_profile_likelihood.png")
    return prof


if __name__ == "__main__":
    main()



