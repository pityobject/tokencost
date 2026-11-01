"""
tc_analyze.py —— 把原始数据汇总成表，并出第一批图。

做三件事：
  1. 读 data/raw_cells.jsonl（每行是一格的一轮），按格取中位数 → data/summary.csv
  2. 打印几张关键表（并发-吞吐、并发-成本、模型-速度）
  3. 出 5 张图 → figures/（图 1–4 + 图 5 Pareto 前沿）

用法：
  & "D:\\llm\\venv\\Scripts\\python.exe" tc_analyze.py
"""

import json

import matplotlib
matplotlib.use("Agg")           # 不弹窗，直接存文件（服务器/无人值守更稳）
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import tc_config as cfg

# 让 matplotlib 能显示中文，否则中文全变成方框
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

METRICS = ["ttft_p50_ms", "ttft_p90_ms", "ttft_cvar90_ms",
           "tpot_p50_ms", "tpot_p90_ms", "tpot_cvar90_ms",
           "total_p50_ms", "throughput_tps", "cost_gpu_s_per_1m",
           "energy_j_per_1m", "power_w_median",
           "n_out_median", "prompt_tokens_median", "temp_c_median", "clock_sm_median"]


def load_raw():
    path = cfg.DATA_DIR / "raw_cells.jsonl"
    if not path.exists():
        raise SystemExit(f"还没有数据：{path}\n先跑 tc_run_grid.py")
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    df = pd.DataFrame(rows)

    # 能效轴：每百万 token 消耗多少焦耳。
    # 用"中位功耗 × 测量窗口时长 ÷ 生成 token 数"估算；功耗是采样中位数，
    # 属于粗估（没有对功率做积分），报告里要注明口径。
    if {"power_w_median", "measure_span_s", "total_tokens"} <= set(df.columns):
        df["energy_j_per_1m"] = (df["power_w_median"] * df["measure_span_s"]
                                 / df["total_tokens"].clip(lower=1) * 1e6)

    # 同一格可能因为断点续跑出现重复轮次，先只保留前 REPEATS 轮
    df = (df.sort_values("repeat")
            .groupby(["cell_id", "repeat"], as_index=False).last())

    ok = df[df["status"] == "ok"].copy()
    failed = df[df["status"] != "ok"].copy()

    # 【regime 分区】llama.cpp 默认 --kv-offload：KV 放不下时会静默搬到主机内存，
    # 吞吐掉 3–4 倍却不报错。这类格与"KV 全在显存"的格不是同一个物理系统，
    # 必须分开统计，否则会系统性地拉低带宽估计。
    def _regime(row):
        return "vram" if cfg.kv_fits_vram((row["model"], row["quant"]),
                                          int(row["ctx"]), int(row["conc"])) else "kv_offload"

    ok["regime"] = [_regime(r) for _, r in ok.iterrows()]
    if failed is not None and len(failed):
        failed["regime"] = [_regime(r) for _, r in failed.iterrows()]
    n_off = int((ok["regime"] == "kv_offload").sum())
    if n_off:
        print(f"[分区] 有 {n_off} 行属于 KV 外溢 regime（KV 装不进 8 GB 显存，被 llama.cpp 静默搬到主机内存），"
              f"已从主统计中剔除，单独在表 4 展示")
    ok = ok[ok["regime"] == "vram"].copy()

    # 测量有效性：时钟与功耗双低 = 降频，数据作废（环境问题，不是物理发现）
    if {"clock_sm_median", "power_w_median"} <= set(ok.columns):
        invalid = ((ok["clock_sm_median"] < cfg.MIN_SM_CLOCK_MHZ)
                   & (ok["power_w_median"] < cfg.MIN_POWER_W))
        n_inv = int(invalid.sum())
        if n_inv:
            print(f"[有效性] {n_inv} 行在降频状态下测得，已剔除："
                  f"{sorted(set(ok.loc[invalid, 'cell_id']))[:6]} ...")
        ok = ok[~invalid].copy()
    return ok, failed


def summarize(ok):
    """每一格：把 REPEATS 轮的中位数当成这一格的指标。

    **必须按 cell_id 分组，不能按 (模型,量化,ctx,并发) 重建**——
    波次对照实验（wave 预设）正是"同配置、不同请求数"，四格的三维度完全相同，
    按维度分组会把它们合并成一行，n_repeats 变成 12，指标全部失真。
    """
    use = [m for m in METRICS if m in ok.columns]
    keys = ["cell_id"] if "cell_id" in ok.columns else ["model", "quant", "ctx", "conc"]
    g = ok.groupby(keys, as_index=False)[use].median()
    # 维度列不参与聚合，单独取回（同一 cell_id 下它们恒定）
    dims = ok.groupby(keys, as_index=False)[["model", "quant", "ctx", "conc"]].first()
    g = dims.merge(g, on=keys, how="left")
    g["n_repeats"] = ok.groupby(keys).size().values
    order = {"0.5B": 0, "1.5B": 1, "3B": 2, "7B": 3}
    g["_o"] = g["model"].map(order).fillna(9)
    g = g.sort_values(["_o", "quant", "ctx", "conc"]).drop(columns="_o")
    return g


def print_tables(g, failed):
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 30)

    print("\n=== 表 1：全部可行格 ===")
    cols = ["cell_id", "conc", "ttft_p50_ms", "tpot_p50_ms", "throughput_tps",
            "cost_gpu_s_per_1m", "temp_c_median", "clock_sm_median", "n_repeats"]
    print(g[cols].to_string(index=False, float_format=lambda x: f"{x:9.1f}"))

    print("\n=== 表 2：并发对成本的边际收益（每个模型×上下文看一行） ===")
    for (m, q, c), sub in g.groupby(["model", "quant", "ctx"]):
        sub = sub.sort_values("conc")
        if len(sub) < 2:
            continue
        first, last = sub.iloc[0], sub.iloc[-1]
        gain = first["cost_gpu_s_per_1m"] / last["cost_gpu_s_per_1m"]
        print(f"  {m:>4} {q:<7} ctx={c:<5} 并发 {int(first['conc'])}→{int(last['conc'])}："
              f"成本 {first['cost_gpu_s_per_1m']:.0f} → {last['cost_gpu_s_per_1m']:.0f} "
              f"（降为 1/{gain:.2f}），TPOT {first['tpot_p50_ms']:.1f} → {last['tpot_p50_ms']:.1f} ms")

    if len(failed):
        print("\n=== 表 3：跑不动的格（真实 OOM 的证据） ===")
        print(failed[["cell_id", "vram_est_gb", "error"]].head(20).to_string(index=False))

    show_efficiency(g)
    show_kv_cliff()


def show_efficiency(g):
    """表 4：服务路径 vs 纯后端上限（"实现效率"）。

    这一节回答一个面试必问的问题：「你的实现离硬件上限还有多远？差在哪？」
    数据来源：data/backend_ref.csv（由 tc_backend_ref.py 用 llama-batched-bench 生成）。
    """
    ref_path = cfg.DATA_DIR / "backend_ref.csv"
    if not ref_path.exists():
        print("\n（没有 data/backend_ref.csv，跳过实现效率对比。"
              "想补：python tc_backend_ref.py）")
        return

    ref = pd.read_csv(ref_path)
    ref = ref.rename(columns={"model": "model", "quant": "quant",
                              "batch": "conc", "backend_tg_tps": "backend_tps"})
    ref = ref[["model", "quant", "conc", "backend_tps"]]

    # 同一个 (模型,量化,并发) 在 ctx 512 下的 server 吞吐最有可比性（短上下文）
    base = g[(g["ctx"] == g["ctx"].min())][["model", "quant", "conc", "throughput_tps"]]
    m = base.merge(ref, on=["model", "quant", "conc"], how="inner")
    if m.empty:
        print("\n（backend_ref.csv 和当前数据的模型/并发对不上，跳过对比）")
        return

    m["efficiency_pct"] = m["throughput_tps"] / m["backend_tps"] * 100
    print(f"\n=== 表 4：服务路径 vs 纯后端上限（ctx={g['ctx'].min()}） ===")
    print(f"{'模型':>6} {'量化':<7} {'并发':>5} {'server t/s':>11} {'后端上限':>10} {'达成率':>8}")
    for _, r in m.sort_values(["model", "quant", "conc"]).iterrows():
        print(f"{r['model']:>6} {r['quant']:<7} {int(r['conc']):>5} "
              f"{r['throughput_tps']:>11.0f} {r['backend_tps']:>10.0f} {r['efficiency_pct']:>7.0f}%")

    worst = m.loc[m["efficiency_pct"].idxmin()]
    print(f"\n最低达成率：{worst['efficiency_pct']:.0f}%（{worst['model']}-{worst['quant']} "
          f"并发 {int(worst['conc'])}）")
    if worst["efficiency_pct"] < 60:
        print("解读：高并发下 llama-server 的 HTTP/采样/推流路径成为瓶颈，")
        print("      显卡在空转（可对照数据里的 util_pct 与 power_w）。")
        print("      这句话可以直接写进报告的『实现效率』一节——它是你的发现，不是缺陷。")


def plot_all(g):
    cfg.FIG_DIR.mkdir(parents=True, exist_ok=True)
    g = g.copy()
    g["combo"] = g["model"] + "-" + g["quant"]

    # 图 1：并发 vs 吞吐（每个上下文一张小图）
    ctxs = sorted(g["ctx"].unique())
    fig, axes = plt.subplots(1, len(ctxs), figsize=(5.2 * len(ctxs), 4.2), squeeze=False)
    for ax, ctx in zip(axes[0], ctxs):
        sub = g[g["ctx"] == ctx]
        for combo, s in sub.groupby("combo"):
            s = s.sort_values("conc")
            ax.plot(s["conc"], s["throughput_tps"], "o-", label=combo)
        ax.set_xscale("log", base=2)
        ax.set_xlabel("并发请求数")
        ax.set_ylabel("吞吐 tokens/s")
        ax.set_title(f"上下文上限 {ctx}")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle("并发 vs 吞吐：越往右越接近机器的批处理上限")
    fig.tight_layout()
    fig.savefig(cfg.FIG_DIR / "fig1_throughput_vs_concurrency.png", dpi=150)
    plt.close(fig)

    # 图 2：并发 vs 单位成本（每百万 token 占用多少 GPU 秒）
    fig, ax = plt.subplots(figsize=(7.5, 5))
    for combo, s in g[g["ctx"] == ctxs[0]].groupby("combo"):
        s = s.sort_values("conc")
        ax.plot(s["conc"], s["cost_gpu_s_per_1m"], "o-", label=combo)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("并发请求数")
    ax.set_ylabel("成本（GPU 秒 / 百万 token，越低越好）")
    ax.set_title(f"并发 vs 单位成本（上下文 {ctxs[0]}）")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(cfg.FIG_DIR / "fig2_cost_vs_concurrency.png", dpi=150)
    plt.close(fig)

    # 图 3：单并发下，TPOT 与模型字节数的关系（应接近一条过原点的直线 → 证明访存受限）
    sub = g[(g["conc"] == 1) & (g["ctx"] == ctxs[0])].copy()
    if len(sub) >= 2:
        w_gb = [cfg.MODEL_SPECS[(m, q)]["weight_bytes"] / 1e9 for m, q in zip(sub["model"], sub["quant"])]
        t_ms = sub["tpot_p50_ms"].values
        fig, ax = plt.subplots(figsize=(6.5, 4.8))
        ax.scatter(w_gb, t_ms, s=70, zorder=3, label="实测")
        # 过原点直线拟合：斜率 = 1/带宽
        k = float(np.dot(w_gb, t_ms) / np.dot(w_gb, w_gb))
        xs = np.linspace(0, max(w_gb) * 1.1, 50)
        ax.plot(xs, k * xs, "--", color="gray",
                label=f"过原点拟合：{1000/k:.0f} GB/s（有效带宽）")
        for x, y, m, q in zip(w_gb, t_ms, sub["model"], sub["quant"]):
            ax.annotate(f"{m}-{q}", (x, y), textcoords="offset points", xytext=(6, -10), fontsize=8)
        ax.set_xlabel("权重文件大小（GB）")
        ax.set_ylabel("TPOT（毫秒/token）")
        ax.set_title("单并发：出字时间正比于权重大小 → 说明瓶颈是读显存，不是算力")
        ax.legend()
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(cfg.FIG_DIR / "fig3_tpot_vs_weights.png", dpi=150)
        plt.close(fig)
        print(f"\n由图 3 过原点拟合得到的有效带宽 ≈ {1000/k:.0f} GB/s")

    print(f"\n图片已保存到 {cfg.FIG_DIR}")


def plot_pareto(g):
    """图 5：Pareto 前沿（报告的核心图）。

    横轴 = 尾部出字延迟（CVaR90，越小越好）
    纵轴 = 单位成本（越小越好）
    => 真正值得选的配置，只落在"左下角那条边界"上，这叫 Pareto 前沿。
    右上方向的所有点都被"更便宜且更快"的配置支配，没有选择价值。
    """
    cfg.FIG_DIR.mkdir(parents=True, exist_ok=True)
    ycol = "tpot_cvar90_ms" if "tpot_cvar90_ms" in g.columns else "tpot_p50_ms"
    d = g.dropna(subset=[ycol, "cost_gpu_s_per_1m"]).copy()
    if len(d) < 3:
        print("数据太少，跳过 Pareto 图")
        return

    # 质量表存在就按质量着色（第三根轴）
    quality = {}
    qpath = cfg.DATA_DIR / "quality.csv"
    if qpath.exists():
        q = pd.read_csv(qpath).dropna(subset=["acc"])
        quality = {(r["model"], r["quant"]): float(r["acc"]) for _, r in q.iterrows()}

    # 计算前沿之前先应用质量约束：质量不达标的点不参与"值不值得选"的比较
    if quality:
        d["acc"] = d.apply(lambda r: quality.get((r["model"], r["quant"]), float("nan")), axis=1)
        best_acc = max(quality.values())
        # 【第三轮审计修正】没做过质量评测的组合（acc 为 NaN）必须**排除**，不能放行。
        # 原来的写法 `acc.isna() | (...)` 把"未评测"当成"通过"，而 0.5B 最便宜又从未测过质量，
        # 结果 6 个 0.5B 格直接支配了整条 Pareto 前沿——图与"前沿全是 3B-Q4"的结论自相矛盾。
        # 正确做法：未评测 = 无法判断 = 不参与，并在图上标注被排除的数量。
        QUALITY_DROP = 0.06   # 与 tc_recommend 一致：≈ 一个置信区间半宽
        d["pass_q"] = d["acc"].notna() & (d["acc"] >= best_acc - QUALITY_DROP)
        n_unevaluated = int(d["acc"].isna().sum())
        if n_unevaluated:
            print(f"[质量] {n_unevaluated} 个格没有质量数据（未评测），已从 Pareto 比较中剔除；"
                  f"如需纳入请先跑 tc_eval.py")
    else:
        d["acc"] = float("nan")
        d["pass_q"] = True

    ok = d[d["pass_q"]].sort_values(ycol)
    front_pts, best = [], float("inf")
    for _, r in ok.iterrows():
        if r["cost_gpu_s_per_1m"] < best:
            best = r["cost_gpu_s_per_1m"]
            front_pts.append(r)
    front = pd.DataFrame(front_pts)

    fig, ax = plt.subplots(figsize=(9.0, 6.0))
    # ---- 图层顺序：剪枝（灰）→ 合格（彩色）→ 前沿（红线）----
    if quality:
        pruned = d[~d["pass_q"]]
        if len(pruned):
            ax.scatter(pruned[ycol], pruned["cost_gpu_s_per_1m"], s=16,
                       facecolors="none", edgecolors="#c4c4c4", linewidths=0.6,
                       zorder=1, label=f"被质量剪枝或未评测（{len(pruned)} 个）")

        # 着色口径：用**全局**质量范围（含被剪枝组合），让色标本身有信息量。
        # 若只用合格点的范围，而 6pp 底线只剩 7B 一个组合，色标会退化成单一颜色。
        cvals, clabel = ok["acc"], "质量（400 题库准确率）"
        v_lo = float(d["acc"].min()) if d["acc"].notna().any() else 0.0
        v_hi = float(d["acc"].max()) if d["acc"].notna().any() else 1.0
        sc = ax.scatter(ok[ycol], ok["cost_gpu_s_per_1m"], c=cvals, cmap="viridis",
                        vmin=v_lo, vmax=v_hi,
                        s=95, edgecolor="white", linewidth=0.8, zorder=3,
                        label=f"满足质量底线（{len(ok)} 个）")
        # 【第三轮复核 3.3】6pp 质量线之后，合格集只剩 7B 一个组合，
        # 16 个合格点的准确率全部相同 → 颜色维度没有任何信息量，色标反而误导。
        # 因此不再画 colorbar，改在图注里写明"合格集全部为 7B-Q4"。
        uniq = sorted({round(float(v), 3) for v in ok["acc"].dropna()})
        if len(uniq) > 1:
            cbar = fig.colorbar(sc, ax=ax, pad=0.02)
            cbar.set_label(clabel, fontsize=9)
        else:
            ax.text(0.99, 0.02, f"合格集全部为 {ok['model'].iloc[0]}-{ok['quant'].iloc[0]}"
                                f"（准确率 {uniq[0]*100:.1f}%）",
                    transform=ax.transAxes, ha="right", va="bottom",
                    fontsize=8, color="#666666")
    else:
        for combo, sub in d.assign(combo=d["model"] + "-" + d["quant"]).groupby("combo"):
            ax.scatter(sub[ycol], sub["cost_gpu_s_per_1m"], s=70, label=combo, zorder=3)

    ax.plot(front[ycol], front["cost_gpu_s_per_1m"], "r-o", lw=1.6, ms=6,
            zorder=4, label="Pareto 前沿")
    # 只标前沿的首尾两点，且强制落在坐标轴内，避免压图例/被裁切
    for i in [j for j in (0, len(front) - 1) if 0 <= j < len(front)]:
        r = front.iloc[i]
        ax.annotate(f"{r['model']}-{r['quant']} ctx{r['ctx']}/c{r['conc']}",
                    (r[ycol], r["cost_gpu_s_per_1m"]),
                    textcoords="offset points",
                    xytext=(8, 8) if i == 0 else (-10, -16),
                    ha="left" if i == 0 else "right",
                    fontsize=7, color="darkred",
                    annotation_clip=True, clip_on=True)

    ax.set_yscale("log")
    ax.set_xlabel("尾部出字延迟 CVaR90（毫秒，越小越好）")
    ax.set_ylabel("单位成本（GPU 秒 / 百万 token，越小越好）")
    if quality:
        ax.set_title(f"只有红色前沿上的配置值得选"
                     f"（合格 {int(d['pass_q'].sum())} / 剪枝 {int((~d['pass_q']).sum())}）")
    else:
        ax.set_title("可行配置的代价：只有红色前沿上的值得选")
    ax.grid(alpha=0.3, which="both")
    ax.margins(x=0.12, y=0.10)          # 留边距，标签不会被裁
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(cfg.FIG_DIR / "fig5_pareto_frontier.png", dpi=150)
    plt.close(fig)

    print(f"\nPareto 前沿上有 {len(front)} 个配置"
          f"（参与比较 {int(d['pass_q'].sum())} 个，被质量剪掉 {int((~d['pass_q']).sum())} 个）")
    print(front[["cell_id", ycol, "cost_gpu_s_per_1m"]]
          .to_string(index=False, float_format=lambda x: f"{x:9.1f}"))
    print("报告图注建议：「在满足质量底线的 __ 个配置中，仅 __ 个位于 Pareto 前沿；"
          "其余均被支配（存在更快且更便宜的替代方案）。」")



def show_kv_cliff():
    """表 5：显存悬崖——KV 溢出到主机内存后吞吐掉多少。

    这是"显存约束不是硬墙而是性能悬崖"的直接证据，也是报告里的一个独立卖点。
    """
    path = cfg.DATA_DIR / "raw_cells.jsonl"
    if not path.exists():
        return
    import json as _json
    rows = [_json.loads(l) for l in path.open(encoding="utf-8") if l.strip()]
    ok = [r for r in rows if r.get("status") == "ok"]

    # 同一 (模型, 并发) 下，比较 ctx 512（KV 在显存）与最大 ctx（可能外溢）
    groups = {}
    for r in ok:
        groups.setdefault((r["model"], r["quant"], r["conc"]), []).append(r)

    print("\n=== 表 5：显存悬崖（同模型、同并发，只改上下文上限） ===")
    print(f"{'模型':>6} {'量化':<7} {'并发':>4} {'ctx512吞吐':>10} {'最大ctx':>7} "
          f"{'该ctx吞吐':>10} {'倍数':>6} {'显存MB':>7}")
    shown = 0
    for (m, q, c), rs in sorted(groups.items()):
        by_ctx = {}
        for r in rs:
            by_ctx.setdefault(r["ctx"], []).append(r["throughput_tps"])
        if 512 not in by_ctx or len(by_ctx) < 2:
            continue
        base = sorted(by_ctx[512])[len(by_ctx[512]) // 2]
        mx = max(by_ctx)
        worst = sorted(by_ctx[mx])[len(by_ctx[mx]) // 2]
        mem = sorted(r["mem_used_mb_median"] for r in rs if r["ctx"] == mx)[0] \
            if any(r["ctx"] == mx for r in rs) else float("nan")
        if cfg.kv_fits_vram((m, q), int(mx), int(c)):
            continue                      # KV 装得下，不算悬崖
        shown += 1
        print(f"{m:>6} {q:<7} {c:>4} {base:>10.0f} {mx:>7} {worst:>10.0f} "
              f"{base/worst:>5.1f}x {mem:>7.0f}")
    if shown:
        print("  解读：显存装不下 KV 时，llama.cpp 会静默把 KV 放到主机内存，")
        print("        吞吐断崖式下跌且**不报错**。所以显存边界不是硬墙，是性能悬崖。")


def main():
    cfg.ensure_dirs()
    ok, failed = load_raw()
    print(f"读到 {len(ok)} 条成功记录，{len(failed)} 条失败记录")
    if not len(ok):
        raise SystemExit("没有任何成功记录，先检查 tc_run_cell.py 能不能单格跑通。")

    g = summarize(ok)
    out = cfg.DATA_DIR / "summary.csv"
    g.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"汇总表已写入 {out}")

    print_tables(g, failed)
    plot_all(g)
    plot_pareto(g)


if __name__ == "__main__":
    main()

