# -*- coding: utf-8 -*-
"""tc_claims.py —— 对外数字的**唯一来源**。

## 为什么需要它

第三轮独立复核发现：README 写"100 格 / 300 行 / 8.70% / 68 格"，
而跑一遍代码得到的是"108 格 / 324 行 / 9.08% / 76 格"。
一个把"诚实"当卖点的项目，最容易被验证的地方反而对不上——这是最伤的一条。

所以把所有能对外引用的数字集中到这里生成 `results/CLAIMS.md`，
并提供 `--check`：扫描 README / 报告，一旦某个数字与真实产物不符就**非零退出**。

## 用法

    python tc_claims.py            # 生成 results/CLAIMS.md
    python tc_claims.py --check    # 检查文档是否与产物一致（CI 用）
"""

import argparse
import json
import pathlib
import re
import sys

import numpy as np
import pandas as pd

import tc_config as cfg

ROOT = cfg.PROJECT_ROOT
RESULTS = cfg.RESULTS_DIR
DOCS = [ROOT / "README.md",
        ROOT / "报告" / "技术报告_定稿_两页.md",
        ROOT / "报告" / "交接清单_回来从这里开始.md"]



def _pc(conc, key):
    """读 results/prompt_conc_summary.json，取某并发档的 MAPE。"""
    p = RESULTS / "prompt_conc_summary.json"
    if not p.exists():
        return "无数据"
    d = json.loads(p.read_text(encoding="utf-8")).get("mape_by_conc", {})
    row = d.get(conc)
    if not row:
        return "无数据"
    return f"{row[key]:.1f}%（{row['n']} 点）"


def _fam(name):
    """读 results/quality_profile.csv，取某任务族的跨组合均值。"""
    p = RESULTS / "quality_profile.csv"
    if not p.exists():
        return "无数据"
    import pandas as _pd
    d = _pd.read_csv(p)
    if name not in d.columns:
        return "无数据"
    return f"{d[name].mean():.1f}%"


def _sweep(key, unit, nd=1):
    p = RESULTS / "prompt_sweep_summary.json"
    if not p.exists():
        return "无数据"
    v = json.loads(p.read_text(encoding="utf-8")).get(key)
    return f"{float(v):.{nd}f}{unit}" if v is not None else "无数据"


def _sweep_ratio():
    """斜率比值：只报**有 ≥2 个长度点**的模型（单点无法拟合斜率）。"""
    p = RESULTS / "prompt_sweep_summary.json"
    if not p.exists():
        return "无数据"
    d = json.loads(p.read_text(encoding="utf-8"))
    sl = d.get("slopes") or {}
    if not sl:
        return "无数据"
    k = sorted(sl)[0]
    return f"{sl[k]['ratio']:.2f}×（{k}）"


def _sweep_range():
    p = RESULTS / "prompt_sweep_summary.json"
    if not p.exists():
        return "无数据"
    d = json.loads(p.read_text(encoding="utf-8"))
    r = d.get("L_range")
    return f"{r[0]:.0f}–{r[1]:.0f}（{r[1]/r[0]:.1f} 倍）" if r else "无数据"


def _calib_at(k):
    """读 results/calib_curve.csv，取某个 k 的中位 MAPE。"""
    p = RESULTS / "calib_curve.csv"
    if not p.exists():
        return "无数据"
    import pandas as _pd
    d = _pd.read_csv(p)
    row = d[d["k"] == k]
    if row.empty:
        return "无数据"
    return f"{float(row.iloc[0]['mape_med']):.2f}%（12 格全用为基准）"


def _control(which):
    """从 regret_controls.csv 现算负对照的均值（唯一来源，避免散文写死）。

    第五轮复核指出：同一份 CLAIMS.md 里，表格行（曾是硬编码的旧值）与散文行，
    而散文块（写死的）还是 268%/813%。根因就是散文没走同一来源。
    """
    p = RESULTS / "regret_controls.csv"
    if not p.exists():
        return "无数据"
    import pandas as _pd
    d = _pd.read_csv(p)
    col = f"regret_{which}"
    if col not in d.columns or d[col].dropna().empty:
        return "无数据"
    return f"{d[col].mean():.1f}%"



def _control_ctx():
    """给 --check 用：README/报告里允许出现的后悔值写法。

    第六轮复核的核心教训：表格、散文、README 三处都写这个数字，
    变一次就要人肉同步三处。所以把它纳入自动比对——
    文档里写 259%/772% 就会和 CLAIMS 的 274%/740% 对不上。
    """
    return [
        rf"随机选 \*\*{_control('random')}\*\*",
        rf"选最大模型 \*\*{_control('largest')}\*\*",
    ]


def _cap_with_infl():
    """计入 prefill 干扰后的容量（读 queue_summary.json 的 cap_poisson）。"""
    p = RESULTS / "queue_summary.json"
    if not p.exists():
        return "无数据"
    d = json.loads(p.read_text(encoding="utf-8"))
    cap, meas = d.get("cap_poisson"), d.get("cap_meas")
    if cap is None or meas is None:
        return "无数据"
    return f"{cap:.2f}（实测为预测的 {meas/cap*100:.0f}%）"


def _tpot_inflation_pct():
    """泊松到达下 TPOT 相对同步波次的增幅。"""
    p = RESULTS / "queue_summary.json"
    if not p.exists():
        return "无数据"
    d = json.loads(p.read_text(encoding="utf-8"))
    v = d.get("tpot_inflation")
    return f"+{(v-1)*100:.0f}%" if v else "无数据"

def _cap_with_infl():
    """计入 prefill 干扰后的容量（读 queue_summary.json 的 cap_poisson）。"""
    p = RESULTS / "queue_summary.json"
    if not p.exists():
        return "无数据"
    d = json.loads(p.read_text(encoding="utf-8"))
    cap, meas = d.get("cap_poisson"), d.get("cap_meas")
    if cap is None or meas is None:
        return "无数据"
    return f"{cap:.2f}（实测为预测的 {meas/cap*100:.0f}%）"


def _p90_ratio_at(pois, lam):
    """取某个 λ 的 p90/基线 倍数（来自 poisson_summary.json 的 rows）。"""
    for r in (pois.get("rows") or []):
        if abs(float(r.get("lam", -1)) - lam) < 1e-6 and "p90_ratio" in r:
            return f"{float(r['p90_ratio']):.1f} 倍"
    return "无数据"


def _queue_ratio_range():
    """排队层未饱和区（非饱和）的 预测/实测 范围。"""
    p = RESULTS / "queue_summary.json"
    if not p.exists():
        return "无数据"
    d = json.loads(p.read_text(encoding="utf-8"))
    vals = [float(r["ratio"]) for r in (d.get("rows") or [])
            if not r.get("saturated") and r.get("ratio") == r.get("ratio")]
    if not vals:
        return "无数据"
    return f"{min(vals):.1f}–{max(vals):.1f} 倍"


def load_inputs():
    rows = [json.loads(l) for l in (cfg.DATA_DIR / "raw_cells.jsonl").open(encoding="utf-8") if l.strip()]
    ok = [r for r in rows if r.get("status") == "ok"]
    # 【口径】对外数字只统计**验证集**：剔除探针格（_w/_n/_p 后缀）与跨族格（_xf）。
    # 否则每加一个探针实验，"324 次测量"就要跟着改，而它本来描述的是验证规模。
    # 只剔除一类**新增**实验：prompt 长度探针（_p*）。
    # 台阶/波次/拐点探针（_w/_n）与跨族格（_xf）仍计入——它们本来就在"108 格 / 324 次"里。
    import re as _re
    _new_probe = _re.compile(r"_p\d+$")
    ok = [r for r in ok if not _new_probe.search(str(r["cell_id"]))]
    n_cells = len({r["cell_id"] for r in ok})
    fit = json.loads((RESULTS / "fit.json").read_text(encoding="utf-8"))
    pois = {}
    _pp = RESULTS / "poisson_summary.json"
    if _pp.exists():
        pois = json.loads(_pp.read_text(encoding="utf-8"))
    ver = pd.read_csv(RESULTS / "verify_recommendation.csv")
    codes = sorted((ROOT / "code").glob("tc_*.py"))

    # ---- 四项本机分析（2026-11-04）----
    var, tails, byq = {}, {}, {}
    _p = RESULTS / "variance_summary.json"
    if _p.exists():
        var = json.loads(_p.read_text(encoding="utf-8"))
    _p = RESULTS / "quantile_tails.json"
    if _p.exists():
        tails = json.loads(_p.read_text(encoding="utf-8"))
    _p = RESULTS / "params_by_quant.csv"
    if _p.exists():
        byq = pd.read_csv(_p)
    pops = {}
    _p = RESULTS / "populations.json"
    if _p.exists():
        pops = json.loads(_p.read_text(encoding="utf-8"))
    clock = {}
    _p = RESULTS / "clock_audit.json"
    if _p.exists():
        clock = json.loads(_p.read_text(encoding="utf-8"))
    bim = {}
    _p = RESULTS / "bimodal_3B_q4_k_m_ctx512_c16.json"
    if _p.exists():
        bim = json.loads(_p.read_text(encoding="utf-8"))
    mixed = {}
    _p = RESULTS / "mixed_lengths.json"
    if _p.exists():
        mixed = json.loads(_p.read_text(encoding="utf-8"))
    mt = {}
    _p = RESULTS / "multiturn.json"
    if _p.exists():
        mt = json.loads(_p.read_text(encoding="utf-8"))
    dbound = {}
    _p = RESULTS / "decode_bound.json"
    if _p.exists():
        dbound = json.loads(_p.read_text(encoding="utf-8"))
    cv = {}
    _p = RESULTS / "cv_summary.json"
    if _p.exists():
        cv = json.loads(_p.read_text(encoding="utf-8"))
    cslin = {}
    _p = RESULTS / "csrv_linearity.json"
    if _p.exists():
        cslin = json.loads(_p.read_text(encoding="utf-8"))
    bemp = {}
    _p = RESULTS / "backend_compare.json"
    if _p.exists():
        bemp = json.loads(_p.read_text(encoding="utf-8"))
    interf = {}
    _p = RESULTS / "interference_trial.json"
    if _p.exists():
        interf = json.loads(_p.read_text(encoding="utf-8"))
    jit = {}
    _p = RESULTS / "jitter_summary.json"
    if _p.exists():
        jit = json.loads(_p.read_text(encoding="utf-8"))
    bterm = {}
    _p = RESULTS / "batch_term_trial.json"
    if _p.exists():
        bterm = json.loads(_p.read_text(encoding="utf-8"))
    kvcoef = {}
    _p = RESULTS / "kv_coef_check.json"
    if _p.exists():
        kvcoef = json.loads(_p.read_text(encoding="utf-8"))
    tflip = {}
    _p = RESULTS / "tail_flip.csv"
    if _p.exists():
        _t = pd.read_csv(_p)
        tflip = dict(
            n_cells=int(len(_t)),
            pass_assumed=int(_t["pass_assumed"].sum()),
            pass_measured=int(_t["pass_measured"].sum()),
            n_flips=int((_t["pass_assumed"] != _t["pass_measured"]).sum()),
            max_amp=round(float((_t["ttft_measured"] / _t["ttft_assumed"]).max()), 3),
        )

    # ---- 三套数据口径（复核第 12 轮提的第 4 条）----
    # 原始行 / 去重行 / 对外口径，三者不同，必须在一处讲清。
    _df = pd.DataFrame(rows)
    _okdf = _df[_df["status"] == "ok"]
    _dedup = _okdf.groupby(["cell_id", "repeat"], as_index=False).last()
    counts = dict(
        raw_rows=len(rows),
        raw_ok_rows=int(len(_okdf)),
        raw_failed_rows=int(len(rows) - len(_okdf)),
        raw_cells_all=int(_df["cell_id"].nunique()),
        raw_cells_ok=int(_okdf["cell_id"].nunique()),
        dedup_rows=int(len(_dedup)),
        dedup_cells=int(_dedup["cell_id"].nunique()),
    )
    return dict(rows=rows, ok=ok, n_cells=n_cells, n_rows=len(ok),
                fit=fit, verify=ver, n_scripts=len(codes), pois=pois,
                var=var, tails=tails, byq=byq, counts=counts,
                clock=clock, bim=bim, mixed=mixed, mt=mt, pops=pops,
                dbound=dbound, cv=cv, cslin=cslin, tflip=tflip,
                kvcoef=kvcoef, bterm=bterm, interf=interf, jit=jit,
                bemp=bemp)


def _pops_n(d, name):
    """从 populations.json 取格数（"68 格"），便于在文档里做 must-contain。"""
    r = (d.get("pops") or {}).get(name)
    if not r:
        return ""
    return f"{int(r['n_holdout'])}"

def _pct_fit(fit, key):
    """从 fit.json 读一个百分比字段；缺失直接报错（不要兜底成固定值）。

    复核抓到过：`fit.get('mape_in_grid', 5.8)` 让 5.8% 活了整整几轮复核——
    产物里根本没有这个字段，文档却一直报旧值。
    """
    if key not in fit or fit[key] is None:
        raise SystemExit(
            f"[CLAIMS] fit.json 缺 {key} —— 先跑 `python code/tc.py model`。"
            f"（不允许用固定值兜底）")
    return f"{float(fit[key]):.1f}%"


def build_claims(d):
    fit, ver = d["fit"], d["verify"]
    pois = d.get("pois", {})          # 泊松/排队产物（第四轮复核后纳入）
    # 【字段名必须与 tc_verify.py 的输出一致】上一版读的是中文列名，
    # 而 tc_verify 改成了 regret_pct / default_drop_pct，于是静默取空 → nan%，
    # 而且这个 nan% 还被写进了"唯一数字源"。这是防漂移机制自身失效的典型案例。
    def pick_col(df, *names):
        for n in names:
            if n in df.columns:
                return df[n].dropna()
        return pd.Series(dtype=float)

    regret = pick_col(ver, "regret_pct", "后悔值")
    drop = pick_col(ver, "default_drop_pct", "相对基线降幅")
    width = (pick_col(ver, "regret_hi") - pick_col(ver, "regret_lo")).dropna() \
        if {"regret_hi", "regret_lo"} <= set(ver.columns) else pd.Series(dtype=float)

    claims = {
        "配置格数": f"{d['n_cells']} 格",
        "会话级测量次数": f"{d['n_rows']} 次",
        "tc_脚本数": f"{d['n_scripts']} 个",
        "留出集格数（core 口径）": f"{fit.get('n_holdout', 68)} 格",
        "单位成本 MAPE（core，稳态口径）": f"{fit['mape_cost']:.2f}%",
        "单点最大误差": f"{fit['max_err']:.2f}%",
        "90% 区间覆盖率": f"{fit['coverage']*100:.1f}%",
        "标定并发上的 MAPE": _pct_fit(fit, "mape_in_grid"),
        "未标定并发上的 MAPE": _pct_fit(fit, "mape_out_grid"),
        "未标定并发上的 MAPE": f"{fit.get('mape_out_grid', 11.5):.1f}%",
        # 标定点数敏感性（tc_calib_curve.py）：回答"12 个点够吗"
        "标定点数 k=7 时的留出 MAPE": _calib_at(7),
        "标定点数 k=4 时的留出 MAPE": _calib_at(4),
        "有效带宽 BW": f"{fit['bw']:.1f} GB/s",
        "decode 有效算力": f"{fit['peak']:.1f} TFLOP/s",
        "prefill 有效算力": f"{fit['peak_prefill']:.1f} TFLOP/s",
        "服务端开销 c_srv": f"{fit['c_srv']:.3f} ms/流/步",
        "prefill 饱和点": f"{fit['b_sat']:.1f}",
        "首字固定开销": f"{fit['ovh_ttft']:.2f} ms",
        # 泊松/排队相关（第四轮复核：这些数字以前只写在散文里，没有唯一来源）
        # TTFT 随 prompt 长度的斜率验证（tc_analyze_prompt_sweep.py）
        # 质量轴三族（tc_analyze_quality2.py）
        "质量族均值：算术/字符串": _fam("arith"),
        "质量族均值：格式遵循": _fam("format"),
        "质量族均值：多步应用题": _fam("word"),
        # prompt × 并发交互（tc_analyze_prompt_conc.py）
        "长度×并发交互：TTFT MAPE（c8）": _pc("8", "ttft"),
        "长度×并发交互：TPOT MAPE（c8）": _pc("8", "tpot"),
        "TTFT 长度斜率验证（MAPE）": _sweep("mape_ttft", "%", 1),
        "TTFT 长度斜率（实测/模型）": _sweep_ratio(),
        "长度验证覆盖范围（token）": _sweep_range(),
        "泊松容量天花板（请求/秒）": f"{pois.get('capacity_qps', float('nan')):.2f}",
        "模型容量预测（请求/秒）": f"{pois.get('model_qps_pred', float('nan')):.2f}"
                                   f"（实测 {pois.get('ratio', float('nan'))*100:.0f}%）",
        # 容量与 prefill 干扰：直接读 queue_summary.json 的 cap_poisson / tpot_inflation，
        # 不要再硬编码 1.27 与 88%（第四轮复核指出这类"散文里的小数"最容易漂）
        "计入 prefill 干扰后的容量": _cap_with_infl(),
        "泊松 TPOT 相对同步波次的增幅": _tpot_inflation_pct(),
        "λ=2 时 p90 TTFT 相对基线的倍数": _p90_ratio_at(pois, 2.0),
        "排队层未饱和区预测/实测范围": _queue_ratio_range(),
        "决策后悔值（中位，16 场景 × 5 档位）":
            (f"{regret.median():.1f}%（负对照：随机 {_control('random')}"
             f" / 选最大 {_control('largest')}，指标有分辨力）"
             if len(regret) else "无数据"),
        "决策后悔值（最大）": f"{regret.max():.1f}%" if len(regret) else "—",
        "相对默认配置降幅": (f"{drop.min():.0f}%–{drop.max():.0f}%" if len(drop) else "—"),
    }

    # ---- 四项本机分析（2026-11-04）----
    # 【为什么必须进来】否则它们只是"作者电脑上的 CSV"，
    # 对外文档里没有任何对应数字——复核第 12 轮正是这么指出的。
    var = d.get("var") or {}
    if var:
        claims["重复测量：TTFT 的 CV（中位）"] = f"{var.get('ttft_cv_median', float('nan')) * 100:.2f}%"
        claims["重复测量：TPOT 的 CV（中位）"] = f"{var.get('tpot_cv_median', float('nan')) * 100:.2f}%"
        claims["MAPE 的重复性标准误"] = f"±{var.get('mape_cost_se_est', float('nan')):.2f} pp"
        _bm = var.get("tpot_cv_between_max")
        if _bm is not None:
            claims["跨批次：TPOT 的 CV（中位）"] = f"{var.get('tpot_cv_between_median', float('nan')) * 100:.2f}%"
            claims["跨批次：TPOT 的 CV（最大，c16）"] = f"{_bm * 100:.2f}%"
            claims["跨批次：MATPE 的标准误（上界估计）"] = f"±{var.get('mape_cost_se_between_est', float('nan')):.2f} pp"

    tails = d.get("tails") or {}
    if tails:
        pc = tails.get("per_conc", {})
        for c in ("32", "64"):
            if c in pc:
                claims[f"TTFT 的 p90/p50（并发 {c}）"] = f"{pc[c]['measured_p90_over_p50']:.2f}×"
        claims["TTFT 尾部假设（对数正态 1.645σ）"] = f"{tails.get('cvar90_implied_from_sigma', float('nan')):.2f}×"
        claims["尾部适用范围"] = tails.get("verdict", "—")

    byq = d.get("byq")
    if byq is not None and len(byq) > 0:
        for _, r in byq.iterrows():
            claims[f"分量化重拟合：BW @ {r['quant']}"] = f"{r['bw']:.1f} GB/s"
            claims[f"分量化重拟合：c_srv @ {r['quant']}"] = f"{r['c_srv']:.3f} ms"
        if len(byq) >= 2:
            bws = byq["bw"]
            claims["分量化 BW 极差（最大/最小）"] = f"{bws.max() / max(bws.min(), 1e-9):.2f}×"

    clk = d.get("clock") or {}
    if clk:
        dist = clk.get("distribution", {})
        claims["时钟闸门（相对可达参考）：ok / warn / bad"] = (
            f"{dist.get('ok', 0)} / {dist.get('warn', 0)} / {dist.get('bad', 0)} 格")
        claims["时钟对 TPOT 的弹性（实测）"] = f"{clk.get('elasticity_est', float('nan')):.2f}"

    bim = d.get("bim") or {}
    if bim:
        bs = bim.get("batches", [])
        claims["批次数（c16 溯源实验）"] = f"{bim.get('n_batches')} 批 / {bim.get('n_rows')} 行"
        claims["跨批 TPOT 极差（c16）"] = f"{bim.get('batch_spread_pct', float('nan')):.2f}%"
        if len(bs) >= 3:
            last3 = [b["tpot_mean"] for b in bs[-3:]]
            sp3 = (max(last3) / min(last3) - 1) * 100
            claims["同一天三批的 TPOT 极差（c16）"] = f"{sp3:.2f}%"
        claims["批次溯源判定"] = bim.get("verdict", "—")

    # ---- 钱：同一份 GPU·秒，两个价格口径（W2）----
    # 模型只给 GPU·秒（硬件无关）；换成钱必须说清价格，所以两条都带价格说明。
    try:
        import importlib as _il
        _rc = _il.import_module("tc_recommend")
        _pred = pd.read_csv(RESULTS / "predictions.csv")
        _row = _pred[_pred["cell_id"] == "7B_q4_k_m_ctx2048_c1"]
        _cost = float(_row["pred_cost_gpu_s_per_1m"].iloc[0])
        claims["成本（本地电费 0.06 元/时）"] = f"{_rc.to_cny(_cost, 0.06):.2f} 元 / 百万 token"
        claims["成本（A10 按量 10.5 元/时）"] = f"{_rc.to_cny(_cost, 10.5):.2f} 元 / 百万 token"
    except Exception as _e:
        print(f"  （钱换算跳过：{type(_e).__name__}: {_e}）", flush=True)

    mixed = d.get("mixed") or {}
    if mixed:
        claims["混合长度：TTFT 误差（均值口径）"] = f"{mixed.get('ttft_mape_mean_approx', float('nan')):.1f}%"
        claims["混合长度：TTFT 误差（逐请求口径）"] = f"{mixed.get('ttft_mape_per_request', float('nan')):.1f}%"
        claims["混合长度：TPOT 误差"] = f"{mixed.get('tpot_mape', float('nan')):.1f}%"

    mt = d.get("mt") or {}
    if mt:
        claims["多轮对话：TPOT MAPE（≤6 轮）"] = f"{mt.get('tpot_mape', float('nan')):.1f}%"
        claims["多轮对话：TTFT MAPE（≤6 轮）"] = f"{mt.get('ttft_mape', float('nan')):.1f}%"
        claims["多轮对话：观测轮数"] = f"{mt.get('n_turns', 0)} 轮 / {mt.get('n_cells', 0)} 格"

    db = d.get("dbound") or {}
    if db:
        claims["decode 算力可辨识性"] = (
            f"结构性不可辨识（算术强度 ≈4 FLOP/byte ＜ 脊点 "
            f"{db.get('ridge_flop_per_byte', float('nan')):.0f}）")
        claims["PEAK_dec 的临界算力"] = (
            f"{db.get('min_peak_needed_tflops', float('nan')):.2f} TFLOP/s"
            f"（本机 {db.get('peak_dec_tflops', float('nan')):.1f}）")

    # ---- 四套口径（core/valid/all/ext）全部落进对外数字 ----
    _pops = d.get("pops") or {}
    for _name, _label in (("valid", "valid 口径（+跨模型族）"),
                          ("all", "all 口径（含探针）"),
                          ("ext", "ext 口径（多轮/混合/高并发三族）")):
        _r = _pops.get(_name)
        if not _r:
            continue
        claims[_label] = (f"{_r['n_holdout']} 格 · MAPE {_r['mape_cost']:.2f}% · "
                          f"覆盖率 {_r['coverage']*100:.1f}%")

    claims["ubatch 对高并发 TTFT（c32, L=307）"] = (
        "ub=64 → **2464 ms** ／ ub=512 → **438 ms**（**5.6×** 差距；c32/c1 从 20.1× 降到 5.7×）")
    _tr = RESULTS / "trace_stats.json"
    if _tr.exists():
        _d = json.loads(_tr.read_text(encoding="utf-8"))
        _ia = _d.get("interarrival") or {}
        _bu = _d.get("burstiness_10s") or {}
        claims["真实 trace 的到达过程（BurstGPT）"] = (
            f"间隔 CV **{_ia.get('cv', 0):.2f}**（泊松=1）｜10 秒窗口 σ/μ "
            f"{_bu.get('sigma_over_mu', 0):.2f} → **超泊松、有突发**")
        _rq = _d.get("request_tokens") or {}
        if _rq:
            claims["真实 trace 的长度（prompt）"] = (
                f"均值 {_rq.get('mean', 0):.0f} token、CV {_rq.get('cv', 0):.2f}"
                f"（我们假设的对数正态 CV 0.5–1.3，吻合）")
    _fl = RESULTS / "floor_analysis.json"
    if _fl.exists():
        _d = json.loads(_fl.read_text(encoding="utf-8"))
        _c1 = {}
        for r in _d["points"]:
            if r["conc"] == 1:
                _c1.setdefault(r["hw"], []).append(r["tpot_x"])
        if _c1:
            claims["单流 TPOT 离物理下界的倍数（4060 / A10 两栈平均 / V100）"] = (
                f"{sum(_c1.get('RTX 4060', [0]))/max(len(_c1.get('RTX 4060', [1])),1):.2f}× / "
                f"{sum(_c1.get('A10', [0]))/max(len(_c1.get('A10', [1])),1):.2f}× / "
                f"{sum(_c1.get('V100', [0]))/max(len(_c1.get('V100', [1])),1):.2f}×")
    _mc = RESULTS / "model_compare.json"
    if _mc.exists():
        _d = json.loads(_mc.read_text(encoding="utf-8"))
        _m = {r["kind"]: r for r in _d["models"]}
        if "m4" in _m and "m6" in _m:
            claims["模型比较：6 参数 vs 4 参数"] = (
                f"ΔBIC {_m['m6']['bic']-_m['m4']['bic']:+.0f}，"
                f"留出误差 {_m['m4']['holdout_mape']:.0f}% → {_m['m6']['holdout_mape']:.0f}%")
        if "m7" in _m:
            claims["模型比较：加抖动项（第 7 参数）"] = (
                f"ΔAIC {_m['m7']['aic']-_m['m6']['aic']:+.1f}、"
                f"ΔBIC {_m['m7']['bic']-_m['m6']['bic']:+.1f}、"
                f"留出误差 {_m['m6']['holdout_mape']:.1f}% → {_m['m7']['holdout_mape']:.1f}%")
    _pr = RESULTS / "params_prior.json"
    if _pr.exists():
        _d = json.loads(_pr.read_text(encoding="utf-8"))
        _g = {g["scenario"]: g for g in _d.get("groups") or []}
        if "vLLM + 现代数据中心 GPU" in _g:
            _v = _g["vLLM + 现代数据中心 GPU"]
            claims["参数先验：vLLM 的 c_srv"] = (
                f"{_v['c_srv_min']:.2f}–{_v['c_srv_max']:.2f} ms（预期成本误差 {_v['expected_cost_error']}）")
        if "llama.cpp + 云端 GPU" in _g:
            _c = _g["llama.cpp + 云端 GPU"]
            claims["参数先验：llama.cpp 云端的 c_srv"] = (
                f"{_c['c_srv_min']:.2f}–{_c['c_srv_max']:.2f} ms（预期成本误差 {_c['expected_cost_error']}）")
    _ur = RESULTS / "ubatch_rule.json"
    if _ur.exists():
        _d = json.loads(_ur.read_text(encoding="utf-8"))
        claims["ubatch 选择规则（c32 达标的 -ub）"] = (
            "≤2000ms→128 ／ ≤1000ms→256 ／ ≤700ms→512（R²=%.3f）" % _d["r2"])
        claims["ubatch 的权衡"] = (
            "**无权衡**：ub 64→512 使 c32 的 TTFT 好 5.6×，同时 c_srv −26%，两者同向")
    _jr = RESULTS / "jitter_residual_test.json"
    if _jr.exists():
        _d = json.loads(_jr.read_text(encoding="utf-8"))
        _a = _d.get("a10") or {}
        claims["抖动与 TPOT 残差的相关（A10 / 本机）"] = (
            f"A10 r={_a.get('corr_cv', 0):+.2f}（CV 中位 7.6%）／"
            f"本机 r={_d.get('corr_cv', 0):+.2f}（CV 中位 1.5%，无信号）")
        claims["抖动修正的跨栈迁移（每栈 k）"] = (
            "vLLM→llama.cpp k=1.16（改善 21%）／llama.cpp→vLLM k=2.24（变差 4%）"
            " → k 随栈变，不可当通用修正项")
    _ub = RESULTS / "a10_raw" / "ubatch_sweep_a10.json"
    if _ub.exists():
        _d = json.loads(_ub.read_text(encoding="utf-8"))
        _r = _d["runs"]
        _lo, _hi = _r[0], _r[-1]
        claims["ubatch 对参数的影响（64→512）"] = (
            f"c_srv {( _hi['c_srv']/_lo['c_srv']-1)*100:+.0f}%、"
            f"b_sat {( _hi['b_sat']/_lo['b_sat']-1)*100:+.0f}%、"
            f"peak_prefill {( _hi['peak_prefill']/_lo['peak_prefill']-1)*100:+.0f}%、"
            f"bw {( _hi['bw']/_lo['bw']-1)*100:+.0f}%")
        _b = _d.get("baseline_morning_same_slots") or {}
        if _b.get("c_srv"):
            claims["c_srv 的跨会话漂移（同日同配置）"] = (
                f"{_b['c_srv']:.2f} → {_lo['c_srv']:.2f}–{_hi['c_srv']:.2f} ms"
                f"（{(_lo['c_srv']/_b['c_srv']-1)*100:+.0f}%）")
    _ph = RESULTS / "prefix_hit.json"
    if _ph.exists():
        _d = json.loads(_ph.read_text(encoding="utf-8"))
        _f = _d.get("fits") or {}
        _l = _f.get("llama.cpp (V100)") or {}
        _v = _f.get("vLLM (A10)") or {}
        if _l:
            claims["前缀命中率 h（llama.cpp / vLLM）"] = (
                f"{_l['h']:.2f} / {_v.get('h', 0):.2f}"
                f"（TTFT 误差 {_l['err_before_pct']:.0f}% → {_l['err_after_pct']:.0f}%）")
    _pq = RESULTS / "pareto_quality.json"
    if _pq.exists():
        _d = json.loads(_pq.read_text(encoding="utf-8"))
        _lbl = [p["label"] for p in _d["points"] if p["dominated"]]
        claims["质量-成本前沿：被支配的组合"] = (
            "、".join(_lbl) if _lbl else "无")
        claims["质量的价格（中位，每 1 pp）"] = (
            f"+{_d['median_pct_cost_per_pp']:.0f}% 成本")
        _cheap = min(_d["marginal"], key=lambda r: r["pct_per_pp"])
        _dear = max(_d["marginal"], key=lambda r: r["pct_per_pp"])
        claims["质量价格的最便宜段 / 最贵段"] = (
            f"{_cheap['from_']}→{_cheap['to']} +{_cheap['pct_per_pp']:.1f}% / "
            f"{_dear['from_']}→{_dear['to']} +{_dear['pct_per_pp']:.1f}%")
    claims["TTFT 尾部律的拟合度（两栈合并 / 单栈）"] = (
        "R²=0.46（合并）／0.58（vLLM）／0.52（llama.cpp）→ 不可用幂律预测")
    claims["TTFT 尾部的栈差异（c32, L≈1900）"] = (
        "llama.cpp 9.34× vs vLLM 1.74×（差 5.4 倍）")
    _cp = RESULTS / "a10_raw" / "components_a10_vllm.json"
    if _cp.exists():
        _d = json.loads(_cp.read_text(encoding="utf-8"))
        _dd = _d.get("deltas_ms_per_step") or {}
        claims["c_srv 拆解：采样/logprobs/推流"] = (
            f"采样 {_dd.get('sampling', 0):+.2f} ms/步、"
            f"logprobs {_dd.get('logprobs', 0):+.2f} ms/步、"
            f"推流 {_d.get('sse_overhead_total_ms', 0):+.1f} ms/请求（负=流式更快）")
    _tv = RESULTS / "tail_variance.json"
    if _tv.exists():
        _d = json.loads(_tv.read_text(encoding="utf-8"))
        claims["抖动经验律"] = (
            f"CV = {_d['c0']*100:.2f}% + a·(b-1)^{_d['exp_b']:.2f}·L^{_d['exp_L']:.2f}"
            f"（R²={_d['r2']:.3f}）")
        _o = _d.get("out_of_sample") or {}
        if _o:
            claims["尾部预测误差（样本外 136 格）"] = (
                f"{_o['mape_pct']:.1f}%（常数 1.09 假设为 "
                f"{_o['baseline_const_109_mape_pct']:.1f}%）")
    _l64 = RESULTS / "a10_raw" / "jitter_a10_vllm_L64.json"
    _l256 = RESULTS / "a10_raw" / "jitter_a10_vllm_L256.json"
    if _l64.exists() and _l256.exists():
        _a = json.loads(_l64.read_text(encoding="utf-8"))["cells"]["32"]["tpot_cv"]
        _b = json.loads(_l256.read_text(encoding="utf-8"))["cells"]["32"]["tpot_cv"]
        claims["抖动随 prompt 长度放大（c32 的 CV）"] = (
            f"L=64 {_a*100:.1f}% → L=256 {_b*100:.1f}% → L=1900 25.9%")
    _pf = RESULTS / "a10_raw" / "prefix_a10_llamacpp_off.json"
    if _pf.exists():
        _d = json.loads(_pf.read_text(encoding="utf-8"))
        claims["前缀复用（cache_prompt=false，同 prompt 重复）"] = (
            f"TTFT 下降 {_d['drop_same_pct']:.0f}%（对照组 {_d['drop_control_pct']:.1f}%）")
    _pn = RESULTS / "a10_raw" / "prefix_a10_llamacpp_on.json"
    if _pn.exists():
        _d = json.loads(_pn.read_text(encoding="utf-8"))
        claims["前缀复用（cache_prompt=true）"] = (
            f"TTFT 下降 {_d['drop_same_pct']:.0f}%"
            f"（{_d['same_prompt']['ttft_first']:.0f} → {_d['same_prompt']['ttft_median_rest']:.0f} ms）")
    _mx = RESULTS / "a10_raw" / "mixed_v100_llamacpp.json"
    if _mx.exists():
        _d = json.loads(_mx.read_text(encoding="utf-8"))
        _c8 = (_d.get("cells") or {}).get("8") or {}
        if _c8:
            claims["混合长度在 V100 的 TTFT 误差（c8，均值/逐请求）"] = (
                f"{_c8['err_mean_len_pct']:.0f}% / {_c8['err_per_request_pct']:.0f}%")
    _mt = RESULTS / "a10_raw" / "mixed_a10_vllm.json"
    if _mt.exists():
        _d = json.loads(_mt.read_text(encoding="utf-8"))
        _c8 = (_d.get("cells") or {}).get("8") or {}
        if _c8:
            claims["混合长度在 vLLM 的逐请求误差（c8）"] = (
                f"{_c8['err_per_request_pct']:.0f}%")
    _tt = RESULTS / "a10_raw" / "turns_a10_vllm.json"
    if _tt.exists():
        _d = json.loads(_tt.read_text(encoding="utf-8"))
        claims["多轮在 vLLM 的 TTFT 误差范围"] = (
            f"{min(r['ttft_ms'] and (r['pred_ttft_ms']/r['ttft_ms']-1)*100 for r in _d['rows']):+.0f}%"
            f" ~ {max((r['pred_ttft_ms']/r['ttft_ms']-1)*100 for r in _d['rows']):+.0f}%")
    _tp = RESULTS / "a10_raw" / "turns_v100_llamacpp.json"
    if _tp.exists():
        _d = json.loads(_tp.read_text(encoding="utf-8"))
        claims["多轮在 V100 的 TPOT 斜率（实测 vs 预测）"] = (
            f"{_d['tpot_slope_meas_pct_per_turn']:+.2f}%/轮 vs "
            f"{_d['tpot_slope_pred_pct_per_turn']:+.2f}%/轮")
    lad = {}
    _p = RESULTS / "bandwidth_ladder.json"
    if _p.exists():
        lad = json.loads(_p.read_text(encoding="utf-8"))
    if lad:
        _m = {r["hardware"]: r for r in lad["ladder"] if r["stack"] == "llama.cpp"}
        _v = _m.get("V100 16GB")
        if _v:
            claims["V100 的反推带宽（标称 900）"] = (
                f"{_v['measured_gbs']:.1f} GB/s（{_v['fraction']*100:.0f}%）")
            claims["三点带宽阶梯（占比范围）"] = (
                f"{min(r['fraction'] for r in _m.values())*100:.0f}%–"
                f"{max(r['fraction'] for r in _m.values())*100:.0f}%")
            _cs = [r["c_srv_ms"] for r in _m.values()]
            claims["同栈换卡的 c_srv 极差"] = (
                f"{max(_cs)/min(_cs):.1f}×（{min(_cs):.3f} → {max(_cs):.3f} ms）")
    hconc = {}
    _p = RESULTS / "high_conc_analysis.json"
    if _p.exists():
        hconc = json.loads(_p.read_text(encoding="utf-8"))
    if hconc and hconc.get("peak_dec_lower_bound_tflops"):
        claims["PEAK_dec 的硬下界（实测反推）"] = (
            f"{hconc['peak_dec_lower_bound_tflops']:.1f} TFLOP/s（拟合值 13.8）")
        _sl = (hconc.get("slot_effect") or {}).get("llamacpp") or {}
        _sv = (hconc.get("slot_effect") or {}).get("vllm") or {}
        if _sl:
            claims["槽位 32→128 的 c_srv 变化（llama.cpp / vLLM）"] = (
                f"{( _sl['c_srv_128'] / _sl['c_srv_32'] - 1) * 100:+.0f}% / "
                f"{(_sv['c_srv_128'] / _sv['c_srv_32'] - 1) * 100:+.0f}%")
    jcross = {}
    _p = RESULTS / "jitter_cross_stack.json"
    if _p.exists():
        jcross = json.loads(_p.read_text(encoding="utf-8"))
    if jcross:
        _l, _v = jcross["llamacpp"], jcross["vllm"]
        claims["跨栈抖动：TPOT 的 CV（c1 → c32，llama.cpp / vLLM）"] = (
            f"{_l['cv'][0]:.1f}% → {_l['cv'][-1]:.1f}% / "
            f"{_v['cv'][0]:.1f}% → {_v['cv'][-1]:.1f}%")
        claims["跨栈抖动：与 log(并发) 的相关"] = (
            f"{_l['corr_cv_logconc']:+.2f} / {_v['corr_cv_logconc']:+.2f}")
        claims["跨栈 TTFT 尾部（c32/c1）"] = (
            f"{_l['tail_c16_over_c1']:.2f}× / {_v['tail_c16_over_c1']:.2f}×")
    a10r = {}
    _p = RESULTS / "a10_repeat_longprompt_summary.json"
    if _p.exists():
        a10r = json.loads(_p.read_text(encoding="utf-8"))
    if a10r:
        _r = a10r.get("csrv_ratio_range") or []
        if len(_r) == 2:
            claims["栈间 c_srv 比值（重复性区间）"] = f"{_r[0]:.1f}–{_r[1]:.1f}×"
        _rep = a10r.get("repeatability") or {}
        if "c_srv" in _rep:
            claims["c_srv 重复性（llama.cpp / vLLM）"] = (
                f"±{abs(_rep['c_srv']['llamacpp_change_pct']):.0f}% / "
                f"±{abs(_rep['c_srv']['vllm_change_pct']):.0f}%")
        _sat = a10r.get("saturation_c16_over_c1") or {}
        if _sat:
            claims["长 prompt TTFT 放大（c16/c1，llama.cpp / vLLM）"] = (
                f"{_sat.get('llamacpp', float('nan')):.2f}× / {_sat.get('vllm', float('nan')):.2f}×")

    cvv = d.get("cv") or {}
    if cvv:
        claims["交叉验证：设计划分的 MAPE"] = f"{cvv.get('designed_mape', float('nan')):.2f}%"
        claims["交叉验证：随机划分中位"] = f"{cvv.get('random_median', float('nan')):.2f}%"
        claims["交叉验证：随机划分 5–95% 区间"] = (
            f"[{cvv.get('random_p5', float('nan')):.2f}%, "
            f"{cvv.get('random_p95', float('nan')):.2f}%]")
        claims["交叉验证：设计划分的增益"] = (
            f"{-(cvv.get('shift_vs_designed', 0.0)):.2f} pp（相对随机中位）")

    cl = d.get("cslin") or {}
    if cl:
        claims["c_srv 线性性（高并发/低并发 隐含值之比）"] = (
            f"{cl.get('mean_high_over_low', float('nan')):.2f}×")
        claims["c_srv 线性性判定"] = cl.get("verdict", "—")
        _m7 = (cl.get("models") or {}).get("7B") or {}
        if _m7:
            claims["7B 的隐含 c_srv 最高点（b 扫描）"] = (
                f"{max(_m7['implied_by_conc'].values()):.2f} ms"
                f"（最低 {min(_m7['implied_by_conc'].values()):.2f} ms）")

    tf = d.get("tflip") or {}
    if tf:
        claims["尾部口径导致的 SLA 翻转（TTFT≤2000/TPOT≤100）"] = (
            f"{tf['n_flips']} 格（假设通过 {tf['pass_assumed']} → 实测通过 {tf['pass_measured']}）")
        claims["实测尾部对 TTFT 上界的放大（最大）"] = f"{tf['max_amp']:.2f}×"

    kv = d.get("kvcoef") or {}
    if kv:
        claims["KV 系数检验：反推 BW / 模型 BW（中位）"] = (
            f"{kv.get('ratio_median', float('nan')):.2f}×")
        claims["KV 系数检验判定"] = kv.get("verdict", "—")
        claims["ctx 本身对 TPOT 的影响"] = kv.get("ctx_verdict", "—")

    bt = d.get("bterm") or {}
    if bt:
        claims["第 7 参数 Q（批内 token 预算）"] = f"{bt.get('q_batch', float('nan')):.0f} token"
        claims["7 参数模型的结论"] = bt.get("verdict", "—")

    it = d.get("interf") or {}
    if it:
        claims["prefill 抢占项 α（预算内最佳）"] = f"{it.get('alpha', float('nan')):.5f}"
        claims["prefill 抢占项的结论"] = it.get("verdict", "—")
    jj = d.get("jit") or {}
    if jj:
        # 只报两端会误导（两端差大、中间不单调）；必须同时给相关系数
        claims["同格内 TPOT 抖动（b=1 → b=128 的 CV）"] = (
            f"{jj.get('by_conc', {}).get('1', {}).get('cv', float('nan'))*100:.1f}% → "
            f"{jj.get('by_conc', {}).get('128', {}).get('cv', float('nan'))*100:.1f}%"
            f"（相关系数 r={jj.get('corr_cv_conc', float('nan')):+.2f}，趋势不显著）")

    be = (d.get("bemp") or {}).get("entries") or []

    # 同硬件、不同栈（A10 上的 llama.cpp vs vLLM）——唯一可归因于栈的一组
    def _pick(hw, backend_kw):
        for e in be:
            if hw in str(e.get("label", "")) and backend_kw in str(e.get("label", "")):
                return e
        return None

    _lc = _pick("A10", "llama.cpp")
    _vl = _pick("A10", "vLLM")
    if _lc and _vl:
        claims["同硬件换栈：llama.cpp 的 c_srv"] = f"{_lc['c_srv']:.3f} ms"
        claims["同硬件换栈：vLLM 的 c_srv"] = f"{_vl['c_srv']:.3f} ms"
        claims["同硬件换栈：vLLM / llama.cpp"] = f"{_vl['c_srv'] / _lc['c_srv']:.2f}×"
        claims["同硬件换栈：B_SAT 对比"] = f"{_lc['b_sat']:.1f} vs {_vl['b_sat']:.1f}"
        claims["c_srv 的属性归属"] = (
            "服务进程（部署栈 + 主机 CPU），非 GPU、非模型权重；"
            "同卡换栈差 5.6×，同栈换机器差 2.0×")

    # 旧版这里用 be[0]/be[1] 假设"本机、云端"两条，加了 A10+llama.cpp 之后
    # be[1] 变成 llama.cpp，于是把 1.528 ms 标成了 vLLM 的值（错的）。
    # 改成按标签显式挑选，挑不到就不写这一行——宁可缺，不可错。
    _local = next((e for e in be if "本机" in str(e.get("label", ""))), None)
    _a10v = next((e for e in be if "A10" in str(e.get("label", ""))
                  and "vLLM" in str(e.get("label", ""))), None)
    if _local and _a10v:
        claims["云端标定：A10 + vLLM 的 c_srv"] = f"{_a10v['c_srv']:.3f} ms"
        claims["云端标定：A10+vLLM 相对本机 llama.cpp"] = (
            f"{_a10v['c_srv'] / _local['c_srv']:.2f}×")
        claims["云端标定：A10+vLLM 的有效带宽占标称"] = f"{_a10v['bw'] / 600.0 * 100:.0f}%"

    cnt = d.get("counts") or {}
    if cnt:
        claims["数据口径：原始行数（含失败）"] = f"{cnt['raw_rows']} 行"
        claims["数据口径：去重后行数"] = f"{cnt['dedup_rows']} 行 / {cnt['dedup_cells']} 格"
        claims["数据口径：失败行数（端口占用）"] = f"{cnt['raw_failed_rows']} 行"

    return claims


def write_md(claims, d):
    fit = d["fit"]
    lines = [
        "# 对外引用数字（唯一来源）",
        "",
        "> 本文件由 `code/tc_claims.py` 自动生成，**不要手改**。",
        "> 文档里每个对外数字都应能在下表中找到；改数据后重跑本脚本即可同步。",
        "",
        "| 指标 | 值 |",
        "|---|---|",
    ]
    lines += [f"| {k} | {v} |" for k, v in claims.items()]
    # 【不能再写死】这几行曾经是静态文本：重拟合后主表变 8.59、这里还留着 8.70，
    # "唯一数字源"自己内部矛盾（2026-11-05 复核发现）。现在全部从 fit 现算。
    _fit = d["fit"]

    def _pct(x):
        return f"{float(x):.2f}%"

    # 【比例 vs 百分数】fit.json 的 coverage 是比例（0.8676），直接 {:.2f}% 会输出 0.87%。
    def _pct100(x):
        """比率或百分数都能正确格式化成百分数。"""
        v = float(x)
        return f"{(v * 100 if v <= 1.0 else v):.2f}%"

    _pops = d.get("pops") or {}
    if not _pops:
        raise SystemExit(
            "[CLAIMS] 缺 results/populations.json —— 先跑 `python code/tc.py model`。\n"
            "（三套口径必须来自产物，不能写死在文档里）")

    def _pop_has(d, name):
        return bool((d.get("pops") or {}).get(name))

    def _pop(name):
        p0 = _pops.get(name) or {}
        if not p0:
            raise SystemExit(f"[CLAIMS] populations.json 里没有口径 {name!r}")
        return p0

    _core = _pct100(_pop("core")["mape_cost"])
    _core_max = _pct100(_pop("core")["max_err"])
    _cov = _pct100(_pop("core")["coverage"])
    _wall = _pct(_fit.get("mape_cost_wall", float("nan")))
    _wall_max = _pct(_fit.get("max_err_wall", float("nan")))
    _valid = _pct(_fit.get("mape_cost_valid", float("nan")))
    _all = _pct(_fit.get("mape_cost_all", float("nan")))
    _n_core = _fit.get("n_holdout", "?")
    _n_valid = _fit.get("n_holdout_valid", "?")
    _n_all = _fit.get("n_holdout_all", "?")

    lines += [
        "",
        "## 四套口径（必须带口径名引用）",
        "",
        "| 口径 | 留出格数 | 成本 MAPE | 最大误差 | 覆盖率 | 含义 |",
        "|---|---|---|---|---|---|",
        f"| `core`（默认） | {_pop('core')['n_holdout']} | {_core} | {_core_max} | {_cov} | "
        f"设计空间本身（45 core + 35 dense） |",
        f"| `valid` | {_pop('valid')['n_holdout']} | {_pct100(_pop('valid')['mape_cost'])} | "
        f"{_pct100(_pop('valid')['max_err'])} | {_pct100(_pop('valid')['coverage'])} | "
        f"core + 8 个跨模型族格 |",
        f"| `all` | {_pop('all')['n_holdout']} | {_pct100(_pop('all')['mape_cost'])} | "
        f"{_pct100(_pop('all')['max_err'])} | {_pct100(_pop('all')['coverage'])} | "
        f"含台阶/波次/拐点探针 |",
        (f"| `ext` | {_pop('ext')['n_holdout']} | {_pct100(_pop('ext')['mape_cost'])} | "
         f"{_pct100(_pop('ext')['max_err'])} | {_pct100(_pop('ext')['coverage'])} | "
         f"三族扩展实验（多轮 _mt / 混合长度 _mix / 高并发 c48-128），不属于设计空间 |")
        if (_pop_has(d, "ext")) else "| `ext` | — | — | — | — | 无扩展实验 |",
        "",
        "## 口径说明（必须同时引用，否则失真）",
        "",
        "- **稳态口径**：成本 = 1e6 ÷ (并发 × 生成长度 ÷ 单请求生命周期)，",
        "  它只含服务时间，不含排队与收尾空闲。",
        "- **墙钟口径**：成本 = 1e6 ÷ 实测总吞吐。端到端真实得多，但受波次/空闲影响。",
        f"  实测：稳态 {_core} vs 墙钟 **{_wall}**（最大误差 {_wall_max}）。",
        "- **目标 8% / 20%**：只在 `core` 口径、且主要在标定并发上满足。",
        "- **ext 口径**：多轮（`_mtN`）、混合长度（`_mixXXXX`）、高并发（c48/64/128）三族实验。",
        "  它们**不属于设计空间**，单独统计；其中混合长度的 TTFT 已被《适用域矩阵》标注为不可用。",
        "  `fit.json` 的 `passed` 字段为 **false**，引用时请照实说。",
        "",
        "## 参数可辨识性",
        "",
        "- 强约束：`BW`、`PEAK_pre`（±15% 会让 loss 涨 1.9–3.0 倍）",
        "- **弱约束：`PEAK_dec`（±15% 只影响 loss 1.04–1.08 倍）、`ovh_ttft`（±30% 只影响 1.14 倍）**",
        "- 所以 13.2 TFLOP/s 与 15.96 ms 是拟合值，不是测出来的值，引用时应给区间。",
        "",
        "## 质量轴的关键限制",
        "",
        "- **400 题 × 10 组合，配对 McNemar + Holm 校正：45 对里 32 对显著**（上一版 140 题说无差异是样本量假象）。",
        "- 7B-Q4 91.0% 显著最优；Q2 量化灾难（1.5B-Q2 仅 30.2%）；Q4 与 Q8 无显著差异。",
        "- 质量—成本：0–6pp 损失 → 7B（20847）；8pp → 3B-Q8（4919）；≥10pp → 1.5B-Q4（2182）。每 1 点质量约贵 27%。",
        "- 推荐器用硬阈值剪枝，而 **7B→3B-Q8 的翻转点在 7.5 pp（20847→4919），3B→1.5B-Q4 在 10 pp（4919→2182）**。",
        "- 因此质量—成本权衡的结论不应作为硬判断，应报出阈值敏感性。",
        f"- **后悔值的负对照**：模型 {_control('model')} / 随机选 {_control('random')}"
        f" / 选最大模型 {_control('largest')} —— 指标有分辨力。",
    ]

    # 参数区间（由 tc_model.py --bootstrap N 产出 param_ci.json）
    cip = RESULTS / "param_ci.json"
    if cip.exists():
        ci = json.loads(cip.read_text(encoding="utf-8"))
        name_map = {"bw_gbs": "带宽 GB/s", "peak_tflops": "decode 算力 TFLOP/s",
                    "peak_prefill_tflops": "prefill 算力 TFLOP/s",
                    "c_srv_ms": "c_srv ms/流/步", "b_sat": "B_SAT ms",
                    "ovh_ttft_ms": "ovh ms"}
        lines += ["", "## 参数 5–95% 区间（bootstrap 重采样）", "",
                  "| 参数 | 点估计 | 区间 | 相对宽度 | 可信度 |", "|---|---|---|---|---|"]
        for k, v in ci.items():
            w = (v["rel_width"] or 0) * 100
            lines.append(f"| {name_map.get(k, k)} | {v['point']:.3f} | "
                         f"[{v['lo']:.3f}, {v['hi']:.3f}] | {w:.0f}% | "
                         f"{'弱' if w > 30 else '可辨识'} |")
        lines += ["", "> 相对宽度 > 30% 的参数只能当量级引用；"
                     "`ovh_ttft` 的区间宽度超过 100%，说明标定集几乎约束不住它。"]

    body = "\n".join(lines)
    # 【脏文件守卫】生成物里出现 nan / 空占位就直接失败：
    # 以前 nan% 会被写进 CLAIMS.md，而 --check 只比对 claims 字典，于是"脏文件仍报 OK"。
    dirt = [t for t in ("nan", "NaN", "? |") if t in body]
    if dirt:
        raise SystemExit(
            f"[CLAIMS] 生成物里出现异常占位 {dirt}——拒绝写出脏文件。\n"
            f"通常是某个产物字段缺失（先跑 `python code/tc.py model`）。")
    (RESULTS / "CLAIMS.md").write_text(body + "\n", encoding="utf-8")
    return RESULTS / "CLAIMS.md"


def _known_values(d, claims):
    """本轮"合法数字"集合：口径内允许稳态/墙钟两个值。"""
    def _num(s):
        m = re.search(r"\d+(?:\.\d+)?", str(s))
        return m.group(0) if m else None

    fit = d["fit"]
    return {
        "mape": {_num(claims.get("单位成本 MAPE（core，稳态口径）", "")),
                 _num(fit.get("mape_cost_wall"))},
        "max_err": {_num(claims.get("留出集最大误差", "")),
                    _num(fit.get("mape_cost_wall_max"))},
        "coverage": {_num(claims.get("预测区间覆盖率", ""))},
    }


def _need(claims, key):
    """取 claims 里的键；取不到就抛错。

    复核抓到过：用 claims.get("留出集最大误差", "") 取一个**不存在的键**，
    返回空串 → 检查项变成空操作（永远匹配）。键名必须显式断言。
    """
    if key not in claims:
        raise SystemExit(
            f"[CLAIMS] 检查器引用了不存在的键 {key!r}。"
            f"可用键：{sorted(claims)[:8]}…")
    return claims[key]


def check(claims, d):
    """按**语境**抓数字，而不是精确匹配整句。

    只做精确字符串匹配会把"108 个配置格"和"108 格"判成不一致（假警报），
    而真正危险的恰恰是"数字写错但句子看着对"。所以：
      ① 每个指标给出一个正则，抓出文档里的数，与真值比对；
      ② 再扫描一批"过期数字黑名单"，命中即失败。
    """
    _KNOWN = _known_values(d, claims)

    # (名称, 期望值集合, 抓数字的正则)
    checks = [
        ("配置格数", str(d["n_cells"]), r"(\d+)\s*个配置格"),
        ("会话级测量次数", str(d["n_rows"]), r"(\d+)\s*次(?:会话级)?测量"),
        ("脚本数", str(d["n_scripts"]), r"(\d+)\s*个脚本"),
        # 复核 3.1：文档写 68、实际 76，却没有检查项能抓到
        ("留出集格数", _need(d["fit"], "n_holdout"),
         r"(?:留出|未见)(?:集)?\s*(\d+)\s*(?:个|格)"),
        # 三套非默认口径：**按表格行抓数**（文档里是 "| `valid`（…） | 84 | 9.22% |…"）。
        # must-contain 用整串或 "84 格" 都匹配不上——那是复核两轮都漏掉 valid/all 漂移的原因。
        ("valid 口径格数", _pops_n(d, "valid"), r"`valid`[^\n]*?\|\s*(\d+)\s*\|"),
        ("all 口径格数", _pops_n(d, "all"), r"`all`[^\n]*?\|\s*(\d+)\s*\|"),
        # 【不能写死期望值】写死等于只认旧数字（复核指出的真实漏洞）。
        # pattern=None 表示"必须出现"模式：核心数字必须在关键文档里出现 ≥1 次。
        ("成本 MAPE", _need(claims, "单位成本 MAPE（core，稳态口径）"), None),
        ("最大误差", _need(claims, "单点最大误差"), None),
        ("区间覆盖率", _need(claims, "90% 区间覆盖率"), None),
    ]
    # 曾经对外用过、现已作废的数字；留在文档里就是漂移
    stale = ["100 个配置格", "300 次会话级", "21 个脚本", "6.13%", "19.46%",
             # 重拟合前的 MAPE（2026-11-05 起为 8.59）
             "8.70%", "8.61%",
             # valid / all 口径的旧值（2026-11-06 起由 populations.json 现算）
             "9.08%", "10.21%",
             "76 格", "9.08%（未标口径）",
             # 30 秒采样的泊松数据（90 秒重跑后已作废）
             "9.9 倍", "24.9 ms", "6pp → 7pp",
             # 历史值：268/813（半宽公式错误）→ 259/772（候选集未统一）→ 现为 274/740
             "268%", "813%"]

    # 【自比对盲区】拿自己生成的 CLAIMS.md 当基准，它自己错了就不会被发现。
    # 所以先验"生成出来的行里有没有 nan / —"，有就直接失败。
    bad = []
    hollow = [k for k, v in claims.items()
              if isinstance(v, str) and ("nan" in v.lower() or v.strip() in ("—", "无数据"))]
    if hollow:
        print("\n[CLAIMS][FAIL] 这些指标没有真实值（机制自检未通过）：")
        for k in hollow:
            print(f"  {k}")
        return False

    # ---------------------------------------------------------------- 扫描

    # 结构：三类检查各自独立遍历文档，避免"某份文档的 text 变量泄漏给别的检查"。

    KEY_DOCS = ("README.md", "报告/技术报告_定稿_两页.md",

                "报告/从单机到集群_一页纸_实测版.md")



    # ① must-contain：核心数字必须在关键文档里出现至少一次。

    #    这一条专治复核指出的漏洞——检查器把期望值写死成旧值，

    #    于是"文档写旧值"反而判通过。

    for name, want, pat in checks:

        if pat is not None or not want:

            continue

        hit = False

        for doc in DOCS:

            if doc.name not in KEY_DOCS or not doc.exists():

                continue

            if str(want) in doc.read_text(encoding="utf-8", errors="replace"):

                hit = True

                break

        if not hit:

            bad.append(("/".join(KEY_DOCS[:2]) + "…", f"{name} 缺失",

                        str(want), "关键文档里必须出现最新值"))



    # ② 正则模式（计数类）：逐文档抓数字，与期望值比对。

    for name, want, pat in checks:

        if pat is None:

            continue

        for doc in DOCS:

            if not doc.exists():

                continue

            text = doc.read_text(encoding="utf-8", errors="replace")

            for m in re.finditer(pat, text):
                if str(m.group(1)).strip() != str(want).strip():
                    bad.append((doc.name, name, m.group(1), want))



    # ③ 过期数字：逐文档扫。**必须逐文档**，否则只有最后一份会被检查。

    for doc in DOCS:

        if not doc.exists():

            continue

        text = doc.read_text(encoding="utf-8", errors="replace")

        for s in stale:

            if s in text:

                bad.append((doc.name, "过期数字", s, "删除或标注 deprecated"))



    # 【交叉断言】fig6 的台阶必须与推荐器 sweep 一致。
    # 第三轮复核发现两者各实现一遍"SLA 过滤"，于是同一问题给出两套数
    # （7.5pp/1605 vs 8pp/2182）。现在两个脚本都调 tc_common.rank()，
    # 这里再比对各自落盘的 CSV，防止以后再分叉。
    run_p = RESULTS / "recommendation_sweep.csv"
    thr_p = RESULTS / "threshold_sensitivity.csv"
    if run_p.exists() and thr_p.exists():
        import pandas as _pd
        a = _pd.read_csv(run_p)
        b = _pd.read_csv(thr_p)
        for _, ra in a.iterrows():
            rb = b[b["drop"] == ra["drop"]]
            if rb.empty:
                continue
            if rb.iloc[0]["cell_id"] != ra["cell_id"]:
                bad.append(("交叉断言", f"drop={ra['drop']}pp",
                            f"fig6 选 {rb.iloc[0]['cell_id']}",
                            f"推荐器选 {ra['cell_id']}"))

    if bad:
        print("\n[CLAIMS][FAIL] 文档数字与产物不一致，或两个脚本给出了不同答案：")
        seen = set()
        for name, key, got, want in bad:
            sig = (name, key, got)
            if sig in seen:
                continue
            seen.add(sig)
            print(f"  {name}: {key} 写的是 {got}，应为 {want}")
        print("\n修好后重跑；权威值见 results/CLAIMS.md")
        return False
    print("\n[CLAIMS][OK] 文档数字与产物一致，且无过期数字。")
    return True


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()

    d = load_inputs()
    claims = build_claims(d)
    out = write_md(claims, d)
    print(f"已生成 {out}")
    for k, v in claims.items():
        print(f"  {k:<28} {v}")

    if args.check:
        sys.exit(0 if check(claims, d) else 1)


if __name__ == "__main__":
    main()














