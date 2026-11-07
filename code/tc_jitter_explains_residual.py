# -*- coding: utf-8 -*-
"""
M8：抖动律能不能解释 TPOT 残差（`tc.py jitterresid`）

## 假说

服务时间有抖动（§5.27 的 `CV(b,L)`，R²=0.994）→ 同批里快请求等慢请求 →
**平均每步耗时被抬高**。若成立：

    TPOT_meas / TPOT_model  ≈  1 + k · CV²

（若按"等待时间与标准差成正比"，则是一次方；若与方差成正比，则是平方。
 两个都试，让数据选。）

## 【预注册判据】先写死，再算

只有**同时**满足下列三条才采纳：

1. **样本外有效**：5 折交叉验证下，TPOT 的 MAPE 相对"不修正"改善 **≥20%**；
2. **k 稳定**：5 折里 k 的变异系数 **<30%**，且符号为正；
3. **不是靠一个点撑起来的**：去掉最大的 5 个 CV 后仍改善 ≥10%。

任一不满足 → 记为负结果，不写进模型。

## 用法

    python code/tc_jitter_explains_residual.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import tc_config as cfg

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
RNG = np.random.default_rng(42)


def fit_k(x, y):
    """过原点的最小二乘：y-1 = k·x，返回 k。"""
    d = np.dot(x, y) / np.dot(x, x) if np.dot(x, x) > 0 else 0.0
    return float(d)


def main():
    p = pd.read_csv(cfg.RESULTS_DIR / "predictions.csv")
    j = pd.read_csv(cfg.RESULTS_DIR / "jitter_by_cell.csv")
    m = p.merge(j[["cell_id", "cv"]], on="cell_id", how="inner")
    m["y"] = m["meas_tpot"] / m["pred_tpot_ms"]      # 实测 / 模型
    m["cv2"] = m["cv"] ** 2
    print("=" * 82)
    print(f"M8：抖动能否解释 TPOT 残差（{len(m)} 格，其中留出 {int((m.group=='留出').sum())} 格）")
    print("=" * 82)
    print(f"  y = 实测/模型：中位 {m['y'].median():.3f}，"
          f"偏差 |y-1| 中位 {np.median(np.abs(m['y']-1))*100:.1f}%")
    print(f"  CV：中位 {m['cv'].median()*100:.2f}%，最大 {m['cv'].max()*100:.1f}%")
    print(f"  corr(y-1, CV)   = {np.corrcoef(m['cv'], m['y']-1)[0,1]:+.3f}")
    print(f"  corr(y-1, CV²)  = {np.corrcoef(m['cv2'], m['y']-1)[0,1]:+.3f}")

    # 全样本拟合（只为看 k 的量级）
    for power, name in ((1, "一次方 k·CV"), (2, "平方 k·CV²")):
        x = m["cv"] ** power
        k = fit_k(x.to_numpy(), (m["y"] - 1).to_numpy())
        pred = 1 + k * x
        mape = float(np.mean(np.abs(pred - m["y"]) / m["y"]) * 100)
        base = float(np.mean(np.abs(1 - m["y"]) / m["y"]) * 100)
        print(f"\n  【{name}】k = {k:+.3f}    全样本 MAPE {mape:.2f}%"
              f"（不修正 {base:.2f}%）")

    # ---------- 5 折交叉验证 ----------
    print("\n" + "=" * 82)
    print("5 折交叉验证（预注册判据：改善≥20%、k 的 CV<30%、去掉最大 5 个 CV 后仍≥10%）")
    print("=" * 82)
    results = {}
    idx = RNG.permutation(len(m))
    folds = np.array_split(idx, 5)
    for power, name in ((1, "k·CV"), (2, "k·CV²")):
        ks, mapes, bases = [], [], []
        for i in range(5):
            te = m.iloc[folds[i]]
            tr = m.iloc[np.concatenate([folds[j] for j in range(5) if j != i])]
            x_tr = (tr["cv"] ** power).to_numpy()
            k = fit_k(x_tr, (tr["y"] - 1).to_numpy())
            pred = 1 + k * (te["cv"] ** power)
            mapes.append(float(np.mean(np.abs(pred - te["y"]) / te["y"]) * 100))
            bases.append(float(np.mean(np.abs(1 - te["y"]) / te["y"]) * 100))
            ks.append(k)
        ks = np.array(ks)
        cv_of_k = float(np.std(ks) / abs(np.mean(ks))) if np.mean(ks) else float("inf")
        imp = (np.mean(bases) - np.mean(mapes)) / np.mean(bases) * 100
        # 去极值检查
        keep = m.nsmallest(len(m) - 5, "cv")
        xk = (keep["cv"] ** power).to_numpy()
        k2 = fit_k(xk, (keep["y"] - 1).to_numpy())
        pred2 = 1 + k2 * xk
        imp2 = float((np.mean(np.abs(1 - keep["y"]) / keep["y"])
                      - np.mean(np.abs(pred2 - keep["y"]) / keep["y"]))
                     / np.mean(np.abs(1 - keep["y"]) / keep["y"]) * 100)
        results[name] = dict(k_mean=round(float(np.mean(ks)), 3),
                             k_cv_pct=round(cv_of_k * 100, 1),
                             mape_cv=round(float(np.mean(mapes)), 2),
                             base_cv=round(float(np.mean(bases)), 2),
                             improvement_pct=round(float(imp), 1),
                             improvement_no_outlier_pct=round(imp2, 1))
        ok = (imp >= 20) and (cv_of_k * 100 < 30) and (imp2 >= 10)
        print(f"\n  【{name}】")
        print(f"    k = {np.mean(ks):+.3f}（5 折变异 {cv_of_k*100:.0f}%）")
        print(f"    样本外 MAPE：{np.mean(mapes):.2f}% vs 不修正 {np.mean(bases):.2f}%"
              f"  → 改善 {imp:.0f}%")
        print(f"    去掉最大 5 个 CV 后：改善 {imp2:.0f}%")
        print(f"    预注册三条判据：改善≥20% [{'✅' if imp>=20 else '❌'}] · "
              f"k 的 CV<30% [{'✅' if cv_of_k*100<30 else '❌'}] · "
              f"去极值后≥10% [{'✅' if imp2>=10 else '❌'}]")
        print(f"    → {'✅ 采纳' if ok else '❌ 不采纳（记负结果）'}")
        results[name]["accepted"] = bool(ok)

    # ---------- 用"律预测的 CV"是否同样有效 ----------
    print("\n" + "=" * 82)
    print("附加检查：不用逐请求抖动，改用 §5.29 的律算 CV，效果是否一样")
    print("=" * 82)
    tv = json.loads((cfg.RESULTS_DIR / "tail_variance.json").read_text(encoding="utf-8"))
    cv_law = tv["c0"] + tv["a"] * np.maximum(m["conc"] - 1, 1e-9) ** tv["exp_b"] \
        * (m["prompt_tokens"] + 64) ** tv["exp_L"]
    for power, name in ((1, "律·CV"), (2, "律·CV²")):
        x = cv_law ** power
        pred = 1 + fit_k(x.to_numpy(), (m["y"] - 1).to_numpy()) * x
        mape = float(np.mean(np.abs(pred - m["y"]) / m["y"]) * 100)
        print(f"  {name}：全样本 MAPE {mape:.2f}%")



    # ---------- A10 数据检验（CV 高，才有统计力） ----------
    print("\n" + "=" * 82)
    print("A10 数据：直接用逐请求探针的 (b, L) 格子与各自标定参数算 y = 实测/模型")
    print("=" * 82)
    import tc_model as tm
    cfg.MODEL_SPECS[("3B", "fp16")] = dict(weight_bytes=6178317312,
                                          n_params=3080000000, n_layers=36,
                                          n_kv_heads=2, head_dim=128)
    sets = [
        ("vLLM", "calib_a10_vllm.json",
         ["jitter_a10_vllm_L64.json", "jitter_a10_vllm_L256.json",
          "jitter_a10_vllm.json", "jitter_a10_vllm_high.json"]),
        ("llama.cpp", "calib_a10_llamacpp.json",
         ["jitter_a10_llamacpp.json", "jitter_a10_llamacpp_high2.json"]),
    ]
    rows = []
    for stack, cal_name, jf in sets:
        cf = cfg.RESULTS_DIR / "a10_raw" / cal_name
        if not cf.exists():
            continue
        cal = json.loads(cf.read_text(encoding="utf-8"))
        params = dict(bw_gbs=cal["bw"], peak_tflops=cal["peak"],
                      peak_prefill_tflops=cal["peak_prefill"],
                      c_srv_ms=cal["c_srv"], b_sat=cal["b_sat"],
                      ovh_ttft_ms=cal["ovh_ttft"])
        for name in jf:
            jf_path = cfg.RESULTS_DIR / "a10_raw" / name
            if not jf_path.exists():
                continue
            d = json.loads(jf_path.read_text(encoding="utf-8"))
            L = float(d["length"])
            for k, v in d["cells"].items():
                b = int(k)
                row = dict(model="3B", quant="fp16", ctx=2304, conc=b,
                           prompt_tokens=L, ctx_avg_tokens=L + 64)
                pr = tm.predict(params, row)
                meas = float(v["tpot_p50"])
                rows.append(dict(stack=stack, b=b, L=L, cv=float(v["tpot_cv"]),
                                 y=meas / pr["pred_tpot_ms"]))
    if len(rows) >= 6:
        a = pd.DataFrame(rows)
        a["cv2"] = a["cv"] ** 2
        print(f"  点数 {len(a)}｜CV 中位 {a['cv'].median()*100:.1f}%（最高 "
              f"{a['cv'].max()*100:.1f}%）")
        print(f"  y 中位 {a['y'].median():.3f}｜|y-1| 中位 "
              f"{np.median(np.abs(a['y']-1))*100:.1f}%")
        print(f"  corr(y-1, CV) = {np.corrcoef(a['cv'], a['y']-1)[0,1]:+.3f}")
        print(f"  corr(y-1, CV²)= {np.corrcoef(a['cv2'], a['y']-1)[0,1]:+.3f}")
        for power, nm in ((1, "k·CV"), (2, "k·CV²")):
            x = (a["cv"] ** power).to_numpy()
            k = fit_k(x, (a["y"] - 1).to_numpy())
            pred = 1 + k * (a["cv"] ** power)
            mape = float(np.mean(np.abs(pred - a["y"]) / a["y"]) * 100)
            base = float(np.mean(np.abs(1 - a["y"]) / a["y"]) * 100)
            print(f"  【{nm}】k = {k:+.3f}   MAPE {mape:.2f}%（不修正 {base:.2f}%）")
        # 留一栈交叉验证：用 llama.cpp 拟、vLLM 验（或反之）
        for tr_stack, te_stack in (("vLLM", "llama.cpp"), ("llama.cpp", "vLLM")):
            tr = a[a["stack"] == tr_stack]
            te = a[a["stack"] == te_stack]
            if len(tr) < 4 or len(te) < 3:
                continue
            x_tr = tr["cv"].to_numpy()
            k = fit_k(x_tr, (tr["y"] - 1).to_numpy())
            pred = 1 + k * te["cv"]
            mape = float(np.mean(np.abs(pred - te["y"]) / te["y"]) * 100)
            base = float(np.mean(np.abs(1 - te["y"]) / te["y"]) * 100)
            print(f"  跨栈：用 {tr_stack} 拟合 k={k:+.3f} → 在 {te_stack} 上 "
                  f"MAPE {mape:.1f}%（不修正 {base:.1f}%）")
        a10_stats = dict(n=len(a),
                         corr_cv=round(float(np.corrcoef(a["cv"], a["y"]-1)[0, 1]), 3),
                         corr_cv2=round(float(np.corrcoef(a["cv2"], a["y"]-1)[0, 1]), 3))
    else:
        a10_stats = None

    out = dict(n_cells=len(m),
               corr_cv=round(float(np.corrcoef(m["cv"], m["y"]-1)[0, 1]), 3),
               corr_cv2=round(float(np.corrcoef(m["cv2"], m["y"]-1)[0, 1]), 3),
               cv_folds=results,
               preregistered=dict(min_improvement_pct=20, max_k_cv_pct=30,
                                  min_improvement_no_outlier_pct=10),
               verdict=("采纳" if any(r["accepted"] for r in results.values())
                        else "负结果：抖动不能解释 TPOT 残差"),
               a10=a10_stats,
               note=("y = 实测/模型；本机 CV 中位仅 1.5%，统计力不足；"
                     "A10 组 CV 最高 26%，是主要检验"))
    (cfg.RESULTS_DIR / "jitter_residual_test.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n→ results/jitter_residual_test.json")


if __name__ == "__main__":
    main()
