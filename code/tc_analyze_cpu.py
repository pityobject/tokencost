# -*- coding: utf-8 -*-
"""CPU 迁移性分析：只重标定硬件常数，看留出格还准不准。

## 检验的那句话

"模型结构跟硬件无关，换机器只需重标定两个常数。"

严格版本（本脚本做的）：
  · 重标定 **三个硬件常数**：带宽 BW、decode 算力、prefill 算力
  · **不动**三个部署栈常数：c_srv、B_SAT、ovh_ttft（它们属于 llama.cpp 的服务实现）
  · 硬件常数用 4 个 CPU 标定格拟合，2 个留出格做检验

如果留出格误差小 → "换硬件只改硬件常数"成立；
如果 c_srv 也必须改 → 说明它含硬件相关成分，报告的表述要修正。
"""
import json
import sys

import numpy as np
import pandas as pd
from scipy.optimize import minimize

import tc_config as cfg
import tc_model as tm

sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def main():
    path = cfg.DATA_DIR / "probe_cpu.csv"
    if not path.exists():
        raise SystemExit(f"没有 {path}，先跑 tc_probe_cpu.py")

    df = pd.read_csv(path)
    df["prompt_tokens"] = float(cfg.PROMPT_TOKENS)
    df["ctx_avg_tokens"] = df["prompt_tokens"] + cfg.GEN_TOKENS / 2

    # 先记下每格属于标定还是留出，再聚合成一格一行。
    # 注意不能直接拿 groupby 产生的布尔序列去索引另一个 DataFrame——
    # 两者的索引不是同一套（一个是 cell_id，一个是 RangeIndex），pandas 会报
    # "Unalignable boolean Series"。用 map 贴回去最稳。
    group_of = df.groupby("cell_id")["group"].first()
    # median(numeric_only=True) 会把 model/quant 这些文本列丢掉，
    # 但下面的 predict() 需要它们，所以维度列单独取回来再合并。
    # median(numeric_only=True) 会把文本列丢掉，但 ctx/conc 是数值列会被保留，
    # 所以这里只补 model/quant（否则合并会出现 conc_x / conc_y 重名）。
    dims = df.groupby("cell_id", as_index=False)[["model", "quant"]].first()
    g = df.groupby("cell_id", as_index=False).median(numeric_only=True)
    g = dims.merge(g, on="cell_id", how="left")
    g["group"] = g["cell_id"].map(group_of)

    gpu = json.loads((cfg.RESULTS_DIR / "fit.json").read_text(encoding="utf-8"))
    base = dict(bw_gbs=gpu["bw"], peak_tflops=gpu["peak"],
                peak_prefill_tflops=gpu.get("peak_prefill", gpu["peak"]),
                c_srv_ms=gpu["c_srv"], b_sat=gpu["b_sat"], ovh_ttft_ms=gpu["ovh_ttft"])

    calib = g[g["group"] == "标定"].copy()
    hold = g[g["group"] == "留出"].copy()
    print(f"CPU 数据：标定 {len(calib)} 格、留出 {len(hold)} 格\n")

    def loss(theta):
        """scipy 只传 theta 一个参数，其余从闭包取。"""
        q = dict(base, bw_gbs=float(np.exp(theta[0])),
                 peak_tflops=float(np.exp(theta[1])),
                 peak_prefill_tflops=float(np.exp(theta[2])))
        s = 0.0
        for _, r in calib.iterrows():
            pr = tm.predict(q, r)
            for a, b in (("pred_tpot_ms", "tpot_p50_ms"), ("pred_ttft_ms", "ttft_p50_ms")):
                v = pr[a] / r[b]
                if not np.isfinite(v) or v <= 0:
                    return 1e9
                s += np.log(v) ** 2
        return s

    x0 = [np.log(40.0), np.log(1.0), np.log(2.0)]     # CPU 起点远低于 GPU
    res = minimize(loss, x0, method="L-BFGS-B",
                   bounds=[(np.log(1), np.log(600)), (np.log(0.01), np.log(300)),
                           (np.log(0.01), np.log(300))])
    cpu = dict(base, bw_gbs=float(np.exp(res.x[0])),
               peak_tflops=float(np.exp(res.x[1])),
               peak_prefill_tflops=float(np.exp(res.x[2])))

    print("=" * 70)
    print("重标定后的硬件常数（c_srv / B_SAT / ovh 保持不变）")
    print("=" * 70)
    print(f"  带宽      GPU {base['bw_gbs']:8.1f} → CPU {cpu['bw_gbs']:8.1f} GB/s"
          f"   （比例 1/{base['bw_gbs']/cpu['bw_gbs']:.1f}）")
    print(f"  decode 算力 GPU {base['peak_tflops']:6.1f} → CPU {cpu['peak_tflops']:6.1f} TFLOP/s"
          f"   （1/{base['peak_tflops']/cpu['peak_tflops']:.1f}）")
    print(f"  prefill 算力 GPU {base['peak_prefill_tflops']:6.1f} → CPU "
          f"{cpu['peak_prefill_tflops']:6.1f} TFLOP/s   （1/"
          f"{base['peak_prefill_tflops']/cpu['peak_prefill_tflops']:.1f}）")
    print(f"  保持不动：c_srv {base['c_srv_ms']:.3f} ms  B_SAT {base['b_sat']:.1f}  "
          f"ovh {base['ovh_ttft_ms']:.1f} ms")

    print("\n" + "=" * 70)
    print("留出格检验（模型没见过这两格）")
    print("=" * 70)
    print(f"{'cell':<28}{'实测TPOT':>9}{'预测TPOT':>9}{'误差':>8}{'实测TTFT':>10}{'预测TTFT':>10}{'误差':>8}")
    errs = []
    for _, r in hold.iterrows():
        pr = tm.predict(cpu, r)
        e1 = pr["pred_tpot_ms"] / r["tpot_p50_ms"] - 1
        e2 = pr["pred_ttft_ms"] / r["ttft_p50_ms"] - 1
        errs += [abs(e1), abs(e2)]
        print(f"{r['cell_id']:<28}{r['tpot_p50_ms']:>9.2f}{pr['pred_tpot_ms']:>9.2f}"
              f"{e1*100:>7.0f}%{r['ttft_p50_ms']:>10.1f}{pr['pred_ttft_ms']:>10.1f}{e2*100:>7.0f}%")
    print(f"\n留出格平均绝对误差 {np.mean(errs)*100:.1f}%")

    # 若允许 c_srv 也重标定，能好多少？
    def loss2(theta):
        q = dict(cpu, c_srv_ms=float(theta[0]))
        s = 0.0
        for _, r in calib.iterrows():
            pr = tm.predict(q, r)
            for a, b in (("pred_tpot_ms", "tpot_p50_ms"), ("pred_ttft_ms", "ttft_p50_ms")):
                s += np.log(pr[a] / r[b]) ** 2
        return s

    r2 = minimize(loss2, [base["c_srv_ms"]], method="L-BFGS-B", bounds=[(0.0, 50.0)])
    c_srv_cpu = float(r2.x[0])
    q2 = dict(cpu, c_srv_ms=c_srv_cpu)
    errs2 = []
    for _, r in hold.iterrows():
        pr = tm.predict(q2, r)
        errs2 += [abs(pr["pred_tpot_ms"] / r["tpot_p50_ms"] - 1),
                  abs(pr["pred_ttft_ms"] / r["ttft_p50_ms"] - 1)]
    print(f"\n若再放开 c_srv：CPU 拟合值 {c_srv_cpu:.3f} ms（GPU {base['c_srv_ms']:.3f}），"
          f"留出误差 {np.mean(errs2)*100:.1f}%")
    print("  判读：若 c_srv 变化大而误差提升小 → 它更像是拟合残差的吸收项，而不是可解释的物理量；")
    print("        若 c_srv 变化小 → 说明它确实与计算后端无关（在 HTTP/采样层）。")

    out = cfg.RESULTS_DIR / "cpu_transfer.json"
    out.write_text(json.dumps(dict(cpu_params=cpu, base_params=base,
                                   c_srv_cpu=c_srv_cpu,
                                   holdout_mae=float(np.mean(errs) * 100),
                                   holdout_mae_with_csrv=float(np.mean(errs2) * 100)),
                              ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n已写入 {out}")

    # ------------------------------------------------------------------
    # 公平检验：留出格 c8 属于"未标定的中间并发"，与"硬件常数是否可迁移"混杂。
    # 把它分开：在**标定过的并发**上做跨模型留一验证——
    #   用 1.5B 的格标定 → 预测 3B；用 3B 标定 → 预测 1.5B。
    # 这样考的才是"换模型后硬件常数还成不成立"。
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("公平检验：同并发、跨模型（标定一个模型，预测另一个）")
    print("=" * 70)
    calib_all = g[g["conc"].isin([1, 4])].copy()
    results = []
    for fit_model, pred_model in (("1.5B", "3B"), ("3B", "1.5B")):
        fit_rows = calib_all[calib_all["model"] == fit_model]
        pred_rows = calib_all[calib_all["model"] == pred_model]
        if len(fit_rows) < 2 or not len(pred_rows):
            continue

        def loss_loo(theta):
            q = dict(base, bw_gbs=float(np.exp(theta[0])),
                     peak_tflops=float(np.exp(theta[1])),
                     peak_prefill_tflops=float(np.exp(theta[2])))
            s = 0.0
            for _, r in fit_rows.iterrows():
                pr = tm.predict(q, r)
                for a, b in (("pred_tpot_ms", "tpot_p50_ms"),
                             ("pred_ttft_ms", "ttft_p50_ms")):
                    v = pr[a] / r[b]
                    if not np.isfinite(v) or v <= 0:
                        return 1e9
                    s += np.log(v) ** 2
            return s

        rr = minimize(loss_loo, [np.log(40.0), np.log(1.0), np.log(2.0)], method="L-BFGS-B",
                      bounds=[(np.log(1), np.log(600)), (np.log(0.01), np.log(300)),
                              (np.log(0.01), np.log(300))])
        q = dict(base, bw_gbs=float(np.exp(rr.x[0])),
                 peak_tflops=float(np.exp(rr.x[1])),
                 peak_prefill_tflops=float(np.exp(rr.x[2])))
        e = []
        for _, r in pred_rows.iterrows():
            pr = tm.predict(q, r)
            e += [abs(pr["pred_tpot_ms"] / r["tpot_p50_ms"] - 1),
                  abs(pr["pred_ttft_ms"] / r["ttft_p50_ms"] - 1)]
        mae = float(np.mean(e) * 100)
        results.append(dict(fit_model=fit_model, pred_model=pred_model, mae=mae,
                            bw=float(np.exp(rr.x[0])), peak_dec=float(np.exp(rr.x[1])),
                            peak_pre=float(np.exp(rr.x[2]))))
        print(f"  用 {fit_model:>5} 标定 → 预测 {pred_model:>5}：平均误差 {mae:5.1f}%"
              f"   (BW {np.exp(rr.x[0]):.1f} GB/s, decode {np.exp(rr.x[1]):.2f} TFLOP/s)")

    if results:
        avg = float(np.mean([r["mae"] for r in results]))
        print(f"\n  跨模型平均误差 {avg:.1f}%")
        print("  判读：<20% → 硬件常数可迁移；>40% → 换机器/换模型都要重测更多常数")
        d = json.loads(out.read_text(encoding="utf-8"))
        d["loo_cross_model"] = results
        d["loo_mae"] = avg
        out.write_text(json.dumps(d, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
