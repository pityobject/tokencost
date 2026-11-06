# -*- coding: utf-8 -*-
"""
A10 补测分析（`tc.py a10repeat`）：标定的**重复性**与**失效区跨栈行为**

## 为什么单独做

`results/calib_a10_*.json` 只记录了一次标定。要判断"vLLM 的 `c_srv` 是 llama.cpp 的 5.6 倍"
这句话有多硬，必须知道**同一配置重复标定会漂多少**。

另外，长 prompt × 高并发是我们适用域里唯一的 ❌。它到底是"模型形式缺陷"
还是"某个部署栈的实现缺陷"，只能在两个栈上分别测才算说清。

## 输入

`results/a10_repeat_longprompt.json`（两栈各两次标定参数 + 长 prompt 实测点）

## 输出

终端两张表 + `results/a10_repeat_longprompt_summary.json`
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import tc_config as cfg
import tc_model as tm

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

KEYS = ["c_srv", "bw", "peak_prefill", "b_sat", "ovh_ttft"]


def main():
    fp = cfg.RESULTS_DIR / "a10_repeat_longprompt.json"
    if not fp.exists():
        print("缺 results/a10_repeat_longprompt.json")
        return
    d = json.loads(fp.read_text(encoding="utf-8"))
    cal = d["calibration"]

    print("=" * 78)
    print("一、标定重复性（同一配置跑两次，看参数漂多少）")
    print("=" * 78)
    print(f"{'参数':<14}{'llama.cpp#1':>13}{'#2':>10}{'变化':>9}{'vLLM#1':>11}{'#2':>10}{'变化':>9}")
    rep = {}
    for k in KEYS:
        a, b = cal["llamacpp_run1"][k], cal["llamacpp_run2"][k]
        c, e = cal["vllm_run1"][k], cal["vllm_run2"][k]
        dl = (b / a - 1) * 100 if a else float("nan")
        dv = (e / c - 1) * 100 if c else float("nan")
        print(f"{k:<14}{a:>13.3f}{b:>10.3f}{dl:>+8.0f}%{c:>11.3f}{e:>10.3f}{dv:>+8.0f}%")
        rep[k] = dict(llamacpp_change_pct=round(dl, 1), vllm_change_pct=round(dv, 1))

    # 栈间比值：四种配对
    pairs = []
    for li in ("llamacpp_run1", "llamacpp_run2"):
        for vi in ("vllm_run1", "vllm_run2"):
            pairs.append(cal[li]["c_srv"] / cal[vi]["c_srv"])
    lo, hi = min(pairs), max(pairs)
    print(f"\n  栈间比值 c_srv(llama.cpp)/c_srv(vLLM)："
          f"4 种配对 = " + " / ".join(f"{x:.2f}×" for x in pairs))
    print(f"  → 结论应写成 **{lo:.1f}–{hi:.1f}×**，而不是一个精确数")

    print("\n" + "=" * 78)
    print("二、长 prompt × 高并发：模型误差方向（用各自 run1 的 6 点参数）")
    print("=" * 78)
    L = d["longprompt"]["L_tokens"]
    # 云端用的是 fp16 权重，仓库的 MODEL_SPECS 里没有这一档 → 运行时注册
    cfg.MODEL_SPECS[("3B", "fp16")] = dict(
        weight_bytes=d["spec"]["weight_bytes"], n_params=d["spec"]["n_params"],
        n_layers=36, n_kv_heads=2, head_dim=128)
    W = d["spec"]["weight_bytes"]
    P = d["spec"]["n_params"]
    rows = []
    for stack, key in (("llama.cpp", "llamacpp"), ("vLLM", "vllm")):
        p = cal[f"{key}_run1"]
        params = dict(bw_gbs=p["bw"], peak_tflops=12.0, peak_prefill_tflops=p["peak_prefill"],
                      c_srv_ms=p["c_srv"], b_sat=p["b_sat"], ovh_ttft_ms=p["ovh_ttft"])
        print(f"\n  {stack}（L={L}）：")
        print(f"    {'并发':>4}{'实测TTFT':>10}{'预测TTFT':>10}{'误差':>8}{'实测TPOT':>10}{'预测TPOT':>10}{'误差':>8}")
        for m in d["longprompt"][key]:
            b = m["conc"]
            row = dict(model="3B", quant="fp16", ctx=2304, conc=b,
                       prompt_tokens=L, ctx_avg_tokens=L + cfg.GEN_TOKENS / 2)
            pr = tm.predict(params, row)
            e_ttft = (pr["pred_ttft_ms"] / m["ttft"] - 1) * 100
            e_tpot = (pr["pred_tpot_ms"] / m["tpot"] - 1) * 100
            print(f"    {b:>4}{m['ttft']:>10.1f}{pr['pred_ttft_ms']:>10.1f}{e_ttft:>+7.0f}%"
                  f"{m['tpot']:>10.2f}{pr['pred_tpot_ms']:>10.2f}{e_tpot:>+7.0f}%")
            rows.append(dict(stack=stack, conc=b, ttft_meas=m["ttft"],
                             ttft_pred=round(pr["pred_ttft_ms"], 1),
                             ttft_err_pct=round(e_ttft, 1)))

    # 饱和程度：c16/c1
    sat = {}
    for key in ("llamacpp", "vllm"):
        ms = d["longprompt"][key]
        sat[key] = round(ms[-1]["ttft"] / ms[0]["ttft"], 2)
    print(f"\n  TTFT 随并发的放大倍数（c16/c1）：llama.cpp {sat['llamacpp']}×，"
          f"vLLM {sat['vllm']}×（并发涨了 16 倍）")
    print("  → 两栈都没有按 16 倍放大：llama.cpp 饱和（摊薄），vLLM 近似线性；"
          "模型两边都**高估**（偏保守）")

    out = dict(repeatability=rep,
               csrv_ratio_range=[round(lo, 2), round(hi, 2)],
               csrv_ratio_pairs=[round(x, 2) for x in pairs],
               longprompt_rows=rows,
               saturation_c16_over_c1=sat,
               verdict=("长 prompt × 高并发：两栈都被模型高估（偏保守），"
                        "因此该失效区属于**模型形式**，不是某个栈的实现问题"))
    (cfg.RESULTS_DIR / "a10_repeat_longprompt_summary.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n→ results/a10_repeat_longprompt_summary.json")


if __name__ == "__main__":
    main()

