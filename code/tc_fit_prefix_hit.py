# -*- coding: utf-8 -*-
"""
前缀命中率 `h` 的估计（`tc.py prefixh`）：把"前缀复用"变成一个模型参数

## 为什么

§5.26 证明了 llama.cpp 会复用前缀 KV（同 prompt 重发省 31%，开缓存省 94%），
§5.25 又看到多轮 TTFT 被模型高估 30–47%。但**模型里没有这个参数**——
所以多轮场景无法预测，只能事后解释。

## 定义

    TTFT = max(W/BW, 2·P·(1-h)·L·min(b,B_SAT)/PEAK_pre) + c_srv·b + ovh_ttft

`h` = **有效前缀命中率**（0 = 完全重算，1 = 全命中）。
它不是模型常数，而是**部署属性**：取决于服务端有没有前缀缓存、请求之间重叠多少。

## 做法

1. 用标定参数 + 多轮实测，网格搜索使 TTFT 误差最小的 `h`；
2. 两个栈各拟一次，看 `h` 是否与"有没有前缀缓存"一致（vLLM 0.5.5 默认无 → 应为 ~0）；
3. 与 §5.26 的重复实验对照（`cache off` 时同 prompt 省 31% → 理论 `h≈0.31`）。

## 用法

    python code/tc_fit_prefix_hit.py
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import tc_config as cfg
import tc_model as tm

sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def load(rel):
    fp = cfg.RESULTS_DIR / rel
    return json.loads(fp.read_text(encoding="utf-8")) if fp.exists() else None


def predict_params(cal):
    return dict(bw_gbs=cal["bw"], peak_tflops=cal["peak"],
                peak_prefill_tflops=cal["peak_prefill"],
                c_srv_ms=cal["c_srv"], b_sat=cal["b_sat"],
                ovh_ttft_ms=cal["ovh_ttft"])


def ttft_with_h(params, L, b, h, gen=128):
    """把有效 prefill 长度缩成 (1-h)·L 后的 TTFT 预测。"""
    row = dict(model="3B", quant="fp16", ctx=2304, conc=b,
               prompt_tokens=float(L) * (1.0 - h),
               ctx_avg_tokens=float(L) * (1.0 - h) + gen / 2)
    return tm.predict(params, row)["pred_ttft_ms"]


def fit_h(params, rows, b=1, gen=128, hi=0.95):
    """网格搜索使 |pred/meas-1| 均值最小的 h；返回 (h, 误差, 逐点)。"""
    hs = np.arange(0.0, hi + 1e-9, 0.01)
    best = None
    for h in hs:
        errs = []
        for r in rows:
            p = ttft_with_h(params, r["prompt_tokens"], b, h, gen)
            errs.append(abs(p / r["ttft_ms"] - 1))
        e = float(np.mean(errs))
        if best is None or e < best[1]:
            best = (float(h), e)
    h, e = best
    detail = []
    for r in rows:
        p = ttft_with_h(params, r["prompt_tokens"], b, h, gen)
        detail.append(dict(turn=r.get("turn"), L=r["prompt_tokens"],
                           measured=round(r["ttft_ms"], 1),
                           pred=round(p, 1),
                           err_pct=round((p / r["ttft_ms"] - 1) * 100, 1)))
    return h, e, detail


def main():
    sets = []
    cal_l = load("a10_raw/calib_v100_llamacpp.json")
    cal_v = load("a10_raw/calib_a10_vllm.json")
    trn_l = load("a10_raw/turns_v100_llamacpp.json")
    trn_v = load("a10_raw/turns_a10_vllm.json")
    if cal_l and trn_l:
        sets.append(("llama.cpp (V100)", cal_l, trn_l))
    if cal_v and trn_v:
        sets.append(("vLLM (A10)", cal_v, trn_v))
    if not sets:
        print("缺标定或多轮数据")
        return

    print("=" * 84)
    print("一、用多轮实测反解前缀命中率 h（第 1 轮是冷启动，排除）")
    print("=" * 84)
    results = {}
    cfg.MODEL_SPECS[("3B", "fp16")] = dict(weight_bytes=6178317312, n_params=3080000000,
                                          n_layers=36, n_kv_heads=2, head_dim=128)
    for name, cal, trn in sets:
        rows = [r for r in trn["rows"] if int(r["ttft_ms"]) > 0 and r.get("turn", 1) >= 2]
        if len(rows) < 2:
            continue
        params = predict_params(cal)
        h0, e0, _ = fit_h(params, rows, 1, hi=0.0) if False else (0.0, None, None)
        # h=0 的基线误差
        base_err = float(np.mean([abs(ttft_with_h(params, r["prompt_tokens"], 1, 0.0) / r["ttft_ms"] - 1)
                                  for r in rows])) * 100
        h, e, detail = fit_h(params, rows, 1)
        results[name] = dict(h=round(h, 2), err_before_pct=round(base_err, 1),
                             err_after_pct=round(e * 100, 1), detail=detail)
        print(f"\n  【{name}】")
        print(f"    h=0（无缓存假设）时 TTFT 平均误差 {base_err:>5.1f}%")
        print(f"    最优 h = **{h:.2f}** → 误差降到 {e*100:>5.1f}%")
        print(f"    {'轮':>3}{'L':>7}{'实测':>9}{'h=0 预测':>10}{'最优 h 预测':>12}{'误差':>8}")
        for d, r in zip(detail, rows):
            p0 = ttft_with_h(params, r["prompt_tokens"], 1, 0.0)
            print(f"    {d['turn']:>3}{d['L']:>7}{d['measured']:>9.1f}"
                  f"{p0:>10.1f}{d['pred']:>12.1f}{d['err_pct']:>7.1f}%")

    # 与重复实验对照
    print("\n" + "=" * 84)
    print("二、与重复实验（§5.26）对照")
    print("=" * 84)
    pfx_off = load("a10_raw/prefix_a10_llamacpp_off.json")
    pfx_on = load("a10_raw/prefix_a10_llamacpp_on.json")
    if pfx_off:
        print(f"  同 prompt 重复（cache_prompt=false）：TTFT 降 {pfx_off['drop_same_pct']:.0f}%"
              f" → 相当于 h≈{pfx_off['drop_same_pct']/100:.2f}（全重叠时）")
    if pfx_on:
        print(f"  同 prompt 重复（cache_prompt=true）： TTFT 降 {pfx_on['drop_same_pct']:.0f}%"
              f" → h≈{pfx_on['drop_same_pct']/100:.2f}")
    for name, r in results.items():
        print(f"  多轮实测反解：{name} h = {r['h']:.2f}（部分重叠，所以低于全重叠的 0.31）")

    out = dict(fits=results,
               reference=dict(
                   cache_off_h=round(pfx_off["drop_same_pct"] / 100, 2) if pfx_off else None,
                   cache_on_h=round(pfx_on["drop_same_pct"] / 100, 2) if pfx_on else None),
               note=("h = 有效前缀命中率，写成 (1-h)·L 的 prefill；"
                     "它是部署属性（有无前缀缓存 + 请求重叠度），不是模型常数"))
    (cfg.RESULTS_DIR / "prefix_hit.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n→ results/prefix_hit.json")


if __name__ == "__main__":
    main()
