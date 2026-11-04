# -*- coding: utf-8 -*-
"""
tc_calibrate.py —— 自标定：在**你自己的硬件 / 后端 / 模型**上跑通这套方法

## 为什么需要它

六参数模型是在一台 RTX 4060 Laptop + llama.cpp 上标定的。别人没法直接用——
但**方法**是通用的：只要在自己的机器上量 4–6 个配置，就能拟合出属于自己那套栈的参数。

这个脚本把"标定"从论文里的图，变成一条命令：

    python tc_calibrate.py my_measurements.json
    python tc_recommend.py --calib ../results/calib_mymachine.json --ttft 600 --tpot 60

## 输入格式（JSON）

```json
{
  "hardware": "RTX 4090 24GB",
  "backend":  "llama.cpp b11669",
  "model": {
    "name": "mymodel", "quant": "q4_k_m",
    "weight_bytes": 2104932768,        // GGUF 文件字节数（多分片求和）
    "n_params": 3090000000,            // 参数量
    "n_layers": 36, "n_kv_heads": 2, "head_dim": 128
  },
  "measurements": [
    {"ctx": 2048, "conc": 1, "prompt_tokens": 512,
     "ttft_ms": 130.0, "tpot_ms": 9.8},
    {"ctx": 2048, "conc": 8, "prompt_tokens": 512,
     "ttft_ms": 320.0, "tpot_ms": 12.5}
  ]
}
```

**建议 4–6 个配置**（每条含并发与提示长度的变化），且至少包含：
  · 一个 conc=1（把带宽和 prefill 算力分开）
  · 一个高并发（把 c_srv 与 B_SAT 暴露出来）
  · 两个不同 prompt 长度（确定 prefill 斜率）

## 关于诚实性

本脚本**不引入任何外部数字**：它只用你输入的实测点拟合。
拟合不出来会直说"标定点不足/病态"，不会给你一个看着像样的假参数。
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import tc_config as cfg
import tc_model as tm

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

REQUIRED_MODEL_FIELDS = ["name", "quant", "weight_bytes", "n_params",
                         "n_layers", "n_kv_heads", "head_dim"]


def load_spec(path):
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    m = d.get("model") or {}
    miss = [k for k in REQUIRED_MODEL_FIELDS if k not in m]
    if miss:
        raise SystemExit(f"model 段缺字段：{miss}\n"
                         f"（weight_bytes 用文件大小；n_layers/n_kv_heads/head_dim "
                         f"可从 GGUF metadata 或 HuggingFace config.json 读到）")
    cells = d.get("measurements") or []
    if len(cells) < 3:
        raise SystemExit(f"至少需要 3 个测量点（建议 4–6 个），现在只有 {len(cells)} 个")
    return d, m, cells


def register_spec(model):
    """把用户模型的规格注入 cfg.MODEL_SPECS，让 predict() 无需改动即可用。"""
    key = (model["name"], model["quant"])
    cfg.MODEL_SPECS[key] = dict(
        files=[f"{model['name']}.gguf"],
        weight_bytes=int(model["weight_bytes"]),
        n_params=float(model["n_params"]),
        n_layers=int(model["n_layers"]),
        n_kv_heads=int(model["n_kv_heads"]),
        head_dim=int(model["head_dim"]),
    )
    return key


def build_df(key, cells):
    rows = []
    for i, c in enumerate(cells):
        lack = [k for k in ("ctx", "conc", "prompt_tokens", "ttft_ms", "tpot_ms") if k not in c]
        if lack:
            raise SystemExit(f"第 {i+1} 个测量点缺字段：{lack}")
        L = float(c["prompt_tokens"])
        rows.append(dict(
            cell_id=f"calib_{i}",
            model=key[0], quant=key[1],
            ctx=int(c["ctx"]), conc=int(c["conc"]),
            prompt_tokens=L,
            ctx_avg_tokens=L + cfg.GEN_TOKENS / 2,
            ttft_p50_ms=float(c["ttft_ms"]),
            tpot_p50_ms=float(c["tpot_ms"]),
            cost_gpu_s_per_1m=np.nan,      # 自标定不需要成本（可由吞吐推出）
        ))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description="在你自己的栈上标定六参数模型")
    ap.add_argument("json_path", help="测量点 JSON 文件")
    ap.add_argument("--out", default=None, help="输出参数文件（默认 results/calib_local.json）")
    args = ap.parse_args()

    d, model, cells = load_spec(args.json_path)
    key = register_spec(model)
    df = build_df(key, cells)

    print("=" * 78)
    print(f"自标定：{d.get('hardware', '（未填硬件）')} · {d.get('backend', '（未填后端）')}")
    print("=" * 78)
    print(f"  模型 {model['name']}-{model['quant']}："
          f"{model['weight_bytes']/1e9:.2f} GB 权重 / {model['n_params']/1e9:.2f}B 参数")
    print(f"  测量点 {len(cells)} 个 → 观测 {2*len(cells)} 个 / 未知参数 6 个")

    params = tm.fit(df)

    # ---- 残差 ----
    print(f"\n{'配置':<26}{'TTFT实测':>10}{'预测':>9}{'误差':>8}"
          f"{'TPOT实测':>10}{'预测':>9}{'误差':>8}")
    errs_t, errs_p = [], []
    for _, r in df.iterrows():
        pr = tm.predict(params, r)
        e1 = pr["pred_ttft_ms"] / r["ttft_p50_ms"] - 1
        e2 = pr["pred_tpot_ms"] / r["tpot_p50_ms"] - 1
        errs_t.append(abs(e1))
        errs_p.append(abs(e2))
        tag = f"c{r['conc']}/L{int(r['prompt_tokens'])}"
        print(f"{tag:<26}{r['ttft_p50_ms']:>9.1f}ms{pr['pred_ttft_ms']:>8.1f}ms"
              f"{e1*100:>7.0f}%{r['tpot_p50_ms']:>9.2f}ms{pr['pred_tpot_ms']:>8.2f}ms"
              f"{e2*100:>7.0f}%")

    n_obs, n_par = 2 * len(df), 6
    dof = n_obs - n_par
    print(f"\n  自由度 = {n_obs} 观测 − {n_par} 参数 = {dof}"
          + ("（刚好够拟合，无法给出误差估计——建议再加 1–2 个点）" if dof <= 0 else ""))
    if dof > 0:
        rms = float(np.sqrt(np.mean(np.square(np.log1p(np.r_[errs_t, errs_p])))))
        print(f"  拟合残差 RMS（对数尺度）≈ {rms:.4f}，"
              f"约合 {100*(np.exp(rms)-1):.1f}% 的相对误差")
        if rms > 0.15:
            print("  ⚠️ 残差偏大：检查是否漏填了真实并发、或请求长度与实测不一致。")

    # ---- 参数 ----
    print("\n  标定结果：")
    print(f"    有效带宽 BW        {params['bw_gbs']:.1f} GB/s")
    print(f"    decode 算力        {params['peak_tflops']:.2f} TFLOP/s")
    print(f"    prefill 算力       {params['peak_prefill_tflops']:.2f} TFLOP/s")
    print(f"    服务开销 c_srv     {params['c_srv_ms']:.3f} ms/流/步")
    print(f"    prefill 饱和点     {params['b_sat']:.1f}")
    print(f"    首字固定开销       {params['ovh_ttft_ms']:.2f} ms")

    # ---- 可识别性检查：decode 算力常常无法从数据里反推 ----
    compute_bound = 0
    for _, r in df.iterrows():
        spec = cfg.MODEL_SPECS[key]
        kv = cfg.kv_bytes_per_token(spec) * r["ctx_avg_tokens"] * r["conc"]
        t_mem = (spec["weight_bytes"] + kv) / (params["bw_gbs"] * 1e9)
        t_cmp = 2 * spec["n_params"] * r["conc"] / (params["peak_tflops"] * 1e12)
        if t_cmp > t_mem:
            compute_bound += 1
    if compute_bound == 0:
        print(f"\n  ⚠️ 可识别性警告：{len(df)} 个测量点里**没有一个是算力受限的**"
              f"（decode 始终访存受限）。")
        print(f"     因此 `decode 算力` 在数学上无约束，拟合值 "
              f"{params['peak_tflops']:.0f} TFLOP/s **不可信、不要引用**。")
        print("     这不影响预测（该分支从不触发），但必须在报告里标明'不可识别'。")

    # 与"参考栈"对比（仅提示量级，不用于校验）
    ref = cfg.RESULTS_DIR / "fit.json"
    if ref.exists():
        r = json.loads(ref.read_text(encoding="utf-8"))
        print(f"\n  （参考：RTX 4060 Laptop 上的标定值是 BW {r['bw']:.0f} GB/s、"
              f"c_srv {r['c_srv']:.2f} ms）")

    out = Path(args.out) if args.out else (cfg.RESULTS_DIR / "calib_local.json")
    out.write_text(json.dumps(dict(
        hardware=d.get("hardware"), backend=d.get("backend"),
        model=model, n_points=len(cells),
        bw=params["bw_gbs"], peak=params["peak_tflops"],
        peak_prefill=params["peak_prefill_tflops"], c_srv=params["c_srv_ms"],
        b_sat=params["b_sat"], ovh_ttft=params["ovh_ttft_ms"],
        residual=dict(ttft_mape=float(np.mean(errs_t) * 100),
                      tpot_mape=float(np.mean(errs_p) * 100)),
    ), ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n已写入 {out}")
    print(f"\n下一步：\n  python tc_recommend.py --calib {out} --ttft 600 --tpot 60")


if __name__ == "__main__":
    main()

