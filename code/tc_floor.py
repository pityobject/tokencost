# -*- coding: utf-8 -*-
"""
理论下界（`tc.py floor`，M10）：把"我们多准"换成"离物理下限多远"

## 为什么

现在对外说的是"成本 MAPE 8.58%"。这说的是**模型和实测的差距**，
但别人真正想问的是：**这台机器理论上能多快？我们的模型离那个极限多远？**

## 两个下界

1. **内存下界**（decode 的硬底）：把权重读一遍的最短时间

       TPOT_min = (W + KV_L·b) / BW_标称

2. **算力下界**（prefill 的硬底）：

       TTFT_min = max( W / BW_标称 , 2·P·L / PEAK_标称 )

然后定义 **离下界倍数** = 实测 / 下界，并把它拆成两部分：

    实测/下界  =  [标称/实测BW]  ×  [1 + 服务开销/(访存下界)]

第一部分是**带宽利用率**（已知 79–92%），第二部分是**服务开销**。
这样就能说清楚："慢的那个 1.7 倍里，有多少是硬件拿不到的带宽，有多少是软件开销"。

## 用法

    python code/tc_floor.py
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import tc_config as cfg

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# 标称显存带宽（GB/s）
NOMINAL_BW = {"RTX 4060": 272.0, "4060": 272.0, "A10": 600.0, "V100": 900.0}
# 标称 fp16 张量算力（TFLOP/s，厂商值；仅 prefill 下界用）
NOMINAL_PEAK = {"RTX 4060": 121.0, "4060": 121.0, "A10": 125.0, "V100": 125.0}

CASES = [
    ("RTX 4060", "llama.cpp", "fit.json", None),
    ("A10", "llama.cpp", "a10_raw/calib_a10_llamacpp.json", None),
    ("V100", "llama.cpp", "a10_raw/calib_v100_llamacpp.json", None),
    ("A10", "vLLM", "a10_raw/calib_a10_vllm.json", None),
]


def nominal(mapping, name):
    for k, v in mapping.items():
        if k.lower() in str(name).lower():
            return v
    return None


def kv_bytes(spec):
    return 2 * 2 * spec["head_dim"] * spec["n_layers"] * 2


def main():
    rows = []
    for hw, stack, rel, _ in CASES:
        fp = cfg.RESULTS_DIR / rel
        if not fp.exists():
            continue
        d = json.loads(fp.read_text(encoding="utf-8"))
        bw_nom = nominal(NOMINAL_BW, hw)
        pk_nom = nominal(NOMINAL_PEAK, hw)
        meas = d.get("measurements") or []
        if not bw_nom:
            continue
        if not meas:
            # fit.json 没有 measurements：用 predictions.csv 的 c1 行代替（本机 4060）
            import pandas as pd
            pr = pd.read_csv(cfg.RESULTS_DIR / "predictions.csv")
            pr = pr[(pr["conc"] == 1) & (pr["model"] == "3B") & (pr["quant"] == "q4_k_m")]
            if pr.empty:
                continue
            meas = [dict(conc=1, L=float(r["prompt_tokens"]), ttft=float(r["meas_ttft"]),
                         tpot=float(r["meas_tpot"])) for _, r in pr.iterrows()]
            spec = dict(weight_bytes=cfg.MODEL_SPECS[("3B", "q4_k_m")]["weight_bytes"],
                        n_params=cfg.MODEL_SPECS[("3B", "q4_k_m")]["n_params"],
                        n_layers=cfg.MODEL_SPECS[("3B", "q4_k_m")]["n_layers"],
                        n_kv_heads=cfg.MODEL_SPECS[("3B", "q4_k_m")]["n_kv_heads"],
                        head_dim=cfg.MODEL_SPECS[("3B", "q4_k_m")]["head_dim"])
            d["bw"] = d.get("bw", 249.1)
            d["c_srv"] = d.get("c_srv", 0.766)
        else:
            spec = d["model"]
        kv = kv_bytes(spec)
        for m in meas:
            b = float(m["conc"])
            L = float(m["L"])
            W = float(spec["weight_bytes"])
            # 访存下界（decode）
            tpot_floor = (W + kv * (L + 64) * b) / (bw_nom * 1e9) * 1e3
            # 算力/访存下界（prefill）
            ttft_mem = W / (bw_nom * 1e9) * 1e3
            ttft_cmp = 2 * spec["n_params"] * L / (pk_nom * 1e12) * 1e3
            ttft_floor = max(ttft_mem, ttft_cmp)
            rows.append(dict(
                hw=hw, stack=stack, conc=m["conc"], L=L,
                tpot=m["tpot"], tpot_floor=tpot_floor,
                tpot_x=m["tpot"] / tpot_floor,
                ttft=m["ttft"], ttft_floor=ttft_floor,
                ttft_x=m["ttft"] / ttft_floor,
                bw_gain=bw_nom / float(d["bw"]),      # 带宽没拿满的倍数
                c_srv=float(d["c_srv"]),
            ))

    if not rows:
        print("没有可用标定")
        return

    print("=" * 92)
    print("一、TPOT 离物理下界多远（c1，纯访存下界 = W/BW_标称）")
    print("=" * 92)
    print(f"  {'硬件':<10}{'栈':<10}{'实测 TPOT':>11}{'物理下界':>10}{'倍数':>8}"
          f"{'带宽利用率':>12}{'服务开销贡献':>14}")
    for r in rows:
        if r["conc"] != 1:
            continue
        # 分解：倍数 = 带宽倍数 × (1 + 服务开销/下界)
        extra = max(r["tpot_x"] / r["bw_gain"] - 1, 0) * 100
        print(f"  {r['hw']:<10}{r['stack']:<10}{r['tpot']:>10.2f}{r['tpot_floor']:>10.2f}"
              f"{r['tpot_x']:>7.2f}×{100/r['bw_gain']:>11.0f}%{extra:>13.0f}%")

    print("\n" + "=" * 92)
    print("二、TTFT 离物理下界多远（c1，max(访存, 算力)）")
    print("=" * 92)
    print(f"  {'硬件':<10}{'栈':<10}{'L':>6}{'实测 TTFT':>11}{'物理下界':>10}{'倍数':>8}")
    for r in rows:
        if r["conc"] != 1:
            continue
        print(f"  {r['hw']:<10}{r['stack']:<10}{r['L']:>6.0f}{r['ttft']:>10.1f}"
              f"{r['ttft_floor']:>10.1f}{r['ttft_x']:>7.2f}×")

    print("\n" + "=" * 92)
    print("三、同一个模型，不同硬件的『离下界倍数』")
    print("=" * 92)
    by_hw = {}
    for r in rows:
        if r["conc"] != 1 or "3B" not in "" and False:
            continue
        by_hw.setdefault((r["hw"], r["stack"]), []).append(r)
    for (hw, stack), rs in by_hw.items():
        x = np.mean([r["tpot_x"] for r in rs])
        print(f"  {hw:<10}{stack:<10} TPOT 平均离下界 **{x:.2f}×**"
              f"（c_srv {rs[0]['c_srv']:.3f} ms）")

    print("\n" + "=" * 92)
    print("四、结论")
    print("=" * 92)
    c1 = [r for r in rows if r["conc"] == 1]
    if c1:
        xs = [r["tpot_x"] for r in c1]
        print(f"  · c1 时 TPOT 离物理下界只有 **{min(xs):.2f}–{max(xs):.2f}×**——"
              "说明单流 decode 已经很接近硬件极限")
        hi = [r for r in rows if r["conc"] >= 16]
        if hi:
            xh = [r["tpot_x"] for r in hi]
            print(f"  · 高并发（≥16）时升到 **{min(xh):.1f}–{max(xh):.1f}×**——"
                  "多出来的部分几乎全是服务开销（`c_srv·b`），不是硬件")
        print("  · 所以『慢』的归因很清楚：**单流看带宽，并发看服务开销**。")
        print("  · 这也解释了为什么 `c_srv` 最重要：它决定了并发时离下界多远。")

    out = dict(points=rows,
               nominal_bw=NOMINAL_BW, nominal_peak=NOMINAL_PEAK,
               note=("物理下界用厂商标称值；实测/下界 = 带宽效率 × (1+服务开销占比)。"
                     "这条把『模型准不准』换成『离硬件极限多远』"))
    (cfg.RESULTS_DIR / "floor_analysis.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n→ results/floor_analysis.json")


if __name__ == "__main__":
    main()
