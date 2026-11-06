# -*- coding: utf-8 -*-
"""
B1：decode 算力（PEAK_dec）为什么在本机不可辨识——用算术强度证明，而不是再跑实验

## 推理

decode 每生成 1 个 token，要读一遍权重（≈ W 字节）并做 2P 次浮点乘加，
所以**算术强度** = 2P / W  FLOP/byte。

- Q4 量化：W ≈ 0.5 字节/参数 → 强度 ≈ **4 FLOP/byte**
- GPU 的"脊点"（ridge point）= PEAK / BW：
  - RTX 4060 Laptop：13.16 TF / 249 GB/s ≈ **53 FLOP/byte**
  - A10：≈ 125 TF / 600 GB/s ≈ **208 FLOP/byte**

因为 **4 ≪ 53 ≪ 208**，decode 在这些卡上**永远访存受限**——
算力项 `max(t_mem, t_cmp)` 里 `t_cmp` 永远取不到。
所以 `PEAK_dec` 在数学上不可辨识，任何拟合值都是"损失函数的一个平坦方向"。

    print("  也就是说：算力项在 max() 里永远取不到 → PEAK_dec 不可辨识是结构性的，")
    print("  不是「数据不够」；多跑实验也修不好，除非换算力/带宽比完全不同的硬件。")
`PEAK > 2P·BW/W`：3B-Q4 在 249 GB/s 上需要 **≈731 TFLOP/s** 才算得动——
比这张卡快 ~55 倍。

## 产物

- `results/decode_bound.json`：每个 (model, quant) 的强度、脊点与临界算力
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import tc_config as cfg

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# 参考硬件（用于对照；本机参数从 fit.json 取）
HW = {
    "RTX 4060 Laptop (本机)": (249.1, 13.16),
    "A10 (云端免费档)": (600.0, 125.0),
    "A100-40G": (1555.0, 312.0),
}


def main():
    fit = json.loads((cfg.RESULTS_DIR / "fit.json").read_text(encoding="utf-8"))
    bw = float(fit["bw"])
    peak = float(fit["peak"])
    print(f"本机拟合值：BW {bw:.0f} GB/s，PEAK_dec {peak:.2f} TFLOP/s，"
          f"脊点 {peak * 1e12 / (bw * 1e9):.0f} FLOP/byte\n")

    rows = []
    for (model, quant), spec in sorted(cfg.MODEL_SPECS.items()):
        W = float(spec["weight_bytes"])
        P = float(spec["n_params"])
        intensity = 2 * P / W                       # FLOP/byte
        need_peak = 2 * P * bw * 1e9 / W            # 临界算力：低于它才转为算力受限
        rows.append(dict(model=model, quant=quant,
                         intensity_flop_per_byte=round(intensity, 2),
                         bytes_per_param=round(W / P, 3),
                         need_peak_tflops=round(need_peak / 1e12, 1),
                         mem_bound=(peak > need_peak / 1e12)))

    print(f"  {'模型':<8}{'量化':<9}{'强度':>8}{'需要算力':>12}{'本机算力':>10}  判定")
    for r in rows:
        print(f"  {r['model']:<8}{r['quant']:<9}"
              f"{r['intensity_flop_per_byte']:>7.1f}{r['need_peak_tflops']:>11.0f}TF"
              f"{peak:>9.1f}TF  {'访存受限' if r['mem_bound'] else '算力受限'}")

    all_mem = all(r["mem_bound"] for r in rows)
    min_need = min(r["need_peak_tflops"] for r in rows)
    print(f"\n  结论：{'全部' if all_mem else '部分'}配置在本机 decode 阶段访存受限。")
    print(f"  临界算力最高只需 {min_need:.2f} TFLOP/s，而本机有 {peak:.1f} TFLOP/s。")
    print(f"  也就是说：算力项在 max() 里永远取不到 → PEAK_dec 不可辨识是**结构性**的，")
    print("  不是「数据不够」；多跑实验也修不好，除非换算力/带宽比完全不同的硬件。")

    print("\n  换成更强的卡会怎样（同样的 Q4 权重、同样的 4 FLOP/byte）：")
    for name, (b, pk) in HW.items():
        print(f"    {name:<24} 脊点 {pk * 1e12 / (b * 1e9):>5.0f} FLOP/byte"
              f"  → {'仍然访存受限' if 4 < pk * 1e12 / (b * 1e9) else '可能算力受限'}")

    out = dict(peak_dec_tflops=peak, bw_gbs=bw,
               ridge_flop_per_byte=round(peak * 1e12 / (bw * 1e9), 1),
               all_decode_memory_bound=bool(all_mem),
               min_peak_needed_tflops=round(min_need, 1),
               hardware=HW,
               verdict=("decode 在这些 GPU 上结构性访存受限 → PEAK_dec 不可辨识；"
                        "任何拟合值都只是损失函数的平坦方向"),
               rows=rows)
    (cfg.RESULTS_DIR / "decode_bound.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n→ results/decode_bound.json")


if __name__ == "__main__":
    main()
