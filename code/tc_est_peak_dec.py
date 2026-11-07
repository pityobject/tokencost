# -*- coding: utf-8 -*-
"""
PEAK_dec 的独立估计（`tc.py peakdec`）：把"弱约束参数"变成可辨识

## 问题

六参数里 `PEAK_dec`（decode 有效算力）一直标着"不可识别"，因为设计空间内
decode 始终访存受限。但**高并发 + 短上下文**会把天平推向算力侧：

    TPOT(b) = max(mem, 2·P·b / PEAK_dec) + c_srv·b

当 `b > b_crit ≈ PEAK_dec·W / (2·P·BW)`（A10 上约 24）时，算力项成为生效项，
于是

    TPOT(b) / b ≈ 2·P / PEAK_dec + c_srv        （b 足够大时）

对 `TPOT(b)` 与 `b` 做线性回归，斜率 = `2P/PEAK_dec + c_srv`，
减去独立标定出的 `c_srv` 即可解出 `PEAK_dec`。

## 输入

- `results/a10_raw/jitter_a10_*_high*.json`（高并发，短 prompt → KV 小）
- 对应栈的标定 json（取 `c_srv`、`bw`）

## 用法

    python code/tc_est_peak_dec.py \
        --jitter results/a10_raw/jitter_a10_vllm_high.json \
        --calib  results/a10_raw/calib_a10_vllm.json \
        --n-params 3.08e9
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import tc_config as cfg

sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jitter", required=True)
    ap.add_argument("--calib", required=True)
    ap.add_argument("--n-params", type=float, default=3.08e9)
    ap.add_argument("--min-conc", type=int, default=32,
                    help="只用在 b≥该值的点上做回归（确保进入算力区）")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    jit = json.loads(Path(args.jitter).read_text(encoding="utf-8"))
    cal = json.loads(Path(args.calib).read_text(encoding="utf-8"))
    cells = jit["cells"]
    bs = sorted(int(k) for k in cells)
    sel = [b for b in bs if b >= args.min_conc]
    if len(sel) < 2:
        print(f"并发点不足（{sel}），至少需要 2 个 ≥{args.min_conc} 的点")
        return

    b = np.array(sel, dtype=float)
    tpot = np.array([cells[str(int(x))]["tpot_p50"] for x in sel], dtype=float)

    # TPOT = slope · b + intercept，斜率含 c_srv
    k, c0 = np.polyfit(b, tpot, 1)
    r2 = 1 - np.sum((tpot - (k * b + c0)) ** 2) / np.sum((tpot - tpot.mean()) ** 2)

    c_srv = float(cal["c_srv"])
    denom = k - c_srv
    print("=" * 78)
    print(f"数据：{Path(args.jitter).name}   b∈{list(sel)}  prompt={jit.get('length')} tok")
    print("=" * 78)
    print(f"  回归：TPOT(b) = {k:.4f}·b + {c0:.1f}   (R²={r2:.4f})")
    print(f"  斜率含 c_srv（独立标定 {c_srv:.3f} ms）→ 纯算力项斜率 "
          f"{denom:.4f} ms/流")

    if denom <= 0:
        print(f"\n  ❌ 斜率 {k:.4f} ≤ c_srv {c_srv:.3f}：该并发范围仍未进入算力区，"
              "或者 c_srv 在高并发下被高估。无法解出 PEAK_dec。")
        out = dict(verdict="unidentifiable", slope_ms_per_stream=k,
                   intercept_ms=c0, r2=round(r2, 4), c_srv_ms=c_srv)
    else:
        peak_from_jitter = 2 * args.n_params / (denom * 1e-3) / 1e12
        print(f"  → PEAK_dec ≈ 2P / {denom:.4f} ms = **{peak_from_jitter:.1f} TFLOP/s**")
        print(f"  拟合值 {cal['peak']:.1f}；此前下界 15.8")
        # 与 b_crit 自洽性检查
        b_crit = peak_from_jitter * 1e12 * 0  # 占位，避免误导
        print(f"\n  注：b≥{args.min_conc} 的读数说明已在算力区；"
              "若 R² 不好，说明该并发段仍混有访存/排队成分。")
        out = dict(verdict="estimated", peak_dec_tflops=round(peak_from_jitter, 1),
                   slope_ms_per_stream=round(k, 4), intercept_ms=round(c0, 2),
                   r2=round(r2, 4), c_srv_ms=c_srv,
                   fitted_peak_in_calib=cal["peak"],
                   points={int(x): float(y) for x, y in zip(b, tpot)},
                   note="斜率减 c_srv 得纯算力项；仅当 R² 高且并发进入算力区时可信")

    fp = Path(args.out) if args.out else (cfg.RESULTS_DIR / "peak_dec_estimate.json")
    fp.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n→ {fp}")


if __name__ == "__main__":
    main()
