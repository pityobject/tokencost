# -*- coding: utf-8 -*-
"""
参数先验表（`tc.py priors`，M4）：没标定也能先得到一个能用的答案

## 为什么

现在的链条是"装服务器 → 8 点标定 → 6 参数 → 推荐"。门槛就在中间那步：
没有 GPU、或者只想先估个数的人，走不下去。

而我们已经有三块 GPU × 两栈的实测（`c_srv` 从 0.27 到 3.85 ms，跨 14 倍）。
把它做成先验表，就能把"必须先标定"变成"可选"。

## 用法

    python code/tc_priors.py                     # 打印先验表
    python code/tc_priors.py --stack vllm --nominal 600   # 给某场景一组参数

## 输出

`results/params_prior.json`
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import tc_config as cfg

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# 标称显存带宽（GB/s）：用于把"实测 bw"折算成"占比"
NOMINAL = {"RTX 4060": 272.0, "4060": 272.0, "A10": 600.0, "V100": 900.0, "V100S": 1130.0}

# 实测来源（栈, 硬件关键字, 文件）
SOURCES = [
    ("llama.cpp", "RTX 4060", "fit.json"),
    ("llama.cpp", "A10", "a10_raw/calib_a10_llamacpp.json"),
    ("llama.cpp", "A10", "a10_raw/calib_a10_llamacpp_rep2.json"),
    ("llama.cpp", "A10", "a10_raw/calib_a10_llamacpp_rep3.json"),
    ("llama.cpp", "V100", "a10_raw/calib_v100_llamacpp.json"),
    ("vLLM", "A10", "a10_raw/calib_a10_vllm.json"),
    ("vLLM", "A10", "a10_raw/calib_a10_vllm_rep2.json"),
    ("vLLM", "A10", "a10_raw/calib_a10_vllm_rep3.json"),
]


def nominal_of(name):
    for k, v in NOMINAL.items():
        if k.lower() in str(name).lower():
            return v
    return None


def collect():
    rows = []
    for stack, hw, rel in SOURCES:
        fp = cfg.RESULTS_DIR / rel
        if not fp.exists():
            continue
        d = json.loads(fp.read_text(encoding="utf-8"))
        nom = nominal_of(hw)
        rows.append(dict(stack=stack, hw=hw, source=rel,
                         c_srv=float(d["c_srv"]), bw=float(d["bw"]),
                         b_sat=float(d.get("b_sat", float("nan"))),
                         nominal=nom,
                         bw_frac=(float(d["bw"]) / nom) if nom else None))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stack", choices=["llamacpp", "vllm"], default=None)
    ap.add_argument("--nominal", type=float, default=None, help="标称带宽 GB/s")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    rows = collect()
    if not rows:
        print("没有可用的标定文件")
        return

    if args.stack and args.nominal:
        st = "vLLM" if args.stack == "vllm" else "llama.cpp"
        sel = [r for r in rows if r["stack"] == st]
        if not sel:
            print(f"没有 {st} 的实测")
            return
        cs = np.array([r["c_srv"] for r in sel])
        bf = np.array([r["bw_frac"] for r in sel if r["bw_frac"]])
        print(f"先验（{st}，标称 {args.nominal:.0f} GB/s）")
        print(f"  c_srv      = {np.median(cs):.3f} ms（实测范围 {cs.min():.3f}–{cs.max():.3f}）")
        if len(bf):
            print(f"  bw         = {np.median(bf)*args.nominal:.0f} GB/s"
                  f"（标称的 {np.median(bf)*100:.0f}%）")
        print(f"\n  预期误差：c_srv 的极差是 {cs.max()/cs.min():.1f}×，"
              "所以成本估算误差约 ±25–40%（取决于落在范围的哪一侧）")
        print("  想更准 → 跑 8 点标定（`python code/tc_profile.py ...`）")
        return

    print("=" * 88)
    print("参数先验表（全部来自实测标定，不是猜的）")
    print("=" * 88)
    print(f"  {'栈':<11}{'硬件':<11}{'c_srv (ms)':>12}{'bw (GB/s)':>11}"
          f"{'占标称':>8}{'b_sat':>8}  来源")
    for r in rows:
        print(f"  {r['stack']:<11}{r['hw']:<11}{r['c_srv']:>12.3f}{r['bw']:>11.1f}"
              f"{(f'{r[chr(98)+chr(119)+chr(95)+chr(102)+chr(114)+chr(97)+chr(99)]*100:.0f}%' if r['bw_frac'] else '—'):>8}"
              f"{r['b_sat']:>8.2f}  {r['source']}")

    # 按 (栈, 场景类) 聚合
    print("\n" + "=" * 88)
    print("二、按场景给先验（供没标定的人直接用）")
    print("=" * 88)
    groups = [
        ("vLLM + 现代数据中心 GPU", [r for r in rows if r["stack"] == "vLLM"]),
        ("llama.cpp + 桌面/笔记本 GPU", [r for r in rows if r["stack"] == "llama.cpp" and "4060" in r["hw"]]),
        ("llama.cpp + 云端 GPU", [r for r in rows if r["stack"] == "llama.cpp" and "4060" not in r["hw"]]),
    ]
    out_groups = []
    print(f"  {'场景':<28}{'c_srv 范围':>18}{'bw 占标称':>12}{'预期成本误差':>14}")
    for name, sel in groups:
        if not sel:
            continue
        cs = np.array([r["c_srv"] for r in sel])
        bf = np.array([r["bw_frac"] for r in sel if r["bw_frac"]])
        spread = cs.max() / cs.min()
        err = "±15%" if spread < 1.6 else ("±25%" if spread < 3 else "±40%")
        out_groups.append(dict(scenario=name, n=len(sel),
                               c_srv_min=round(float(cs.min()), 3),
                               c_srv_max=round(float(cs.max()), 3),
                               bw_frac=round(float(np.median(bf)), 3) if len(bf) else None,
                               expected_cost_error=err))
        print(f"  {name:<28}{f'{cs.min():.2f}–{cs.max():.2f} ms':>18}"
              f"{(f'{np.median(bf)*100:.0f}%' if len(bf) else '—'):>12}{err:>14}")

    print("\n  用法：先按场景取一组先验跑推荐；若结论影响决策，再花 10 分钟做 8 点标定。")
    print("  命令：python code/tc_priors.py --stack vllm --nominal 600")

    out = dict(points=rows, groups=out_groups,
               note=("先验来自 8 份实测标定；bw_frac = 实测/标称。"
                     "c_srv 同场景内可差 1.5–3×，所以给的是范围不是点"))
    fp = Path(args.out) if args.out else (cfg.RESULTS_DIR / "params_prior.json")
    fp.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n→ {fp}")


if __name__ == "__main__":
    main()
