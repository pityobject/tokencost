# -*- coding: utf-8 -*-
"""
标定自检诊断（`tc.py diagnose`，M6）：把 24 条失败记录里最贵的三条做成自动提示

## 为什么

标定完最容易出的三类问题，我们**自己全都踩过**：

| 症状 | 原因 | 我们踩的坑 |
|---|---|---|
| `bw` 只有标称的 40% | `-ngl` 没给满 / 降频 | F2（显存边界）、`报告/A10_上机手册.md` §10 |
| 高并发 TTFT 随并发线性爆炸 | **`-ub` 太小** | §5.34（c32 差 5.6 倍） |
| `c_srv` 两次差 30%+ | 同配置漂移 / 配置变了 | 同配置 +28%；32→128 槽 +64% |

把这三条做成检查，比再写一篇文档有用——它会在**出错的那一刻**说话。

## 用法

    python code/tc_diagnose.py --calib results/a10_raw/calib_a10_llamacpp.json
    python code/tc_diagnose.py --calib ... --nominal 600 --stack llamacpp
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import tc_config as cfg

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PRIOR_C_SRV = {           # (栈) -> (下限, 上限)，来自 tc_priors 的实测范围
    "llamacpp": {"desktop": (0.6, 1.7), "cloud": (1.8, 4.2)},
    "vllm": {"desktop": (0.20, 0.45), "cloud": (0.20, 0.45)},
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calib", required=True)
    ap.add_argument("--nominal", type=float, default=None,
                    help="该卡的标称带宽 GB/s（给不出就跳过第一条检查）")
    ap.add_argument("--stack", choices=["llamacpp", "vllm"], default="llamacpp")
    ap.add_argument("--class", dest="cls", choices=["desktop", "cloud"],
                    default="cloud", help="机器类别，用于 c_srv 先验比较")
    args = ap.parse_args()

    fp = Path(args.calib)
    if not fp.exists():
        print(f"❌ 找不到 {fp}")
        return
    d = json.loads(fp.read_text(encoding="utf-8"))
    meas = d.get("measurements") or []
    issues = []
    ok = []

    print("=" * 78)
    print(f"标定自检：{fp.name}")
    print("=" * 78)
    print(f"  硬件标记：{d.get('hardware')}｜后端：{d.get('backend')}")
    print(f"  bw={d['bw']:.1f} GB/s｜c_srv={d['c_srv']:.3f} ms｜"
          f"b_sat={d.get('b_sat', float('nan')):.2f}｜"
          f"TTFT/TPOT MAPE={d.get('residual', {}).get('ttft_mape', 0):.1f}%/"
          f"{d.get('residual', {}).get('tpot_mape', 0):.1f}%")

    # ① BW 占比
    if args.nominal:
        frac = d["bw"] / args.nominal
        if frac < 0.45:
            issues.append(
                f"**`bw` 只有标称的 {frac*100:.0f}%（{d['bw']:.0f} / {args.nominal:.0f} GB/s）**\n"
                f"     最可能的原因：`-ngl` 没给满（有层留在 CPU）或 GPU 降频。\n"
                f"     先跑：`nvidia-smi --query-gpu=clocks.sm,clocks.max.sm,power.draw --format=csv`\n"
                f"     再确认服务日志里有 `offloaded N/N layers to GPU`")
        elif frac > 1.05:
            issues.append(f"`bw` 超过标称（{frac*100:.0f}%）——检查标称值是显存带宽还是 L2 带宽")
        else:
            if frac < 0.75 and args.stack == "vllm":
                ok.append(f"`bw` 占标称 {frac*100:.0f}%——对 vLLM 属正常"
                          f"（其访存实现本身有约 35% 开销，实测 63–64%）")
            else:
                ok.append(f"`bw` 占标称 {frac*100:.0f}%（正常区间 75–95%）")

    # ② 高并发 TTFT 是否线性爆炸
    small = [m for m in meas if m["conc"] <= 1]
    big_all = [m for m in meas if m["conc"] >= 16]
    # 取**最大并发**那一格（ub 太小的问题在高并发最明显）
    big = [max(big_all, key=lambda m: m["conc"])] if big_all else []
    if small and big:
        s = min(small, key=lambda m: abs(m["L"] - 300))
        b = min(big, key=lambda m: abs(m["L"] - 300))
        if s["L"] > 0 and b["ttft"] > 0:
            ratio = b["ttft"] / s["ttft"]
            if ratio > 10:
                issues.append(
                    f"**c{b['conc']} 的 TTFT 是 c1 的 {ratio:.0f} 倍**"
                    f"（{b['ttft']:.0f} vs {s['ttft']:.0f} ms）\n"
                    f"     这是 `-ub`（ubatch）太小的典型特征。实测：ub=64 时该比值 **20×**，"
                    f"ub=512 时只有 **5.7×**。\n"
                    + (f"     建议：`-ub 512 -b 512` 重启服务后重新标定（c32 的 TTFT 可降 5.6 倍）"
                   if args.stack == "llamacpp" else
                   f"     建议：vLLM 的对应旋钮是 `--max-num-batched-tokens`（调大）与 "
                   f"`--enable-chunked-prefill`；CUDAGraph 也可能影响"))
            elif ratio > 7:
                issues.append(
                    f"c{b['conc']}/c1 的 TTFT 比值 {ratio:.0f}× 偏高（正常 5–6×）——"
                    f"考虑把 `-ub` 调大")
            else:
                ok.append(f"c{b['conc']}/c1 的 TTFT 比值 {ratio:.1f}×（正常）")

    # ③ c_srv 与同类先验比较
    lo, hi = PRIOR_C_SRV[args.stack][args.cls]
    cs = d["c_srv"]
    if cs < lo * 0.5 or cs > hi * 2:
        issues.append(
            f"**`c_srv` = {cs:.2f} ms 落在同类先验（{lo:.2f}–{hi:.2f}）之外**\n"
            f"     可能是：换了服务端配置（槽数/`-ub`/缓存）、同配置重复漂移（实测 +28%）或改了槽位（+64%）、"
            f"或这台机器负载不稳。\n"
            f"     记下来：参数必须与服务端配置一起记录")
    else:
        ok.append(f"`c_srv` = {cs:.2f} ms 在同类先验 {lo:.2f}–{hi:.2f} 之内")

    # ④ b_sat 是否偏小（配置产物）
    bsat = d.get("b_sat")
    if bsat is not None and bsat < 5:
        issues.append(
            f"`b_sat` = {bsat:.1f} 偏小——这个参数**随 `-ub` 变**（ub=512 时 4.6，ub=64 时 13.4），"
            f"不要当硬件常数引用")

    # ⑤ 拟合质量
    r = d.get("residual") or {}
    if r.get("tpot_mape", 0) > 15:
        issues.append(f"TPOT 拟合误差 {r['tpot_mape']:.1f}% 偏大——检查是否有格子的请求数不足")
    elif r.get("tpot_mape", 999) < 5:
        ok.append(f"TPOT 拟合误差 {r['tpot_mape']:.1f}%（很好）")

    print()
    if ok:
        print("  ✅ 正常项：")
        for o in ok:
            print(f"     · {o}")
    if issues:
        print("\n  ⚠️ 需要处理：")
        for i in issues:
            print(f"     · {i}")
    if not issues:
        print("\n  ✅ 没有发现问题——参数可以用，记得和服务端配置一起记录。")
    print()


if __name__ == "__main__":
    main()
