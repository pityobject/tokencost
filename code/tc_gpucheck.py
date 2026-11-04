# -*- coding: utf-8 -*-
"""
tc_gpucheck.py —— 开跑前的一句话体检：GPU 上有没有"别人的活"

背景程序对延迟/成本测量的污染有三种（按杀伤力排序）：

1. **显存**：把格子推过悬崖阈值，llama.cpp 静默把 KV 搬到内存
   → 慢 2.6–4.3 倍且**不报错**（本项目踩过，见 失败与修复记录 F2）
2. **显存带宽**：直接压低拟合出的"有效带宽 BW"
3. **SM 抢占与帧节奏**：抬高尾部延迟（p90/p99）

服务没起时采样，看到的占用全部来自别的程序——这才是干净的判据。

    python tc_gpucheck.py            # 单次体检
    python tc_gpucheck.py --seconds 5
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import tc_gpu as gpu

sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def main():
    ap = argparse.ArgumentParser(description="开跑前的 GPU 背景体检")
    ap.add_argument("--seconds", type=float, default=2.0)
    args = ap.parse_args()

    b = gpu.idle_baseline(seconds=args.seconds)
    print("GPU 背景体检")
    print(f"  显存已用     {b.get('bg_mem_mb')} MB（含 Windows 桌面）")
    print(f"  空载利用率   p95 = {b.get('bg_util_p95')}%")
    print(f"  SM 时钟      {b.get('bg_clock_mhz')} MHz（空闲态本来就低，正常）")
    print(f"  判定         {b.get('bg_level')} —— {b.get('bg_notes')}")
    print(json.dumps(b, ensure_ascii=False))

    if b.get("bg_level") == "bad":
        print("\n建议先关掉这些再开跑：")
        print("  · 别的 DSH 会话（可能自己起了 llama-server → 抢显存 + 抢端口 8090）")
        print("  · 浏览器（尤其是开了硬件加速的页面/视频）")
        print("  · 游戏及其启动器、动态壁纸、音乐播放器的可视化")
        return 1
    if b.get("bg_level") == "warn":
        print("\n⚠️ 轻度背景占用：结果可用，但尾部延迟可能略偏高。")
        return 0
    print("\n✅ 背景干净，可以开跑。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
