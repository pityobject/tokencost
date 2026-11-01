"""
tc_reset_data.py —— 归档并清空实验数据，为一次干净的全量重跑做准备。

为什么要归档而不是直接删：万一新一次又出问题，旧数据还能用来对照
（比如"这次是不是比上次快"）。归档放在 _selftest/archive_<时间>/，不占主目录。

它会清空：
    data/raw_cells.jsonl      逐格原始数据
    data/raw_requests.jsonl   逐请求明细
    data/summary.csv          汇总表
    data/quality.csv          质量表（可选保留，见 --keep-quality）
    results/*                 拟合参数与预测
    figures/*                 图

它**不会**动：
    data/probe.csv            槽位上下文体检（与重跑无关）
    data/backend_ref.csv      纯后端上限（与重跑无关）
    logs/                     保留，方便对照

用法：
  & "D:\\llm\\venv\\Scripts\\python.exe" tc_reset_data.py
  & "D:\\llm\\venv\\Scripts\\python.exe" tc_reset_data.py --keep-quality   # 保留质量表
"""

import argparse
import shutil
import sys
from datetime import datetime

import tc_config as cfg

FILES = ["raw_cells.jsonl", "raw_requests.jsonl", "summary.csv", "cells_to_rerun.txt"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--keep-quality", action="store_true",
                    help="保留 data/quality.csv（质量评测与延迟测量无关，可复用）")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    cfg.ensure_dirs()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    arch = cfg.PROJECT_ROOT / "_selftest" / f"archive_{stamp}"
    arch.mkdir(parents=True, exist_ok=True)

    moved = []
    for name in FILES:
        p = cfg.DATA_DIR / name
        if p.exists():
            moved.append(p)
    if not args.keep_quality:
        q = cfg.DATA_DIR / "quality.csv"
        if q.exists():
            moved.append(q)
        qd = cfg.DATA_DIR / "quality_details.jsonl"
        if qd.exists():
            moved.append(qd)
    for d in (cfg.RESULTS_DIR, cfg.FIG_DIR):
        if d.exists():
            moved += [p for p in d.iterdir() if p.is_file()]

    print(f"归档到 {arch}")
    for p in moved:
        print(f"   {p.relative_to(cfg.PROJECT_ROOT)}  ({p.stat().st_size/1024:.0f} KB)")

    if args.dry_run:
        print("\n[dry-run] 未实际移动")
        return

    for p in moved:
        shutil.move(str(p), str(arch / p.name))

    print(f"\n已归档 {len(moved)} 个文件，主目录已清空。")
    print("\n保留的：")
    for name in ("probe.csv", "backend_ref.csv", "README.md"):
        if (cfg.DATA_DIR / name).exists():
            print(f"   data/{name}")
    if args.keep_quality:
        print("   data/quality.csv（--keep-quality）")
    print("\n下一步：确认电源状态后，按顺序重跑：")
    print("   python tc_run_grid.py                 # 核心 45 格")
    print("   python tc_run_grid.py --preset dense  # 35 格")
    print("   python tc_run_grid.py --preset high   # 10 格")
    print("   python tc_run_grid.py --preset ctx    # 25 格")
    print("   python tc_run_grid.py --preset quant  # 28 格（需先下载 Q2/Q3）")


if __name__ == "__main__":
    main()
