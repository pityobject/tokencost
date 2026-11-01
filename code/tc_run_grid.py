"""
tc_run_grid.py —— 按顺序把设计空间里的格全部跑一遍，支持"断点续跑"。

断点续跑很重要：这套实验要跑两三个小时，中途你可能要关机、断电、去上课。
默认行为是读已有的 data/raw_cells.jsonl，已经跑满 REPEATS 轮的格自动跳过。

用法：
  # 先试一格，确认整条链路通
  & "D:\\llm\\venv\\Scripts\\python.exe" tc_run_grid.py --only 3B q4_k_m 2048 4 --repeats 1

  # 正式跑（会跳过已完成的格，随时 Ctrl+C 再重跑）
  & "D:\\llm\\venv\\Scripts\\python.exe" tc_run_grid.py

  # 只跑物理可行的格（默认），或连装不下的格也试一次（为了记录失败证据）
  & "D:\\llm\\venv\\Scripts\\python.exe" tc_run_grid.py --include-infeasible
"""

import argparse
import json
import time

import tc_config as cfg
from tc_run_cell import RAW_CELLS, run_cell


def load_done():
    """已有数据里，每格已经成功跑了几轮。"""
    done = {}
    if not RAW_CELLS.exists():
        return done
    with RAW_CELLS.open(encoding="utf-8") as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("status") == "ok":
                done[r["cell_id"]] = done.get(r["cell_id"], 0) + 1
    return done



def check_power_plan():
    """开跑前检查电源计划。笔记本的 Silent/节能 模式会把 GPU 压到 300–1500 MHz。"""
    import subprocess
    try:
        out = subprocess.run(["powercfg", "/getactivescheme"],
                             capture_output=True, text=True, timeout=15).stdout.strip()
    except Exception as e:
        print(f"[自检] 读不到电源计划（{e}），跳过")
        return
    print(f"[自检] 当前电源计划：{out.split(':')[-1].strip() if ':' in out else out}")
    low = out.lower()
    bad = any(k in low for k in ("silent", "静音", "节能", "powersaver", "power saver"))
    if bad:
        print("=" * 72)
        print("[拒绝开跑] 当前是静音/节能电源计划，GPU 会被压到 300–1500 MHz，")
        print("          跑出来的数据慢 3–7 倍却不会报错。先切到高性能：")
        print("          设置 → 系统 → 电源和电池 → 电源模式：最佳性能")
        print("          或命令行：powercfg /setactive 8c5e7fda-e8bf-4a96-9a85-a6e23a8c635c")
        print("          （若提示方案不存在：powercfg -duplicatescheme 8c5e7fda-e8bf-4a96-9a85-a6e23a8c635c）")
        print("          另外：部分笔记本还有厂商控制中心（联想 Vantage / 华硕 Armoury Crate）")
        print("          的独立性能档，也要设成「性能/狂暴」，它会覆盖 Windows 电源计划。")
        print("注意：方案名不代表真实性能——本机实测 Silent 也能跑到 2340 MHz / 80 W。")
        print("真正的判据是加载后的实测频率，由 tc_preflight.py 与每格的 valid 标记负责。")
        print("=" * 72)
        # 不阻断；继续跑，但每格结束会打 [无效] 标记
    else:
        print("[自检] 电源计划正常。")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=cfg.REPEATS)
    ap.add_argument("--include-infeasible", action="store_true",
                    help="连显存估计装不下的格也试一次（会失败，但失败本身是证据）")
    ap.add_argument("--only", nargs=4, metavar=("MODEL", "QUANT", "CTX", "CONC"),
                    help="只跑指定的一格")
    ap.add_argument("--fresh", action="store_true", help="忽略已有数据，从头跑")
    ap.add_argument("--limit", type=int, default=0,
                    help="本次最多跑几个格（先跑 6 个验证数据正常，再跑全部）")
    ap.add_argument("--models", default="",
                    help="只跑指定模型，逗号分隔，例如 3B 或 1.5B,3B")
    ap.add_argument("--preset", default=None,
                    help="预设：core/dense/high/ctx/quant/all + step（定位台阶）/wave（波次对照）/knee（算力拐点）")
    args = ap.parse_args()

    cfg.ensure_dirs()

    # 开跑前自检。
    # 注意：空闲 GPU 永远是 210 MHz / 2.5 W / P8，拿它判断降频是错的（我第一版就错在这）。
    # 真正能提前查的是「电源计划」和「是否插电」；频率要在第一格跑起来之后看。
    check_power_plan()

    if args.only:
        m, q, c, n = args.only
        run_cell(m, q, int(c), int(n), repeats=args.repeats)
        return

    done = {} if args.fresh else load_done()
    if args.preset:
        cells = cfg.preset_cells(args.preset)
        print(f"[preset={args.preset}] 共 {len(cells)} 个可行格（离群格已自动剔除）")
    else:
        cells = cfg.all_cells(specs=cfg.CORE_SPECS)
    if not args.include_infeasible:
        cells = [c for c in cells if c["feasible"]]
    if args.models:
        wanted = {m.strip() for m in args.models.split(",") if m.strip()}
        cells = [c for c in cells if c["model"] in wanted]

    todo = [c for c in cells
            if done.get(cfg.cell_id(c["model"], c["quant"], c["ctx"], c["conc"],
                                    c.get("suffix", "")), 0) < args.repeats]
    if args.limit and len(todo) > args.limit:
        print(f"[--limit {args.limit}] 本次只跑前 {args.limit} 个格（其余下次续跑）")
        todo = todo[:args.limit]

    print("=" * 72)
    print(f"设计空间共 {len(cfg.all_cells())} 格；物理可行 {len([c for c in cfg.all_cells() if c['feasible']])} 格")
    print(f"本次待跑 {len(todo)} 格，每格 {args.repeats} 轮")
    print("=" * 72)

    t_all = time.time()
    for i, c in enumerate(todo, 1):
        cid = cfg.cell_id(c["model"], c["quant"], c["ctx"], c["conc"], c.get("suffix", ""))
        print(f"\n[{i}/{len(todo)}] {cid}   估计显存 {c['vram_est_gb']} GB   "
              f"已用时 {(time.time()-t_all)/60:.1f} min")
        t0 = time.time()
        rows = run_cell(c["model"], c["quant"], c["ctx"], c["conc"],
                        repeats=args.repeats, warmup=2,
                        requests=c.get("requests"), suffix=c.get("suffix", ""))
        st = rows[-1].get("status")
        print(f"    → {st}，耗时 {(time.time()-t0):.0f}s")

    print("\n" + "=" * 72)
    print(f"全部完成，总用时 {(time.time()-t_all)/60:.1f} min")
    print(f"原始数据：{RAW_CELLS}")
    print("下一步： python tc_analyze.py  （汇总 + 出图）")


if __name__ == "__main__":
    main()
