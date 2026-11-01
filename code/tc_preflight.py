"""
tc_preflight.py —— 开跑前的真实环境自检。

做法：跑**一格真实测量**（3B-Q4 / ctx512 / 并发1，3B 加载快、耗时约 40 秒），
然后看这一格记录的 SM 时钟与功耗，判断机器是否处于满血状态。

为什么不用"空闲频率"判断：空闲 GPU 永远是 210 MHz / 2.5 W / P8 状态。
必须加载后看才准。这也是 tc_run_cell 每格都记录频率/功耗的原因。

判定：SM ≥ 2000 MHz 且 功耗 ≥ 50 W → 通过。

用法：
  & "D:\\llm\\venv\\Scripts\\python.exe" tc_preflight.py
"""

import json
import subprocess
import sys

import tc_config as cfg
from tc_run_cell import RAW_CELLS, run_cell


def check_power_plan():
    try:
        out = subprocess.run(["powercfg", "/getactivescheme"], capture_output=True,
                             text=True, timeout=15).stdout.strip()
    except Exception as e:
        print(f"[1] 电源计划：读取失败（{e}），跳过")
        return True
    name = out.split(":")[-1].strip() if ":" in out else out
    low = out.lower()
    bad = any(k in low for k in ("silent", "静音", "节能", "saver"))
    print(f"[1] 电源计划：{name}" + ("   [!!] 静音/节能，会限功耗" if bad else "   [OK]"))
    return not bad


def main():
    print("=" * 68)
    print("TokenCost 开跑前自检（跑一格真实测量，看频率与功耗）")
    print("=" * 68)
    plan_ok = check_power_plan()   # 仅作提示，不参与判定

    print("[2] 跑一格：3B-Q4 / ctx512 / 并发1（约 40 秒）...")
    rows = run_cell("3B", "q4_k_m", 512, 1, repeats=1, save_requests=False, warmup=1)
    r = rows[-1]
    if r.get("status") != "ok":
        print(f"    这一格没跑成：{r.get('error')}")
        sys.exit(1)

    clk = r.get("clock_sm_median") or 0
    pwr = r.get("power_w_median") or 0
    tmp = r.get("temp_c_median") or 0
    tpot = r.get("tpot_p50_ms") or 0
    print(f"    实测：SM {clk:.0f} MHz   功耗 {pwr:.1f} W   温度 {tmp:.0f} °C   TPOT {tpot:.1f} ms")
    print(f"    判据：SM ≥ {cfg.MIN_SM_CLOCK_MHZ} MHz 且 功耗 ≥ {cfg.MIN_POWER_W} W")
    print(f"    参考：满血 3B-Q4 在并发1 的 TPOT 约 9.5 ms")

    ok = clk >= cfg.MIN_SM_CLOCK_MHZ and pwr >= cfg.MIN_POWER_W
    # 判定只看实测：实测满血就放行，哪怕方案名是 Silent（这个方案只是风扇策略）
    print("=" * 68)
    if ok:
        print("通过。环境可以跑正式实验：")
        print("  python tc_run_grid.py")
        print("  python tc_run_grid.py --preset dense / high / ctx / quant")
        # 自检产生的这一格留着也行（数据有效），但会混进数据集
        print("\n提示：自检这一格已经写进 data/raw_cells.jsonl（数据是有效的）。")
        print("      如果想从干净状态开始，跑 python tc_reset_data.py 归档后重来。")
    else:
        print("未通过。按顺序修：")
        print("  1) 插上电源适配器")
        print("  2) 设置 → 系统 → 电源和电池 → 电源模式：最佳性能")
        print("     控制面板 → 电源选项 → 显示附加计划 → 高性能")
        print("  3) 厂商控制中心切「性能/野兽」：联想 Fn+Q、华硕 Armoury Crate")
        print("     ——它会覆盖 Windows 电源计划，很多人只改 Windows 那层就以为好了")
        print("  4) 关掉浏览器/录屏/杀毒扫描，等 1 分钟再跑本脚本")
        print(f"  本次测得 {clk:.0f} MHz / {pwr:.0f} W，正常应是 2300–2500 MHz / 80–95 W")
        sys.exit(2)


if __name__ == "__main__":
    main()

