# -*- coding: utf-8 -*-
"""
tc_killsrv.py —— 释放被占用的 llama-server 端口

## 为什么需要它

`tc_run_cell.py` 会为每一格**自己起一个** llama-server。如果之前手工起的服务
还没关，端口 8090 被占，整批实验会全部 failed（每格几百毫秒就报错）。

手工 `Stop-Process` 也能解决，但容易杀错进程。这里按**端口**精确定位：
只杀正在监听 cfg.PORT 的那个进程。

    python tc_killsrv.py            # 关掉占用 8090 的 llama-server
    python tc_killsrv.py --port 8081
    python tc_killsrv.py --dry-run  # 只看是谁占着，不杀
"""
import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import tc_config as cfg

sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def find_pids_windows(port):
    """用 netstat 找监听该端口的 PID（不依赖 psutil）。"""
    try:
        out = subprocess.run(["netstat", "-ano"], capture_output=True, text=True,
                             encoding="cp936", errors="replace", timeout=30)
    except Exception as e:
        print(f"  netstat 失败：{e}")
        return []
    pids = set()
    for line in out.stdout.splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        local, state, pid = parts[1], parts[3], parts[-1]
        if local.endswith(f":{port}") and state.upper() == "LISTENING":
            if pid.isdigit():
                pids.add(int(pid))
    return sorted(pids)


def proc_name(pid):
    """进程名。tasklist 在受限环境会 Access denied，所以多路兜底。"""
    # ① tasklist（最快，但可能被沙箱拒绝）
    try:
        out = subprocess.run(
            ["tasklist", "/FI", f"PID eq {pid}", "/FO", "CSV", "/NH"],
            capture_output=True, text=True, encoding="cp936", errors="replace", timeout=20)
        first = [ln for ln in out.stdout.strip().splitlines() if ln and "INFO" not in ln]
        if first and out.returncode == 0:
            return first[0].split(",")[0].strip('"')
    except Exception:
        pass
    # ② PowerShell Get-Process（几乎总能读到名字）
    try:
        out = subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             f"(Get-Process -Id {pid} -ErrorAction Stop).ProcessName"],
            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=25)
        name = out.stdout.strip().splitlines()
        if out.returncode == 0 and name:
            return name[0].strip()
    except Exception:
        pass
    return "?"


def main():
    ap = argparse.ArgumentParser(description="按端口释放 llama-server")
    ap.add_argument("--port", type=int, default=cfg.PORT)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force", action="store_true",
                    help="即使读不到进程名也杀（只杀监听该端口的进程）")
    args = ap.parse_args()

    if sys.platform != "win32":
        print("本脚本只处理 Windows；Linux/macOS 请用 lsof -ti:%d | xargs kill" % args.port)
        return 0

    pids = find_pids_windows(args.port)
    if not pids:
        print(f"端口 {args.port} 空闲，无需清理。")
        return 0

    for pid in pids:
        name = proc_name(pid)
        if args.dry_run:
            print(f"  占用 {args.port}：PID {pid}（{name}）")
            continue
        if name != "?" and "llama" not in name.lower():
            print(f"  ⚠️ PID {pid}（{name}）不像 llama-server，跳过——请手工确认")
            continue
        r = subprocess.run(["taskkill", "/F", "/PID", str(pid)],
                           capture_output=True, text=True, encoding="cp936", errors="replace")
        ok = r.returncode == 0
        print(f"  {'已结束' if ok else '结束失败'} PID {pid}（{name}）")
        if not ok:
            print("   " + (r.stdout or r.stderr).strip()[:160])

    left = find_pids_windows(args.port)
    if left:
        print(f"\n仍有占用：{left}（可能需要管理员权限）")
        return 1
    print(f"\n端口 {args.port} 已释放。")
    return 0


if __name__ == "__main__":
    sys.exit(main())

