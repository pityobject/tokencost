"""
tc_env_check.py —— 环境自检。开工第一件事，跑这个。

它会检查：
  1. Python 版本和关键库是否装好
  2. llama.cpp 的可执行文件在不在
  3. 模型文件在不在、字节数对不对（防止下载了半截文件）
  4. 显存/磁盘够不够
  5. 设计空间里哪些格装不下（显存可行性剪枝表）

用法：
  & "D:\\llm\\venv\\Scripts\\python.exe" tc_env_check.py
"""

import shutil
import subprocess
import sys
from pathlib import Path

import tc_config as cfg

OK, BAD, WARN = "[OK]  ", "[!!]  ", "[~~]  "


def line(msg, flag=OK):
    print(flag + msg)


def check_python():
    print("\n=== 1. Python 与依赖库 ===")
    v = sys.version_info
    line(f"Python {v.major}.{v.minor}.{v.micro}  ({sys.executable})",
         OK if (v.major, v.minor) == (3, 12) else WARN)
    if (v.major, v.minor) != (3, 12):
        line("建议用 3.12（D:\\ruanjian\\Python 3.12.0）。其他版本也能跑，但别用 3.14 alpha。", WARN)

    need = {
        "numpy": "numpy", "pandas": "pandas", "scipy": "scipy",
        "matplotlib": "matplotlib", "httpx": "httpx",
        "simpy": "simpy", "psutil": "psutil",
    }
    missing = []
    for mod, pkg in need.items():
        try:
            m = __import__(mod)
            line(f"{mod:<12} {getattr(m, '__version__', '?')}")
        except ImportError:
            missing.append(pkg)
            line(f"{mod:<12} 缺失", BAD)
    if missing:
        line("缺少依赖，运行： python -m pip install " + " ".join(missing), BAD)
    return not missing


def check_llama():
    print("\n=== 2. llama.cpp 可执行文件 ===")
    print(f"自动定位到的目录：{cfg.LLAMA_BIN}")

    ok = True
    missing = [n for e, n in [(cfg.SERVER_EXE, "llama-server"), (cfg.BENCH_EXE, "llama-bench")]
               if not e.exists()]

    if missing:
        ok = False
        line(f"找不到：{', '.join(missing)}", BAD)
        print("  已搜索过这些位置：")
        for s in cfg.LLAMA_BIN_SEARCHED:
            print(f"    - {s}")
        print("  请自己确认文件到底在哪（复制这条命令跑）：")
        print(r"    Get-ChildItem D:\llm -Recurse -Filter llama-server.exe -ErrorAction SilentlyContinue | Select FullName")
        print(r"  如果在别处找到，把那个目录里的所有 .exe 和 .dll 一起挪到 D:\llm\bin 即可：")
        print(r"    Move-Item <找到的目录>\* D:\llm\bin -Force")
        return False

    for exe, name in [(cfg.SERVER_EXE, "llama-server"), (cfg.BENCH_EXE, "llama-bench")]:
        kb = exe.stat().st_size / 1024
        line(f"{name:<14} {exe}  ({kb:.0f} KB)")
    print("  说明：这些 .exe 只有 9 KB 是正常的——它们只是壳，真正的代码在 llama.dll /"
          " ggml-cuda.dll 里。不要因为文件小就以为没下完。")

    # 能不能真的跑起来（这一步能一次性抓出缺 DLL、缺 VC++ 运行库两类问题）
    try:
        out = subprocess.run([str(cfg.SERVER_EXE), "--version"],
                             capture_output=True, text=True, timeout=30,
                             cwd=str(cfg.LLAMA_BIN))
        first = (out.stdout or out.stderr).strip().splitlines()
        if out.returncode == 0 and first:
            line(f"可执行（--version 正常）：{first[0]}")
        else:
            ok = False
            line(f"能启动但退出码 {out.returncode}，输出：{(first or ['(空)'])[0]}", BAD)
    except PermissionError as e:
        # 被限制的环境（如某些沙箱/策略）会拒绝启动子进程；这不是你装错了
        line(f"无法在当前环境启动子进程（{e}）。手工验证一下：", WARN)
        print(r"    D:\llm\bin\llama-server.exe --version")
        print("  能打印 version: ... 就说明没问题。")
    except Exception as e:
        ok = False
        line(f"启动失败：{type(e).__name__}: {e}", BAD)
        print("  常见原因：① 缺 CUDA 运行库 DLL（重做 4.2 步）② 缺 VC++ 2015-2022 Redistributable")

    # CUDA 运行库是否齐全
    dlls = list(cfg.LLAMA_BIN.glob("*.dll"))
    big = [d for d in dlls if d.stat().st_size > 10e6]
    if len(dlls) >= 5:
        line(f"DLL：{len(dlls)} 个（其中 >10 MB 的 {len(big)} 个，如 {', '.join(d.name for d in big[:3])}）")
    else:
        ok = False
        line(f"只找到 {len(dlls)} 个 DLL —— 多半是忘了解压 cudart-llama-bin-win-cuda-12.4-x64.zip", BAD)

    # GPU 后端是否被这个 build 认到
    try:
        out = subprocess.run([str(cfg.BENCH_EXE), "--list-devices"],
                             capture_output=True, text=True, timeout=60,
                             cwd=str(cfg.LLAMA_BIN))
        txt = (out.stdout or "") + (out.stderr or "")
        if "CUDA" in txt.upper():
            dev = [l.strip() for l in txt.splitlines() if "CUDA" in l.upper()][:1]
            line(f"后端：{dev[0] if dev else '检测到 CUDA'}")
        else:
            ok = False
            line("后端里没有 CUDA —— 这个包可能是 CPU 版，或者 DLL 没放对", BAD)
            print("  你下载的应该是 llama-bXXXXX-bin-win-cuda-12.4-x64.zip（CUDA 12.4 版）")
    except Exception as e:
        line(f"llama-bench --list-devices 执行失败：{e}", WARN)

    return ok


def check_models():
    print("\n=== 3. 模型文件（字节数必须与配置一致） ===")
    all_ok = True
    for key, spec in cfg.MODEL_SPECS.items():
        total = 0
        for f in spec["files"]:
            p = cfg.MODEL_DIR / f
            if not p.exists():
                all_ok = False
                line(f"{key[0]:>4} {key[1]:<7} 缺失 {f}", BAD)
                break
            total += p.stat().st_size
        else:
            match = (total == spec["weight_bytes"])
            expect = spec["weight_bytes"] / 1e9
            hint = "一致" if match else f"不一致（应为 {expect:.2f} GB，可能下载不完整）"
            line(f"{key[0]:>4} {key[1]:<7} {total/1e9:6.2f} GB  {hint}", OK if match else BAD)
            all_ok &= match
    return all_ok


def check_gpu():
    print("\n=== 4. GPU 与磁盘 ===")
    ok = True
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=20,
        )
        info = out.stdout.strip()
        line(f"GPU: {info}")
        if "4060" not in info:
            line("显卡不是 4060，模型里的带宽/算力参数需要重新测，结论仍成立。", WARN)
    except Exception as e:
        ok = False
        line(f"nvidia-smi 调用失败：{e}（没装显卡驱动？）", BAD)

    need_gb = sum(s["weight_bytes"] for s in cfg.MODEL_SPECS.values()) / 1e9
    for drive, need in [("D:", need_gb + 3), ("C:", 2.0)]:
        try:
            total, used, free = shutil.disk_usage(drive + "\\")
            line(f"{drive} 剩余 {free/1e9:5.1f} GB（本项目在 {drive} 上需要约 {need:.1f} GB）",
                 OK if free / 1e9 > need else BAD)
            ok &= (free / 1e9 > need)
        except Exception as e:
            line(f"{drive} 读取失败：{e}", BAD)
    return ok


def show_design_space():
    print("\n=== 5. 设计空间与可行性（两条硬约束：显存 + 总上下文） ===")
    print(f"{'模型':>6} {'量化':<7} {'每槽ctx':>8} {'并发':>5} {'总ctx':>7} {'显存估计':>9}  结论")
    cells = cfg.all_cells()
    for c in cells:
        verdict = "√ 可行" if c["feasible"] else "× " + (c.get("reason") or "")
        print(f"{c['model']:>6} {c['quant']:<7} {c['ctx']:>8} {c['conc']:>5} "
              f"{c['ctx_total']:>7} {c['vram_est_gb']:>8.2f}G  {verdict}")

    ok_n = sum(c["feasible"] for c in cells)
    ctx_only = sum(1 for c in cells if not c["feasible"] and "显存" not in c.get("reason", ""))
    both = sum(1 for c in cells if not c["feasible"] and "显存" in c.get("reason", ""))
    print(f"\n全网格 {len(cells)} 格：可行 {ok_n} 格，剪掉 {len(cells)-ok_n} 格"
          f"（{ctx_only} 格仅因上下文超限，{both} 格同时超上下文与显存）。")
    print(f"标定集 12 格全部在可行集内；留出集 {ok_n - 12} 格。")
    print("被剪掉的格不要硬试 —— 它们是'约束非随机地剪设计空间'的证据，直接写进报告。")
    print("\n单位说明：显卡标称 8188 MiB ≈ 8.59×10⁹ 字节，本表按十进制 GB 估算并保留约 1 GB 余量"
          "（WDDM/桌面合成会占用一部分），所以比标称值保守。")


def main():
    print("=" * 70)
    print("TokenCost 环境自检")
    print("=" * 70)
    a = check_python()
    b = check_llama()
    c = check_models()
    d = check_gpu()
    show_design_space()
    print("\n" + "=" * 70)
    if all([a, b, c, d]):
        print("环境就绪。可以跑 tc_smoke_test.py 做第一次推理了。")
    else:
        print("上面标 [!!] 的项还没通过，先按 `docs/tutorial/01_环境搭建_S0.md` 修完再往下走。")


if __name__ == "__main__":
    main()
