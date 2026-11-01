"""
tc_server.py —— 管理 llama-server 的启停。

为什么需要这个文件？
  我们的实验要反复换组合（模型、量化、上下文、并发），每换一次都必须重启服务，
  因为上下文上限和并发槽位数是在启动时定死的。
  手动开窗口太容易出错，所以写个函数把它包起来。

新手须知：
  这里用 subprocess 起了外部程序。它不是 Python 函数，是一个独立进程。
  terminate() 相当于"体面地请它退出"，kill() 是"强制关掉"。
"""

import subprocess
import time
from pathlib import Path

import httpx

import tc_config as cfg

LOG_DIR = cfg.PROJECT_ROOT / "logs"


def _cmd(key, ctx, conc, extra=None, ngl=99, kv_cache_type=None):
    """拼服务端命令行。

    ngl：放到 GPU 的层数。99 = 全放；**0 = 纯 CPU**（迁移性实验用）。
    kv_cache_type：KV cache 量化档位，例如 "q8_0"。KV 字节数减半 → 显存悬崖移远。
    """
    cmd = [
        str(cfg.SERVER_EXE),
        "--model", str(cfg.model_path(key)),
        "--host", cfg.HOST,
        "--port", str(cfg.PORT),
        # 【最容易写错的一行】--ctx-size 是服务端【总】上下文，会按 --parallel 平分给槽位。
        # 要让每个槽位都有 ctx 个 token，必须传 ctx × conc。写成 str(ctx) 会让
        # 32 路并发时每槽位只剩 ctx/32 个 token，所有请求都被截断——实验直接废掉。
        "--ctx-size", str(cfg.total_ctx(ctx, conc)),
        "--parallel", str(conc),       # 槽位数 = 我们的"批大小"
        "--n-gpu-layers", str(ngl),    # 99 = 全放 GPU；0 = 纯 CPU
        "--metrics",                   # 打开 /metrics，方便以后加监控
    ]
    if kv_cache_type:
        # llama.cpp 的 K/V 缓存类型是分开的两个开关，这里统一设成同一档
        cmd += ["--cache-type-k", kv_cache_type, "--cache-type-v", kv_cache_type]
    if extra:
        cmd += extra
    return cmd


def free_port(port=None):
    """端口被占用时不要瞎猜，先查清楚。"""
    import socket
    port = port or cfg.PORT
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex((cfg.HOST, port)) != 0


def start(key, ctx, conc, timeout_s=180, extra=None, tag="", ngl=99, kv_cache_type=None):
    """启动服务并等它真正就绪。返回 (进程对象, 日志路径)。

    注意：就绪的判据不是"进程起来了"，而是 /health 返回 ok。
    模型加载要好几秒，提前发请求会失败。
    """
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f"server_{tag or 'run'}.log"
    if not free_port():
        raise RuntimeError(f"端口 {cfg.PORT} 被占用，先关掉别的 llama-server。")

    log_f = open(log_path, "w", encoding="utf-8", errors="replace")
    proc = subprocess.Popen(
        _cmd(key, ctx, conc, extra, ngl=ngl, kv_cache_type=kv_cache_type),
        stdout=log_f, stderr=subprocess.STDOUT,
        cwd=str(cfg.LLAMA_BIN),          # 让程序能找到同目录下的 DLL
    )

    t0 = time.time()
    while time.time() - t0 < timeout_s:
        if proc.poll() is not None:      # 进程自己退出了 → 启动了但失败
            log_f.close()
            tail = log_path.read_text(encoding="utf-8", errors="replace")[-2000:]
            raise RuntimeError(f"llama-server 启动即退出（多半是显存不够）。日志尾部：\n{tail}")
        try:
            r = httpx.get(f"{cfg.BASE_URL}/health", timeout=2.0)
            if r.status_code == 200:
                return proc, log_path
        except Exception:
            pass
        time.sleep(0.5)

    stop(proc)
    log_f.close()
    raise TimeoutError(f"等待服务就绪超时（{timeout_s}s），看日志：{log_path}")


def stop(proc):
    """关掉服务。Windows 上 terminate 有时不彻底，所以补一刀 kill。"""
    if proc is None or proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=15)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=10)


def server_info():
    """读 /props 拿服务端自报的信息（模型名、上下文等），用于写进数据表。"""
    try:
        return httpx.get(f"{cfg.BASE_URL}/props", timeout=5.0).json()
    except Exception:
        return {}


# 允许直接命令行测："python tc_server.py 3B q4_k_m 2048 4"
if __name__ == "__main__":
    import sys
    key = (sys.argv[1], sys.argv[2])
    ctx, conc = int(sys.argv[3]), int(sys.argv[4])
    p, lp = start(key, ctx, conc, tag="manual")
    print(f"服务已就绪：{cfg.BASE_URL}  日志：{lp}")
    print("浏览器打开 http://127.0.0.1:8090 就能直接对话。按 Ctrl+C 结束。")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        stop(p)
        print("已关闭。")
