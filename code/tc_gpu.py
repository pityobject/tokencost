"""
tc_gpu.py —— 在实验过程中持续记录 GPU 温度和频率。

为什么必须记这个？
  笔记本是移动端功耗管理：同一份代码，插电/不插电、风扇转不转、芯片热不热，
  跑出来的 tokens/s 能差 20% 以上。不记录这些，你的数据就无法复现，
  也无法回答"你那 40 tokens/s 是不是刚好赶在降频前"这种致命追问。
"""

import statistics
import subprocess
import time
import threading
import time


QUERY = ("temperature.gpu,clocks.sm,clocks.mem,power.draw,"
         "utilization.gpu,memory.used")


def _sample_once():
    try:
        out = subprocess.run(
            ["nvidia-smi", f"--query-gpu={QUERY}", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
        parts = [p.strip() for p in out.stdout.strip().split(",")]
        if len(parts) < 6:
            return None
        return dict(
            temp_c=float(parts[0]), clock_sm=float(parts[1]), clock_mem=float(parts[2]),
            power_w=float(parts[3]), util_pct=float(parts[4]), mem_used_mb=float(parts[5]),
        )
    except Exception:
        return None


class GpuMonitor:
    """用 with 语句包住一段要监控的代码，退出时自动给统计结果。

    用法：
        with GpuMonitor(interval_s=2.0) as mon:
            ...跑实验...
        print(mon.summary())      # {'temp_c_median': 71.0, ...}
    """

    def __init__(self, interval_s=2.0):
        self.interval_s = interval_s
        self.samples = []
        self._stop = threading.Event()
        self._thread = None

    def __enter__(self):
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=self.interval_s + 3)

    def _run(self):
        while not self._stop.is_set():
            s = _sample_once()
            if s:
                self.samples.append(s)
            self._stop.wait(self.interval_s)

    def summary(self):
        if not self.samples:
            return {}
        keys = self.samples[0].keys()
        return {f"{k}_median": statistics.median(s[k] for s in self.samples) for k in keys} | {
            f"{k}_max": max(s[k] for s in self.samples) for k in keys
        }


if __name__ == "__main__":
    print("单独测试：采 5 次 GPU 状态")
    with GpuMonitor(interval_s=1.0) as mon:
        time.sleep(5)
    for k, v in mon.summary().items():
        print(f"  {k:<22} {v:.2f}")


# ---------------------------------------------------------------- 外来占用归因
# 背景程序（壁纸引擎、浏览器硬件加速、QQ 音乐、游戏启动器、**别的 DSH 会话**）
# 对测量有三重污染：
#   ① 显存 → 把格子推过悬崖阈值，llama.cpp 静默 KV offload（慢 2.6–4.3 倍，不报错）
#   ② 显存带宽 → 直接压低我们拟合的"有效带宽 BW"
#   ③ SM 抢占 + 帧节奏 → 抬高尾部延迟（p90/p99）
# 服务未启动时采样，能干净地看到"本进程之外的占用"。

BG_MEM_WARN_MB = 500      # Windows 桌面本身约 200–400 MB，超过就说明有别的程序在吃显存
BG_MEM_BAD_MB = 1200      # 浏览器/游戏/别的推理进程
BG_UTIL_WARN_PCT = 5.0    # 空载时 GPU 利用率应当接近 0


def idle_baseline(seconds=1.5, interval_s=0.3):
    """在没有我们自己的服务时采样，返回外来占用的统计。

    返回 dict：
        bg_mem_mb   显存已用（中位）— 含 Windows 桌面
        bg_util_p95 利用率 p95
        bg_clock_mhz  SM 时钟中位
        bg_level    'ok' / 'warn' / 'bad'
        bg_notes    人话说明
    """
    samples = []
    end = time.time() + seconds
    while time.time() < end:
        s = _sample_once()
        if s:
            samples.append(s)
        time.sleep(interval_s)
    if not samples:
        return dict(bg_mem_mb=None, bg_util_p95=None, bg_clock_mhz=None,
                    bg_level="unknown", bg_notes="nvidia-smi 采样失败")

    mem = sorted(x["mem_used_mb"] for x in samples)
    util = sorted(x["util_pct"] for x in samples)
    clk = sorted(x["clock_sm"] for x in samples)
    p95 = util[max(0, int(round(0.95 * (len(util) - 1))))]
    notes, level = [], "ok"
    if mem[len(mem) // 2] >= BG_MEM_BAD_MB:
        level = "bad"
        notes.append(f"显存被占 {mem[len(mem)//2]:.0f} MB（≥{BG_MEM_BAD_MB}）——很可能有浏览器/游戏/别的推理进程")
    elif mem[len(mem) // 2] >= BG_MEM_WARN_MB:
        level = "warn"
        notes.append(f"显存被占 {mem[len(mem)//2]:.0f} MB（≥{BG_MEM_WARN_MB}）——建议关掉壁纸/浏览器硬件加速")
    if p95 >= BG_UTIL_WARN_PCT:
        level = "bad" if level == "bad" else "warn"
        notes.append(f"空载利用率 p95={p95:.0f}%（≥{BG_UTIL_WARN_PCT:.0f}%）——有程序在持续用 GPU")
    if not notes:
        notes.append("背景干净")
    return dict(bg_mem_mb=float(mem[len(mem) // 2]), bg_util_p95=float(p95),
                bg_clock_mhz=float(clk[len(clk) // 2]), bg_level=level,
                bg_notes="；".join(notes))

