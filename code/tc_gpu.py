"""
tc_gpu.py —— 在实验过程中持续记录 GPU 温度和频率。

为什么必须记这个？
  笔记本是移动端功耗管理：同一份代码，插电/不插电、风扇转不转、芯片热不热，
  跑出来的 tokens/s 能差 20% 以上。不记录这些，你的数据就无法复现，
  也无法回答"你那 40 tokens/s 是不是刚好赶在降频前"这种致命追问。
"""

import statistics
import subprocess
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
