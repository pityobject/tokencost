"""
tc_gpu.py —— 在实验过程中持续记录 GPU 温度和频率。

为什么必须记这个？
  笔记本是移动端功耗管理：同一份代码，插电/不插电、风扇转不转、芯片热不热，
  跑出来的 tokens/s 能差 20% 以上。不记录这些，你的数据就无法复现，
  也无法回答"你那 40 tokens/s 是不是刚好赶在降频前"这种致命追问。
"""

import statistics
import subprocess
from pathlib import Path
import time
import threading
import time


LAST_ERROR = [None]   # 最近一次采样失败的原因（供日志用）

QUERY = ("temperature.gpu,clocks.sm,clocks.mem,power.draw,"
         "utilization.gpu,memory.used")


def _sample_once():
    try:
        out = subprocess.run(
            ["nvidia-smi", f"--query-gpu={QUERY}", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
        if out.returncode != 0:
            LAST_ERROR[0] = f"nvidia-smi 退出码 {out.returncode}: {(out.stderr or '').strip()[:120]}"
            return None
        parts = [p.strip() for p in out.stdout.strip().split(",")]
        if len(parts) < 6:
            LAST_ERROR[0] = f"nvidia-smi 输出异常: {out.stdout.strip()[:120]!r}"
            return None
        return dict(
            temp_c=float(parts[0]), clock_sm=float(parts[1]), clock_mem=float(parts[2]),
            power_w=float(parts[3]), util_pct=float(parts[4]), mem_used_mb=float(parts[5]),
        )
    except Exception as e:      # 记下真实原因，别让调用方只能写"采样失败"
        LAST_ERROR[0] = f"{type(e).__name__}: {e}"
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
    # nvidia-smi 偶发失败（驱动忙、被安全软件拦、PATH 抖动），重试 3 次
    for attempt in range(3):
        end = time.time() + seconds
        while time.time() < end:
            s = _sample_once()
            if s:
                samples.append(s)
            time.sleep(interval_s)
        if samples:
            break
        time.sleep(0.5)
    if not samples:
        why = LAST_ERROR[0] or "nvidia-smi 无输出"
        print(f"    （背景采样失败，原因：{why}）", flush=True)
        return dict(bg_mem_mb=None, bg_util_p95=None, bg_clock_mhz=None,
                    bg_level="unknown", bg_notes=f"采样失败：{why}")

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



# ---------------------------------------------------------------- 时钟相对判据（W1）
# 背景程序/功耗墙会让 SM 时钟掉下来。旧的绝对闸门（<2000 且 <50W）太松：
# 2280 MHz / 93 W 的格子照样放行，但它比 2490 MHz 的格子慢约 8%。
#
# 参考值不能用 clocks.max.sm（那是标称 3105 MHz，笔记本永远达不到），
# 而要用"同一模型在历史上达到过的最高时钟"——可达参考。

CLOCK_RATIO_OK = 0.95      # ≥95% 参考值：正常
CLOCK_RATIO_WARN = 0.90    # 90–95%：偏低，标注
# <90%：明显偏低（标注为 bad）


def _ref_path():
    return Path(__file__).resolve().parent.parent / "results" / "clock_ref.json"


def clock_reference(rebuild=False):
    """每个模型的可达参考时钟（MHz）。结果缓存在 results/clock_ref.json。

    来源：data/raw_cells.jsonl 里所有成功格的 clock_sm_median 最大值。
    这是**经验参考**，不是标称值——后者在这台机器上不可达。
    """
    import json
    p = _ref_path()
    if p.exists() and not rebuild:
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            pass
    data = Path(__file__).resolve().parent.parent / "data" / "raw_cells.jsonl"
    best = {}
    if data.exists():
        for line in data.open(encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except Exception:
                continue
            if r.get("status") != "ok":
                continue
            c = r.get("clock_sm_median")
            if c:
                m = r.get("model", "?")
                best[m] = max(best.get(m, 0.0), float(c))
    if best:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(best, ensure_ascii=False, indent=2), encoding="utf-8")
    return best


def clock_gate(median_clock, model, ref=None):
    """返回 (ratio, level, note)。ref 为 None 时判定为 unknown。"""
    ref = ref if ref is not None else clock_reference()
    base = (ref or {}).get(model)
    if not median_clock or not base:
        return None, "unknown", "没有可达参考时钟（先跑任意一格即可自动建立）"
    ratio = float(median_clock) / float(base)
    if ratio >= CLOCK_RATIO_OK:
        return round(ratio, 4), "ok", f"时钟 {median_clock:.0f}/{base:.0f} MHz"
    if ratio >= CLOCK_RATIO_WARN:
        return round(ratio, 4), "warn", (
            f"时钟只有参考值的 {ratio*100:.0f}%（{median_clock:.0f}/{base:.0f} MHz）——标注，不作废")
    return round(ratio, 4), "bad", (
        f"时钟仅参考值的 {ratio*100:.0f}%（{median_clock:.0f}/{base:.0f} MHz）——"
        f"实测弹性约 0.3，仍不作废但需在分析里标注")
