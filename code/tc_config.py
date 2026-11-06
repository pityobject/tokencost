"""
tc_config.py —— 全局配置。所有脚本只从这里读路径和参数，不要在别处硬编码。

新手须知：
  * 这个文件是整套代码的"总控台"。改模型、改实验网格、改端口，都在这里改。
  * MODEL_SPECS 里每个数字都不是猜的：
      - weight_bytes = 实际查到的 GGUF 文件字节数（可用文件属性核对）
      - n_layers / n_kv_heads / head_dim = 模型结构参数（决定 KV cache 大小）
      - n_params = 参数量（决定每步要做多少浮点运算）
"""

from pathlib import Path
import os

# ---------------------------------------------------------------- 路径
PROJECT_ROOT = Path(__file__).resolve().parent.parent   # tokencost/
DATA_DIR     = PROJECT_ROOT / "data"                    # 原始实验数据
RESULTS_DIR  = PROJECT_ROOT / "results"                 # 拟合结果、汇总表
FIG_DIR      = PROJECT_ROOT / "figures"                 # 图
LOG_DIR      = PROJECT_ROOT / "logs"

# ---------------------------------------------------------------- 外部依赖位置
# 【可移植性】换机器时不需要改代码，设两个环境变量即可：
#   TOKENCOST_LLAMA_BIN   llama.cpp 的 bin 目录（含 llama-server.exe）
#   TOKENCOST_MODEL_DIR   GGUF 权重目录
# 没设就回退到作者本机的默认位置，再退到磁盘搜索。
import os as _os

_ENV_LLAMA = _os.environ.get("TOKENCOST_LLAMA_BIN", "").strip()
_ENV_MODELS = _os.environ.get("TOKENCOST_MODEL_DIR", "").strip()

LLAMA_BIN_SEARCH = ([Path(_ENV_LLAMA)] if _ENV_LLAMA
                    else [Path(r"D:\llm\bin\llama-cuda"), Path(r"D:\llm\bin")])
LLAMA_SEARCH_ROOTS = ([Path(_ENV_LLAMA).parent, Path(_ENV_LLAMA)] if _ENV_LLAMA
                      else [Path(r"D:\llm\bin"), Path(r"D:\llm")])
MODEL_DIR = (Path(_ENV_MODELS) if _ENV_MODELS else Path(r"D:\llm\models"))


def _find_exe_dir(exe_name="llama-server.exe", candidates=None, roots=None, max_depth=5):
    """自动定位 llama.cpp 的目录（兼容你实际解压到哪一层）。

    为什么不在配置里写死一个路径：解压时文件可能落在 bin 根目录、也可能落在
    bin\\llama-cuda 或 bin\\build\\bin。写死一处，另一处就报"找不到"，
    而那个报错看起来很像"你没解压"——我被这个坑过一次。

    返回 (目录, 搜索过的位置列表)。
    """
    searched = []
    for c in (candidates if candidates is not None else LLAMA_BIN_SEARCH):
        c = Path(c)
        searched.append(str(c))
        if (c / exe_name).exists():
            return c, searched

    for root in (roots if roots is not None else LLAMA_SEARCH_ROOTS):
        root = Path(root)
        if not root.exists():
            continue
        searched.append(f"{root} (递归 {max_depth} 层)")
        for dirpath, dirnames, filenames in os.walk(root):
            depth = len(Path(dirpath).relative_to(root).parts)
            if depth >= max_depth:
                dirnames[:] = []
            # 跳过体积巨大的无关目录，避免遍历几万个文件
            dirnames[:] = [d for d in dirnames
                           if d.lower() not in ("venv", "site-packages", ".git", "__pycache__")]
            if exe_name in filenames:
                return Path(dirpath), searched

    return Path(r"D:\llm\bin"), searched        # 找不到时的默认值（仅用于报错展示）


LLAMA_BIN, LLAMA_BIN_SEARCHED = _find_exe_dir()          # 自动定位，不再写死

SERVER_EXE = LLAMA_BIN / "llama-server.exe"
BENCH_EXE  = LLAMA_BIN / "llama-bench.exe"

HOST = "127.0.0.1"
PORT = 8090                                             # 显式指定端口，不依赖默认值
BASE_URL = f"http://{HOST}:{PORT}"

# ---------------------------------------------------------------- 模型清单
# key = (模型规模标签, 量化标签)。多分片模型只需第一个分片，llama.cpp 会自动找后续分片。
MODEL_SPECS = {
    ("1.5B", "q4_k_m"): dict(
        files=["qwen2.5-1.5b-instruct-q4_k_m.gguf"],
        weight_bytes=1_117_320_736,
        n_params=1.54e9, n_layers=28, n_kv_heads=2, head_dim=128,
    ),
    ("1.5B", "q8_0"): dict(
        files=["qwen2.5-1.5b-instruct-q8_0.gguf"],
        weight_bytes=1_894_532_128,
        n_params=1.54e9, n_layers=28, n_kv_heads=2, head_dim=128,
    ),
    ("3B", "q4_k_m"): dict(
        files=["qwen2.5-3b-instruct-q4_k_m.gguf"],
        weight_bytes=2_104_932_768,
        n_params=3.09e9, n_layers=36, n_kv_heads=2, head_dim=128,
    ),
    ("3B", "q8_0"): dict(
        files=["qwen2.5-3b-instruct-q8_0.gguf"],
        weight_bytes=3_616_088_480,
        n_params=3.09e9, n_layers=36, n_kv_heads=2, head_dim=128,
    ),
    ("1.5B", "q2_k"): dict(
        files=["qwen2.5-1.5b-instruct-q2_k.gguf"],
        weight_bytes=752_880_160,
        n_params=1.54e9, n_layers=28, n_kv_heads=2, head_dim=128,
    ),
    ("1.5B", "q3_k_m"): dict(
        files=["qwen2.5-1.5b-instruct-q3_k_m.gguf"],
        weight_bytes=924_455_968,
        n_params=1.54e9, n_layers=28, n_kv_heads=2, head_dim=128,
    ),
    ("3B", "q2_k"): dict(
        files=["qwen2.5-3b-instruct-q2_k.gguf"],
        weight_bytes=1_376_856_480,
        n_params=3.09e9, n_layers=36, n_kv_heads=2, head_dim=128,
    ),
    ("3B", "q3_k_m"): dict(
        files=["qwen2.5-3b-instruct-q3_k_m.gguf"],
        weight_bytes=1_724_178_848,
        n_params=3.09e9, n_layers=36, n_kv_heads=2, head_dim=128,
    ),
    ("0.5B", "q4_k_m"): dict(
        files=["qwen2.5-0.5b-instruct-q4_k_m.gguf"],
        weight_bytes=491_400_032,
        # 架构：24 层 / 14 注意力头 / 2 个 KV 头 / head_dim = 896/14 = 64
        n_params=0.49e9, n_layers=24, n_kv_heads=2, head_dim=64,
    ),
    ("Llama1B", "q4_k_m"): dict(
        files=["Llama-3.2-1B-Instruct-Q4_K_M.gguf"],
        weight_bytes=807_694_368,
        # Llama-3.2-1B：16 层 / 32 注意力头 / 8 个 KV 头（GQA）/ head_dim = 2048/32 = 64
        # KV = 2 × 16 × 8 × 64 × 2 = 32,768 B/token
        n_params=1.24e9, n_layers=16, n_kv_heads=8, head_dim=64,
    ),
    ("7B", "q4_k_m"): dict(
        files=["qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf",
               "qwen2.5-7b-instruct-q4_k_m-00002-of-00002.gguf"],
        weight_bytes=4_683_073_632,     # 3,993,201,344 + 689,872,288
        n_params=7.62e9, n_layers=28, n_kv_heads=4, head_dim=128,
    ),
    # 7B q8_0 权重 8.10 GB > 8 GB 显存，物理装不下，不列进网格。
}

# ---------------------------------------------------------------- 实验网格
# 核心网格（已跑完，45 格可行，是历史结论的基础，**不要动**）
CTX_SIZES   = [512, 2048, 8192]        # 【每槽位】上下文上限（token），不是总上下文
CONCURRENCY = [1, 4, 16, 32]           # 并发请求数 = llama-server 的 --parallel（槽位数）

# ---- C 档扩展维度（2026-10-31 追加；标定集不变，新增格全部进留出集）----
CTX_EXT   = [1024, 4096]               # 细化上下文，用来检验 KV 项
CONC_EXT  = [2, 8, 12, 24, 48, 64]     # 细化 / 拉高并发
# 为什么没有 128：ctx×并发 ≤ 32768 且提示词 296 token 要求 ctx ≥ 424
# → 512 × 64 = 32768 已是物理上限，128 不可能。


# 【关键·别改错】llama-server 的 --ctx-size 是**整个服务端的总上下文**，会按 --parallel
# 平分给各槽位。要让每个槽位都有 ctx 个 token，必须传 --ctx-size = ctx × 并发数。
# 又因为模型原生最大上下文是 32768，总上下文不能超过它（除非开 YaRN，本项目不开），
# 于是产生第二条硬约束：ctx × 并发 ≤ N_CTX_TRAIN。
# 这条约束会剪掉「2048×32」「8192×16」「8192×32」这些组合。
N_CTX_TRAIN = 32768

GEN_TOKENS    = 128                    # 每个请求固定生成多少 token
PROMPT_TOKENS = 256                    # 每个请求的提示长度（近似；实际以 usage 为准）
REPEATS       = 3                      # 每格重复次数（取中位数）

# ---- 测量有效性闸门（2026-10-31 加）----
# 笔记本没插电或电源计划变节能时，GPU 会从 2490 MHz 掉到 300–1500 MHz，
# 数据慢 3–7 倍却毫无提示。判据必须是「时钟低 且 功耗低」：只看功耗会把
# 「高并发下 GPU 空转」误判成降频（那种情况时钟仍是满频）。
MIN_SM_CLOCK_MHZ = 2000
MIN_POWER_W      = 50
PREFLIGHT_CLOCK_MHZ = 1500    # 开跑前自检阈值，低于它就拒绝开跑


GPU_OVERHEAD_GB = 0.6                  # 计算缓冲区等额外显存估计


def requests_per_repeat(conc: int) -> int:
    """并发越高，单次要发的请求越多，才能让测量稳定。"""
    # 【高并发要更多请求】原来 c48/c64/c128 都回落到 32 个请求——
    # 32 个请求在 conc=64 时连一波都不满，测到的是"启动瞬间"而不是稳态。
    explicit = {1: 16, 2: 32, 4: 24, 8: 32, 12: 48, 16: 48, 24: 72, 32: 64}
    if conc in explicit:
        return explicit[conc]
    return max(64, 2 * conc)   # 至少两波，保证进入稳态


# ---------------------------------------------------------------- 标定集 / 留出集
# 只允许模型看这 12 个格；其余全部是留出集，用来算"它没见过的格预测得准不准"。
CALIB_CELLS = [
    ("3B",   "q4_k_m", 512,  1),
    ("3B",   "q4_k_m", 2048, 16),
    ("3B",   "q4_k_m", 8192, 4),
    ("7B",   "q4_k_m", 512,  32),
    ("7B",   "q4_k_m", 2048, 4),
    ("7B",   "q4_k_m", 8192, 1),
    ("1.5B", "q8_0",   512,  16),
    ("1.5B", "q8_0",   2048, 1),
    ("1.5B", "q4_k_m", 8192, 4),
    ("3B",   "q8_0",   512,  4),
    ("3B",   "q8_0",   8192, 4),      # 原为 (2048, 32)，因 ctx×并发超 32k 已换掉
    ("1.5B", "q4_k_m", 2048, 16),
]


# ---------------------------------------------------------------- 结构辅助函数
def kv_bytes_per_token(spec) -> int:
    """每个上下文 token 占多少字节的 KV cache（fp16）。

    每层要为每个 KV 头保存 K 和 V 两份，各 head_dim 个数，每个数 2 字节。
    这就是 GQA（分组查询注意力）省显存的地方：KV 头数远少于 Q 头数。
    """
    return 2 * spec["n_layers"] * spec["n_kv_heads"] * spec["head_dim"] * 2


def kv_gb(spec, ctx: int, conc: int) -> float:
    """总 KV 占用。总上下文 = ctx × 并发，所以 KV 也是按总量算。"""
    return kv_bytes_per_token(spec) * ctx * conc / 1e9


def total_ctx(ctx: int, conc: int) -> int:
    """要传给 llama-server 的 --ctx-size（总上下文）。

    为什么必须乘：--ctx-size 是服务端总上下文，服务端会按 --parallel 平分给槽位。
    只传 ctx 而不乘，8 路并发时每槽位只剩 ctx/8，短提示都会被截断。
    """
    return ctx * conc


def estimate_vram_gb(key, ctx: int, conc: int) -> float:
    spec = MODEL_SPECS[key]
    return spec["weight_bytes"] / 1e9 + kv_gb(spec, ctx, conc) + GPU_OVERHEAD_GB


def ctx_cap_ok(ctx: int, conc: int) -> bool:
    """上下文上限：总上下文不能超过模型原生最大长度（未开 YaRN）。"""
    return total_ctx(ctx, conc) <= N_CTX_TRAIN


def infeasible_reason(key, ctx: int, conc: int, vram_gb: float = 8.0):
    """返回全部不可行原因（可行则返回 None）。

    两条约束可能同时违反（例如 7B × 8k × 16 并发：总上下文超限、显存也不够）。
    报告里要能分别统计"被上下文剪掉几个""被显存剪掉几个""同时被两者剪掉几个"。
    """
    reasons = []
    if not ctx_cap_ok(ctx, conc):
        reasons.append(f"总上下文 {total_ctx(ctx, conc)} > {N_CTX_TRAIN}")
    vram = estimate_vram_gb(key, ctx, conc)
    if vram >= vram_gb * 0.95:
        reasons.append(f"显存估计 {vram:.2f} GB ≥ {vram_gb*0.95:.1f} GB")
    return "；".join(reasons) if reasons else None


def kv_fits_vram(key, ctx: int, conc: int, vram_gb: float = 8.0) -> bool:
    """只按显存判断 KV 是否装得下（不管上下文窗口那条约束）。

    这条判据决定了一个格属于哪个 regime：
      True  → KV 全在显存里，属于本项目建模的物理系统；
      False → llama.cpp 会静默把 KV 搬到主机内存（--kv-offload 默认开），
              吞吐断崖式下跌，**不能混进模型拟合**。
    """
    return estimate_vram_gb(key, ctx, conc) < vram_gb * 0.95


def feasible(key, ctx: int, conc: int, vram_gb: float = 8.0) -> bool:
    """两条硬约束都要满足：显存装得下 + 总上下文不超模型上限。"""
    return infeasible_reason(key, ctx, conc, vram_gb) is None


def all_cells(concs=None, ctxs=None, specs=None):
    """设计空间里的所有格，附带显存估计、总上下文与可行性。

    concs / ctxs / specs 可覆盖，方便做扩展实验而不动核心网格。
    """
    concs = CONCURRENCY if concs is None else concs
    ctxs = CTX_SIZES if ctxs is None else ctxs
    specs = list(MODEL_SPECS) if specs is None else list(specs)
    out = []
    for key in specs:
        for ctx in ctxs:
            for conc in concs:
                out.append(dict(
                    model=key[0], quant=key[1], ctx=ctx, conc=conc,
                    ctx_total=total_ctx(ctx, conc),
                    vram_est_gb=round(estimate_vram_gb(key, ctx, conc), 2),
                    feasible=feasible(key, ctx, conc),
                    reason=infeasible_reason(key, ctx, conc),
                ))
    return out


def model_path(key) -> Path:
    return MODEL_DIR / MODEL_SPECS[key]["files"][0]


def cell_id(model, quant, ctx, conc, suffix="") -> str:
    """格标识。suffix 用于区分"同配置不同请求数"的实验（例如波次对照）。"""
    return f"{model}_{quant}_ctx{ctx}_c{conc}{suffix}"


def ensure_dirs():
    for d in (DATA_DIR, RESULTS_DIR, FIG_DIR, LOG_DIR):
        d.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------- 实验预设（C 档）
# 核心 5 个组合 = 已完成 45 格的那套，**冻结**。必须写死，
# 不能用"排除法"——后来加的 0.5B 会被顺带卷进来，把 45 格变成 54 格。
CORE_SPECS = [("1.5B", "q4_k_m"), ("1.5B", "q8_0"),
              ("3B", "q4_k_m"), ("3B", "q8_0"), ("7B", "q4_k_m")]
# 纪律：CORE 的划分是历史结论的基础，**冻结不改**；所有扩展格一律进留出集。

def _with(cells, requests=None, suffix=""):
    """给一批格挂上"显式请求数"与"格名后缀"。"""
    out = []
    for c in cells:
        c = dict(c)
        if requests is not None:
            c["requests"] = requests(c["conc"]) if callable(requests) else requests
        c["suffix"] = suffix(c["conc"]) if callable(suffix) else suffix
        out.append(c)
    return out


def _step_preset():
    """定位台阶：并发 4→16 逐点加密，**固定 4 波**。

    为什么要固定波次：上一轮发现 TPOT 会被"新请求的 prefill 与旧请求的 decode 重叠"污染，
    而各格请求数不一致导致波次从 16 到 1.33 不等。固定 4 波后，
    并发是唯一变量，台阶位置才有意义。
    """
    concs = [4, 5, 6, 7, 8, 10, 12, 16]
    cells = all_cells(concs=concs, ctxs=[512], specs=[("3B", "q4_k_m")])
    return _with(cells, requests=lambda b: 4 * b, suffix="_w4")


def _wave_preset():
    """波次假设对照：并发固定 8，只改请求数（1/2/4/8 波）。

    如果"波次越多 TPOT 越高"，这里会看到单调上升；
    如果 TPOT 与波次无关，那上一轮的台阶就是真·并发效应。
    """
    cells = all_cells(concs=[8], ctxs=[512], specs=[("3B", "q4_k_m")])
    out = []
    for n_wave in (1, 2, 4, 8):
        for c in cells:
            c = dict(c)
            c["requests"] = 8 * n_wave
            c["suffix"] = f"_n{8 * n_wave}"
            out.append(c)
    return out


def _knee_preset():
    """算力拐点：0.5B 太小，访存 2.6 ms 而算力 4.7 ms（并发 64），计算先到。

    这是唯一能在 8 GB 单卡上真正进入"算力受限区"的机会，
    用来把 PEAK_decode 从反推变成实测。
    """
    concs = [1, 2, 4, 8, 16, 32, 48, 64]
    cells = all_cells(concs=concs, ctxs=[512], specs=[("0.5B", "q4_k_m")])
    return _with(cells, requests=lambda b: max(16, 4 * b), suffix="_w4")



def _xfamily_preset():
    """跨模型族验证：Llama-3.2-1B，覆盖并发与上下文两个维度。

    **不参与任何重新拟合**：直接用 Qwen 上标定的 6 个参数预测。
    """
    cells = all_cells(concs=[1, 4, 16, 32], ctxs=[512], specs=[("Llama1B", "q4_k_m")])
    cells += all_cells(concs=[1, 4], ctxs=[2048, 8192], specs=[("Llama1B", "q4_k_m")])
    return _with(cells, requests=lambda b: max(16, 4 * b), suffix="_xf")


def preset_cells(name: str):
    """返回某个预设下的格列表。

    core    已完成的 45 格（不要重跑）
    dense   核心 5 组合 × ctx{512,2048} × 并发{2,8,12,24}      —— 让后悔值有区分度
    high    核心 5 组合 × ctx512 × 并发{48,64}                 —— 逼近物理并发上限
    ctx     核心 5 组合 × ctx{1024,4096} × 并发{1,4,16}        —— 检验 KV 项
    quant   1.5B/3B × {q2_k,q3_k_m} × ctx{512,2048} × 并发{1,4,16,32} —— 质量-成本曲线陡峭段
    """
    core = CORE_SPECS
    lowq = [k for k in MODEL_SPECS if k[1] in ("q2_k", "q3_k_m")]
    if name == "xfamily":
        return [c for c in _xfamily_preset() if c["feasible"]]
    if name == "step":
        return [c for c in _step_preset() if c["feasible"]]
    if name == "wave":
        return [c for c in _wave_preset() if c["feasible"]]
    if name == "knee":
        return [c for c in _knee_preset() if c["feasible"]]
    if name == "core":
        return [c for c in all_cells(specs=CORE_SPECS) if c["feasible"]]
    if name == "dense":
        cells = all_cells(concs=[2, 8, 12, 24], ctxs=[512, 2048], specs=core)
    elif name == "high":
        cells = all_cells(concs=[48, 64], ctxs=[512], specs=core)
    elif name == "ctx":
        cells = all_cells(concs=[1, 4, 16], ctxs=CTX_EXT, specs=core)
    elif name == "quant":
        if not lowq:
            print("[提示] 还没下载 q2_k / q3_k_m，quant 预设为空")
        cells = all_cells(concs=[1, 4, 16, 32], ctxs=[512, 2048], specs=lowq)
    elif name == "all":
        cells = preset_cells("dense") + preset_cells("high") + preset_cells("ctx") + preset_cells("quant")
    else:
        raise SystemExit(f"未知预设：{name}")
    return [c for c in cells if c["feasible"]]
