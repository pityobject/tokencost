# C 档扩展计划（冻结版）

> 制定：2026-10-31 ｜ 目标：把报告里自己写的 5 条局限逐条变成实验，并新增能效、迁移性、排队三根轴
> **纪律：核心 12 标定格 + 45 格结论冻结不动；所有新增格一律进留出集。**
> 因此扩展后的结论仍是"模型没见过的新配置"，说服力更强而不是更弱。

---

## 为什么要这么做（一句话）

现在的报告有 5 条自己承认的局限。**每一条都是一个可以做的实验**——
把"局限"改写成"我测过"，比把局限描述得更漂亮有价值得多。

| 报告里的局限 | 本计划的哪一阶段解决 |
|---|---|
| `PEAK` 未进入算力受限区 | P1-2 后端饱和扫描 |
| 15 个不可行格只是估算 | P1-1 真跑一遍拿 OOM |
| 后悔值 0%（候选太稀） | P2-1 密集并发网格 |
| 负载画像只有同步波次 | P4-1 泊松到达 + 排队验证 |
| 质量基准 140 题、任务单一 | P4-2 扩到 300+ 题 + 中文/JSON/长上下文 |

新增轴：**能效（P1-3，已可用）**、**迁移性（P3-1）**、**量化档位 Q2/Q3（P2-3）**。

---

## P1：今晚，约 40 分钟机时

### P1-1 真实的 OOM 证据（15 分钟）

```powershell
cd C:\Users\王钟垚\Desktop\find_career\tokencost\code
& "D:\llm\venv\Scripts\python.exe" tc_run_grid.py --include-infeasible
```

- 会依次尝试 15 个"按估计不可行"的格，预期全部失败。
- **失败本身就是数据**：`raw_cells.jsonl` 里会出现 `status=failed` 与 CUDA 的原始报错，
  报告里的"15 格不可行"就从估算变成实测。
- 预计 10–20 分钟（7B 的格加载模型较慢）。

**跑完检查**：

```powershell
(Get-Content ..\data\raw_cells.jsonl | Select-String '"status": "failed"').Count
```

期望 ≥ 10。

### P1-2 后端饱和扫描（验证 PEAK，5 分钟）

为什么不在 server 上测：并发 64 时 `c_srv × 64 ≈ 54 ms`，服务开销会**淹没**算力拐点，
从服务路径永远看不到 compute-bound。必须绕开 HTTP。

```powershell
& "D:\llm\venv\Scripts\python.exe" tc_backend_ref.py --models 1.5B,3B,7B --npl 32,48,64
```

预期看到吞吐在某个并发上**停止线性增长**，饱和值 ≈ `PEAK / (2·P)`：

| 模型 | P | 预测饱和吞吐 |
|---|---|---|
| 1.5B | 1.54e9 | ≈ 10,300 t/s |
| 3B | 3.09e9 | ≈ 5,100 t/s |
| 7B | 7.62e9 | ≈ 2,100 t/s |

如果实测饱和值与之接近 → **PEAK 首次被直接验证**，报告里那条"外推"的局限可以删掉；
如果差很多 → 说明 `PEAK` 应改名为"DECODE 有效算力"，并说明与 prefill 的有效算力不同。

### P1-3 能效轴（10 分钟，零新实验）

代码已就绪（`tc_analyze.py` 会算 `energy_j_per_1m`）。跑：

```powershell
& "D:\llm\venv\Scripts\python.exe" tc_analyze.py
```

已可见的初步结果：

| 配置 | 能效（MJ / 百万 token） |
|---|---|
| 1.5B-Q4 / 并发 32 | **0.10** |
| 7B-Q4 / 并发 1 | **1.9** |

**19 倍差距**——这是"小模型 + 高并发"的第二个理由，和成本结论互相独立。

> 口径说明（报告必须写）：用"中位功耗 × 测量窗口 ÷ token 数"估算，未对功率积分，属粗估。

---

## P2：1 小时机时，约 98 个新格

先下载两个低比特量化（约 3.4 GB）：

```powershell
$MS = "D:\llm\venv\Scripts\modelscope.exe"
& $MS download --model Qwen/Qwen2.5-1.5B-Instruct-GGUF --include "qwen2.5-1.5b-instruct-q2_k.gguf" --local_dir D:\llm\models
& $MS download --model Qwen/Qwen2.5-1.5B-Instruct-GGUF --include "qwen2.5-1.5b-instruct-q3_k_m.gguf" --local_dir D:\llm\models
& $MS download --model Qwen/Qwen2.5-3B-Instruct-GGUF --include "qwen2.5-3b-instruct-q2_k.gguf" --local_dir D:\llm\models
& $MS download --model Qwen/Qwen2.5-3B-Instruct-GGUF --include "qwen2.5-3b-instruct-q3_k_m.gguf" --local_dir D:\llm\models
```

然后逐档跑（**一次一档，跑完立刻看数据是否正常**）：

| 命令 | 格数 | 目的 |
|---|---|---|
| `tc_run_grid.py --preset dense` | 35 | 并发 2/8/12/24，让后悔值有区分度 |
| `tc_run_grid.py --preset high` | 10 | 并发 48/64，逼近物理上限 |
| `tc_run_grid.py --preset ctx` | 25 | 上下文 1024/4096，检验 KV 项 |
| `tc_run_grid.py --preset quant` | 28 | Q2/Q3，质量-成本曲线的陡峭段 |

预计每格 40–60 秒，合计约 80 分钟。

---

## P3：迁移性验证（2–3 小时，含写代码）

**目标**：把"只需重标定两个常数"从**主张**变成**证据**。

做法：同一批模型在 **CPU 后端**（`-ngl 0`）上跑 6 格，用 CPU 的 BW/PEAK 重新标定，
再看留出格是否仍能预测。CPU 的带宽/算力与 GPU 差 10 倍以上，是天然的"另一台机器"。

- 需要改：`tc_server.start(..., ngl=0)` 参数、`tc_run_grid --preset cpu`
- 预期产物：`data/cpu_ref.csv` + 一张"GPU vs CPU 同一模型"的对照表
- **风险**：CPU 上 7B 极慢（可能 2–3 token/s），只跑 1.5B/3B

---

## P4：从"选配置"到"容量规划"（1–2 天）

### P4-1 泊松到达 + 排队验证

现在的负载是**同步波次**（b 个请求同时发），TTFT 属最坏情况。
改成**泊松到达、固定 QPS** 后，可以验证：

- Little 定律：`L = λW`（并发数 ≈ 到达率 × 平均逗留时间）
- 预测 P99 与实测 P99 的偏差
- **系统容量曲线的膝点**：QPS 超过多少后延迟爆炸

这是唯一能让项目从"课程项目"跨到"研究课题"的改动，也直接对应鹏城云脑的调度问题。

需要：`tc_loadgen` 增加到达过程模式 + 一个排队分析脚本。

### P4-2 质量基准扩到 300+ 题

新增三类，补上现有基准的盲区：

| 新题型 | 判分方式 | 为什么加 |
|---|---|---|
| 中文指令遵循（"用 JSON 输出三个字段"） | 解析 JSON + 字段校验 | 真实应用最常见的需求 |
| 长上下文检索（在 4k 文本里找一句话） | 子串匹配 | 检验"质量随上下文衰减" |
| 多步算术 | 精确匹配 | 区分"会算"与"会推理" |

同时把 `--n-per-kind` 提到 30，把区间压到 ±5pp 以内。

---

## 阶段验收与产出

| 阶段 | 新增结论 | 报告改动 |
|---|---|---|
| P1 | PEAK 是否被验证、真实 OOM 计数、能效 19 倍差距 | 删/改局限第 1 条，新增"能效"小节 |
| P2 | 密集网格下的后悔值、Q2/Q3 的质量悬崖、KV 项是否成立 | 图 5 重画，质量表扩到 7 个组合 |
| P3 | 迁移性的正面证据 | 一页纸从"主张"升级为"已验证" |
| P4 | 容量曲线与 P99 预测 | 新增"容量规划"小节 |

---

## 一条不可违反的纪律

**P2/P3/P4 的所有新格，标定集一个字都不许动。**
新增数据只用来做留出验证。理由：现在的 12 格标定是历史结论的基础，
一旦为了"拟合得更好"而调整，之前所有结论都要重算，而且失去了"预注册"的意义。
