# TokenCost

> **GitHub** <https://github.com/pityobject/tokencost> ｜ **Gitee** <https://gitee.com/pityobject/tokencost><br>
> 全部数据与脚本可复现；对外数字由 `code/tc_claims.py` 统一生成，并在 CI 里自动校验（漂移即失败）。

**在单张 8 GB 消费级 GPU 上，把"大模型推理有多贵"写成一个可被实测证伪的解析模型。**

给定（模型规模, 量化档位, 上下文上限, 并发数），预测单位 token 成本与延迟，
并给出满足 SLA 与质量底线的最低成本配置。

| 指标 | 结果 |
|---|---|
| 留出集（68 个未见配置，core 口径）单位成本 MAPE | **8.58%**（稳态）/ 11.28%（墙钟） |
| 其中**标定过的并发**（33 格） | **5.5%** |
| 单点最大误差 | 24.2% |
| 90% 预测区间实测覆盖率 | 86.8% |
| 跨模型族（Llama-3.2-1B，零重标定） | **11.6%**（成本 MAPE） |
| 决策后悔值（16 场景 × 5 档位） | 模型 **0.0%** · 随机选 **274%** · 选最大模型 **740%**（负对照证明指标有分辨力） |
| 质量—成本权衡 | 0–6pp 损失→20847 · 8pp→4919 · ≥10pp→2182（每 1 点质量贵约 27%） |

数据规模：**118 个配置格 / 388 次会话级测量**（验证集口径）；全仓共 **141 格 / 417 次**（含 23 个长度/并发探针，各 1 轮），全部在本机实测（RTX 4060 Laptop 8 GB）。

---

## 一句话结论

> decode 是**访存受限**的（有效带宽 249 GB/s，标称的 92%），
> 但并发一高，瓶颈就**从显存带宽转移到 llama.cpp 解码循环内每步的固定开销**——
> 并发 ≥16 时 TPOT 的 58%、最高 77% 不是访存。
> 实测 HTTP/SSE 交付开销 ≈ 0，所以换传输层救不了，要换的是 decode 调度实现。

---

## 模型

```
TPOT(b) = max( (W + KV·L·b) / BW ,  2·P·b / PEAK_dec ) + c_srv · b
TTFT(L) = max( W / BW ,  2·P·L·min(b, B_SAT) / PEAK_pre ) + c_srv · b + ovh
throughput = b · GEN / (TTFT + GEN · TPOT)      cost = 1e6 / throughput   [GPU·s / 1M tokens]
```

| 参数 | 值 | 含义 |
|---|---|---|
| `BW` | 249.1 GB/s | **有效**访存带宽（标称 272 的 92%；含反量化与内核开销，非纯硬件常数） |
| `PEAK_dec` | 13.2 TFLOP/s ⚠️**弱约束** | decode 有效算力（瘦长矩阵乘，张量核喂不满） |
| `PEAK_pre` | 31.9 TFLOP/s | prefill 有效算力（大 GEMM） |
| `c_srv` | 0.766 ms / 并发流 / 步 | 服务端每步开销 |
| `B_SAT` | 13.3 | prefill 并行饱和点 |
| `ovh` | 15.96 ms ⚠️**弱约束** | 首字固定开销 |

**为什么 decode 和 prefill 要分开两个算力**：decode 是 batch×1 的瘦长矩阵乘，
prefill 是一次大 GEMM。用后端基准反推，两者差 2.4 倍（9 与 31.6 TFLOP/s）。
合成一个参数时最大误差 19.5%，拆开后降到 15.3%。

---


### 质量轴（400 题 × 10 组合 × 配对检验）

| 组合 | 准确率 | Wilson 下界 |
|---|---|---|
| **7B-Q4** | **91.0%** | 87.8% |
| 3B-Q8 | 86.8% | 80.4% |
| 1.5B-Q4 / 3B-Q4 | 82.2% | 78.2% |
| 1.5B-Q8 | 80.2% | 76.1% |
| 3B-Q3_K_M | 80.8% | 76.6% |
| 1.5B-Q3_K_M | 61.5% | 56.6% |
| 0.5B-Q4 | 57.2% | 52.4% |
| 3B-Q2_K | 53.0% | 48.1% |
| 1.5B-Q2_K | 30.2% | 26.0% |

**配对 McNemar（Holm 校正）：45 对里 32 对显著。**

三条可直接用的结论：
1. **Q2 是灾难**（1.5B-Q2 只有 30.2%），**Q4 就够**（Q4 与 Q8 无显著差异，p=1.00，而 Q8 权重多 1.7 倍）。
2. **7B 是唯一显著更强的档位**，但成本约 5 倍。
3. `reverse`（单词倒序）题型**所有模型 0%**，无区分度，是本基准的已知缺陷。

### 质量—成本权衡（替代"推荐一个配置"）

| 可接受的质量损失 | 最便宜配置 | 成本 | 实际质量 |
|---|---|---|---|
| 0–6 pp | 7B-Q4 / ctx2048 / c1 | **20847** | 87.8% |
| 8 pp | 3B-Q8 / ctx2048 / c4 | **4919** | 80.4% |
| ≥10 pp | 1.5B-Q4 / ctx2048 / c4 | **2182** | 78.2% |

**成本差 9.6 倍换 9.6 个百分点质量 ≈ 每 1 个百分点的质量贵 27%。**

而且：**在 8 GB 卡上，"7B 质量 + 300 ms 首字"这个组合不存在**
（7B 最便宜的合规配置预测 TTFT 372 ms）。质量与延迟之间是硬取舍，不是调参能绕过的。

---


---

## 快速开始

```powershell
# 0) 依赖
pip install numpy pandas scipy matplotlib httpx psutil

# 1) 告诉程序 llama.cpp 和模型在哪（换机器只需改这两个环境变量，不用改代码）
$env:TOKENCOST_LLAMA_BIN = "D:\llm\bin"        # 含 llama-server.exe 的目录
$env:TOKENCOST_MODEL_DIR = "D:\llm\models"     # 含 *.gguf 的目录

# 2) 自检（不通过就不要往下跑）
python code/tc.py check
python code/tc.py preflight        # 跑一格真实测量，确认机器满频

# 3) 采集与分析
python code/tc.py run              # 核心 45 格（约 50 分钟）
python code/tc.py analyze          # 汇总 + 5 张图
python code/tc.py steady           # 稳态口径重算
python code/tc.py model            # 六参数拟合 + 留出验证
python code/tc.py recommend        # 配置推荐
python code/tc.py sweep            # 质量—成本权衡表
python code/tc.py verify           # 决策后悔值（16 场景）

# 4) 一致性检查（改文档数字后必跑）
python code/tc.py claims --check
```

**没有 GPU 也想跑通全链路？** 先 `python code/_make_fake_data.py` 生成合成数据
（它会拒绝覆盖真实数据），再跑 `analyze → model`，可验证代码逻辑。

**测试**（零依赖，不需要装 pytest）：

```powershell
python tests/test_core.py
```

覆盖三件事：`predict` 的黄金值、`aggregate` 的分位数口径、
以及 `fit` 的**恢复实验**（用已知参数生成合成数据再拟合回去）。

## 这个模型能信到哪（适用边界）

这是本仓库最重要的表——**知道哪里能信，比多一个漂亮数字有用**。

| 情形 | 能用吗 | 实测精度 |
|---|---|---|
| 同机器、同部署栈、**标定过的并发**（1/4/16/32） | ✅ | 5.5% |
| 同机器、**未标定的中间并发**（2/8/12/24） | ⚠️ 必须实测 | 11.4% |
| **换模型族**（Qwen → Llama，零重标定） | ✅ | 11.6%（成本 MAPE） |
| **换硬件**（GPU → CPU，重标定 3 个硬件常数） | ⚠️ 方向对 | 27% |
| **真实到达过程**（泊松到达） | ⚠️ 容量可给上界；排队延迟只到量级 | 天花板 2.47 vs 预测 3.43（**72%**，计入干扰后 **88%**）；排队层 1.2–1.6 倍偏保守；TPOT +27% |
| KV 装不进显存（触发 `--kv-offload`） | ❌ 另一套物理 | 吞吐掉 2.6–4.3 倍且不报错 |

**注意最后一条**：显存边界不是硬墙，是**性能悬崖**。
KV 放不下时 llama.cpp 会静默把 KV 搬到主机内存，不报错、只是慢 3–4 倍。
KV 量化（`--cache-type-k/v q8_0`）能救回一部分：1.5B 上吞吐 +238%，
但当权重本身占满显存时（7B）无效。

---

> English README: [README.en.md](README.en.md)

> **这个项目为什么可信**：22 次被数据推翻的记录、以及它们沉淀成的机制，见 `报告/方法论_测量纪律.md`。
> **每个数据文件的列含义**见 `报告/数据字典.md`（自动生成、CI 校验，18 个数据文件全覆盖）。

## 打开一个网页就能用

`site/index.html` 是一个**单文件、离线可用**的配置计算器：输入 SLA 与 GPU 价格，
直接告诉你哪个配置最省钱。

**三种打开方式**

```powershell
# 1) 直接双击（Windows）
start site\index.html

# 2) 起一个本地服务（端口随你）
python -m http.server 8000 --directory site

# 3) GitHub Pages（见 .github/workflows/pages.yml）
#    仓库 Settings → Pages → Source 选 "GitHub Actions"，
#    之后每次 push 到 main 都会自动发布 site/ 目录
```

页面里**没有任何物理公式**——所有数字由 `python code/tc.py web` 从仓库产物生成，
并且页面加载时会用 Python 预算的 4 个用例**自检筛选结果**，不一致就在页脚显示红字。
CI 第 7 步还会重新生成并比对，保证网页与模型不漂移。

## 适用域：先看这张表

| 工况 | 状态 | 证据 |
|---|---|---|
| 同步波次 + 设计空间内并发 | ✅ 可信 | 留出 68 格成本 MAPE **8.58%** |
| 短 prompt（≤2000）+ 低并发（1–4） | ✅ 可信 | TTFT MAPE **6.9%**（跨 11 倍长度） |
| 换模型族（Llama，零重标定） | ✅ 可用 | 成本 MAPE **11.6%** |
| 真实泊松到达 | ⚠️ 只能给上界 | 实测/预测 **72%**（计入干扰 88%） |
| 中间并发（8/12） | ⚠️ TPOT 偏低 20–25% | c8 实测 19.4→43.3 vs 模型 14.9→17.0 |
| **长 prompt（>1000）+ 高并发（≥8）** | ❌ **不可用** | TTFT 误差最高 **224%** |
| 换硬件 / 换后端 / 多卡 | ❌ 未测 | 需按《自标定指南》重标定 |

完整矩阵（含按输出分级的可信度）见 `报告/适用域矩阵_一页纸.md`。

## 常见误用（照着改就行）

| 症状 | 原因 | 正确做法 |
|---|---|---|
| `can't open file ...\code\code\tc_recommend.py` | 在 `code/` 里又加了 `code/` 前缀 | 在 `code/` 里用 `python tc_recommend.py`；在根目录用 `python code/tc.py recommend` |
| `ModuleNotFoundError: pandas` | 用了系统 Python | 用 `D:\llm\venv\Scripts\python.exe` |
| `端口 8090 被占用` | 之前手工起的 llama-server 还在 | `python code/tc.py stop --force` |

## 四项补充分析（2026-11）

| 分析 | 结论 | 产物 |
|---|---|---|
| **重复性** | **同批次** TPOT CV **0.23%**（连续重复）；**跨批次**中位 0.50%、最大 **6.36%**（c16，分两批跑）→ 聚合 MAPE 的标准误 **±0.02 pp（同批）/ ±0.6 pp（跨批上界）** | `results/variance_summary.json` |
| **延迟尾部** | 实测 `p90/p50` 从并发 1 的 **1.01** 升到并发 64 的 **3.75**；而模型用的尾部假设恒定 **1.08×** → **并发 ≥10 时 SLA 判定偏乐观 1.3–3.5 倍** | `figures/fig11_tail_ratio.png`、`results/quantile_tails.csv` |
| **按量化重拟合** | `c_srv` 在 Q4/Q8 上几乎相同（**0.680 vs 0.697**，差 2%）→ 每步开销确实对精度不敏感；但 `BW` 差 **1.22×**（214 vs 261）→ 见下方"BW 的口径" | `results/params_by_quant.csv` |
| **参数可辨识性** | profile likelihood 给出 **4 个可辨识 + 2 个弱约束**（`peak_tflops`、`ovh`），与第一轮复核的 Hessian 结论一致 | `figures/fig12_profile_likelihood.png` |

> **BW 的口径（重要）**：249.1 GB/s 是**有效带宽**，它吸收了反量化、内核与调度开销，
> **不是纯硬件常数**。同一张卡按量化档分别重拟合会得到 214（Q4）与 261（Q8）——
> 差 22%。所以正确用法是：**在同一部署栈内比较用合并值**；
> 一旦换量化/换后端，BW 与 `c_srv` 都要重新标定（见《自标定指南》）。

> **环境标注**：每格记录背景指纹（`bg_*`）与时钟相对值（`clock_ratio` / `clock_gate`）。
> 133 格里 111 格 `ok`、16 格 `warn`、6 格 `bad`（7B 的 c4/c8 撞功耗墙，约影响 4%）；
> 实测**时钟对 TPOT 的弹性仅 0.29**，所以低时钟只标注、不作废。

> **数据口径（三个数都对，含义不同）**：原始 **370 行**（含 8 行端口占用导致的失败，如实保留）
> → 去重 **356 行 / 132 格**（同一格多次运行时 `repeat` 会重编号，同格同轮取最后一次）
> → 对外口径 **118 格 / 388 次**（再剔除 23 个长度/并发探针）。

> ### 运行前先确认两件事
>
> **① 目录**：下面的命令都在**仓库根目录**执行（`cd tokencost`）。
> 如果你已经在 `code/` 目录里，就直接写脚本名；再加一层 `code/` 前缀会解析到不存在的路径。
>
> **② 解释器**：必须用装了依赖的 venv，不要用系统 Python。
> ```powershell
> $py = "D:\llm\venv\Scripts\python.exe"   # 本仓库唯一支持的解释器
> ```

## 在你自己的机器上用（自标定）

```bash
python code/tc_profile.py --base-url http://127.0.0.1:8080/v1 --spec 3B:q4_k_m
#   ↑ 一条命令：自动发 8 组请求 → 测 TTFT/TPOT → 拟合 → 出标定文件

# 或手工准备测量点（格式见报告/自标定指南）：
python code/tc_calibrate.py my_measurements.json        # 8 个实测点 → 你的 6 个参数
python code/tc_recommend.py --calib results/calib_mymachine.json --ttft 600 --tpot 60
```

不需要 RTX 4060、不需要复现我的实验——量 8 个配置（并发 1/4/16/32 + 两种提示长度）即可。
完整流程见 `报告/自标定指南_一页纸.md`（含"为什么随便 4 个点不行"的实测对比）。

## 体检（一条命令）

```bash
python code/tc_audit.py          # 全量：卫生 / 自包含(裸克隆) / 数据链 / 契约
python code/tc_audit.py --fast   # 跳过裸克隆
```

它会模拟"别人 clone 后只有已跟踪文件"的环境，验证 `claims --check` 与单元测试仍能通过。
当前状态：**14/14 项通过**。

## 复现

```powershell
# 环境
python code/tc_env_check.py        # 依赖/显存/llama.cpp 自检
python code/tc_preflight.py        # 跑一格真实测量，确认机器满频（关键！）

# 采集（约 90 分钟，108 格 × 3 轮）
python code/tc_run_grid.py                 # 核心 45 格
python code/tc_run_grid.py --preset dense  # 35 格（并发加密）
python code/tc_run_grid.py --preset step   # 探针：定位非线性位置
python code/tc_run_grid.py --preset wave   # 探针：波次对照
python code/tc_run_grid.py --preset knee   # 探针：0.5B 算力拐点
python code/tc_run_grid.py --preset xfamily# 探针：跨模型族

# 分析
python code/tc_analyze.py          # 汇总 + 5 张图
python code/tc_recompute_steady.py # 稳态口径重算（消除波次污染）
python code/tc_model.py            # 6 参数拟合 + 留出验证
python code/tc_recommend.py --ttft 600 --tpot 60
python code/tc_recommend.py --ttft 600 --tpot 60 --gpu-price a10
#   ↑ 加上价格就给出「元/百万 token」：本地电费口径 0.35 元，云端 A10 按量 60.92 元
#     同一份 GPU·秒，两个价格差 175 倍——所以「多少钱」必须和「什么价」一起说
python code/tc_verify.py           # 决策后悔值
python code/tc_decompose_csrv.py   # c_srv 三向分解
```

**测量纪律**（这几条决定数据能不能用）：
- 每格 3 轮取中位数；每轮注入随机前缀并关 `cache_prompt`（否则 prompt cache 会把 TTFT 打成 0）
- `ignore_eos: true` 强制生成 128 token（否则各格长度不同，TPOT 不可横向比较）
- 每格记录温度/频率/功耗；**时钟与功耗双低判为降频，整格作废**
  （笔记本没插电时会掉到 1/7 频率且不报错——本项目中招过一次，44 格数据报废）
- `--ctx-size` 必须传 `ctx × 并发`（服务端按 `--parallel` 平分给槽位，传错会静默截断所有请求）

---

## 目录

```
code/          67 个脚本（采集 / 分析 / 拟合 / 推荐 / 探针 / 对账）
data/          原始数据（raw_cells.jsonl 300 行、raw_requests.jsonl 逐请求明细）
results/       fit.json（6 参数）、predictions.csv、recommendation.csv
figures/       fig1–fig5（吞吐、成本、带宽、预测vs实测、Pareto）
报告/          技术报告定稿、迁移一页纸、简历条目与答辩稿、交接清单
手册/          从零复现的 10 篇教程（含每一步的"为什么"）；写于 45 格时期，数字以 results/CLAIMS.md 为准
notes/         学习笔记与踩坑记录
```

---

## 已知局限（主动列出）

1. **单机单栈**：数据来自一张 4060 + llama.cpp；跨硬件只做了 CPU 一项（27%）。
2. **模型只算服务时间，不算排队时间**：真实负载下 TTFT 必须另加排队延迟。
3. **算力拐点在 8 GB 上不可达**：并发 64 时 `c_srv×b = 49 ms` 是算力时间（4.75 ms）的 10 倍，
   计算永远被服务开销淹没。
4. **预测区间偏窄**：覆盖率 86.8%，略低于 82–96% 目标区间中位。

---

## 这个项目最值钱的部分

不是 5.5%，是**它被改过多次的过程**——每一次都是数据推翻了假设：

| 版本 / 发现 | 假设 | 结果 |
|---|---|---|
| v1 | 3 参数够（固定开销、TTFT 不随并发变） | MAPE **33.5%**，并发 32 低估到 0.39 |
| v2 | 加 prefill 并发项 | TTFT 仍系统性偏差 |
| v3 | 加 `c_srv·b` 与 `B_SAT` | MAPE **6.1%** |
| v4 | 拆开 decode/prefill 算力 | 最大误差 19.5% → **15.3%** |
| 发现 1 | 显存边界是硬墙 | 其实是**性能悬崖**：14 个"不可行格"能跑，7 个静默降级 3–4 倍 |
| 发现 2 | 并发对 TPOT 线性 | 其实是**陡—平—陡**，划出 5.5% / 11.4% 的适用边界 |
| 发现 3 | 瓶颈在 HTTP/推流 | 其实是**decode 循环内部**，交付开销 ≈ 0 |
| 发现 4 | 同步波次是最坏情况 | 其实是最**乐观**情形（无排队）；λ=2 时 p90 达基线 **13.0 倍**，λ=1.5 就到 7.1 倍 |
| 发现 5 | 泊松探针"生成器饱和"、模型不能做容量规划 | 都是误判：那是服务器饱和；模型能给容量**上界**（实测/预测 73%） |
| 发现 6 | 泊松与同步波次 TPOT 相同 | 泊松下 TPOT 高 **25%**（prefill 打断 decode） |

完整审计记录见 `报告/审计记录_十五轮对抗性复核.md`（十五轮，含每次被推翻的推理）。


---

## 口径与可辨识性（引用数字前必读）

**三套留出集口径**（同一组参数，只改留出范围）：

| 口径 | 留出格数 | 成本 MAPE | 最大误差 | 覆盖率 |
|---|---|---|---|---|
| `core`（默认，设计空间 45+35 格） | 68 | **8.58%** | 24.11% | 86.8% |
| `valid`（+8 个跨模型族格） | 84 | 9.22% | 24.11% | 83.3% |
| `all`（含台阶/波次/拐点探针） | 124 | 9.80% | 26.57% | 73.4% |
| `ext`（多轮/混合长度/高并发三族扩展） | 8 | 11.62% | 23.42% | 62.5% |

切换方式：`python code/tc_model.py --population core|valid|all|ext`。
**引用时必须带口径名**，否则就是在制造漂移。

**两种成本口径**：
- **稳态**（模型自身定义）= 1e6 ÷ (并发 × 生成长度 ÷ 单请求生命周期)，**8.58%**；
- **墙钟**（端到端真实吞吐）= 1e6 ÷ 实测总吞吐，**11.28%**（最大误差 35.1%）。
  两者差 2.6 个百分点，来源是收尾空闲与波次效应。**对外应同时报这两个数。**

**目标 8% / 20% 只在 `core` 口径的部分子集上满足**，`fit.json` 的 `passed` 字段是 `false`。
不要写成"全部达标"。

**6 个参数里有 2 个是弱约束**（Hessian 条件数 2.8×10⁴）：
`PEAK_dec` ±15% 只让 loss 变 1.04–1.08 倍，`ovh_ttft` ±30% 只变 1.14 倍。
所以 13.2 TFLOP/s 与 15.96 ms 是**拟合值不是实测值**，正式场合应给区间。

**质量轴在统计上是空的**：五组 95% 置信区间全部重叠；
而推荐器用的硬阈值会把结论整档翻过去：**7.5 pp 时从 7B 翻到 3B-Q8（20847→4919），10 pp 时从 3B 翻到 1.5B-Q4（4919→2182）**。
所以"质量—成本权衡"应报敏感性，不应作为硬结论。

> 以上数字由 `python code/tc_claims.py` 生成 `results/CLAIMS.md`；
> `python code/tc_claims.py --check` 会在文档数字漂移时**非零退出**。

---

## 第三方复核

`报告/第三方复核意见_20261101.md` 是一份独立通读代码 + 复算全部对外数字的审查意见，它的结论是「内核是真的，外壳（对外数字）是虚的」。
本仓库的 P0 修复（三套口径、`tc_claims.py` 防漂移、Pareto 的 NaN 逻辑）全部来自那份意见，修复过程记在 `报告/审计记录_十五轮对抗性复核.md` 的第 12–15 轮。





































