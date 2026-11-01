# TokenCost

**在单张 8 GB 消费级 GPU 上，把"大模型推理有多贵"写成一个可被实测证伪的解析模型。**

给定（模型规模, 量化档位, 上下文上限, 并发数），预测单位 token 成本与延迟，
并给出满足 SLA 与质量底线的最低成本配置。

| 指标 | 结果 |
|---|---|
| 留出集（68 个未见配置，core 口径）单位成本 MAPE | **8.70%**（稳态）/ 11.28%（墙钟） |
| 其中**标定过的并发**（33 格） | **5.8%** |
| 单点最大误差 | 24.2% |
| 90% 预测区间实测覆盖率 | 86.8% |
| 跨模型族（Llama-3.2-1B，零重标定） | **12.7%** |
| 决策后悔值（16 场景 × 5 质量档位，bootstrap 区间） | 中位 0.0% / 最大 3.9%，**区间宽度中位 1.1 pp** |
| 质量—成本权衡 | 0–6pp 损失→20847 · 8pp→4919 · ≥10pp→2182（每 1 点质量贵约 27%） |

数据规模：**108 个配置格 / 324 次会话级测量**，全部在本机实测（RTX 4060 Laptop 8 GB）。

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
| `BW` | 249.1 GB/s | 有效访存带宽（标称 272 的 92%） |
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
| 3B-Q8 | 84.2% | 80.4% |
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
| 同机器、同部署栈、**标定过的并发**（1/4/16/32） | ✅ | 5.8% |
| 同机器、**未标定的中间并发**（2/8/12/24） | ⚠️ 必须实测 | 11.5% |
| **换模型族**（Qwen → Llama，零重标定） | ✅ | 12.7% |
| **换硬件**（GPU → CPU，重标定 3 个硬件常数） | ⚠️ 方向对 | 27% |
| **真实到达过程**（泊松到达，含排队） | ❌ 不能用于 TTFT | TPOT 准，TTFT 差 2–8 倍 |
| KV 装不进显存（触发 `--kv-offload`） | ❌ 另一套物理 | 吞吐掉 2.6–4.3 倍且不报错 |

**注意最后一条**：显存边界不是硬墙，是**性能悬崖**。
KV 放不下时 llama.cpp 会静默把 KV 搬到主机内存，不报错、只是慢 3–4 倍。
KV 量化（`--cache-type-k/v q8_0`）能救回一部分：1.5B 上吞吐 +238%，
但当权重本身占满显存时（7B）无效。

---

## 复现

```powershell
# 环境
python code/tc_env_check.py        # 依赖/显存/llama.cpp 自检
python code/tc_preflight.py        # 跑一格真实测量，确认机器满频（关键！）

# 采集（约 90 分钟，100 格 × 3 轮）
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
code/          28 个脚本（采集 / 分析 / 拟合 / 推荐 / 探针 / 对账）
data/          原始数据（raw_cells.jsonl 300 行、raw_requests.jsonl 逐请求明细）
results/       fit.json（6 参数）、predictions.csv、recommendation.csv
figures/       fig1–fig5（吞吐、成本、带宽、预测vs实测、Pareto）
报告/          技术报告定稿、迁移一页纸、简历条目与答辩稿、交接清单
手册/          从零复现的 10 篇教程（含每一步的"为什么"）
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

不是 5.8%，是**它被改过多次的过程**——每一次都是数据推翻了假设：

| 版本 / 发现 | 假设 | 结果 |
|---|---|---|
| v1 | 3 参数够（固定开销、TTFT 不随并发变） | MAPE **33.5%**，并发 32 低估到 0.39 |
| v2 | 加 prefill 并发项 | TTFT 仍系统性偏差 |
| v3 | 加 `c_srv·b` 与 `B_SAT` | MAPE **6.1%** |
| v4 | 拆开 decode/prefill 算力 | 最大误差 19.5% → **15.3%** |
| 发现 1 | 显存边界是硬墙 | 其实是**性能悬崖**：14 个"不可行格"能跑，7 个静默降级 3–4 倍 |
| 发现 2 | 并发对 TPOT 线性 | 其实是**陡—平—陡**，划出 5.8% / 11.5% 的适用边界 |
| 发现 3 | 瓶颈在 HTTP/推流 | 其实是**decode 循环内部**，交付开销 ≈ 0 |
| 发现 4 | 同步波次是最坏情况 | 其实是最**乐观**情形（无排队），泊松下 TTFT 差 2–8 倍 |

完整审计记录见 `../审计记录_项目可行性复核_20261019.md`（十一轮，含每次被推翻的推理）。


---

## 口径与可辨识性（引用数字前必读）

**三套留出集口径**（同一组参数，只改留出范围）：

| 口径 | 留出格数 | 成本 MAPE | 最大误差 | 覆盖率 |
|---|---|---|---|---|
| `core`（默认，设计空间 45+35 格） | 68 | **8.70%** | 24.19% | 86.8% |
| `valid`（+8 个跨模型族格） | 76 | 9.08% | 24.19% | 85.5% |
| `all`（含台阶/波次/拐点探针） | 96 | 10.21% | 26.64% | 76.0% |

切换方式：`python code/tc_model.py --population core|valid|all`。
**引用时必须带口径名**，否则就是在制造漂移。

**两种成本口径**：
- **稳态**（模型自身定义）= 1e6 ÷ (并发 × 生成长度 ÷ 单请求生命周期)，**8.70%**；
- **墙钟**（端到端真实吞吐）= 1e6 ÷ 实测总吞吐，**11.28%**（最大误差 35.1%）。
  两者差 2.6 个百分点，来源是收尾空闲与波次效应。**对外应同时报这两个数。**

**目标 8% / 20% 只在 `core` 口径的部分子集上满足**，`fit.json` 的 `passed` 字段是 `false`。
不要写成"全部达标"。

**6 个参数里有 2 个是弱约束**（Hessian 条件数 2.8×10⁴）：
`PEAK_dec` ±15% 只让 loss 变 1.04–1.08 倍，`ovh_ttft` ±30% 只变 1.14 倍。
所以 13.2 TFLOP/s 与 15.96 ms 是**拟合值不是实测值**，正式场合应给区间。

**质量轴在统计上是空的**：五组 95% 置信区间全部重叠；
而推荐器用的硬阈值 **6pp → 7pp 会让推荐从 3B-Q4 翻成 1.5B-Q4（成本差 36%）**。
所以"质量—成本权衡"应报敏感性，不应作为硬结论。

> 以上数字由 `python code/tc_claims.py` 生成 `results/CLAIMS.md`；
> `python code/tc_claims.py --check` 会在文档数字漂移时**非零退出**。



