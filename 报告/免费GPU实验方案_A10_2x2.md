# 免费 GPU 实验方案：A10 上的 2×2 对照

> 结论先说：**B1（换后端）和 B2（换硬件）可以完全用免费额度做完，不需要花钱租卡。**
> 依据：阿里云 PAI-DSW 免费试用提供 `ecs.gn7i-c8g1.2xlarge`（**1× NVIDIA A10，24 GB 显存**，8 vCPU，30 GiB 内存）。

---

## 一、要领的免费资源（按重要性）

| # | 产品 | 免费额度 | 折算可用时长 | 用途 |
|---|---|---|---|---|
| **①** | **PAI-DSW** | 750 计算时 / 3 个月（每月 250） | **A10 ≈ 35.8 小时/月**（3 个月约 107 小时） | **B1 + B2 主战场** |
| ② | PAI-EAS | 500 元抵扣金 / 1 个月 | A10 按 10.50 元/h ≈ **47 小时** | 备选（部署在线服务） |
| ③ | PAI-DLC | 100 计算时 / 3 个月 | A10 ≈ **14.4 小时** | 备选（跑批式实验） |
| ④ | 云工开物学生认证 | **300 元优惠券 / 12 个月** | 可买 ECS/OSS 等 | 托管静态站、对象存储 |
| ⑤ | 免费试用中心 | 160+ 产品，各自独立资格 | — | ECS / OSS / CDN |

**换算表**（官方）：`ecs.gn7i-c8g1.2xlarge` 消耗 **6.991 计算时/小时**；`gn6v`（V100）14.553；`g6`（纯 CPU）0.573。

> 入口：<https://free.aliyun.com/>，搜 "PAI"，**先领后用**——一旦创建过 DSW/DLC/EAS 实例就不能再领。
> 细则：<https://help.aliyun.com/zh/pai/product-overview/free-trial-guide>

### 必须注意的三件事（都写在官方文档里）

1. **实例"运行中"就在计费**，哪怕没跑代码。用完**手动停止**。
2. **停止 ≠ 不收费**：如果扩容过系统盘，停止后系统盘仍计费。只扩容了 100 GB 免费盘则没事。
3. **闲置 3 小时低利用率会自动关机**——长实验要在调度配置里加"排除策略"（按实例名），否则跑到一半被停。

---

## 二、为什么 A10 是理想的第二块硬件

| | RTX 4060 Laptop（现有） | **A10**（免费试用） |
|---|---|---|
| 显存 | 8 GB | **24 GB** |
| 显存带宽 | ~256 GB/s | **~600 GB/s** |
| 定位 | 消费级 | **数据中心卡** |
| 显存/带宽比 | — | 差 **2.3 倍** |

带宽差 2.3 倍、显存差 3 倍——**这是真正意义上的"换硬件"**，不是同一类卡的小改。

---

## 三、2×2 实验设计（这才是重点）

免费额度足够做一个**受控的二维对照**，这是把"现象"变成"规律"的关键：

|  | **llama.cpp** | **vLLM** |
|---|---|---|
| **RTX 4060 Laptop** | ✅ **已有**（108 格，6 参数） | ❌ 8 GB 跑不动 |
| **A10（免费）** | **① 只换硬件** | **② 只换后端** |

- **① A10 + llama.cpp**：与现有基准唯一差别是硬件 → 直接检验"换硬件只需重标定 BW/PEAK"
- **② A10 + vLLM**：与 ① 唯一差别是后端 → 直接检验"`c_srv` 是部署栈属性"

**两个对照各自只有一个变量。** 这是整个项目里第一次做真正的受控实验。

### 每组要测什么（复用 `tc_profile`，8 个点约 10 分钟）

```bash
# A10 上（DSW 终端里）
python code/tc_profile.py --base-url http://127.0.0.1:8080/v1 \
  --spec-json ~/qwen3b_a10.json --hardware "A10 24GB" \
  --out results/calib_a10_llamacpp.json
```

**每组 8 点**：并发 1/4/16/32 × L512，加 L128 与 L2048（定 prefill 斜率）、c8（弱区）。
四组合计约 40 分钟——**远低于每月 35.8 小时的额度**。

### 验收标准（这才是要写进报告的）

| 检验 | 期望 | 依据 |
|---|---|---|
| ① 的 `BW` 反推 vs A10 官方带宽 | 在 60–80% 量级（有效带宽） | 4060 上是 249 vs 标称 272（92%） |
| ① 的 **`c_srv` 与 4060 接近** | 差 <30% | 同一个 llama.cpp 构建，栈相同 |
| ② 的 **`c_srv` 与 ① 明显不同** | 差 >50% | vLLM 的调度/采样实现完全不同 |
| ② 的 **参数形式不变** | 6 参数仍能拟合，MAPE <15% | 这是"规律"的核心 |
| 跨硬件预测 | 用 4060 参数直接预测 A10，误差 ≈ 带宽比 | 检验"只重标定两个常数"主张 |

> **如果 ② 的 `c_srv` 明显更小，且 6 参数形式仍成立**，你就能写出这句话：
> **"每步服务开销属于服务进程（部署栈 + 主机 CPU），不是 GPU 或模型权重的属性——同一张 A10 上 vLLM 的 `c_srv` 是 llama.cpp 的 18%。换个后端，只要重测这一个参数。"**

---

## 四、具体操作（DSW 上）

### 4.0 一条命令跑完（推荐）

仓库里已有对齐好的脚本，**不需要手抄命令**：

```bash
cd ~ && git clone https://gitee.com/pityobject/tokencost.git
cd tokencost && bash scripts/a10_run.sh          # 全流程（约 70 分钟）
#   bash scripts/a10_run.sh llamacpp              # 只跑组 ①
#   bash scripts/a10_run.sh vllm                  # 只跑组 ②（新信息，额度紧就先跑这个）
```

脚本会依次：校验 GPU → 跑 `tc_profile --selftest`（链路闸门）→ 下模型 →
**分别生成 GGUF / HF 两份 spec** → 起服务 → **校验 `n_ctx_slot` 语义并按需重启** →
标定 → 打印两后端参数对照表。日志在 `logs/`，结果在 `results/`。

---


### 4.1 建实例

1. PAI 控制台 → 工作空间 → 交互式建模（DSW）→ 创建实例
2. **资源类型选"公共资源"** → 资源规格弹窗里选 **"免费试用"** 页签 → `ecs.gn7i-c8g1.2xlarge`
3. 镜像：选**预装 PyTorch 2.x + CUDA 12.x** 的官方镜像（省 30 分钟装环境）
4. 系统盘保持 **100 GB**（免费；扩容要另收费）

### 4.2 组 ①：A10 + llama.cpp

```bash
# 编译（8 vCPU，约 5–10 分钟）
git clone https://github.com/ggerganov/llama.cpp && cd llama.cpp
cmake -B build -DGGML_CUDA=ON && cmake --build build -j8

# 下载 Qwen 的 GGUF（国内用 ModelScope 更快）
pip install modelscope
modelscope download --model Qwen/Qwen2.5-3B-Instruct-GGUF \
  qwen2.5-3b-instruct-q4_k_m.gguf --local_dir ~/models

# 起服务（注意 -c 与 -np 的语义：本 build 是 n_ctx_slot = ctx）
~/llama.cpp/build/bin/llama-server -m ~/models/qwen2.5-3b-instruct-q4_k_m.gguf \
  # 对齐口径：每槽 2304 × 32 槽 = 73728（脚本会自动校验 ctx 语义）
  -c 73728 --parallel 32 --host 0.0.0.0 --port 8080 &

# 标定
python code/tc_profile.py --base-url http://127.0.0.1:8080/v1 \
  --spec 3B:q4_k_m --hardware "A10 24GB / llama.cpp" \
  --out results/calib_a10_llamacpp.json
```

### 4.3 组 ②：A10 + vLLM

```bash
pip install vllm
# 下 HF 格式权重（不是 GGUF！vLLM 吃不了 GGUF）
modelscope download --model Qwen/Qwen2.5-3B-Instruct --local_dir ~/hf/qwen3b

# 起 OpenAI 兼容服务
python -m vllm.entrypoints.openai.api_server \
  --model ~/hf/qwen3b --served-model-name qwen3b \
  # 对齐口径：每序列 2304、并发 32（与 llama.cpp 的 --parallel 一致）
  --max-model-len 2304 --max-num-seqs 32 --port 8081 &

python code/tc_profile.py --base-url http://127.0.0.1:8081/v1 \
  --wire-model qwen3b --spec-json ~/qwen3b_spec.json \
  --hardware "A10 24GB / vLLM" --out results/calib_a10_vllm.json
```

> `--spec-json` 里填：`weight_bytes`（HF safetensors 总字节）、`n_params=3080000000`、
> `n_layers=36`、`n_kv_heads=2`、`head_dim=128`（从 `config.json` 读）。

### 4.4 一个必须留意的坑（我们已经踩过一次）

`llama.cpp` 的 `--ctx-size` 语义在**不同 build 之间不一样**（本项目实测是 `n_ctx_slot = ctx`，不按并发切分）。
vLLM 又是另一套（`--max-model-len` 是每序列上限）。

**脚本已经自动做了这件事**：启动后读日志里的 `n_ctx_slot`，若发现它等于总 ctx
（大于 `PER_SLOT×2`）就自动用 `-c PER_SLOT` 重启。

另外两条同样重要：
1. **两个后端不能用同一份 `weight_bytes`**——GGUF 文件含对齐/元数据，比 HF safetensors 大几个百分点；脚本会分别生成 `spec_gguf.json` / `spec_hf.json`。
2. **先跑 `tc_profile.py --selftest`**：不连服务器的链路自检，不过就别开始测。

**所以每次都要先看日志确认**，并且：
- `tc_profile` 会用**服务端返回的 `prompt_tokens`** 作为真实 L（不信任名义值）；
- 如发现 TTFT 不随 L 增长，先查日志里的 ctx/slot 语义，别急着说"模型失效"。

---

## 五、额度与时间预算

| 内容 | 用时 |
|---|---|
| 建实例 + 装环境 | 30–60 分钟 |
| 组 ① 编译 llama.cpp + 下载模型 | 20–40 分钟 |
| 组 ① 测量（8 点 × 1 次） | 15 分钟 |
| 组 ② 装 vLLM + 下 HF 权重 | 20–30 分钟 |
| 组 ② 测量 | 15 分钟 |
| 复测两次（看方差） | 30 分钟 |
| **合计** | **约 3 小时** |

**每月 35.8 小时，3 小时一次绰绰有余**，剩下的额度还能做：
- 换量化档（Q4/Q8）看 `c_srv` 是否变化
- 换模型尺寸（1.5B/7B，A10 24 GB 能放 7B）
- 复现 108 格中的一部分做交叉验证

---

## 六、这组实验做完能多出什么

| 之前 | 之后 |
|---|---|
| "换硬件只需重标定两个常数"（**主张**） | A10 实测反推 BW/PEAK，与规格对照（**证据**） |
| "`c_srv` 是部署栈属性"（**推测**） | llama.cpp vs vLLM 的 `c_srv` 实测差 N 倍（**证据**） |
| 单机单栈（**边界**） | 2 硬件 × 2 后端（**受控对照**） |

**最后一句可以直接写进报告**：

> "同一张 A10 上，vLLM 的每流每步开销是 llama.cpp 的 X 分之一；
> 同一套 llama.cpp 上，A10 的有效带宽是 4060 的 Y 倍。
> 六参数模型在两个维度上都只需重标定对应参数——**这就是'部署栈属性'与'硬件属性'的分离**。"

---

## 七、顺序建议

```
Day 1   领 PAI-DSW 试用（A10）→ 建实例 → 装环境 → 跑组 ①（llama.cpp）
Day 2   同一实例上跑组 ②（vLLM）→ 两组参数对比 → 出第一版结论
Day 3   复测 + 换量化档 → 写「跨硬件 × 跨后端」一节
```

**每天只用 1–2 小时额度，3 个月共 107 小时，完全够。**
