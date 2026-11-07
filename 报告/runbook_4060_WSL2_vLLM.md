# 4060 + WSL2 + vLLM：消费级卡上的第二栈对照（runbook）

> 目的：回答"**消费级卡上，`c_srv` 的栈差异还在不在？**"
> 现有证据只有 A10（24 格，留出 MAPE 4.26%）。
> 3B fp16 在 8 GB 上装不进（6.18 GB 权重 + 2.25 GB KV > 8.19 GB），所以用 **1.5B fp16**
> 与 llama.cpp 的 **1.5B q8_0**（1.76 GB，精度接近）做同卡对照。

## 0. 前置事实

| 项 | 值 |
|---|---|
| GPU | RTX 4060 Laptop **8188 MiB** |
| vLLM 在 Windows 原生 | ❌ 不支持，必须 WSL2 |
| 1.5B fp16 权重 | ~3.1 GB（+KV，8 GB 可容纳） |
| WSL2 | 本机**尚未安装** |

## 1. 装 WSL2（约 15–30 分钟，需重启）

管理员 PowerShell：

```powershell
wsl --install
# 重启电脑；首次启动会要求设置 Linux 用户名/密码
wsl -l -v          # 应显示 VERSION 2
```

## 2. WSL 内装 vLLM

```bash
sudo apt update && sudo apt install -y python3-pip python3-venv
python3 -m venv ~/venv && source ~/venv/bin/activate
pip install -U pip
pip install vllm -i https://pypi.tuna.tsinghua.edu.cn/simple
python -c "import vllm,torch;print(vllm.__version__, torch.cuda.is_available())"
```

> 若 `torch.cuda.is_available()` 为 False：WSL2 的 CUDA 由驱动透传，确认 Windows 驱动 ≥ 535。

## 3. 下载 1.5B fp16 权重（ModelScope，国内快）

```bash
pip install modelscope
modelscope download --model Qwen/Qwen2.5-1.5B-Instruct --local_dir ~/models/qwen1.5b
ls ~/models/qwen1.5b
```

## 4. 起 vLLM（32 并发槽）

```bash
nohup python -m vllm.entrypoints.openai.api_server \
  --model ~/models/qwen1.5b --served-model-name local \
  --dtype float16 --port 8000 --host 127.0.0.1 \
  --max-model-len 8192 --max-num-seqs 32 \
  --gpu-memory-utilization 0.90 --disable-log-requests \
  > ~/vllm_4060.log 2>&1 &

for i in $(seq 1 60); do curl -s http://127.0.0.1:8000/v1/models >/dev/null && { echo 就绪; break; }; sleep 5; done
```

## 5. 跑网格（16 格：8 并发 × 2 个长度）

在 **Windows 侧**跑（`tc_profile.py` 在仓库里；WSL 里也能访问 `/mnt/c/...`）：

```powershell
cd C:\Users\王钟垚\Desktop\find_career\tokencost
"D:\llm\venv\Scripts\python.exe" code\tc_profile.py `
  --base-url http://127.0.0.1:8000/v1 --spec-json spec_15b_fp16.json `
  --backend vllm --wire-model local `
  --concs 1,2,4,8,12,16,24,32 --lengths 297 `
  --ctx 2048 --slots 32 --requests 96 --gen 128 `
  --hardware "4060 8GB / vLLM" --out results\a10_raw\vllm_4060_ctx2048.json
```

`spec_15b_fp16.json`：

```json
{"name":"1.5B","quant":"fp16","weight_bytes":3087007744,"n_params":1540000000,
 "n_layers":28,"n_kv_heads":2,"head_dim":128}
```

## 6. 对照（本地已有）

llama.cpp 的 1.5B q8_0 在 4060 上的格已在 `data/summary_steady.csv` 里。
分析时比较**同卡同尺寸**的 `c_srv` 与 `bw`，而不是绝对延迟。

## 预期结论的三种可能

| 若… | 说明 |
|---|---|
| vLLM `c_srv` ≈ 0.2–0.3（远低于 llama.cpp 0.766） | **栈差异在消费卡上同样存在** → 跨栈结论得到第二个硬证据 |
| vLLM `c_srv` ≈ 0.6–0.9 | 栈差异是云端特性（大 batch 调度），消费卡上不明显 |
| 起不来 / OOM | 8 GB 的硬限制，如实记录 |

## 风险

- WSL2 装完需要重启，中途会打断工作；
- vLLM wheel ~200 MB，本地网络若 <500 KB/s 也要 10 分钟以上；
- 8 GB 下 `--max-num-seqs 32` 可能 OOM → 先试 16。
