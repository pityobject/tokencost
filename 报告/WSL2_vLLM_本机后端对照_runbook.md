# WSL2 + vLLM：在本机做"纯后端对照"

> **目的**：同一张 4060、同一个模型、同一个精度，**只换后端**，看 `c_srv` 是否变化。
> 这是"每步服务开销是部署栈属性"这个主张最干净的证明——硬件完全不变。
>
> **前置检查（已确认）**
> - Windows 10（构建 26200）→ WSL2 GPU 透传支持
> - NVIDIA 驱动 **566.26** → 满足 WSL CUDA 要求（≥ 470）
> - 磁盘：C 剩 20.8 GB、D 剩 29.5 GB（WSL 装 C，建议之后导出到 D）
> - **WSL 未安装** ← 第一步要做的

---

## 0. 硬件与显存预算（先算清，避免白装）

用 **Qwen2.5-1.5B-Instruct，fp16**（两个后端都用 fp16，否则测到的是量化差异）：
权重 **3.1 GB**，KV 每 token **28.7 KB**（28 层 × 2 KV 头 × 128 维 × 2 字节 × 2）。

| 配置 | KV 总量 | KV | 权重 | 合计 | 8 GB 卡可用 ~7.2 GB |
|---|---|---|---|---|---|
| 32 槽 × 2112 ctx | 67584 token | 1.9 GB | 3.1 GB | **5.0 GB** | ✅ |
| 8 槽 × 2112 ctx | 16896 token | 0.5 GB | 3.1 GB | 3.6 GB | ✅（更保险） |

> **3B 的 fp16 是 6.2 GB 权重**，8 GB 装不下 → 本机只能用 1.5B。
> 想在 3B 上做后端对照，必须去 A10（24 GB）。

---

## 1. 安装 WSL2 + Ubuntu

**以管理员身份**打开 PowerShell：

```powershell
wsl --install -d Ubuntu-22.04
```

- 会自动启用"虚拟机平台 / WSL"两个 Windows 功能，**然后要求重启**
- 重启后 Ubuntu 首次启动会让你设用户名密码
- 若 `wsl --install` 报错，先手动启功能再重启：

```powershell
dism.exe /online /enable-feature /featurename:Microsoft-Windows-Subsystem-Linux /all /norestart
dism.exe /online /enable-feature /featurename:VirtualMachinePlatform /all /norestart
# 重启后
wsl --set-default-version 2
```

**验证**：

```powershell
wsl --status          # 应显示 默认版本: 2
wsl -l -v             # Ubuntu-22.04  State=Running  VERSION=2
```

---

## 2. 限制 WSL 内存（笔记本必做）

WSL2 默认最多吃一半内存，会和你的 Windows 抢。新建 `C:\Users\王钟垚\.wslconfig`：

```ini
[wsl2]
memory=6GB
processors=8
swap=2GB
```

然后 `wsl --shutdown`，下次启动生效。

---

## 3. 验证 WSL 里能看到显卡（关键一步）

```powershell
wsl -d Ubuntu-22.04 -- nvidia-smi
```

**必须看到 RTX 4060 Laptop / 8188 MiB**。

- 看不到 → 驱动太旧或 WSL 版本不对：升级 NVIDIA 驱动（≥ 566）后 `wsl --update`
- 看到但报 CUDA 错误 → 正常，容器里没装 CUDA toolkit，靠驱动透传即可

> **这一步失败就别往下走了**（1 小时规则，见文末）。

---

## 4. 在 WSL 里装 vLLM（用国内镜像）

```bash
# 进入 WSL
wsl -d Ubuntu-22.04

# 换 pip 源（清华）
pip3 config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

# vLLM 会自带匹配的 torch（约 2 GB，镜像下 5–10 分钟）
pip3 install --upgrade pip
pip3 install vllm
```

**验证**：

```bash
python3 -c "import vllm, torch; print(vllm.__version__, torch.__version__, torch.cuda.is_available())"
# 期望: 0.x.x  2.x.x  True
```

---

## 5. 下模型（fp16，两个后端共用一份权重）

```bash
pip3 install -U modelscope
python3 - <<'PY'
from modelscope import snapshot_download
p = snapshot_download('Qwen/Qwen2.5-1.5B-Instruct',
                      local_dir='/home/$USER/models/qwen1.5b')
print(p)
PY
```

> 目录里应有 `config.json` + `model-*.safetensors`（约 3.1 GB）。

---

## 6. 起 vLLM（参数必须与 llama.cpp 对齐）

```bash
cd ~
python3 -m vllm.entrypoints.openai.api_server \
  --model ~/models/qwen1.5b \
  --served-model-name qwen1.5b \
  --max-model-len 2112 \
  --max-num-seqs 32 \
  --gpu-memory-utilization 0.85 \
  --no-enable-prefix-caching \
  --port 8080
```

**对照表（两个后端必须逐项一致）**

| 概念 | llama.cpp（Windows 侧） | vLLM（WSL 侧） |
|---|---|---|
| 槽位上限 | `--parallel 32` | `--max-num-seqs 32` |
| 每槽上下文 | `--ctx-size = 2112 × 32`（脚本会自动乘） | `--max-model-len 2112` |
| 前缀缓存 | 请求里 `cache_prompt:false` | `--no-enable-prefix-caching` |
| 生成长度 | 请求里 `max_tokens=128` | 同（请求里） |
| 精度 | **F16 GGUF** | **fp16 权重** |

> `tc_profile` 已自动发送 `cache_prompt:false` 与唯一 nonce，所以两边都不会命中缓存。

**验证服务**（Windows 侧直接访问 WSL 的 8080）：

```powershell
curl http://127.0.0.1:8080/v1/models
```

---

## 7. 起 llama.cpp（Windows 侧，F16 权重）

```powershell
# 下 F16 的 GGUF（不是 Q4！）
cd D:\llm\models
& "D:\llm\venv\Scripts\python.exe" -m modelscope download `
  --model Qwen/Qwen2.5-1.5B-Instruct-GGUF `
  qwen2.5-1.5b-instruct-f16.gguf --local_dir D:\llm\models
```

```powershell
cd C:\Users\王钟垚\Desktop\find_career\tokencost\code
$py = "D:\llm\venv\Scripts\python.exe"
# 32 槽 × 2112 ctx（脚本内部会传 --ctx-size 67584 --parallel 32）
& $py tc_server.py 1.5B f16 2112 32
```

> 若 `MODEL_SPECS` 里没有 `("1.5B","f16")`，用第 8 步的 `--spec-json` 自己填即可。

---

## 8. 两条标定命令（唯一变量是后端）

**先写一份 spec（两个后端共用，因为同一个模型同一个精度）** `D:\llm\qwen15b_fp16.json`：

```json
{
  "model": {
    "name": "Qwen1.5B", "quant": "fp16",
    "weight_bytes": 3090000000,
    "n_params": 1540000000,
    "n_layers": 28, "n_kv_heads": 2, "head_dim": 128
  }
}
```

> `weight_bytes` 用实际文件大小（`dir` 看 safetensors 总字节），别估。

```powershell
cd C:\Users\王钟垚\Desktop\find_career\tokencost\code
$py = "D:\llm\venv\Scripts\python.exe"

# ① vLLM（WSL 里跑着，端口 8080）
& $py tc_profile.py --base-url http://127.0.0.1:8080/v1 `
  --spec-json D:\llm\qwen15b_fp16.json --wire-model qwen1.5b `
  --slots 32 --per-slot-ctx 2112 --gen 128 `
  --hardware "4060 Laptop 8GB / vLLM" --out ..\results\calib_vllm.json

# ② llama.cpp（Windows 上跑着，端口 8090）
& $py tc_profile.py --base-url http://127.0.0.1:8090/v1 `
  --spec-json D:\llm\qwen15b_fp16.json `
  --slots 32 --per-slot-ctx 2112 --gen 128 `
  --hardware "4060 Laptop 8GB / llama.cpp" --out ..\results\calib_llamacpp.json
```

---

## 9. 结果怎么看（这才是有价值的部分）

| 参数 | 期望 | 含义 |
|---|---|---|
| `BW` | 两者相差 <10% | 同一张卡 → **硬件属性确实不变** |
| `PEAK_pre` | 两者相差 <20% | 同上 |
| **`c_srv`** | **若相差 >30% → 证明了"服务开销是栈属性"** | 这是核心结论 |
| `b_sat` | 可能不同 | 两个后端的批处理策略不同，正常 |

**如果 `c_srv` 差得明显**，你能写进报告：

> 同一张 RTX 4060、同一个 Qwen2.5-1.5B fp16 权重、同一组并发与长度，
> llama.cpp 的每流每步开销是 vLLM 的 **N 倍**；
> 而两者反推出的有效带宽相差不到 X%——**说明带宽是硬件属性，服务开销是栈属性**。

---

## 10. 时间与失败回退

| 步骤 | 预计 |
|---|---|
| `wsl --install` + 重启 + 建用户 | 20 分钟 |
| 装 vLLM（镜像） | 10–20 分钟 |
| 下模型 | 5–10 分钟 |
| 起服务 + 两条标定 | 20 分钟 |
| **合计** | **1–1.5 小时** |

**1 小时规则**：如果卡在"WSL 里看不到 GPU"超过 1 小时，**直接放弃本机方案，改用 A10**。
免费额度足够，不要在环境上耗掉一整天——**环境问题不是研究问题**。

---

## 附：常见坑

| 现象 | 原因 | 处理 |
|---|---|---|
| `wsl --install` 后仍提示要装 | 没重启 | 重启 Windows |
| WSL 里 `nvidia-smi` 报错 | 驱动旧 / WSL 旧 | 升级驱动 + `wsl --update` |
| vLLM 报 CUDA OOM | `gpu-memory-utilization` 太高或 ctx 太大 | 降到 0.8 或 `--max-model-len 1024` |
| vLLM 启动极慢 | 首次编译 CUDA graph | 等 2–5 分钟；可加 `--enforce-eager` 跳过（会略慢） |
| Windows 侧连不上 WSL 的 8080 | 服务只听 127.0.0.1 且端口转发失效 | 在 WSL 里用 `--host 0.0.0.0` 起，或 `wsl --shutdown` 重来 |
| llama.cpp 载入 F16 慢 | 3.1 GB 权重，正常 | 约 10–20 秒 |
