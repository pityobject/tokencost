#!/usr/bin/env bash
# =============================================================================
# A10 一条龙：在同一张 A10 上跑 llama.cpp 与 vLLM 两条标定（口径严格对齐）
#
# 用途：证明「每步服务开销 c_srv 是部署栈属性」——同一张卡、同一个模型、
#       同一精度、同一组并发与长度，只换后端。
#
# 环境：PAI-DSW 免费试用 ecs.gn7i-c8g1.2xlarge（1×A10 24GB, 8 vCPU）
# 用法：
#   bash a10_run.sh              # 全流程
#   bash a10_run.sh llamacpp     # 只跑 llama.cpp
#   bash a10_run.sh vllm         # 只跑 vLLM
#
# 约定（两个后端必须一致，改一处就要改另一处）：
#   每槽上下文 PER_SLOT=2304   （2048 prompt + 128 生成 + 128 余量；2112 会截断 2048 点）
#   槽位上限   SLOTS=32        （llama.cpp --parallel / vLLM --max-num-seqs）
#   生成长度   GEN=128         （模型的设计点，见 失败与修复记录 F10）
# =============================================================================
set -euo pipefail

# DSW 的"工作目录"是 /mnt/systemdisk（`$HOME=/root` 是容器覆盖层，重启即丢）
if [ -z "${WORKDIR:-}" ]; then
  # DSW 实际把工作目录挂在 /mnt/workspace（控制台里可能显示为"系统盘"）
  if [ -d /mnt/workspace ] && [ -w /mnt/workspace ]; then
    WORKDIR=/mnt/workspace/tc_a10
  elif [ -d /mnt/systemdisk ] && [ -w /mnt/systemdisk ]; then
    WORKDIR=/mnt/systemdisk/tc_a10
  else
    WORKDIR="$HOME/tc_a10"
  fi
fi
echo "WORKDIR = $WORKDIR"
export WORKDIR
MODEL_GGUF_FILE="qwen2.5-3b-instruct-fp16.gguf"

PER_SLOT=2304
SLOTS=32
GEN=128
REPO_URL="https://gitee.com/pityobject/tokencost.git"

PHASE="${1:-all}"
# 硬件标签：在 A10 与 V100 上用同一脚本时，用于区分文件名与报告里的 hardware 字段
HW_LABEL="${HW_LABEL:-A10 24GB}"
# GPU 层数：99 = 全部卸载到 GPU。**必须显式给**，否则可能测到 CPU 推理而不报错。
NGL="${NGL:-99}"
mkdir -p "$WORKDIR"/{models,logs,results}
cd "$WORKDIR"

# root 还是 sudo？DSW 里通常是 root，但别假设
SUDO=""
if [ "$(id -u)" -ne 0 ]; then SUDO="sudo"; fi

log() { echo -e "\n\033[1;36m=== $* ===\033[0m"; }

die() { echo -e "\n\033[1;31m❌ $*\033[0m"; exit 1; }

# ---------------------------------------------------------------- 0. 依赖
log "0. 检查环境与依赖"
python3 -c "import torch; print('torch', torch.__version__, '| cuda', torch.version.cuda, '| available', torch.cuda.is_available())" 2>&1 || true
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader \
  || die "看不到 GPU。确认实例规格是 gn7i（A10）"
nvidia-smi --query-gpu=clocks.sm,clocks.max.sm,power.draw,temperature.gpu \
  --format=csv,noheader | tee "$WORKDIR/logs/clock_start.txt"

if [ ! -d "$WORKDIR/tokencost" ]; then
  git clone "$REPO_URL" "$WORKDIR/tokencost"
fi
( cd "$WORKDIR/tokencost" && git pull --ff-only ) || echo "（git pull 跳过）"

command -v cmake >/dev/null || {
  $SUDO apt-get update -qq && $SUDO apt-get install -y -qq cmake build-essential
}
pip3 config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple >/dev/null 2>&1 || true
python3 -m pip install -q --upgrade pip
python3 -m pip install -q modelscope

# ---------------------------------------------------------------- 1. 下模型
log "0.5 链路自检（不连服务器，验证 tc_profile 本身可用）"
( cd "$WORKDIR/tokencost/code" && python3 tc_profile.py --selftest )   || die "tc_profile 自检失败——后面所有数字都不可信，先修环境"

log "1. 下载模型（HF fp16 与 GGUF F16 各一份，精度相同）"
python3 - <<'PY'
import os
from modelscope import snapshot_download
base = os.path.join(os.environ["WORKDIR"], "models")
snapshot_download('Qwen/Qwen2.5-3B-Instruct', local_dir=os.path.join(base, 'hf'))
snapshot_download('Qwen/Qwen2.5-3B-Instruct-GGUF',
                  allow_patterns=['*fp16*', '*f16*'], local_dir=os.path.join(base, 'gguf'))
print("模型下载完成")
PY

GGUF_PATH="$WORKDIR/models/gguf/${MODEL_GGUF_FILE}"
if [ ! -f "$GGUF_PATH" ]; then
  GGUF_PATH=$(find "$WORKDIR/models/gguf" -name '*fp16*.gguf' | head -1)
fi
if [ ! -f "$GGUF_PATH" ]; then
  GGUF_PATH=$(find "$WORKDIR/models/gguf" -name '*f16*.gguf' | head -1)
fi
if [ ! -f "$GGUF_PATH" ]; then
  echo "目录里现有的 GGUF："; ls -lh "$WORKDIR/models/gguf" | head -20
  die "没找到 fp16/f16 的 GGUF（见上面清单；也可手工建软链到 $WORKDIR/models/gguf/${MODEL_GGUF_FILE}）"
fi
WEIGHT_BYTES=$(stat -c%s "$GGUF_PATH")
echo "GGUF: $GGUF_PATH  ($((WEIGHT_BYTES/1024/1024)) MB)"

# 【不要共用】GGUF 文件含对齐/元数据，比 HF safetensors 大几个百分点；
# 两个后端各用自己那份 weight_bytes，否则 BW 会把封装开销算进去。
HF_BYTES=$(python3 - <<'PY'
import glob, os
files = glob.glob(os.path.join(os.environ["WORKDIR"], "models", "hf", "*.safetensors"))
print(sum(os.path.getsize(f) for f in files))
PY
)
echo "GGUF F16: $((WEIGHT_BYTES/1024/1024)) MB | HF safetensors: $((HF_BYTES/1024/1024)) MB"

make_spec() {  # $1=out $2=weight_bytes
  cat > "$1" <<JSON
{
  "model": {
    "name": "Qwen3B", "quant": "fp16",
    "weight_bytes": $2,
    "n_params": 3080000000,
    "n_layers": 36, "n_kv_heads": 2, "head_dim": 128
  }
}
JSON
}
make_spec "$WORKDIR/spec_gguf.json" "$WEIGHT_BYTES"
make_spec "$WORKDIR/spec_hf.json"   "$HF_BYTES"
echo "spec: $WORKDIR/spec_gguf.json（llama.cpp） / $WORKDIR/spec_hf.json（vLLM）"

# 跑一条标定：失败就打印日志尾部并退出（不要静默通过）
run_profile() {
  local url="$1" spec="$2" out="$3" logf="$4"; shift 4
  ( cd "$WORKDIR/tokencost/code" && \
    python3 tc_profile.py --base-url "$url" \
      --spec-json "$spec" \
      --slots "$SLOTS" --per-slot-ctx "$PER_SLOT" --gen "$GEN" \
      --out "$out" "$@" > "$logf" 2>&1 ) || {
    echo "❌ 标定失败，日志尾部："; tail -40 "$logf"; exit 1; }
  tail -25 "$logf"
}

# ---------------------------------------------------------------- 2. llama.cpp
if [ "$PHASE" = "all" ] || [ "$PHASE" = "llamacpp" ]; then
  log "2. 编译 llama.cpp（CUDA；8 vCPU 约 8–15 分钟）"
  if [ ! -d "$WORKDIR/llama.cpp" ]; then
    git clone --depth 1 https://github.com/ggerganov/llama.cpp "$WORKDIR/llama.cpp"
  fi
  ( cd "$WORKDIR/llama.cpp" \
    && cmake -B build -DGGML_CUDA=ON -DLLAMA_CURL=OFF >/dev/null \
    && cmake --build build -j"$(nproc)" >/dev/null )

  log "3. 起 llama.cpp（--ctx-size = $PER_SLOT × $SLOTS = $((PER_SLOT*SLOTS))）"
  "$WORKDIR/llama.cpp/build/bin/llama-server" \
    -m "$GGUF_PATH" -ngl "$NGL" \
    --ctx-size $((PER_SLOT*SLOTS)) --parallel "$SLOTS" \
    --host 127.0.0.1 --port 8090 \
    > "$WORKDIR/logs/llamacpp.log" 2>&1 &
  SRV=$!
  ok=0
  for _ in $(seq 1 120); do
    curl -sf http://127.0.0.1:8090/health >/dev/null && { ok=1; break; }
    sleep 2
  done
  [ "$ok" = 1 ] || { kill $SRV 2>/dev/null || true; tail -30 "$WORKDIR/logs/llamacpp.log"; die "llama.cpp 没起来"; }

  # 【必须校验 ctx 语义】不同 build 的 -c 含义不同（本项目踩过，见 F8）：
  #   切成并发：n_ctx_slot = -c / --parallel   → 传 PER_SLOT*SLOTS
  #   不切分  ：n_ctx_slot = -c                → 必须传 PER_SLOT，否则显存爆
  # 【必须校验】模型是否真的卸载到 GPU。llama.cpp 在显存放不下或参数缺失时会回退 CPU，
  # 而 CPU 推理的数字"看起来正常"，只是慢一个数量级——不报错，只换机器。
  if ! grep -qiE "offload|CUDA0 model|using device CUDA" "$WORKDIR/logs/llamacpp.log"; then
    tail -30 "$WORKDIR/logs/llamacpp.log"
    die "日志里看不到 CUDA 卸载迹象——可能跑在 CPU 上，拒绝继续"
  fi
  echo "GPU 卸载确认：$(grep -iE 'offload|CUDA0 model' "$WORKDIR/logs/llamacpp.log" | tail -1)"

  SLOT_CTX=$(grep -oE "n_ctx_slot *= *[0-9]+" "$WORKDIR/logs/llamacpp.log" | grep -oE "[0-9]+" | head -1)
  echo "日志里的 n_ctx_slot = ${SLOT_CTX:-（未找到）}，期望 ≈ $PER_SLOT"
  if [ -n "${SLOT_CTX:-}" ] && [ "$SLOT_CTX" -gt $((PER_SLOT * 2)) ]; then
    echo "⚠️ 该 build 不按并发切分 ctx → 用 -c $PER_SLOT 重启"
    kill $SRV 2>/dev/null || true; sleep 3
    "$WORKDIR/llama.cpp/build/bin/llama-server" \
      -m "$GGUF_PATH" -ngl "$NGL" --ctx-size "$PER_SLOT" --parallel "$SLOTS" \
      --host 127.0.0.1 --port 8090 > "$WORKDIR/logs/llamacpp.log" 2>&1 &
    SRV=$!
    for _ in $(seq 1 120); do curl -sf http://127.0.0.1:8090/health >/dev/null && break; sleep 2; done
  fi

  log "4. 标定 llama.cpp（约 5–10 分钟）"
  run_profile "http://127.0.0.1:8090/v1" "$WORKDIR/spec_gguf.json" \
              "$WORKDIR/results/calib_llamacpp.json" \
              "$WORKDIR/logs/profile_llamacpp.log" \
              --hardware "$HW_LABEL / llama.cpp"

  kill $SRV 2>/dev/null || true
  sleep 5
fi

# ---------------------------------------------------------------- 3. vLLM
if [ "$PHASE" = "all" ] || [ "$PHASE" = "vllm" ]; then
  log "5. 检查 vLLM（PAI 镜像常已预装）"
  if [ "$HW_LABEL" != "A10 24GB" ]; then
    echo "⚠️ 当前硬件标签是「$HW_LABEL」。V100(sm70) 不被新版 vLLM 支持，"
    echo "   若下面起不来，属预期——A10 已经拿到 vLLM 数据，这里只跑 llama.cpp 即可。"
  fi
  if ! python3 -c "import vllm" 2>/dev/null; then
    # 镜像自带 torch 2.3.1+cu121；最新 vLLM 会强升 torch（几 GB，容易把环境搞坏）
    python3 -m pip install -q "vllm==0.6.6.post1" \
      || python3 -m pip install -q "vllm==0.6.3" \
      || python3 -m pip install -q vllm
  fi
  python3 -c "import vllm, torch; print('vllm', vllm.__version__, '| torch', torch.__version__, '| cuda', torch.cuda.is_available())"

  log "6. 起 vLLM（--max-model-len=$PER_SLOT --max-num-seqs=$SLOTS，关前缀缓存）"
  python3 -m vllm.entrypoints.openai.api_server \
    --model "$WORKDIR/models/hf" \
    --served-model-name qwen3b \
    --max-model-len "$PER_SLOT" \
    --max-num-seqs "$SLOTS" \
    --gpu-memory-utilization 0.9 \
    --no-enable-prefix-caching \
    --host 127.0.0.1 --port 8081 \
    > "$WORKDIR/logs/vllm.log" 2>&1 &
  SRV=$!
  ok=0
  for _ in $(seq 1 180); do
    curl -sf http://127.0.0.1:8081/v1/models >/dev/null && { ok=1; break; }
    sleep 2
  done
  [ "$ok" = 1 ] || { kill $SRV 2>/dev/null || true; tail -40 "$WORKDIR/logs/vllm.log"; die "vLLM 没起来"; }

  log "7. 标定 vLLM（约 5–10 分钟）"
  run_profile "http://127.0.0.1:8081/v1" "$WORKDIR/spec_hf.json" \
              "$WORKDIR/results/calib_vllm.json" \
              "$WORKDIR/logs/profile_vllm.log" \
              --wire-model qwen3b --hardware "$HW_LABEL / vLLM"

  kill $SRV 2>/dev/null || true
fi

# ---------------------------------------------------------------- 4. 汇总
log "8. 汇总对比"
python3 - <<'PY'
import json, os, pathlib
d = pathlib.Path(os.environ["WORKDIR"]) / "results"
rows = []
for name, tag in (("calib_llamacpp.json", "llama.cpp"),
                  ("calib_vllm.json", "vLLM")):
    p = d / name
    if p.exists():
        rows.append((tag, json.loads(p.read_text(encoding="utf-8"))))
if len(rows) < 2:
    print("（只有一个后端的结果，先补齐另一个）")
if rows:
    keys = ["bw", "peak", "peak_prefill", "c_srv", "b_sat", "ovh_ttft"]
    print(f"{'参数':<14}" + "".join(f"{n:>13}" for n, _ in rows) + f"{'比值':>9}")
    for k in keys:
        vals = [j.get(k) for _, j in rows]
        line = f"{k:<14}" + "".join(
            f"{v:>13.3f}" if isinstance(v, (int, float)) else f"{'—':>13}" for v in vals)
        if len(vals) == 2 and all(isinstance(v, (int, float)) and v for v in vals):
            line += f"{vals[0]/vals[1]:>9.2f}"
        print(line)
    print("\n判读：bw / peak 应接近（同一张卡）；")
    print("      c_srv 若差 >30% → 证明『每步服务开销是部署栈属性』。")
PY

nvidia-smi --query-gpu=clocks.sm,clocks.max.sm,power.draw,temperature.gpu \
  --format=csv,noheader | tee "$WORKDIR/logs/clock_end.txt"

echo -e "\n全部完成。结果在 $WORKDIR/results/，日志在 $WORKDIR/logs/"
echo "把上面这张表贴回来即可。"
