#!/usr/bin/env bash
# =============================================================================
# V100 一条龙：带宽阶梯第三点（249 → 472 → ~700 GB/s）
#
# 目的：检验 `TPOT ∝ W/BW` 是否线性。前两点是 4060（249）与 A10（472）。
# V100 标称 900 GB/s，是这个检验唯一缺的一格。
#
# 环境假设：PAI-DSW 免费实例 `ecs.gn6v-c8g1.2xlarge`（1×V100 16 GB, 8 vCPU），
#           镜像 PyTorch 2.x + CUDA 12.x。全新空盘，脚本自己把一切装好。
#
# 与 A10 口径严格一致：
#   每槽上下文 PER_SLOT=2304（2048 prompt + 128 生成 + 余量）
#   槽位 SLOTS=32  → 总 ctx 73728
#   生成长度 GEN=128，后端只跑 llama.cpp（V100 是 sm70，新版 vLLM 不支持 Volta）
#
# 用法：
#   curl -sSL https://gitee.com/pityobject/tokencost/raw/main/scripts/v100_run.sh -o v100_run.sh
#   sed -i 's/\r$//' v100_run.sh
#   bash v100_run.sh
# =============================================================================
set -euo pipefail

WORKDIR="${WORKDIR:-/mnt/workspace/tc_a10}"
PER_SLOT=2304
SLOTS=32
GEN=128
TOTAL_CTX=$((PER_SLOT * SLOTS))
ARCH=70                     # V100 = sm70；不指定会编所有架构，慢很多
MIRROR="https://mirrors.aliyun.com/pypi/simple/"
GITEE="https://gitee.com/pityobject/tokencost/raw/main"

log() { echo -e "\n\033[1;36m=== $* ===\033[0m"; }
die() { echo -e "\n\033[1;31m❌ $*\033[0m"; exit 1; }

mkdir -p "$WORKDIR"/{models,gguf,logs,results,code}

# ---------------------------------------------------------------- 0. 环境
log "0. 检查 GPU"
GPU_INFO=$(nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader) \
  || die "看不到 GPU。确认规格是 ecs.gn6v（V100）"
echo "$GPU_INFO"

# 【必须断言型号】脚本名是 V100，在 A10 上跑会白费 20 分钟编译（实测发生过一次）
GPU_NAME=$(echo "$GPU_INFO" | cut -d, -f1 | xargs)
GPU_FREE_MB=$(echo "$GPU_INFO" | cut -d, -f3 | grep -oE "[0-9]+" | head -1)
if ! echo "$GPU_NAME" | grep -qi "V100"; then
  if [ "${ALLOW_OTHER_GPU:-0}" != "1" ]; then
    die "当前 GPU 是「$GPU_NAME」，不是 V100。要跑带宽阶梯请新建 ecs.gn6v 实例；
        若只是想在别的卡上跑同一套标定，加 ALLOW_OTHER_GPU=1 重跑。"
  fi
  echo "⚠️ 非 V100（$GPU_NAME），已按 ALLOW_OTHER_GPU=1 继续"
fi
[ "${GPU_FREE_MB:-0}" -ge 9000 ] || die "空闲显存只有 ${GPU_FREE_MB} MB（需 ≥9000）。先停掉占显存的进程：
        pkill -f api_server; pkill -f vllm; sleep 5; nvidia-smi"
which nvcc || ls /usr/local/cuda/bin/nvcc || die "缺 nvcc，换带 CUDA 的镜像"

log "0.5 装依赖（阿里云源，实测比清华源稳）"
python3 -m pip install -q -i "$MIRROR" --upgrade pip
python3 -m pip install -q -i "$MIRROR" numpy pandas httpx modelscope gguf sentencepiece transformers

log "0.6 取仓库代码（只拉需要的 4 个文件，避免整仓克隆）"
for f in tc_config.py tc_loadgen.py tc_model.py tc_profile.py; do
  curl -sSL "$GITEE/code/$f" -o "$WORKDIR/code/$f"
done
ls -l "$WORKDIR/code/" | tail -5
python3 -c "import ast,sys
for f in ('tc_config.py','tc_loadgen.py','tc_model.py','tc_profile.py'):
    ast.parse(open(f'/mnt/workspace/tc_a10/code/{f}',encoding='utf-8').read())
print('  语法 OK')"

# ---------------------------------------------------------------- 1. 模型
log "1. 下载 HF fp16 权重（约 6 GB）"
python3 - <<PY
from modelscope import snapshot_download as d
d('Qwen/Qwen2.5-3B-Instruct', local_dir='$WORKDIR/models/hf')
print('模型下载完成')
PY

# ---------------------------------------------------------------- 2. llama.cpp 源码
log "2. 从 PyPI 源码包取 llama.cpp（GitHub/Gitee 的 git 协议在该网络被阻断）"
rm -rf /tmp/lcp && mkdir -p /tmp/lcp
python3 -m pip download llama-cpp-python --no-binary :all: --no-deps -d /tmp/lcp -i "$MIRROR" -q
cd /tmp/lcp && tar xzf llama_cpp_python-*.tar.gz
SRC=$(ls -d /tmp/lcp/llama_cpp_python-*/vendor/llama.cpp | head -1)
[ -f "$SRC/CMakeLists.txt" ] || die "sdist 里没有 vendor/llama.cpp"
rm -rf "$WORKDIR/llama.cpp"
cp -r "$SRC" "$WORKDIR/llama.cpp"
echo "源码: $WORKDIR/llama.cpp ($(du -sh "$WORKDIR/llama.cpp" | cut -f1))"

log "3. 编译 llama-server（CUDA sm$ARCH，约 8–15 分钟）"
cd "$WORKDIR/llama.cpp"
cmake -B build -DGGML_CUDA=ON -DLLAMA_CURL=OFF \
  -DCMAKE_CUDA_ARCHITECTURES=$ARCH \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc > "$WORKDIR/logs/cmake.log" 2>&1 \
  || { tail -20 "$WORKDIR/logs/cmake.log"; die "cmake 配置失败"; }
cmake --build build -j"$(nproc)" --target llama-server > "$WORKDIR/logs/build.log" 2>&1 \
  || { tail -30 "$WORKDIR/logs/build.log"; die "编译失败"; }
ls -lh build/bin/llama-server

# ---------------------------------------------------------------- 3. 转 GGUF
log "4. HF → GGUF fp16（约 1 分钟）"
cd "$WORKDIR"
python3 llama.cpp/convert_hf_to_gguf.py models/hf \
  --outfile gguf/qwen3b-fp16.gguf --outtype f16 > "$WORKDIR/logs/convert.log" 2>&1 \
  || { tail -20 "$WORKDIR/logs/convert.log"; die "转换失败"; }
ls -lh gguf/

# ---------------------------------------------------------------- 4. 起服务
log "5. 起 llama-server（--ctx-size 是总上下文 = $TOTAL_CTX）"
# 清掉所有可能占显存的服务（vLLM 是上一轮实验最容易残留的）
pkill -f llama-server 2>/dev/null || true
pkill -f api_server 2>/dev/null || true
sleep 4
nvidia-smi --query-gpu=memory.free --format=csv,noheader | sed 's/^/  起服务前空闲显存: /' 
setsid nohup "$WORKDIR/llama.cpp/build/bin/llama-server" \
  -m "$WORKDIR/gguf/qwen3b-fp16.gguf" -ngl 99 \
  --ctx-size "$TOTAL_CTX" --parallel "$SLOTS" \
  --host 127.0.0.1 --port 8092 > "$WORKDIR/logs/server.log" 2>&1 &

for i in $(seq 1 90); do
  curl -sf http://127.0.0.1:8092/health >/dev/null && break
  sleep 2
done
curl -sf http://127.0.0.1:8092/health >/dev/null || { tail -30 "$WORKDIR/logs/server.log"; die "服务没起来"; }

grep -iE "offloaded|CUDA0 model" "$WORKDIR/logs/server.log" | tail -2 \
  || die "日志里没有 CUDA 卸载迹象——可能跑在 CPU 上"
SLOT_CTX=$(grep -oE "n_ctx_slot *= *[0-9]+" "$WORKDIR/logs/server.log" | grep -oE "[0-9]+" | head -1)
echo "n_ctx_slot = ${SLOT_CTX:-（未找到）}，期望 $PER_SLOT"
[ "${SLOT_CTX:-0}" = "$PER_SLOT" ] || die "ctx 语义不符：该 build 可能不按并发切分，检查 --ctx-size"

# ---------------------------------------------------------------- 5. 标定
log "6. 生成 spec 并标定（6 点，约 3–5 分钟）"
GG=$(stat -c%s "$WORKDIR/gguf/qwen3b-fp16.gguf")
python3 - <<PY
import json
spec = dict(name='Qwen3B', quant='fp16', weight_bytes=$GG,
            n_params=3080000000, n_layers=36, n_kv_heads=2, head_dim=128)
json.dump({'model': spec}, open('$WORKDIR/spec_gguf.json','w'), indent=2)
print('weight_bytes =', $GG)
PY

cd "$WORKDIR/code"
python3 -u tc_profile.py --base-url http://127.0.0.1:8092/v1 \
  --backend llamacpp --spec-json "$WORKDIR/spec_gguf.json" \
  --slots "$SLOTS" --per-slot-ctx "$PER_SLOT" --gen "$GEN" \
  --hardware "V100 16GB / llama.cpp" \
  --out "$WORKDIR/results/calib_v100_llamacpp.json" 2>&1 | tail -20

# ---------------------------------------------------------------- 6. 收尾
log "7. 打包"
cd "$WORKDIR/results"
tar czf /mnt/workspace/v100_results.tgz calib_v100_llamacpp.json
ls -lh /mnt/workspace/v100_results.tgz
nvidia-smi --query-gpu=clocks.sm,clocks.max.sm,power.draw,temperature.gpu \
  --format=csv,noheader | tee "$WORKDIR/logs/clock_end.txt"

echo -e "\n✅ 完成。结果：$WORKDIR/results/calib_v100_llamacpp.json"
echo "   下载 /mnt/workspace/v100_results.tgz，然后停实例（不是释放）。"
echo "   本地对比：python code/tc.py backends"
