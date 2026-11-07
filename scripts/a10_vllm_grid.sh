#!/usr/bin/env bash
# =============================================================================
# A10 + vLLM 全网格（补上"vLLM 只有 6 个标定点"的缺口）
#
# 目标：在 A10 上用 vLLM 跑一套设计网格，使得可以像 llama.cpp 那样做
#       "自己标定 / 自己留出"的验证（而不是拿 6 个点算跨栈比值）。
#
# 网格：ctx ∈ {512, 2048, 8192} × conc ∈ {1,2,4,8,12,16,24,32} = 24 格
#       prompt 长度按 ctx 取（与 llama.cpp 侧一致：ctx8192 用 1973 token）
#
# 用法（在云上实例里）：
#   bash a10_vllm_grid.sh
#
# 预计：启动 3 分钟 + 24 格 × 3 轮 × ~40 s ≈ 40 分钟
# =============================================================================
set -u
WORK=/mnt/workspace/tc_grid
mkdir -p "$WORK"
cd "$WORK" || exit 1

A10_GPUS=${A10_GPUS:-1}
MODEL_PATH=${MODEL_PATH:-/mnt/workspace/tc_a10/models/Qwen2.5-3B-Instruct}
PORT=8000
BASE="http://127.0.0.1:${PORT}/v1"

echo "=== 0) 环境自检 ==="
python3 -c "import numpy,pandas,scipy,httpx;print(' deps ok')" || {
  echo "缺依赖：pip install numpy pandas scipy httpx psutil"; exit 1; }
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -2

echo "=== 1) 停掉旧的 vLLM（避免端口/显存冲突）==="
pkill -f "vllm.entrypoints" 2>/dev/null || true
sleep 3
nvidia-smi --query-gpu=memory.used --format=csv,noheader

echo "=== 2) 启动 vLLM（32 并发槽，最大长度 16384）==="
nohup python3 -m vllm.entrypoints.openai.api_server \
  --model "$MODEL_PATH" \
  --served-model-name local \
  --port $PORT --host 127.0.0.1 \
  --max-model-len 16384 \
  --max-num-seqs 32 \
  --gpu-memory-utilization 0.85 \
  --disable-log-requests \
  > "$WORK/vllm.log" 2>&1 &
echo "  pid=$!"

echo "=== 3) 等就绪（最多 5 分钟）==="
for i in $(seq 1 60); do
  if curl -s "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1; then
    echo "  就绪（${i}0 秒）"; break
  fi
  sleep 5
done
curl -s "$BASE/models" | head -c 300; echo

echo "=== 4) 跑网格（3 个 ctx × 8 个并发）==="
run_grid() {
  local ctx=$1 L=$2 tag=$3
  echo "--- ctx=$ctx L=$L"
  python3 tc_profile.py \
    --base-url "$BASE" --spec 3B:fp16 --backend vllm --wire-model local \
    --concs 1,2,4,8,12,16,24,32 --lengths "$L" \
    --ctx "$ctx" --slots 32 --requests 3 --gen 128 \
    --hardware "A10 24GB / vLLM" \
    --out "grid_${tag}.json" 2>&1 | tail -5
}

# 注意：每槽 ctx 要能装下 prompt+gen；ctx=512 时 L 只能用 297
run_grid 512  297   ctx512
run_grid 2048 297   ctx2048
run_grid 8192 1973  ctx8192

echo "=== 5) 汇总 ==="
ls -lh "$WORK"/grid_*.json
python3 - <<'PY'
import glob, json
for f in sorted(glob.glob("grid_*.json")):
    d = json.load(open(f))
    print(f"  {f}: {d.get('n_points')} 点, c_srv={d.get('c_srv')}")
PY

echo ""
echo "=== 完成。把 grid_*.json 和 vllm.log 拷回本地 ==="
echo "  本地命令："
echo "    scp root@<实例IP>:$WORK/grid_*.json tokencost/results/a10_raw/vllm_grid/"
echo "    scp root@<实例IP>:$WORK/vllm.log   tokencost/results/a10_raw/vllm_grid/"
