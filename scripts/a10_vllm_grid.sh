#!/usr/bin/env bash
# =============================================================================
# A10 + vLLM 全网格（补"vLLM 只有 6 个标定点"的缺口）
#
# 目标：在 A10 上用 vLLM 跑 24 格设计网格，使 vLLM 能像 llama.cpp 一样
#       做"自己标定 / 自己留出"的验证，而不是拿 6 个点算跨栈比值。
#
# 网格：ctx ∈ {512, 2048, 8192} × conc ∈ {1,2,4,8,12,16,24,32}
#       prompt 长度按 ctx 取（与 llama.cpp 侧一致：ctx8192 用 1973 token）
#
# 前置：把本地 code/ 下 4 个文件放到 $WORK/code/
#       tc_profile.py tc_loadgen.py tc_model.py tc_config.py
#       （本脚本会自动找；找不到就用 Gitee 拉，失败则报错退出）
#
# 用法：
#   bash a10_vllm_grid.sh            # 全网格（约 40 分钟）
#   bash a10_vllm_grid.sh --smoke    # 只跑 1 格，验证链路（3 分钟）
# =============================================================================
set -u
WORK=${WORK:-/mnt/workspace/tc_grid}
SMOKE=0
[ "${1:-}" = "--smoke" ] && SMOKE=1
mkdir -p "$WORK/code"
cd "$WORK" || exit 1

PORT=${PORT:-8000}
BASE="http://127.0.0.1:${PORT}/v1"

echo "=== 0) 找到 tc_profile.py ==="
PROFILE=""
for p in "$WORK/code/tc_profile.py" "$WORK/tc_profile.py" \
         /mnt/workspace/tc_a10/code/tc_profile.py ./tc_profile.py; do
  [ -f "$p" ] && PROFILE="$p" && break
done
if [ -z "$PROFILE" ]; then
  echo "  本地没有，尝试从 Gitee 拉（需要网络）"
  for f in tc_profile.py tc_loadgen.py tc_model.py tc_config.py; do
    curl -fsSL "https://gitee.com/pityobject/tokencost/raw/main/code/$f" \
      -o "$WORK/code/$f" || echo "  [warn] $f 拉取失败"
  done
  PROFILE="$WORK/code/tc_profile.py"
fi
if [ ! -f "$PROFILE" ]; then
  echo "  ✗ 找不到 tc_profile.py。请先在本地上传："
  echo "    scp code/tc_profile.py code/tc_loadgen.py code/tc_model.py code/tc_config.py \\"
  echo "        root@<实例IP>:$WORK/code/"
  exit 1
fi
echo "  使用 $PROFILE"

echo "=== 1) 依赖自检 ==="
python3 -c "import numpy,pandas,scipy,httpx;print('  deps ok')" || {
  echo "  缺依赖：pip install numpy pandas scipy httpx psutil"; exit 1; }

echo "=== 2) 找到 3B 模型目录 ==="
MODEL_PATH=${MODEL_PATH:-}
if [ -z "$MODEL_PATH" ]; then
  for d in /mnt/workspace/tc_a10/models/*3B* /mnt/workspace/models/*3B* \
           /root/models/*3B* /mnt/workspace/tc_grid/models/*3B*; do
    [ -d "$d" ] && MODEL_PATH="$d" && break
  done
fi
if [ -z "$MODEL_PATH" ] || [ ! -d "$MODEL_PATH" ]; then
  echo "  ✗ 没找到 3B 模型目录。手动指定："
  echo "    MODEL_PATH=/path/to/Qwen2.5-3B-Instruct bash a10_vllm_grid.sh"
  echo "  现有目录："; ls -d /mnt/workspace/*/models/* 2>/dev/null | head
  exit 1
fi
echo "  MODEL_PATH=$MODEL_PATH"
ls "$MODEL_PATH" | head -4

echo "=== 3) 停旧 vLLM、起新实例（32 槽，最大长度 16384）==="
pkill -f "vllm.entrypoints" 2>/dev/null || true
sleep 3
nohup python3 -m vllm.entrypoints.openai.api_server \
  --model "$MODEL_PATH" --served-model-name local \
  --port $PORT --host 127.0.0.1 \
  --max-model-len 16384 --max-num-seqs 32 \
  --gpu-memory-utilization 0.85 --disable-log-requests \
  > "$WORK/vllm.log" 2>&1 &
echo "  pid=$!"

echo "=== 4) 等就绪（最多 5 分钟）==="
ok=0
for i in $(seq 1 60); do
  if curl -s "$BASE/models" >/dev/null 2>&1; then ok=1; echo "  就绪（约 ${i}0 秒）"; break; fi
  sleep 5
done
if [ $ok -eq 0 ]; then
  echo "  ✗ 启动超时，看日志：tail -50 $WORK/vllm.log"; exit 1
fi
curl -s "$BASE/models" | head -c 200; echo

# 3B fp16 的规格（本地 cfg 里没有这一项，用 --spec-json 传）
cat > "$WORK/spec_3b_fp16.json" <<'JSON'
{"name": "3B", "quant": "fp16",
 "weight_bytes": 6178317312, "n_params": 3080000000,
 "n_layers": 36, "n_kv_heads": 2, "head_dim": 128}
JSON

run_grid() {
  local ctx=$1 L=$2 tag=$3
  echo "--- ctx=$ctx  L=$L"
  python3 "$PROFILE" \
    --base-url "$BASE" --spec-json "$WORK/spec_3b_fp16.json" \
    --backend vllm --wire-model local \
    --concs 1,2,4,8,12,16,24,32 --lengths "$L" \
    --ctx "$ctx" --slots 32 --requests 3 --gen 128 \
    --hardware "A10 24GB / vLLM" \
    --out "$WORK/grid_${tag}.json" 2>&1 | tail -6
}

echo "=== 5) 跑网格 ==="
if [ $SMOKE -eq 1 ]; then
  echo "  [smoke] 只跑 ctx=512 / conc=1"
  python3 "$PROFILE" --base-url "$BASE" --spec-json "$WORK/spec_3b_fp16.json" \
    --backend vllm --wire-model local --concs 1 --lengths 297 \
    --ctx 512 --slots 32 --requests 1 --gen 128 \
    --hardware "A10 24GB / vLLM" --out "$WORK/grid_smoke.json" 2>&1 | tail -8
else
  run_grid 512  297   ctx512
  run_grid 2048 297   ctx2048
  run_grid 8192 1973  ctx8192
fi

echo "=== 6) 汇总 ==="
ls -lh "$WORK"/grid_*.json 2>/dev/null
python3 - "$WORK" <<'PY'
import glob, json, sys
for f in sorted(glob.glob(sys.argv[1] + "/grid_*.json")):
    d = json.load(open(f))
    print(f"  {f.split('/')[-1]}: {d.get('n_points')} 点, c_srv={d.get('c_srv')}")
PY

echo ""
echo "=== 完成 ==="
echo "  结果在 $WORK/grid_*.json"
echo "  拷回本地："
echo "    mkdir -p tokencost/results/a10_raw/vllm_grid"
echo "    scp root@<实例IP>:$WORK/grid_*.json tokencost/results/a10_raw/vllm_grid/"
echo "    scp root@<实例IP>:$WORK/vllm.log   tokencost/results/a10_raw/vllm_grid/"
echo "  然后本地：python code/tc.py vllmgrid"
