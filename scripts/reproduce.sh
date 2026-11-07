#!/usr/bin/env bash
# =============================================================================
# 第三方复现套件（one command, no GPU needed）
#
# 目标：任何人 clone 仓库后，用一条命令验证：
#   ① 环境自检（依赖、Python）
#   ② 数据完整性（行数、格数、无静默失败）
#   ③ 数字一致性（文档与产物，CLAIMS --check）
#   ④ 报告/doc lint（无过期数字）
#   ⑤ 审计 14 项（含裸克隆检查）
#   ⑥ 护栏测试 6 项
#   ⑦ 关键结果可重算（重新拟合 → 对比已发布参数）
#   ⑧ 图标/站点一致性
#
# 用法：
#   bash scripts/reproduce.sh
#   bash scripts/reproduce.sh --fast     # 跳过重拟合（约 30 秒）
#
# 判据：最后一行必须是 "REPRODUCE PASS"。
# =============================================================================
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1

FAST=0
[ "${1:-}" = "--fast" ] && FAST=1

PY=""
# Windows 上 `python3` 常常是应用商店的占位符（运行会弹商店而不是执行），
# 所以优先用 python，并**验证它能真正打印版本**再采用。
for c in python python3 "D:/llm/venv/Scripts/python.exe"; do
  if command -v "$c" >/dev/null 2>&1; then
    if "$c" -c "import sys;print(sys.version_info[0])" >/dev/null 2>&1; then
      PY="$c"; break
    fi
  fi
done
[ -z "$PY" ] && { echo "❌ 找不到 python"; exit 1; }

PASS=0; FAIL=0
ok()   { echo "  ✅ $*"; PASS=$((PASS+1)); }
bad()  { echo "  ❌ $*"; FAIL=$((FAIL+1)); }
step() { echo; echo "=== $* ==="; }

echo "复现套件｜$(date '+%Y-%m-%d %H:%M')｜python = $PY"

# ---------------------------------------------------------------- ① 环境
step "① 环境"
$PY -c "import sys; assert sys.version_info>=(3,9)" 2>/dev/null \
  && ok "Python $( $PY -c 'import sys;print("%d.%d"%sys.version_info[:2])' )" \
  || bad "Python < 3.9"
for m in numpy pandas scipy httpx; do
  $PY -c "import $m" 2>/dev/null && ok "$m" || bad "$m 缺失"
done
command -v git >/dev/null 2>&1 && ok "git 可用" || bad "git 不可用"

# ---------------------------------------------------------------- ② 数据
step "② 数据完整性"
$PY - <<'PY'
import json, sys
from pathlib import Path
sys.path.insert(0, "code")
import tc_config as cfg
raw = Path("data/raw_cells.jsonl")
rows = [json.loads(l) for l in raw.read_text(encoding="utf-8").splitlines() if l.strip()]
cells = {r.get("cell_id") for r in rows}
print(f"  原始行数 {len(rows)}，去重格数 {len(cells)}")
assert len(rows) > 300, "行数异常"
assert len(cells) > 100, "格数异常"
for f in ("summary_steady.csv", "quality.csv"):
    assert (Path("data") / f).exists(), f"{f} 缺失"
print("  OK")
PY
[ $? -eq 0 ] && ok "数据文件与规模正常" || bad "数据检查失败"

# ---------------------------------------------------------------- ③ 数字一致性
step "③ 文档数字与产物一致"
$PY code/tc.py claims --check >/tmp/_claims.log 2>&1
if grep -q "OK" /tmp/_claims.log; then ok "claims --check 通过"
else bad "claims --check 失败（看 /tmp/_claims.log）"; fi

# ---------------------------------------------------------------- ④ lint
step "④ 对外文档 lint"
$PY code/tc.py lint >/tmp/_lint.log 2>&1
if grep -q "LINT\]\[OK\]" /tmp/_lint.log; then ok "无过期/损坏数字"
else bad "lint 失败（看 /tmp/_lint.log）"; fi

# ---------------------------------------------------------------- ⑤ 审计
step "⑤ 审计 14 项"
$PY code/tc.py audit >/tmp/_audit.log 2>&1
tail -3 /tmp/_audit.log | grep -q "全部通过\|14/14" && ok "审计通过" || bad "审计未通过"

# ---------------------------------------------------------------- ⑥ 护栏测试
step "⑥ 护栏测试"
$PY tests/test_guardrails.py >/tmp/_guard.log 2>&1
grep -q "0 failed" /tmp/_guard.log && ok "$(tail -1 /tmp/_guard.log)" || bad "护栏测试失败"

# ---------------------------------------------------------------- ⑦ 重拟合
step "⑦ 重新拟合（验证参数可复算）"
if [ "$FAST" = "1" ]; then
  echo "  （--fast：跳过）"
else
  cp results/fit.json /tmp/_fit_before.json 2>/dev/null
  $PY code/tc_model.py --population core >/tmp/_fit.log 2>&1
  if [ $? -eq 0 ]; then
    $PY - <<'PY'
import json, sys
from pathlib import Path
sys.stdout.reconfigure(encoding="utf-8")
a = json.loads(Path("/tmp/_fit_before.json").read_text(encoding="utf-8"))
b = json.loads(Path("results/fit.json").read_text(encoding="utf-8"))
keys = ("bw", "peak", "c_srv", "b_sat", "ovh_ttft", "mape_cost")
bad = []
for k in keys:
    if k not in a or k not in b:
        continue
    d = abs(b[k] - a[k]) / max(abs(a[k]), 1e-9) * 100
    flag = "OK" if d < 1.0 else "DIFF"
    if flag == "DIFF":
        bad.append(k)
    print(f"  {k:<10} 发布 {a[k]:>10.4f}  重算 {b[k]:>10.4f}  差 {d:>5.2f}%  {flag}")
sys.exit(1 if bad else 0)
PY
    if [ $? -eq 0 ]; then ok "重拟合复现（参数差 <1%）"
    else bad "重拟合结果与发布值不同（>1%）"; fi
    cp /tmp/_fit_before.json results/fit.json 2>/dev/null   # 还原，避免污染工作区
  else
    bad "重拟合失败（看 /tmp/_fit.log）"
  fi
fi

# ---------------------------------------------------------------- ⑧ 站点
step "⑧ 计算器站点"
$PY code/tc_web.py >/tmp/_web.log 2>&1
grep -q "score\|计算器\|<!DOCTYPE" site/index.html 2>/dev/null \
  && ok "site/index.html 已生成" || bad "站点文件异常"

# ---------------------------------------------------------------- 汇总
echo
echo "================================================================"
echo "  通过 $PASS 项，失败 $FAIL 项"
if [ "$FAIL" -eq 0 ]; then
  echo "  REPRODUCE PASS"
else
  echo "  REPRODUCE FAIL —— 把上面的 ❌ 行贴到 issue 里"
fi
echo "================================================================"
exit $FAIL
