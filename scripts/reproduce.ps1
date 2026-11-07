# =============================================================================
# 第三方复现套件（Windows 版）
#
# 与 scripts/reproduce.sh 等价，但不需要 bash —— 本项目是 Windows 为主的，
# 让复现者为了跑 8 组检查去装 Git Bash/WSL 是不必要的门槛。
#
# 用法：
#   powershell -ExecutionPolicy Bypass -File scripts/reproduce.ps1
#   powershell -ExecutionPolicy Bypass -File scripts/reproduce.ps1 -Fast
#
# 判据：最后一行必须是 REPRODUCE PASS
# =============================================================================
param(
    [switch]$Fast
)

$ErrorActionPreference = "Continue"
$env:PYTHONUTF8 = "1"
Set-Location (Split-Path $PSScriptRoot -Parent)

$pass = 0; $fail = 0
function Ok($m)  { Write-Host "  [OK] $m" -ForegroundColor Green;  $script:pass++ }
function Bad($m) { Write-Host "  [FAIL] $m" -ForegroundColor Red;  $script:fail++ }
function Step($m) { Write-Host ""; Write-Host "=== $m ===" -ForegroundColor Cyan }

# 找 python：优先仓库约定路径，其次 PATH 里的 python
$py = $null
foreach ($c in @("D:\llm\venv\Scripts\python.exe", "python", "python3")) {
    try {
        $v = & $c -c "import sys;print('%d.%d'%sys.version_info[:2])" 2>$null
        if ($LASTEXITCODE -eq 0) { $py = $c; $pyv = $v; break }
    } catch { }
}
Write-Host "复现套件（Windows）｜$(Get-Date -Format 'yyyy-MM-dd HH:mm')"
if (-not $py) { Write-Host "找不到 python" -ForegroundColor Red; exit 1 }

Step "1/8 环境"
$maj, $min = $pyv.Split(".")
if ([int]$maj -ge 3 -and [int]$min -ge 9) { Ok "Python $pyv" } else { Bad "Python $pyv < 3.9" }
foreach ($m in @("numpy", "pandas", "scipy", "httpx")) {
    & $py -c "import $m" 2>$null
    if ($LASTEXITCODE -eq 0) { Ok $m } else { Bad "$m 缺失" }
}

Step "2/8 数据完整性"
& $py -c @"
import json,sys
from pathlib import Path
sys.path.insert(0,'code')
rows=[json.loads(l) for l in Path('data/raw_cells.jsonl').read_text(encoding='utf-8').splitlines() if l.strip()]
cells={r.get('cell_id') for r in rows}
assert len(rows)>300 and len(cells)>100, (len(rows),len(cells))
for f in ('summary_steady.csv','quality.csv'):
    assert (Path('data')/f).exists(), f
print(f'  行数 {len(rows)}，格数 {len(cells)}')
"@
if ($LASTEXITCODE -eq 0) { Ok "数据文件与规模正常" } else { Bad "数据检查失败" }

Step "3/8 文档数字与产物一致"
$out = & $py code\tc.py claims --check 2>&1
if ($out -match "OK") { Ok "claims --check 通过" } else { Bad "claims --check 失败" }

Step "4/8 对外文档 lint（含 README 覆盖度）"
$out = & $py code\tc.py lint 2>&1
if ($LASTEXITCODE -eq 0 -and ($out -match "LINT..OK" -or $out -match "LINT\]\[OK")) { Ok "无作废数字，关键结论都在 README" }
else { Bad "lint 失败（作废数字 或 README 缺关键结论）" }

Step "5/8 审计"
$out = & $py code\tc.py audit 2>&1
if ($out -match "12/12|14/14|全部通过") { Ok "审计通过" } else { Bad "审计未通过" }

Step "6/8 护栏测试"
$out = & $py tests\test_guardrails.py 2>&1
if ($out -match "0 failed") { Ok ($out | Select-Object -Last 1) } else { Bad "护栏测试失败" }

Step "7/8 重拟合（与发布值比对，然后还原）"
if ($Fast) {
    Write-Host "  （-Fast：跳过）"
} else {
    Copy-Item results\fit.json "$env:TEMP\_fit_before.json" -Force
    & $py code\tc_model.py --population core 2>&1 | Out-Null
    if ($LASTEXITCODE -eq 0) {
        & $py -c @"
import json,sys
from pathlib import Path
sys.stdout.reconfigure(encoding='utf-8')
a=json.loads(Path(r'$env:TEMP\_fit_before.json').read_text(encoding='utf-8'))
b=json.loads(Path('results/fit.json').read_text(encoding='utf-8'))
bad=[]
for k in ('bw','peak','c_srv','b_sat','ovh_ttft','mape_cost'):
    if k in a and k in b:
        d=abs(b[k]-a[k])/max(abs(a[k]),1e-9)*100
        print(f'  {k:<10} 发布 {a[k]:>10.4f}  重算 {b[k]:>10.4f}  差 {d:>5.2f}%')
        if d>=1.0: bad.append(k)
sys.exit(1 if bad else 0)
"@
        if ($LASTEXITCODE -eq 0) { Ok "重拟合复现（参数差 <1%）" } else { Bad "重拟合与发布值不同（>1%）" }
    } else { Bad "重拟合失败" }
    Copy-Item "$env:TEMP\_fit_before.json" results\fit.json -Force   # 还原，避免污染工作区
}

Step "8/8 计算器站点"
& $py code\tc_web.py 2>&1 | Out-Null
if (Test-Path site\index.html) { Ok "site/index.html 可生成" } else { Bad "站点异常" }

Write-Host ""
Write-Host ("=" * 64)
Write-Host ("  通过 $pass 项，失败 $fail 项")
if ($fail -eq 0) { Write-Host "  REPRODUCE PASS" -ForegroundColor Green }
else { Write-Host "  REPRODUCE FAIL —— 把上面的 [FAIL] 行贴到 issue 里" -ForegroundColor Red }
Write-Host ("=" * 64)
exit $fail
