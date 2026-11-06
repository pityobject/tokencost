# CI checks (Windows PowerShell 5.1+ compatible, ASCII only)
#
# Runs two checks; any failure returns non-zero:
#   1. tc_claims.py --check   docs numbers must match artifacts
#   2. tests/test_core.py     three zero-dependency unit tests
#
# Usage:
#   powershell -ExecutionPolicy Bypass -File scripts\ci.ps1
#
# Exit code: 0 = all pass, 1 = something failed

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

# Pick an interpreter: project venv first, then PATH python
$candidates = @(
    "D:\llm\venv\Scripts\python.exe",
    (Join-Path $root ".venv\Scripts\python.exe")
)
$py = $null
foreach ($c in $candidates) {
    if (Test-Path $c) { $py = $c; break }
}
if (-not $py) {
    $cmd = Get-Command python -ErrorAction SilentlyContinue
    if ($cmd) { $py = "python" }
}
if (-not $py) {
    Write-Host "[CI] no python interpreter found" -ForegroundColor Red
    exit 1
}

Write-Host "[CI] interpreter: $py"
Write-Host ""

$fail = 0

Write-Host "[CI] 1/7 regenerate claims table ..."
& $py "code\tc_claims.py" | Out-Null
if ($LASTEXITCODE -ne 0) { $fail = 1 }

Write-Host "[CI] 2/7 verify docs match artifacts ..."
& $py "code\tc_claims.py" --check
if ($LASTEXITCODE -ne 0) { $fail = 1 }

Write-Host ""
Write-Host "[CI] 3/7 unit tests ..."
& $py "tests\test_core.py"
if ($LASTEXITCODE -ne 0) { $fail = 1 }
& $py "tests\test_guardrails.py"
if ($LASTEXITCODE -ne 0) { Write-Host "[FAIL] 护栏测试未通过"; $fail = 1 }
if ($LASTEXITCODE -ne 0) { $fail = 1 }

Write-Host ""
Write-Host "[CI] 4/7 doc number lint (warning only) ..."
& $py "code\tc_lint_docs.py"
$lintExit = $LASTEXITCODE
if ($lintExit -ne 0) {
    Write-Host "[CI] lint reported suspects (not failing the build)" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "[CI] 5/7 project audit (hygiene / self-contained / data chain / contract) ..."
& $py "code\tc_audit.py" --fast
if ($LASTEXITCODE -ne 0) { $fail = 1 }

Write-Host ""
Write-Host "[CI] 6/7 self-calibration pipeline selftest (synthetic data, no server) ..."
& $py "code\tc_profile.py" --selftest
if ($LASTEXITCODE -ne 0) { $fail = 1 }

Write-Host ""
Write-Host "[CI] 7/7 static calculator matches artifacts ..."
& $py "code\tc_web.py" | Out-Null
if (-not (Test-Path "site\index.html")) {
  Write-Host "[FAIL] site/index.html 没有生成"; $fail = 1
} else {
  & git -c core.quotepath=false diff --quiet -- site/index.html
  if ($LASTEXITCODE -ne 0) {
    Write-Host "[FAIL] site/index.html 与模型产物不一致 —— 重新生成后提交"
    $fail = 1
  } else {
    Write-Host "[ OK ] 网页与产物一致"
  }
}

Write-Host ""
if ($fail -eq 0) {
    Write-Host "[CI] CI PASS" -ForegroundColor Green
    exit 0
} else {
    Write-Host "[CI] CI FAIL - see messages above" -ForegroundColor Red
    exit 1
}

