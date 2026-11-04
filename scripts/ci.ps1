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

Write-Host "[CI] 1/6 regenerate claims table ..."
& $py "code\tc_claims.py" | Out-Null
if ($LASTEXITCODE -ne 0) { $fail = 1 }

Write-Host "[CI] 2/6 verify docs match artifacts ..."
& $py "code\tc_claims.py" --check
if ($LASTEXITCODE -ne 0) { $fail = 1 }

Write-Host ""
Write-Host "[CI] 3/6 unit tests ..."
& $py "tests\test_core.py"
if ($LASTEXITCODE -ne 0) { $fail = 1 }

Write-Host ""
Write-Host "[CI] 4/6 doc number lint (warning only) ..."
& $py "code\tc_lint_docs.py"
$lintExit = $LASTEXITCODE
if ($lintExit -ne 0) {
    Write-Host "[CI] lint reported suspects (not failing the build)" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "[CI] 5/6 project audit (hygiene / self-contained / data chain / contract) ..."
& $py "code\tc_audit.py" --fast
if ($LASTEXITCODE -ne 0) { $fail = 1 }

Write-Host ""
Write-Host "[CI] 6/6 self-calibration pipeline selftest (synthetic data, no server) ..."
& $py "code\tc_profile.py" --selftest
if ($LASTEXITCODE -ne 0) { $fail = 1 }

Write-Host ""
if ($fail -eq 0) {
    Write-Host "[CI] CI PASS" -ForegroundColor Green
    exit 0
} else {
    Write-Host "[CI] CI FAIL - see messages above" -ForegroundColor Red
    exit 1
}

