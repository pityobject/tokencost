# TokenCost

**A six-parameter cost model for LLM serving on a single 8 GB consumer GPU —
with every number measured, and every failure recorded.**

[中文 README](README.md) · [Data dictionary](报告/数据字典.md) ·
[Measurement discipline](报告/方法论_测量纪律.md) ·
[Applicability matrix](报告/适用域矩阵_一页纸.md) ·
[Interactive calculator](site/index.html)

---

## What it is

A roofline-style model that predicts **latency and cost** of LLM inference from six
hardware/deployment parameters:

```
TPOT(b) = max( (W + KV·L·b)/BW , 2·P·b/PEAK_dec ) + c_srv·b
TTFT(L) = max( W/BW , 2·P·L·min(b, B_SAT)/PEAK_pre ) + c_srv·b + ovh
```

| parameter | fitted value | meaning |
|---|---|---|
| `BW` | 249.1 GB/s | **effective** bandwidth (absorbing quant/dequant overhead) |
| `PEAK_dec` | 13.2 TFLOP/s | decode compute — **structurally unidentifiable on this GPU**, do not quote |
| `PEAK_pre` | 31.9 TFLOP/s | prefill compute |
| `c_srv` | 0.766 ms | per-stream per-step serving overhead |
| `B_SAT` | 13.3 | prefill saturation concurrency |
| `ovh` | 16.0 ms | fixed first-token overhead |

## Headline results (measured, not assumed)

| metric | value |
|---|---|
| Held-out cost MAPE (core, 12 calib → 68 unseen cells) | **8.90%** |
| — same, at the calibrated concurrencies | **5.8%** |
| — at uncalibrated concurrencies | 11.5% |
| — random 12-point re-splits (30 runs, 5–95%) | **9.1%–13.9%** |
| Prediction-interval coverage (target 90%) | 84.2% |
| Decision regret vs oracle (16 scenarios) | **0.0%** (random baseline 274%) |
| Cross-family transfer (Llama-3.2-1B, zero recalibration) | 11.6% |

Data: **115 configuration cells / 385 session-level runs**, 0 silent failures
(11 recorded port-conflict failures are kept in the raw data).

## Where the model does **not** work (measured boundaries)

| condition | status | evidence |
|---|---|---|
| long prompt (>1000 tok) × high concurrency (≥8) | ❌ unusable | TTFT error up to **224%** |
| mixed prompt lengths (long+short interleaved) | ❌ TTFT unusable | +45…+109% vs measured; TPOT still fine (17%) |
| middle concurrency (8/12) TPOT | ⚠️ 20–25% low | serving overhead has a step at b≈8 (7B: 6× peak) |
| SLA tail (p90) at concurrency ≥10 | ⚠️ optimistic ~2× | 28/132 cells flip from pass to fail at TTFT≤2000 ms |
| different hardware / different backend | ❌ untested | use the self-calibration guide |
| `PEAK_dec` | ❌ unidentifiable | decode arithmetic intensity ≈4 FLOP/byte vs GPU ridge ≈53 |

## Quickstart

```powershell
$py = "D:\llm\venv\Scripts\python.exe"     # this repo's venv
cd tokencost

# 1) reproduce every headline number (no GPU needed)
& $py code/tc.py claims            # regenerate results/CLAIMS.md
& $py code/tc.py claims --check    # verify docs match artifacts
& $py code/tc.py audit             # 14 project checks (incl. bare-clone)

# 2) get a recommendation
& $py code/tc.py recommend --ttft 600 --tpot 45 --gpu-price a10
#   → 7B_q4_k_m_ctx2048_c1 · 20847 GPU·s/1M tok · 60.8 CNY/1M tok @ 10.5 CNY/h

# 3) open the calculator
start site/index.html
```

Optional (enables the `tokencost` command):

```powershell
& $py -m pip install -e . --no-deps --no-build-isolation
tokencost recommend --ttft 600 --tpot 45 --gpu-price a10
```

## Calibrate on your own machine

```bash
python code/tc_profile.py --base-url http://127.0.0.1:8080/v1 --spec 3B:q4_k_m
```

Auto-probes an OpenAI-compatible endpoint, fits the six parameters, and prints residuals.
A synthetic self-test runs in CI (`tc_profile.py --selftest`), so the calibration path
cannot silently rot.

## Why you can trust the numbers

Every figure above is generated into [`results/CLAIMS.md`](results/CLAIMS.md) and
checked by CI (8 steps): claims regeneration + drift check, 8 unit/guardrail tests,
doc-number lint, 14-point project audit (including a bare-clone self-containment test),
calibration self-test, calculator consistency, and data-dictionary consistency.

The project also keeps a public record of **15 times measurement overturned an
assumption** — including three times the checker itself was silently broken
([measurement discipline](报告/方法论_测量纪律.md), [failure log](报告/失败与修复记录.md)).

> Verifiability is not "the results look good" — it is **errors being visible**.

## Repository layout

```
code/      60 scripts (measurement, fitting, analysis, guards)
data/      18 raw/derived data files (see data dictionary)
results/   artifacts: fit.json, CLAIMS.md, audits, analyses
figures/   16 figures
tests/     3 core + 5 guardrail tests
site/      single-file offline calculator
报告/      reports: one-pager, appendix A–K, applicability matrix, methodology
scripts/   ci.ps1 (8-step CI) and remote-GPU runbook
```

## License

MIT. See [LICENSE](LICENSE) and [NOTICE](NOTICE).

