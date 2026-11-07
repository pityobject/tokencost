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
| Held-out cost MAPE (core, 12 calib → 68 unseen cells) | **8.58%** |
| — same, at the calibrated concurrencies | **5.5%** |
| — at uncalibrated concurrencies | 11.4% |
| — random 12-point re-splits (30 runs, 5–95%) | **9.1%–13.9%** |
| Prediction-interval coverage (target 90%) | 86.8% |
| Decision regret vs oracle (16 scenarios) | **0.0%** (random baseline 274%) |
| Cross-family transfer (Llama-3.2-1B, zero recalibration) | 11.6% |
| **Same-parameter form on a second stack** (A10 + vLLM, independent calibration) | TPOT error **0.8%** |
| **Serving overhead is a property of the serving *process*** | `c_srv` vLLM / llama.cpp = **5.3–7.2×** on the same A10 |
| Calibration repeatability (same config, two runs) | `bw` **±3%** · `c_srv` **±28%** (llama.cpp) / **±6%** (vLLM) |
| **`BW` is a physical quantity** (three GPUs, same stack) | measured/nominal = **79–92%** — 249 / 472 / 776 GB/s vs 272 / 600 / 900 |
| **Price of quality** (400-question benchmark) | **+10% cost per +1 pp** (cheapest segment 3.6%/pp, dearest 24.4%/pp) |
| One quality tier is strictly dominated | `3B-q4_k_m` = `1.5B-q4_k_m` in accuracy (82.2%) but **48% pricier** |

Data: **118 configuration cells / 388 session-level runs**, 0 silent failures
(11 recorded port-conflict failures are kept in the raw data).

## Where the model does **not** work (measured boundaries)

| condition | status | evidence |
|---|---|---|
| long prompt (>1000 tok) × high concurrency (≥8) | ❌ unusable | **both stacks overestimate TTFT** (llama.cpp 2.33×, vLLM 8.49× growth at c16/c1) and underestimate TPOT by 20–40% → a model-form defect, not a stack defect |
| mixed prompt lengths (long+short interleaved) | ❌ TTFT unusable — **confirmed on both stacks** | per-request error **121%** (llama.cpp 4060), **146%** (llama.cpp V100), **121%** (vLLM A10) |
| middle concurrency (8/12) TPOT | ⚠️ 20–25% low | serving overhead has a step at b≈8 (7B: 6× peak) |
| SLA tail (p90) at concurrency ≥10 | ⚠️ optimistic ~2× | 28/132 cells flip from pass to fail at TTFT≤2000 ms |
| different hardware / different backend | ⚠️ **attributable, not yet predictive** | now 3 GPUs × 2 stacks: `bw` 249 / 472 / 776 GB/s against 272 / 600 / 900 nominal (79–92%); `c_srv` spans **5.0×** across GPUs for one stack |
| `PEAK_dec` | ❌ unidentifiable | arithmetic intensity ≈4 FLOP/byte vs ridge ≈53; **three independent designs all failed** (see appendix §5.28) |
| TTFT p90 tail | ❌ **must be measured** | no power law fits: R² = 0.46 pooled, 0.58 vLLM, 0.52 llama.cpp; at c32 the stack gap is **9.34× vs 1.74×** |
| TPOT p90 tail | ✅ **predictable** | `CV = 1.09% + 1.21e-4·(b-1)^0.52·L^0.78` (R²=0.994); lognormal shape error 0.8% |
| multi-turn TTFT | ⚠️ overestimated 30–47% on llama.cpp only | cause: **prefix KV reuse even with `cache_prompt=false`** (TTFT −31%; with cache on −94%); vLLM has no prefix cache → error only −3…−13% |

## Cross-hardware and cross-stack (2026-11-06, free A10 trial)

The six-parameter form was re-fitted from scratch on a **second stack** (vLLM 0.5.5)
on a **second GPU** (A10 24 GB), same model, same request protocol, 6 probes each.

| parameter | RTX 4060 + llama.cpp | **A10 + llama.cpp** | **A10 + vLLM** |
|---|---|---|---|
| `BW` (GB/s) | 249.1 (92% of nominal) | 471.5 (79%) | 384.9 (64%) |
| `PEAK_pre` (TFLOP/s) | 31.8 | 39.8 | 76.7 |
| **`c_srv` (ms)** | 0.766 | **1.529 / 1.963** (two runs) | **0.273 / 0.290** |
| `B_SAT` | 13.2 | 7.2 | 21.9 |
| fit error | — | TTFT 3.7% / TPOT 1.0% | TTFT 9.5% / TPOT 0.8% |

**Three claims this supports**

1. **The parameter form transfers across stacks** — TPOT error 0.8% on vLLM, with no
   structural change to the model.
2. **`c_srv` belongs to the serving *process*** (stack + host CPU), not to the GPU or the
   weights: same A10, swapping only the stack changes it by **5.3–7.2×**; the same stack on a
   different machine still changes it by **2.0×**.
3. **Repeatability must be quoted**: two runs of the same configuration move `c_srv` by
   **±28%** (llama.cpp) and **±6%** (vLLM), while `BW` only moves **±3%** —
   because `BW` is a physical quantity and `c_srv` is a sum of implementation details.

A third point (**V100 16 GB**) was added the next day: `BW` = **775.5 GB/s** (86% of nominal 900).

| GPU (all llama.cpp) | nominal | measured | ratio | `c_srv` |
|---|---|---|---|---|
| RTX 4060 Laptop | 272 | 249.1 | 92% | 0.766 |
| A10 24 GB | 600 | 471.5 | 79% | 1.529 |
| **V100 16 GB** | **900** | **775.5** | **86%** | **3.848** |

**Per-request jitter is cross-stack** (500+ tracked requests per cell, TPOT CV):

| concurrency | 1 | 4 | 8 | 16 | 32 | r(log b) |
|---|---|---|---|---|---|---|
| llama.cpp | 1.5% | 7.0% | 13.7% | 18.0% | **20.8%** | **+0.989** |
| vLLM | 1.1% | 7.9% | 13.2% | 19.4% | **25.9%** | **+0.989** |

And it scales with prompt length: at c32 the CV goes **3.3% (L=64) → 7.2% (L=256) → 25.9% (L=1900)**.
That single law explains why our earlier 136-cell local dataset (mostly L≈297) saw no trend.

Raw calibration files (unmodified JSON) are archived under `results/a10_raw/`.

## Quickstart

```bash
bash scripts/reproduce.sh        # ~60 s, 8 groups of checks
bash scripts/reproduce.sh --fast # skip refit, ~20 s
```

The last line must read `REPRODUCE PASS`. No GPU, no network, and it never
modifies a tracked file (the refit is compared and then restored).

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

