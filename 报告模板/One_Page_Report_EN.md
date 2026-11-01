# Cost–Latency–Quality Pareto Frontier for LLM Inference: A Falsifiable Analytical Model

<!--
Usage: replace every ______ with your own measured number.
Rules: no hedges ("probably", "roughly speaking"), no claim without a number,
       every figure caption states the finding, not what the figure "shows".
Target: 2 pages, single column, 10.5 pt.
-->

**Author:** Wang Zhongyao · Harbin Institute of Technology, Weihai · School of Science
**Contact:** ______ · **Code:** ______

---

## One-sentence result

> A three-parameter analytical model calibrated on **12** configurations predicts the cost per million
> tokens on **______** unseen configurations with **______%** MAPE and **______%** worst-case error.

---

## 1. Problem

Every decoded token requires streaming the full weight matrix from memory. Inference cost is therefore
governed by two machine constants — effective memory bandwidth and effective compute throughput —
rather than by nominal FLOPs. The engineering question is:

> Given a latency SLA and a quality floor, which (model size, quantization, context limit, concurrency)
> minimizes cost per million tokens?

---

## 2. Method

### 2.1 Analytical model

```
TPOT(b) = max( (W + KV(ctx)) / BW ,  2·P·b / PEAK ) + ovh
TTFT(L) = max( W / BW ,  2·P·L / PEAK ) + ovh
knee    b* = w · PEAK / (2 · BW)
```

Three fitted parameters only: effective bandwidth `BW`, effective compute `PEAK`, fixed overhead `ovh`
(parameter-to-sample ratio 1:4).

### 2.2 Calibration and holdout

- Calibration: ______ configurations (design of experiments: extremes and centre of each factor)
- Holdout: ______ configurations, **never used for fitting**
- Loss: `Σ [ln(predicted/measured)]²` — relative, symmetric, scale-free across two orders of magnitude
- Prediction interval: two error layers (parameter uncertainty × dof correction, plus measurement
  noise ÷ √k), quantile from a **t** distribution because the calibration set is small

---

## 3. Setup

| Item | Value |
|---|---|
| GPU | RTX 4060 Laptop, ______ MiB, driver ______ |
| Stack | llama.cpp ______ (CUDA ______), Python ______ |
| Models | Qwen2.5 ______, GGUF ______ |
| Grid | ______ model×quant × context ______ × concurrency ______ |
| Load profile | ______ prompt tokens, ______ generated tokens per request |
| Conditions | AC power, best-performance mode, ______ °C, no throttling observed |

Protocol: ______ warm-up requests discarded per cell; three repeats, median reported; GPU temperature and
clocks logged per cell; a random non-repeating prefix is injected and `cache_prompt` disabled to prevent
prompt-cache hits from corrupting TTFT.

---

## 4. Results

### 4.1 Decode is bandwidth-bound

> TPOT at concurrency 1 is ______ / ______ / ______ ms for 1.12 / 2.10 / 4.68 GB models,
> i.e. proportional to weight bytes (Figure 1, R² = ______). Implied effective bandwidth
> ______ GB/s, i.e. ______% of the ~260 GB/s nameplate.

**Figure 1** — Decode time scales with weight bytes (effective bandwidth ______ GB/s), confirming a
bandwidth-bound decode stage.

### 4.2 Batching amortises cost, up to a knee

> Cost per million tokens drops from ______ to ______ (1/______) as concurrency rises from 1 to 32,
> while per-request TPOT changes by ______%. The fitted knee lies at concurrency ______; beyond it,
> cost **saturates** rather than increasing, while latency keeps degrading.

### 4.3 Memory and quality prune the design space into a non-rectangular region

> Of 60 combinations, ______ are infeasible because KV cache exceeds 8 GB, concentrated in the
> "large model × long context × high concurrency" corner; the quality floor removes ______ more.

**Figure 5** — Of the ______ configurations meeting the quality floor, only ______ lie on the Pareto
frontier.

### 4.4 Accuracy and honesty of the model

| Metric | Result | Target |
|---|---|---|
| Holdout MAPE (cost) | ______% | ≤ 8% |
| Worst single-cell error | ______% | ≤ 20% |
| Empirical coverage of the 90% interval | ______% | 82–96% |

### 4.5 Decision quality

> Across ______ business scenarios, the recommender's choice (based on the calibrated model only)
> differs from the post-hoc optimum (based on all measured points) by a median **regret of ______%**,
> while cutting cost by ______%–______% relative to the default configuration.

---

## 5. Discussion

### 5.1 A hypothesis that died

> I initially assumed ______. The measurements showed ______, because ______.
> This changed the conclusion from "______" to "______".

### 5.2 Limitations

- Single 8 GB GPU, single node; load profile fixed at ______-token prompts and ______-token outputs;
- Multi-tenant interference, intra-batch length skew, KV quantization and flash-attention are not modelled;
- Beyond concurrency ______ the model is unvalidated; the applicability domain is stated in §4.4.

### 5.3 Transferability

The model structure is hardware-independent; porting to A100 or Ascend 910B requires **re-calibrating
two constants only** (BW, PEAK). See the attached one-pager, *From One GPU to a Cluster*.

---

## 6. Reproducibility

```
repo: ______
python code/tc_env_check.py
python code/tc_run_grid.py      # ~ ______ minutes
python code/tc_analyze.py && python code/tc_model.py && python code/tc_verify.py
```
