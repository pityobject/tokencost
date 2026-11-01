"""
tests/test_core.py —— 三个最小但关键的测试（P2-13 的第一步）。

**零依赖**：不要求装 pytest。
    python tests/test_core.py        # 直接跑（内置微型 runner）
    python -m pytest tests -q        # 装了 pytest 也能被收集

为什么只测这三件：
  ① `predict` 的黄金值——改了公式或参数键名，这里立刻红；
  ② `aggregate` 的统计口径——TTFT/TPOT 是从逐请求时间戳算出来的，
     不同人对 p50 / CVaR90 的实现容易不一致；
  ③ `fit` 的恢复实验——用已知参数生成合成数据再拟合回去，
     确认"优化器 + 损失函数"这条路本身是对的，
     而不是靠真实数据碰巧拟合得好。
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "code"))

import tc_config as cfg          # noqa: E402
import tc_loadgen as lg          # noqa: E402
import tc_model as tm            # noqa: E402


def approx(got, want, rel=0.05, abs_tol=None):
    """简易 approx：pytest 没装时也能用。"""
    if abs_tol is not None:
        ok = abs(got - want) <= abs_tol
    else:
        ok = abs(got - want) <= abs(want) * rel
    assert ok, f"期望 {want}±{rel*100 if abs_tol is None else abs_tol}，实际 {got}"


# ---------------------------------------------------------------- ① predict 黄金值
def test_predict_golden():
    """3B-Q4 / ctx512 / 并发 4 的预测值应与当前标定一致（改公式必须同步改这里）。"""
    params = dict(bw_gbs=249.13, peak_tflops=13.16, peak_prefill_tflops=31.92,
                  c_srv_ms=0.766, b_sat=13.26, ovh_ttft_ms=15.96)
    row = dict(model="3B", quant="q4_k_m", ctx=512, conc=4,
               prompt_tokens=296.0, ctx_avg_tokens=296.0 + cfg.GEN_TOKENS / 2)
    pr = tm.predict(params, row)

    # 物理合理性：TPOT 必须大于纯访存时间（约 8.6 ms），TTFT 必须大于 20 ms
    assert pr["pred_tpot_ms"] > 8.0, "TPOT 低于访存下限，公式或参数错了"
    assert pr["pred_ttft_ms"] > 20.0
    assert pr["pred_cost_gpu_s_per_1m"] > 0

    # 黄金值（相对误差内），防止公式被无意改坏
    approx(pr["pred_tpot_ms"], 11.7, rel=0.05)
    # 3B-Q4/ctx512/并发4 的预测成本约 3416 GPU·秒/百万 token
    approx(pr["pred_cost_gpu_s_per_1m"], 3416, rel=0.10)


# ---------------------------------------------------------------- ② aggregate 口径
def test_aggregate_quantiles():
    """p50 / P90 / CVaR90 的实现必须与定义一致。"""
    rows = []
    for i in range(20):
        rows.append(dict(ok=True, ttft_ms=100 + i * 5, n_out=cfg.GEN_TOKENS,
                         prompt_tokens=296,
                         tpot_ms=10.0, total_ms=100 + i * 5 + cfg.GEN_TOKENS * 10,
                         t_start_rel=float(i), t_end_rel=float(i) + 1.4,
                         n_chunks=cfg.GEN_TOKENS))
    agg = lg.aggregate(rows, 4)          # 签名是位置参数 concurrency

    assert agg["n_ok"] == 20, agg.get("n_ok")
    assert 145 <= agg["ttft_p50_ms"] <= 155, agg["ttft_p50_ms"]
    assert 185 <= agg["ttft_p90_ms"] <= 196, agg["ttft_p90_ms"]
    # CVaR90 = 最差 10%（2 个）的均值 = (190 + 195) / 2 = 192.5
    assert 190 <= agg["ttft_cvar90_ms"] <= 195, agg["ttft_cvar90_ms"]
    approx(agg["tpot_p50_ms"], 10.0, abs_tol=0.01)


# ---------------------------------------------------------------- ③ fit 恢复实验
def test_fit_recovery():
    """从已知参数生成合成数据，再拟合回去——验证拟合流程本身。"""
    true = dict(bw_gbs=250.0, peak_tflops=13.0, peak_prefill_tflops=32.0,
                c_srv_ms=0.8, b_sat=13.0, ovh_ttft_ms=16.0)

    rows = []
    for model in ("1.5B", "3B", "7B"):
        for conc in (1, 4, 16, 32):
            for ctx in (512, 2048):
                if (model, "q4_k_m") not in cfg.MODEL_SPECS:
                    continue
                row = dict(cell_id=cfg.cell_id(model, "q4_k_m", ctx, conc),
                           model=model, quant="q4_k_m", ctx=ctx, conc=conc,
                           prompt_tokens=296.0,
                           ctx_avg_tokens=296.0 + cfg.GEN_TOKENS / 2)
                pr = tm.predict(true, row)
                row["tpot_p50_ms"] = pr["pred_tpot_ms"]
                row["ttft_p50_ms"] = pr["pred_ttft_ms"]
                row["cost_gpu_s_per_1m"] = pr["pred_cost_gpu_s_per_1m"]
                rows.append(row)

    df = pd.DataFrame(rows)
    calib = df.iloc[::3]                      # 1/3 标定，其余留出
    fitp = tm.fit(calib)

    approx(fitp["bw_gbs"], true["bw_gbs"], rel=0.25)
    approx(fitp["peak_prefill_tflops"], true["peak_prefill_tflops"], rel=0.30)
    approx(fitp["c_srv_ms"], true["c_srv_ms"], rel=0.40)

    hold = df.iloc[1::3]
    errs = []
    for _, r in hold.iterrows():
        pr = tm.predict(fitp, r)
        errs.append(abs(pr["pred_tpot_ms"] / r["tpot_p50_ms"] - 1))
    assert np.mean(errs) < 0.15, f"留出误差 {np.mean(errs)*100:.1f}% 过大（合成数据应接近 0）"


# ---------------------------------------------------------------- 微型 runner
if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    tests = [(n, f) for n, f in sorted(globals().items())
             if n.startswith("test_") and callable(f)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"  [PASS] {name}")
        except AssertionError as e:
            failed += 1
            print(f"  [FAIL] {name}: {e}")
        except Exception as e:
            failed += 1
            print(f"  [ERROR] {name}: {type(e).__name__}: {e}")
    print(f"\n{len(tests)-failed}/{len(tests)} 通过")
    sys.exit(1 if failed else 0)

