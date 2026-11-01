"""
_make_fake_data.py —— 造一份"假的实验数据"，用来在真跑之前先验证整条分析链路。

为什么需要它：
  真实实验要跑两三个小时。如果你把 tc_analyze / tc_model / tc_recommend 留到最后才第一次运行，
  一旦有 bug，你面对的是两小时数据 + 一堆报错，很难判断"是数据问题还是代码问题"。
  先用假数据把下游全部跑通，之后真数据进来就是换个文件而已。

假数据按 tc_model.py 的物理模型生成，再叠加 4% 噪声，模拟真实测量的抖动。

用法：
  & "D:\\llm\\venv\\Scripts\\python.exe" _make_fake_data.py
  & "D:\\llm\\venv\\Scripts\\python.exe" _make_fake_data.py --force    # 覆盖已有数据（危险）

⚠️ 安全保护：如果 data/raw_cells.jsonl 里已有真实实验数据，本脚本会**拒绝执行**。
   它用 "w" 模式写文件，会静默覆盖真数据——这是本项目最危险的一个动作。
"""

import argparse
import json
import sys

import numpy as np

import tc_config as cfg
from tc_model import predict

# 假数据背后的"真"参数（用来验证 tc_model 能不能反推回来）
TRUE = dict(bw_gbs=251.0, peak_tflops=13.0, peak_prefill_tflops=31.6,
            c_srv_ms=0.85, b_sat=14.0, ovh_ttft_ms=16.0)
NOISE_SIGMA = 0.04


def _looks_like_real_data(path) -> bool:
    """已有数据是真实实验产生的，还是以前跑的假数据？

    真实数据带 client_cpu_pct / temp_c_median 这类运行时字段；
    假数据由 predict() 生成，没有硬件传感器字段。
    """
    try:
        first = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
    except Exception:
        return True                       # 读不懂就当真实数据，宁可不覆盖
    return first.get("client_cpu_pct") is not None or "temp_c_median" in first


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true",
                    help="即使已有真实数据也强行覆盖")
    args = ap.parse_args()

    cfg.ensure_dirs()
    out = cfg.DATA_DIR / "raw_cells.jsonl"

    if out.exists() and out.stat().st_size > 0 and _looks_like_real_data(out) and not args.force:
        print("=" * 72)
        print("拒绝执行：data/raw_cells.jsonl 里已经有真实实验数据。")
        print("本脚本会用假数据覆盖它，而且不会报错。要保留真数据，先归档：")
        print("  Move-Item ../data/raw_cells.jsonl ../_selftest/real_data_backup.jsonl")
        print("确认不要了，才加 --force 运行。")
        print("=" * 72)
        sys.exit(1)

    rng = np.random.default_rng(7)
    n = 0

    with out.open("w", encoding="utf-8") as f:
        for c in cfg.all_cells():
            if not c["feasible"]:
                continue
            spec = cfg.MODEL_SPECS[(c["model"], c["quant"])]
            row = dict(model=c["model"], quant=c["quant"], conc=c["conc"],
                       ctx_avg_tokens=cfg.PROMPT_TOKENS + cfg.GEN_TOKENS / 2,
                       prompt_tokens=cfg.PROMPT_TOKENS)
            pr = predict(TRUE, row)
            for rep in range(cfg.REPEATS):
                tpot = pr["pred_tpot_ms"] * float(np.exp(rng.normal(0, NOISE_SIGMA)))
                ttft = pr["pred_ttft_ms"] * float(np.exp(rng.normal(0, NOISE_SIGMA)))
                thr = pr["pred_throughput_tps"] * float(np.exp(rng.normal(0, NOISE_SIGMA)))
                rec = dict(
                    cell_id=cfg.cell_id(c["model"], c["quant"], c["ctx"], c["conc"]),
                    model=c["model"], quant=c["quant"], ctx=c["ctx"], conc=c["conc"],
                    weight_bytes=spec["weight_bytes"], n_params=spec["n_params"],
                    ctx_avg_tokens=row["ctx_avg_tokens"], vram_est_gb=c["vram_est_gb"],
                    requests=cfg.requests_per_repeat(c["conc"]), gen_tokens=cfg.GEN_TOKENS,
                    repeat=rep, status="ok", wall_s=25.0,
                    n_ok=cfg.requests_per_repeat(c["conc"]), n_fail=0,
                    n_out_median=cfg.GEN_TOKENS, prompt_tokens_median=cfg.PROMPT_TOKENS,
                    ttft_p50_ms=ttft, ttft_p90_ms=ttft * 1.15, ttft_cvar90_ms=ttft * 1.25,
                    ttft_max_ms=ttft * 1.4,
                    tpot_p50_ms=tpot, tpot_p90_ms=tpot * 1.03, tpot_cvar90_ms=tpot * 1.06,
                    total_p50_ms=ttft + tpot * cfg.GEN_TOKENS,
                    throughput_tps=thr, cost_gpu_s_per_1m=1e6 / thr,
                    mean_total_ms=ttft + tpot * cfg.GEN_TOKENS,
                    measure_span_s=25.0, total_tokens=thr * 25.0,
                    per_slot_tps=thr / c["conc"],
                    temp_c_median=68.0, clock_sm_median=2400.0,
                )
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n += 1

    print(f"已生成 {n} 行假数据 -> {out}")
    print(f"背后的真参数：带宽 {TRUE['bw_gbs']} GB/s、算力 {TRUE['peak_tflops']} TFLOP/s、"
          f"开销 {TRUE['ovh_ms']} ms（tc_model 应能大致反推）")
    print("下一步： python tc_analyze.py && python tc_model.py")
    print("\n跑完记得清走四处产物：data/raw_cells.jsonl、data/summary.csv、results/*、figures/*")


if __name__ == "__main__":
    main()
