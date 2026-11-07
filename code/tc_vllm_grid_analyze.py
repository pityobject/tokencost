# -*- coding: utf-8 -*-
"""
A10+vLLM 全网格的本地分析（`tc.py vllmgrid`）

把云端跑回来的 `results/a10_raw/vllm_grid/grid_*.json` 变成两条结论：

1. **vLLM 自己的标定 / 留出验证**：用 12 个"标定式"格拟合六参数，
   其余格当留出，算成本/TPOT/TTFT 的 MAPE——和 llama.cpp 的 8.58% 同口径。
2. **跨栈比值在网格上稳不稳**：同 (ctx, conc) 下 vLLM vs llama.cpp 的 c_srv 比值，
   从 6 个点扩展到 24 个点，报范围与离散度。

用法：python code/tc_vllm_grid_analyze.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import tc_config as cfg
import tc_model as tm

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
GRID_DIR = cfg.RESULTS_DIR / "a10_raw" / "vllm_grid"


def load_grid():
    """把云端多个 grid_*.json 拼成一张测量表（每行 = 一个 (ctx, conc)）。"""
    rows = []
    for fp in sorted(GRID_DIR.glob("grid_*.json")):
        d = json.loads(fp.read_text(encoding="utf-8"))
        ctx = None
        tag = fp.stem  # grid_ctx512
        for part in tag.split("_"):
            if part.startswith("ctx") and part[3:].isdigit():
                ctx = int(part[3:])
        for m in d.get("measurements") or []:
            rows.append(dict(
                ctx=ctx or m.get("ctx"),
                conc=int(m["conc"]),
                L=float(m.get("L") or 0),
                ttft=float(m["ttft"]),
                tpot=float(m["tpot"]),
                # tc_profile 不写 throughput：用稳态定义从 ttft/tpot 推
                throughput=float(m.get("throughput") or
                                 (int(m["conc"]) * int(m.get("gen") or 128) /
                                  ((float(m["ttft"]) + int(m.get("gen") or 128) * float(m["tpot"])) / 1e3))),
                gen=int(m.get("gen") or 128),
            ))
    df = pd.DataFrame(rows)
    return df


def main():
    if not GRID_DIR.exists() or not list(GRID_DIR.glob("grid_*.json")):
        print(f"⚠️ 还没有云端网格数据：{GRID_DIR}")
        print("  先在 A10 上跑：bash scripts/a10_vllm_grid.sh")
        print("  再把 grid_*.json 拷到该目录后重跑本脚本。")
        return

    df = load_grid()
    print("=" * 78)
    print(f"A10 + vLLM 全网格：{len(df)} 个测量点")
    print("=" * 78)
    print(f"  ctx 取值: {sorted(df['ctx'].dropna().unique())}")
    print(f"  conc 取值: {sorted(df['conc'].unique())}")
    print(df.groupby('ctx')[['L', 'ttft', 'tpot']].median().round(1).to_string())

    # ---- 用 tc_model 的口径做标定/留出 ----
    # 标定集：conc ∈ {1,4,16,32}（与 llama.cpp 侧一致）
    calib_mask = df["conc"].isin([1, 4, 16, 32])
    calib, hold = df[calib_mask].copy(), df[~calib_mask].copy()
    print(f"\n  标定 {len(calib)} 格 / 留出 {len(hold)} 格"
          f"（注意：这不等于 llama.cpp 的 12/68，后者含多模型多量化）")

    # tc_model.fit 需要 model/quant/ctx/conc/prompt_tokens 等列
    def to_cells(d):
        out = d.copy()
        out["model"] = "3B"
        out["quant"] = "fp16"
        out["prompt_tokens"] = out["L"]
        out["ctx_avg_tokens"] = out["L"] + out["gen"] / 2
        out["ttft_p50_ms"] = out["ttft"]
        out["tpot_p50_ms"] = out["tpot"]
        return out

    cfg.MODEL_SPECS.setdefault(("3B", "fp16"), dict(
        weight_bytes=6178317312, n_params=3080000000, n_layers=36,
        n_kv_heads=2, head_dim=128))

    cal = to_cells(calib)
    hp = to_cells(hold)
    params = tm.fit(cal)
    print("\n  vLLM 自己拟合出的参数：")
    for k, v in params.items():
        print(f"    {k:<22} {v:>10.3f}")

    def mape(d):
        errs = []
        for _, r in d.iterrows():
            pr = tm.predict(params, r)
            life = pr["pred_ttft_ms"] + r["gen"] * pr["pred_tpot_ms"]
            thr = r["conc"] * r["gen"] / life * 1e3
            pred_cost = 1e6 / thr
            meas_cost = 1e6 / r["throughput"] if r["throughput"] == r["throughput"] else np.nan
            if meas_cost == meas_cost and pred_cost > 0:
                errs.append(abs(pred_cost / meas_cost - 1))
        return float(np.mean(errs) * 100) if errs else float("nan")

    print(f"\n  留出集成本 MAPE（vLLM 自己标定）：**{mape(hp):.2f}%**")
    print(f"  对照：llama.cpp 在 4060 上的留出 MAPE = 8.58%（68 格，含多模型多量化）")
    print("\n  读法：若 vLLM 的留出 MAPE 与 8.58% 同量级，说明"
          "\"六参数形式跨栈成立\"不再只靠 6 个标定点。")

    # ---- 跨栈 c_srv 比值的网格稳定性 ----
    print("\n" + "=" * 78)
    print("跨栈比值：同 (ctx, conc) 下 vLLM / llama.cpp 的 c_srv")
    print("=" * 78)
    print("  提示：逐格比值需要两栈同网格；当前 vLLM 网格是 24 格、"
          "llama.cpp 侧是 12 标定 + 68 留出，交集有限。")
    print("  现有可靠结论仍是 6 点上的 5.3–7.2×（见 CLAIMS）。")
    print("\n  建议：若想在网格上比，llama.cpp 侧也要跑同样的 24 格（可在 A10 上补一遍）。")


if __name__ == "__main__":
    main()
