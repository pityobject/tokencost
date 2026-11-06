# -*- coding: utf-8 -*-
"""
跨硬件 / 跨部署栈参数对比（`tc.py backends`）

## 为什么需要

`c_srv` 到底属于硬件还是部署栈，是这个项目对外最重要的一句话。
但手抄表格无法复现。本脚本从**标定产物**生成对比表：

- 本机：`results/fit.json`（4060 + llama.cpp，12 格标定）
- 云端：`results/calib_*.json`（`tc_profile` 输出）

## 判读规则（写在代码里，避免事后解释）

| 对比 | 差异大说明 |
|---|---|
| 同一硬件、不同栈 → `c_srv` 差异 | **`c_srv` 是部署栈属性**（这就是我们要的结论） |
| 不同硬件、同一栈 → `c_srv` 差异 | 说明它还混有硬件/CPU 因素 |
| `bw` 差异 | 应与标称带宽比值同向 |
| `peak_prefill` 差异 | 应与 fp16 算力同向 |

**诚实条款**：本机是 4060+llama.cpp，云端是 A10+vLLM —— 硬件与栈**同时变了**，
所以两者的 `c_srv` 差只能"提示"、不能"证明"栈属性。要证明，需要 A10+llama.cpp 那一格。
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import tc_config as cfg

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# 标称值（GB/s, TFLOP/s fp16），用于判断"有效值占标称的比例是否合理"
NOMINAL = {
    "RTX 4060 Laptop": (272.0, 15.0),
    "A10 24GB": (600.0, 125.0),
    "V100 16GB": (900.0, 125.0),
    "T4 16GB": (320.0, 65.0),
}


def load_entries():
    """返回 [{label, hw, backend, bw, c_srv, peak_prefill, b_sat, ovh, mape}]。"""
    out = []

    fit_fp = cfg.RESULTS_DIR / "fit.json"
    if fit_fp.exists():
        d = json.loads(fit_fp.read_text(encoding="utf-8"))
        out.append(dict(label="本机 4060 + llama.cpp", hw="RTX 4060 Laptop",
                        source="fit.json", bw=d.get("bw"),
                        c_srv=d.get("c_srv"), peak_prefill=d.get("peak_prefill"),
                        b_sat=d.get("b_sat"), ovh=d.get("ovh_ttft"),
                        mape=d.get("mape_cost"), n=d.get("n_holdout")))

    for p in sorted(cfg.RESULTS_DIR.glob("calib_a10_*.json")):   # 只读本次云端产物
        d = json.loads(p.read_text(encoding="utf-8"))
        hw_full = str(d.get("hardware", ""))
        # "A10 24GB / vLLM" → hw="A10 24GB", backend="vLLM"
        hw, _, backend = hw_full.partition(" / ")
        res = d.get("residual") or {}
        out.append(dict(label=f"{hw} + {backend or '?'}", hw=hw,
                        source=p.name, bw=d.get("bw"), c_srv=d.get("c_srv"),
                        peak_prefill=d.get("peak_prefill"), b_sat=d.get("b_sat"),
                        ovh=d.get("ovh_ttft"),
                        mape=res.get("tpot_mape"), n=d.get("n_points")))
    return out


def main():
    rows = load_entries()
    if len(rows) < 2:
        print("需要至少两份标定产物（results/fit.json + results/calib_*.json）")
        return

    print("=" * 92)
    print("跨硬件 / 跨部署栈的参数对比")
    print("=" * 92)
    hdr = f"{'配置':<28}{'BW GB/s':>10}{'有效%':>8}{'PEAK_pre':>10}{'c_srv ms':>10}{'B_SAT':>8}{'ovh ms':>8}"
    print(hdr)
    for r in rows:
        nom = NOMINAL.get(r["hw"])
        eff = (r["bw"] / nom[0] * 100) if (nom and r["bw"]) else float("nan")
        print(f"{r['label']:<28}{r['bw']:>10.1f}{eff:>7.0f}%{r['peak_prefill']:>10.1f}"
              f"{r['c_srv']:>10.3f}{r['b_sat']:>8.1f}{r['ovh']:>8.2f}")

    base = rows[0]
    print("\n相对第一行（本机 4060 + llama.cpp）的比值：")
    for r in rows[1:]:
        if not (r["c_srv"] and base["c_srv"]):
            continue
        same_hw = (r["hw"] == base["hw"])
        print(f"  {r['label']:<28}c_srv {r['c_srv']/base['c_srv']:>5.2f}×   "
              f"BW {r['bw']/base['bw']:>5.2f}×   "
              f"PEAK_pre {r['peak_prefill']/base['peak_prefill']:>5.2f}×"
              f"   {'（同硬件，只变栈 → 可归因于栈）' if same_hw else '（硬件也变了 → 不可单独归因）'}")

    # 有效带宽占标称比例——判断标定是否合理
    print("\n有效带宽占标称值的比例（应落在 55–95%）：")
    for r in rows:
        nom = NOMINAL.get(r["hw"])
        if nom and r["bw"]:
            print(f"  {r['label']:<28}{r['bw']:>7.1f} / {nom[0]:>6.1f} GB/s = {r['bw']/nom[0]*100:>5.0f}%")

    out = dict(entries=[{k: (round(v, 4) if isinstance(v, float) else v)
                         for k, v in r.items()} for r in rows],
               note=("本机与云端改的是「硬件+栈」两个因素，c_srv 的差只能提示不能证明；"
                     "要证明需 A10 + llama.cpp 这一格"))
    (cfg.RESULTS_DIR / "backend_compare.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n→ results/backend_compare.json")


if __name__ == "__main__":
    main()
