# -*- coding: utf-8 -*-
"""
带宽阶梯（`tc.py ladder`）：三块 GPU 的反推 BW vs 标称 BW

## 要回答什么

`TPOT = max((W + KV·L·b)/BW, 2·P·b/PEAK_dec) + c_srv·b` 里的 `BW` 是**物理量**还是
"为了让数据拟合而凑出来的补丁参数"？

如果它是物理量，那么在不同 GPU 上独立标定出的 `BW` 应当与硬件的**标称带宽成比例**。
两个点只能连线；三个点才能看出是否线性。

## 数据

- 本机 RTX 4060：`results/fit.json`
- A10 + llama.cpp：`results/a10_raw/calib_a10_llamacpp.json`
- V100 + llama.cpp：`results/a10_raw/calib_v100_llamacpp.json`

（可选）A10 + vLLM：`results/a10_raw/calib_a10_vllm.json`，用来看"同卡不同栈时 BW 是否也变"
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import tc_config as cfg

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# 标称显存带宽（GB/s，厂商值）
NOMINAL = {
    "RTX 4060 Laptop": 272.0,
    "A10 24GB": 600.0,
    "V100 16GB": 900.0,
}


def load(rel):
    fp = cfg.RESULTS_DIR / rel
    return json.loads(fp.read_text(encoding="utf-8")) if fp.exists() else None


def main():
    rows = []
    fit = load("fit.json")
    if fit:
        rows.append(dict(hw="RTX 4060 Laptop", stack="llama.cpp",
                         bw=float(fit["bw"]), c_srv=float(fit["c_srv"]),
                         mape=float(fit.get("mape_cost", float("nan")))))
    for rel, hw, stack in (
        ("a10_raw/calib_a10_llamacpp.json", "A10 24GB", "llama.cpp"),
        ("a10_raw/calib_v100_llamacpp.json", "V100 16GB", "llama.cpp"),
        ("a10_raw/calib_a10_vllm.json", "A10 24GB", "vLLM"),
    ):
        d = load(rel)
        if not d:
            continue
        res = d.get("residual") or {}
        rows.append(dict(hw=hw, stack=stack, bw=float(d["bw"]),
                         c_srv=float(d["c_srv"]),
                         mape=float(res.get("tpot_mape", float("nan")))))

    if not rows:
        print("没有任何标定结果")
        return

    print("=" * 84)
    print("一、反推带宽 vs 标称带宽（同一个栈 llama.cpp，换硬件）")
    print("=" * 84)
    print(f"  {'硬件':<18}{'栈':<12}{'标称 BW':>9}{'反推 BW':>10}{'占比':>7}"
          f"{'c_srv':>9}{'TPOT 误差':>10}")
    mains = [r for r in rows if r["stack"] == "llama.cpp"]
    for r in mains:
        nom = NOMINAL.get(r["hw"])
        pct = f"{r['bw'] / nom * 100:.0f}%" if nom else "—"
        print(f"  {r['hw']:<18}{r['stack']:<12}"
              f"{(f'{nom:.0f}' if nom else '—'):>9}{r['bw']:>10.1f}{pct:>7}"
              f"{r['c_srv']:>9.3f}{r['mape']:>9.1f}%")

    # 线性检验：反推 BW / 标称 BW 的离散程度
    ratios = [r["bw"] / NOMINAL[r["hw"]] for r in mains if r["hw"] in NOMINAL]
    if len(ratios) >= 2:
        import statistics as st
        print(f"\n  占比 {min(ratios)*100:.0f}%–{max(ratios)*100:.0f}%，"
              f"均值 {st.mean(ratios)*100:.0f}%，标准差 {st.pstdev(ratios)*100:.1f} pp")
        print("  → 若占比高度一致，说明反推 BW 与标称成比例（即它是物理量）；")
        print("    占比差异反映各平台的真实可达带宽（A10 常见 75–85%，笔记本卡 90%+）。")

    vllm = [r for r in rows if r["stack"] == "vLLM"]
    if vllm and mains:
        a10l = next((r for r in mains if r["hw"] == "A10 24GB"), None)
        a10v = vllm[0]
        if a10l:
            print(f"\n  同卡换栈（A10）：llama.cpp {a10l['bw']:.1f} vs "
                  f"vLLM {a10v['bw']:.1f} GB/s → "
                  f"vLLM 为 llama.cpp 的 {a10v['bw']/a10l['bw']*100:.0f}%")
            print("  （说明 BW 不只是硬件属性，也受影响于推理引擎的访存实现——"
                  "但变化幅度远小于 c_srv 的 5–7 倍）")

    print("\n" + "=" * 84)
    print("二、c_srv 随硬件/栈的迁移（同一个 llama.cpp 栈）")
    print("=" * 84)
    for r in mains:
        print(f"  {r['hw']:<18} c_srv = {r['c_srv']:>6.3f} ms")
    if len(mains) >= 2:
        lo = min(mains, key=lambda r: r["c_srv"])
        hi = max(mains, key=lambda r: r["c_srv"])
        print(f"\n  极差 {hi['c_srv']/lo['c_srv']:.1f}×（{lo['hw']} → {hi['hw']}）")
        print("  → 同一个部署栈、同一份模型权重，只换 GPU，服务开销差 "
              f"{hi['c_srv']/lo['c_srv']:.1f} 倍。")
        print("    这进一步说明 c_srv 不是一个'模型常数'，而是服务进程在该机器上的实测。")

    out = dict(ladder=[dict(hardware=r["hw"], stack=r["stack"],
                            nominal_gbs=NOMINAL.get(r["hw"]),
                            measured_gbs=round(r["bw"], 1),
                            fraction=round(r["bw"] / NOMINAL[r["hw"]], 3)
                            if r["hw"] in NOMINAL else None,
                            c_srv_ms=round(r["c_srv"], 3),
                            tpot_mape_pct=round(r["mape"], 1)
                            if r["mape"] == r["mape"] else None)
                       for r in rows],
               note="反推 BW 来自各自独立标定；标称值为厂商规格")
    (cfg.RESULTS_DIR / "bandwidth_ladder.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n→ results/bandwidth_ladder.json")


if __name__ == "__main__":
    main()
