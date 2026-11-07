# -*- coding: utf-8 -*-
"""
高并发扩档分析（`tc.py highconc`）：c48/c64/c128 上的隐含 c_srv 与 PEAK_dec 下界

## 三个问题

1. **上端是否膨胀**：本机（4060）在 c48/64/128 的隐含 `c_srv` 是 0.71/0.72/0.84（平坦）。
   A10 两栈在高并发是否同样平坦？
2. **`PEAK_dec` 的下界**：vLLM 在 c128 的 TPOT 只有 49.96 ms，而用拟合的
   `PEAK_dec=13.83` 算出的算力项是 57 ms——**实测低于模型下限**，说明 `PEAK_dec` 偏小。
   由此可反推一个硬下界。
3. **槽位配置的影响**：llama.cpp 在 32 槽与 128 槽下各标定过一次，
   `c_srv` 从 1.53–1.96 涨到 2.51——这是"服务进程属性"的又一证据。

## 数据

- `results/a10_raw/calib_a10_{llamacpp,vllm}.json`（32 槽基准，run1）
- `results/a10_raw/calib_a10_{llamacpp,vllm}_rep3.json`（128 槽）
- `results/a10_raw/jitter_a10_{llamacpp_high2,vllm_high}.json`（c48/64/128 实测）

## 输出

终端三张表 + `results/high_conc_analysis.json`
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import tc_config as cfg

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
RAW = cfg.RESULTS_DIR / "a10_raw"
L_AVG = 300 + cfg.GEN_TOKENS / 2      # 高并发探针的 prompt=300


def load(name):
    fp = RAW / name
    return json.loads(fp.read_text(encoding="utf-8")) if fp.exists() else None


def floor_ms(b, bw_gbs, peak_tflops, weight_bytes, kv_per_tok):
    """模型下限 = max(访存, 算力)，单位 ms。"""
    t_mem = (weight_bytes + kv_per_tok * L_AVG * b) / (bw_gbs * 1e9) * 1e3
    t_cmp = 2 * 3.08e9 * b / (peak_tflops * 1e12) * 1e3
    return max(t_mem, t_cmp), t_mem, t_cmp


def main():
    cal_l, cal_v = load("calib_a10_llamacpp.json"), load("calib_a10_vllm.json")
    rep_l, rep_v = load("calib_a10_llamacpp_rep3.json"), load("calib_a10_vllm_rep3.json")
    j_l = load("jitter_a10_llamacpp_high2.json") or load("jitter_a10_llamacpp_high.json")
    j_v = load("jitter_a10_vllm_high.json")
    if not all([cal_l, cal_v, j_l, j_v]):
        print("缺数据：需要 calib_a10_{llamacpp,vllm}.json 与 jitter_a10_*_high*.json")
        return

    kv = cfg.kv_bytes_per_token(cfg.MODEL_SPECS[("3B", "q4_k_m")]) if \
        ("3B", "q4_k_m") in cfg.MODEL_SPECS else 36864
    # 3B fp16 的 KV：36 层 × 2 kv-head × 128 head_dim × 2 字节 × 2(K,V)
    spec = cal_l["model"]
    kv = 2 * 2 * 128 * spec["n_layers"] * 2

    print("=" * 86)
    print("一、高并发下的隐含 c_srv（用各栈 run1 的 6 点参数算 floor）")
    print("=" * 86)
    print(f"  {'并发':>4}{'栈':>12}{'TPOT':>9}{'floor':>9}{'隐含 c_srv':>12}{'标定值':>9}")
    rows = []
    for tag, cal, jj in (("llama.cpp", cal_l, j_l), ("vLLM", cal_v, j_v)):
        for k in sorted(jj["cells"], key=lambda x: int(x)):
            c = int(k)
            tpot = jj["cells"][k]["tpot_p50"]
            fl, t_mem, t_cmp = floor_ms(c, cal["bw"], cal["peak"],
                                        spec["weight_bytes"], kv)
            implied = (tpot - fl) / c
            rows.append(dict(stack=tag, conc=c, tpot=tpot, floor=round(fl, 1),
                             implied_csrv=round(implied, 3),
                             fitted_csrv=round(cal["c_srv"], 3),
                             binding=("compute" if t_cmp > t_mem else "memory")))
            print(f"  {c:>4}{tag:>12}{tpot:>9.1f}{fl:>9.1f}{implied:>12.3f}"
                  f"{cal['c_srv']:>9.3f}")

    print("\n" + "=" * 86)
    print("二、PEAK_dec 的硬下界（实测 TPOT 不得低于算力项，由此反推）")
    print("=" * 86)
    best = None
    for r in rows:
        if r["tpot"] <= 0:
            continue
        need = 2 * 3.08e9 * r["conc"] / (r["tpot"] * 1e-3) / 1e12
        if best is None or need > best[1]:
            best = (r, need)
    if best:
        r, need = best
        print(f"  最紧的一格：{r['stack']} c{r['conc']}，TPOT 实测 {r['tpot']:.2f} ms")
        print(f"  → PEAK_dec ≥ 2·P·b / TPOT = **{need:.1f} TFLOP/s**")
        print(f"  而当前拟合值是 {cal_v['peak']:.1f}（vLLM）/ {cal_l['peak']:.1f}（llama.cpp）")
        print("  说明：快栈高并发时算力项成为生效项，而拟合值偏小——"
              "这正是'PEAK_dec 弱约束'的具体后果。")

    print("\n" + "=" * 86)
    print("三、槽位配置的影响（llama.cpp 32 槽 vs 128 槽）")
    print("=" * 86)
    if rep_l and rep_v:
        print(f"  {'参数':<14}{'llama.cpp 32 槽':>18}{'128 槽':>12}{'变化':>10}"
              f"{'vLLM 32 槽':>14}{'128 槽':>10}{'变化':>10}")
        for k in ("c_srv", "peak_prefill", "b_sat", "ovh_ttft", "bw"):
            a, b = cal_l[k], rep_l[k]
            c, d = cal_v[k], rep_v[k]
            print(f"  {k:<14}{a:>18.3f}{b:>12.3f}{(b/a-1)*100:>9.0f}%"
                  f"{c:>14.3f}{d:>10.3f}{(d/c-1)*100:>9.0f}%")
        print("\n  → llama.cpp 的 c_srv 在 128 槽下涨了 "
              f"{(rep_l['c_srv']/cal_l['c_srv']-1)*100:.0f}%，vLLM 只变 "
              f"{(rep_v['c_srv']/cal_v['c_srv']-1)*100:.0f}%——"
              "**槽位配置本身也是'服务进程属性'的一部分**。")

    out = dict(prompt_len=L_AVG, kv_bytes_per_token=kv,
               high_conc=rows,
               peak_dec_lower_bound_tflops=round(best[1], 2) if best else None,
               slot_effect=(dict(llamacpp=dict(c_srv_32=cal_l["c_srv"],
                                               c_srv_128=rep_l["c_srv"]),
                                 vllm=dict(c_srv_32=cal_v["c_srv"],
                                           c_srv_128=rep_v["c_srv"]))
                            if (rep_l and rep_v) else None),
               note="隐含 c_srv 对 PEAK_dec 敏感；其走势比绝对值更可靠")
    (cfg.RESULTS_DIR / "high_conc_analysis.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n→ results/high_conc_analysis.json")


if __name__ == "__main__":
    main()
