"""
tc_bandwidth.py —— 用一次测量就估出你机器的"有效带宽"，顺便验证访存受限这件事。

原理（这是整个项目的第一个洞察）：
  decode 阶段生成 1 个 token，必须把全部权重读一遍。
  所以：  时间/token ≈ 权重字节 ÷ 带宽   →   带宽 ≈ 权重字节 ÷ 时间
  在并发=1、短上下文时，KV cache 很小，可以忽略，上式最干净。

它为什么重要：
  带宽是唯一能迁移到别的硬件的机器常数。换到 A100 或昇腾上，
  只需要重测这一个数，模型就能继续用 —— 这就是面试时"你到我们这儿怎么落地"的答案。

用法：
  & "D:\\llm\\venv\\Scripts\\python.exe" tc_bandwidth.py
"""

import pandas as pd

import tc_config as cfg


def main():
    path = cfg.DATA_DIR / "summary.csv"
    if not path.exists():
        raise SystemExit(f"没有 {path}，先跑 tc_analyze.py")

    df = pd.read_csv(path)
    sub = df[(df["conc"] == min(cfg.CONCURRENCY)) & (df["ctx"] == min(cfg.CTX_SIZES))].copy()
    if sub.empty:
        raise SystemExit(f"没有并发={min(cfg.CONCURRENCY)}、ctx={min(cfg.CTX_SIZES)} 的数据。")

    print(f"用 并发={min(cfg.CONCURRENCY)}、上下文={min(cfg.CTX_SIZES)} 的格估计有效带宽\n")
    print(f"{'模型':>6} {'量化':<7} {'权重GB':>7} {'TPOT ms':>9} {'有效带宽 GB/s':>14}")
    rows = []
    for _, r in sub.iterrows():
        spec = cfg.MODEL_SPECS[(r["model"], r["quant"])]
        w_gb = spec["weight_bytes"] / 1e9
        t_s = r["tpot_p50_ms"] / 1e3
        bw = w_gb / t_s
        rows.append(bw)
        print(f"{r['model']:>6} {r['quant']:<7} {w_gb:7.2f} {r['tpot_p50_ms']:9.1f} {bw:14.1f}")

    bw_mean = sum(rows) / len(rows)
    print(f"\n有效带宽均值 ≈ {bw_mean:.1f} GB/s")
    print("（RTX 4060 笔记本的标称带宽是 272 GB/s；实测通常只有七到九成，剩下的就是开销。）")
    print("各模型估出来的带宽如果彼此接近，就证明'时间正比于权重字节'成立，即访存受限成立；")
    print("如果差得很远，说明有别的因素（CPU offload、KV cache、context 太短预热不足），要回去查。")


if __name__ == "__main__":
    main()
