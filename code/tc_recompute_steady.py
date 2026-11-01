"""
tc_recompute_steady.py —— 用逐请求数据重算"稳态吞吐"，消除波次数带来的测量污染。

背景（本轮发现）：
  原来的吞吐 = 总 token ÷ 墙钟窗口。但每格的请求数/并发并不一致
  （requests_per_repeat 给的是 16/24/32/48/64），导致波次数从 16 波到 1.33 波不等。
  波次少时，测量窗口里包含大量"收尾空闲"，吞吐被系统性低估。

  实测证据（1.5B-Q4，ctx512）：
    并发  8：32 请求 = 4.00 波，吞吐 502
    并发 12：32 请求 = 2.67 波，吞吐 477   ← 并发更高反而更慢（非物理）
    并发 16：48 请求 = 3.00 波，吞吐 830
    并发 24：32 请求 = 1.33 波，吞吐 796

  而 TPOT 完全单调（5.7→6.3→7.9→13.8→14.6→16.0→20.7→28.2），说明物理没变，是口径的问题。

修正后的定义（与波次无关）：
    稳态吞吐 = 并发 × 生成长度 ÷ 平均单请求生命周期
             = conc × GEN ÷ mean(TTFT + 生成×TPOT)
    稳态成本 = 1e6 ÷ 稳态吞吐

  这个量只用"一条请求自己经历了什么"，不看我们这台机器同时跑了几波。
  它正是解析模型所预测的东西（模型算的就是单请求生命周期）。

用法：
  & "D:\\llm\\venv\\Scripts\\python.exe" tc_recompute_steady.py
产出：
  data/summary_steady.csv     （在 summary.csv 基础上多了 steady_* 两列）
"""

import json
import sys

import pandas as pd

import tc_config as cfg

RAW_REQ = cfg.DATA_DIR / "raw_requests.jsonl"


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if not RAW_REQ.exists():
        raise SystemExit(f"找不到 {RAW_REQ}——逐请求明细是重算的前提")

    per = {}
    with RAW_REQ.open(encoding="utf-8") as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not r.get("ok") or not r.get("total_ms"):
                continue
            key = (r["cell_id"], r["repeat"])
            per.setdefault(key, []).append(r["total_ms"])

    rows = []
    for (cid, rep), tot in per.items():
        rows.append(dict(cell_id=cid, repeat=rep, n_req=len(tot),
                         mean_total_ms=sum(tot) / len(tot)))
    df = pd.DataFrame(rows)
    # 每格取各轮中位数（与其他指标一致）
    agg = df.groupby("cell_id", as_index=False).median(numeric_only=True)

    summary = pd.read_csv(cfg.DATA_DIR / "summary.csv")
    m = summary.merge(agg[["cell_id", "mean_total_ms", "n_req"]], on="cell_id",
                      how="left", suffixes=("", "_req"))

    # 稳态吞吐：只看单请求生命周期，与波次无关
    gens = m["gen_tokens"].fillna(cfg.GEN_TOKENS) if "gen_tokens" in m.columns else cfg.GEN_TOKENS
    m["steady_throughput_tps"] = m["conc"] * gens / (m["mean_total_ms"] / 1e3)
    m["steady_cost_gpu_s_per_1m"] = 1e6 / m["steady_throughput_tps"]
    m["wave_count"] = m["n_req"] / m["conc"]
    m["wall_vs_steady_pct"] = (m["throughput_tps"] / m["steady_throughput_tps"] - 1) * 100

    out = cfg.DATA_DIR / "summary_steady.csv"
    m.to_csv(out, index=False, encoding="utf-8-sig")

    print(f"{'cell':<28}{'波次':>6}{'墙钟吞吐':>9}{'稳态吞吐':>9}{'偏差':>8}")
    for _, r in m.sort_values(["model", "quant", "ctx", "conc"]).iterrows():
        print(f"{r['cell_id']:<28}{r['wave_count']:>6.2f}{r['throughput_tps']:>9.0f}"
              f"{r['steady_throughput_tps']:>9.0f}{r['wall_vs_steady_pct']:>7.0f}%")
    print(f"\n已写入 {out}")
    print("偏差 > 10% 的那些格，就是墙钟口径被收尾空闲污染的格。")


if __name__ == "__main__":
    main()
