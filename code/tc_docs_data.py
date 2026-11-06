# -*- coding: utf-8 -*-
"""
C2：自动生成数据字典（`报告/数据字典.md`）

## 为什么需要

现在有 18 个数据文件 + 48 个结果文件。外人（或半年后的自己）打开 `results/`
会看到一堆 `csrv_linearity.csv`、`tail_flip.csv`，不知道：
- 每列是什么、什么单位；
- 由哪条命令生成；
- 有没有口径陷阱（比如 steady vs wall 两列成本）。

## 做法

**从产物现读**（列名、类型、示例值都自动抓），再叠一份**人工维护的释义表**
（列名 → 含义/单位）。这样：
- 列清单永远和真实文件一致（不会漏列）；
- 释义只需要维护一份映射；
- 生成结果进 CI 比对，防止文档过期。

## 用法

    python tc_docs_data.py            # 生成 报告/数据字典.md
    python tc_docs_data.py --check    # 只校验是否与当前产物一致
"""
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import tc_config as cfg

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "报告" / "数据字典.md"

# ---------------------------------------------------------------- 释义表
COL_DOC = {
    # raw_cells.jsonl
    "cell_id": "配置格唯一标识：模型_量化_上下文档_并发行（+后缀）",
    "status": "ok / failed；failed 也保留（诚实记录）",
    "repeat": "同一格的轮次；**换批会从 0 重新开始**，跨批分析须按回退切分",
    "conc": "并发（同时活跃的槽位数 b）",
    "ctx": "每槽上下文上限（token）；总上下文 = ctx × conc",
    "requests": "该格发出的请求总数",
    "gen_tokens": "每请求生成 token 数",
    "prompt_tokens_nominal": "名义提示长度（实际以 prompt_tokens_median 为准）",
    "prompt_tokens_median": "服务端返回的提示 token 中位数（真实值）",
    "ctx_avg_tokens": "解码时平均上下文 = 提示 + 生成/2",
    "ttft_p50_ms": "首字延迟中位数（ms）",
    "tpot_p50_ms": "出字间隔中位数（ms）",
    "throughput_tps": "吞吐（token/s），= 总生成 token ÷ 墙钟",
    "cost_gpu_s_per_1m": "**墙钟**口径成本（GPU·秒/百万 token），含收尾空闲",
    "steady_cost_gpu_s_per_1m": "**稳态**口径成本；`tc_model` 用这一列，对外 8.59% 基于它",
    "vram_est_gb": "显存估计（GB）= 权重 + KV",
    "clock_sm_median": "SM 时钟中位（MHz）；相对可达参考判据见 clock_gate",
    "power_w_median": "整卡功耗中位（W）",
    "temp_c_median": "温度中位（°C）",
    "bg_level": "背景占用判定：ok / warn / bad / unknown（unknown≠干净）",
    "bg_mem_mb": "起服务前显存已用（MB，含 Windows 桌面）",
    "bg_util_p95": "起服务前 GPU 利用率 p95（%）",
    "clock_gate": "时钟相对判据：ok / warn / bad / unknown",
    "clock_ratio": "SM 时钟 ÷ 该模型历史最高时钟",
    "length_mix": "混合长度分布（如 256:0.7000,1024:0.2000），仅 _mix 格有",
    "turns": "多轮轮数，仅 _mtN 格有",
    "error": "失败原因（failed 行）",
    # raw_requests.jsonl
    "req_id": "请求序号",
    "n_out": "该请求实际生成 token 数",
    "ttft_ms": "单请求首字延迟（ms）",
    "total_ms": "到最后一个 token 的时间（ms）",
    "wall_ms": "到连接关闭的时间（ms）——TPOT 不能用它算（曾因此偏高 2–3 倍）",
    "t_start_rel": "相对整轮实验起点的开始时刻（s）",
    "t_end_rel": "相对整轮实验起点的结束时刻（s）",
    "ok": "该请求是否成功",
    # summary_steady.csv / summary.csv
    "ttft_p90_ms": "首字延迟 p90（ms）",
    "tpot_p90_ms": "出字间隔 p90（ms）",
    "measure_span_s": "纯测量窗口长度（s）",
    "per_slot_tps": "单槽吞吐（token/s）",
    "total_tokens": "该格总生成 token 数",
    "client_cpu_pct": "客户端单核 CPU 占用（%）；>90% 说明可能是脚本限制",
    # fit.json
    "bw_gbs": "有效带宽（GB/s）；**吸收了反量化/内核开销，不是纯硬件常数**",
    "peak_tflops": "decode 算力（TFLOP/s）；**结构性不可辨识**（见 decode_bound.json）",
    "peak_prefill_tflops": "prefill 算力（TFLOP/s）",
    "c_srv_ms": "每流每步服务开销（ms）；对并发**线性假设只在两端成立**",
    "b_sat": "prefill 饱和并发数",
    "ovh_ttft_ms": "首字固定开销（ms）",
    "mape_cost": "留出集成本 MAPE（%）——core 口径的稳态值",
    "mape_cost_wall": "墙钟口径 MAPE（%）",
    "max_err": "留出集最大成本误差（%）",
    "coverage": "90% 预测区间覆盖率（比例，0–1）",
    "half_width_log": "预测区间半宽（对数尺度）",
    "n_holdout": "留出格数",
    "calc_bound_cells": "标定集里算力受限的格数（用于可辨识性判断）",
    # quantile_tails.csv
    "ttft_p90_over_p50": "实测 TTFT 的 p90/p50（反映抖动）",
    "tpot_p90_over_p50": "实测 TPOT 的 p90/p50",
    "ttft_p99_over_p50": "实测 TTFT 的 p99/p50",
    "ttft_sigma_log": "TTFT 的对数标准差",
    # csrv_linearity.csv
    "implied_csrv": "反推的每步开销 = (TPOT − 硬件下限)/b（ms）",
    "service": "服务时间 = TPOT − 硬件下限（ms）",
    "floor": "硬件下限 = max(访存时间, 算力时间)（ms）",
    # tail_flip.csv
    "factor_ttft": "实测尾部因子（p90/p50）",
    "ttft_assumed": "假设口径的 90% 上界（ms）",
    "ttft_measured": "实测口径的 90% 上界（ms）",
    "pass_assumed": "假设口径是否通过 SLA",
    "pass_measured": "实测口径是否通过 SLA",
    # clock_audit.csv
    "clock_mhz": "该格 SM 时钟中位（MHz）",
    "clock_ref_mhz": "该模型的可达参考时钟（历史最高）",
    # cv_splits.csv
    "mape_cost": "这次划分下的留出 MAPE（%）",
    # mixed_lengths.csv
    "L_cv": "提示长度的变异系数（越大越混合）",
    "L_mean": "实测平均提示长度（token）",
    "ttft_meas_p50": "实测首字延迟中位（ms）",
    # quality.csv
    "acc": "该组合在质量基准上的准确率",
    "n_items": "题目数",
    "acc_lo": "Wilson 95% 下界",
}

FILE_DOC = {
    "raw_cells.jsonl": ("每格每轮一行（含失败行）", "python code/tc.py run / steady"),
    "raw_requests.jsonl": ("每请求一行（分位数/尾部分析的基础）", "同上（--save-requests 默认开）"),
    "summary_steady.csv": ("每格汇总（稳态口径），模型的输入", "python code/tc.py steady"),
    "summary.csv": ("每格汇总（墙钟口径）", "python code/tc.py analyze"),
    "quality.csv": ("质量基准（400 题）逐组合准确率", "python code/tc.py eval"),
    "results/fit.json": ("六参数 + 留出集指标", "python code/tc.py model"),
    "results/populations.json": ("core/valid/all 三套口径的指标", "python code/tc.py model"),
    "results/quantile_tails.csv": ("实测延迟分位数（尾部证据）", "python code/tc.py tails"),
    "results/cv_summary.json": ("随机重划分的 MAPE 分布", "python code/tc.py cv"),
    "results/csrv_linearity.csv": ("逐并发反推 c_srv（线性假设检验）", "python code/tc.py csrv"),
    "results/decode_bound.json": ("decode 算术强度与硬件脊点（可辨识性证明）", "python code/tc.py decodebound"),
    "results/tail_flip.csv": ("两种尾部口径下的 SLA 判定（翻转统计）", "python code/tc.py tailflip"),
    "results/clock_audit.csv": ("逐格时钟相对判定", "python code/tc.py clockaudit"),
    "results/mixed_lengths.csv": ("混合长度 vs 平均长度近似", "python code/tc.py mixed"),
    "results/multiturn.csv": ("多轮逐轮前缀与延迟", "python code/tc.py multiturn"),
    "results/bimodal_*.json": ("批次/热状态溯源", "python code/tc.py bimodal"),
}


def describe(col):
    return COL_DOC.get(col, "")


def table_for_csv(path, max_cols=40):
    df = pd.read_csv(path)
    lines = ["| 列 | 类型 | 示例 | 含义 |", "|---|---|---|---|"]
    for c in list(df.columns)[:max_cols]:
        s = df[c].dropna()
        sample = str(s.iloc[0])[:28] if len(s) else ""
        lines.append(f"| `{c}` | {df[c].dtype} | {sample} | {describe(c)} |")
    if len(df.columns) > max_cols:
        lines.append(f"| … | | | 还有 {len(df.columns) - max_cols} 列 |")
    return "\n".join(lines)


def table_for_json(path, max_keys=40):
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    lines = ["| 字段 | 示例 | 含义 |", "|---|---|---|"]
    n = 0
    for k, v in d.items():
        if isinstance(v, (dict, list)) and n >= max_keys:
            lines.append(f"| `{k}` | … | （嵌套结构，见文件） |")
            continue
        sample = str(v)[:40]
        lines.append(f"| `{k}` | {sample} | {describe(k)} |")
        n += 1
    return "\n".join(lines)


def build():
    parts = ["# 数据字典", "",
             "> 本文件由 `python code/tc_docs_data.py` **自动生成**（列清单来自真实产物），",
             "> 由 CI 比对，过期即失败。共 18 个数据文件 + results/ 产物。", "",
             "## 口径提醒（最容易踩的两条）", "",
             "1. **成本有两列**：`cost_gpu_s_per_1m`（墙钟）与 `steady_cost_gpu_s_per_1m`（稳态）。",
             "   `tc_model` 用后者；混用会让 MAPE 从 8.59% 变成 11.28%（F15）。",
             "2. **`repeat` 是局部编号**：同一格换批会从 0 重来，跨批分析须按回退切分（F11）。",
             ""]

    data_dir = cfg.DATA_DIR
    for name in sorted(p.name for p in data_dir.iterdir() if p.is_file()):
        if not (name.endswith(".csv") or name.endswith(".jsonl")):
            continue
        path = data_dir / name
        title, gen = FILE_DOC.get(name, ("", ""))
        parts.append(f"## data/{name}")
        parts.append("")
        if title:
            parts.append(f"**内容**：{title}")
        if gen:
            parts.append(f"**生成**：`{gen}`")
        parts.append("")
        try:
            if name.endswith(".jsonl"):
                first = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
                n_lines = sum(1 for _ in path.open(encoding="utf-8"))
                parts.append(f"**行数**：{n_lines}")
                parts.append("")
                parts.append("| 字段 | 示例 | 含义 |")
                parts.append("|---|---|---|")
                for k, v in first.items():
                    parts.append(f"| `{k}` | {str(v)[:34]} | {describe(k)} |")
            else:
                parts.append(table_for_csv(path))
        except Exception as e:
            parts.append(f"（解析失败：{type(e).__name__}: {e}）")
        parts.append("")

    res = cfg.RESULTS_DIR
    parts += ["## results/ 产物", "",
              "| 文件 | 内容 | 生成命令 |", "|---|---|---|"]
    for p in sorted(res.iterdir()):
        if not p.is_file() or p.suffix not in (".csv", ".json", ".md"):
            continue
        rel = f"results/{p.name}"
        key = p.name
        meta = FILE_DOC.get(rel) or FILE_DOC.get(key)
        if key.endswith(".md"):
            continue
        parts.append(f"| `{key}` | {meta[0] if meta else '—'} | "
                     f"`{meta[1]}` |" if meta else f"| `{key}` | — | — |")

    parts += ["", "## 关键产物的列释义（逐个展开）", ""]
    for key in ("results/fit.json", "results/populations.json",
                "results/quantile_tails.csv", "results/csrv_linearity.csv",
                "results/tail_flip.csv", "results/clock_audit.csv",
                "results/cv_splits.csv", "results/mixed_lengths.csv"):
        p = ROOT / key
        if not p.exists():
            continue
        parts.append(f"### `{key}`")
        parts.append("")
        try:
            if key.endswith(".json"):
                parts.append(table_for_json(p))
            else:
                parts.append(table_for_csv(p, max_cols=14))
        except Exception as e:
            parts.append(f"（解析失败：{e}）")
        parts.append("")

    return "\n".join(parts) + "\n"


def main():
    check = "--check" in sys.argv
    body = build()
    if check:
        old = OUT.read_text(encoding="utf-8") if OUT.exists() else ""
        if old.strip() != body.strip():
            print("[FAIL] 数据字典与产物不一致——重新生成后提交")
            return 1
        print("[ OK ] 数据字典与产物一致")
        return 0
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(body, encoding="utf-8")
    print(f"已生成 {OUT}（{len(body.splitlines())} 行）")
    return 0


if __name__ == "__main__":
    sys.exit(main())


