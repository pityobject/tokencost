# -*- coding: utf-8 -*-
"""
多轮对话测量（`tc.py turnsprobe`）：KV 增长斜率是否跨栈一致

## 为什么需要

本机 E3 的结论：多轮对话**能拟合**（TPOT MAPE 5.2%、TTFT 6.3%），
但模型**低估 KV 增长斜率**（实测 +2.44%/turn vs 预测 +0.53%/turn，差 4.6 倍）。

问题：这是模型形式缺陷，还是 llama.cpp 的 KV cache 管理特性？
在 vLLM 上重测同一协议即可分辨。

## 做法

连续 t=1..N 轮，每轮把上一轮回复追加进历史，测量每轮的：
- `prompt_tokens`（服务端返回，含历史）
- TTFT / TPOT

再用标定参数预测每轮的 TPOT（模型的 KV 项会随 ctx 增长），
比较**实测斜率**与**预测斜率**（%/轮）。

## 用法

    python code/tc_measure_turns.py --base-url http://127.0.0.1:8081/v1 \
        --wire-model qwen3b --backend vllm \
        --calib results/a10_raw/calib_a10_vllm.json \
        --turns 6 --gen 128 --conc 1 \
        --hardware "A10 24GB / vLLM" --out results/turns_a10_vllm.json
"""
import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

import httpx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import tc_config as cfg
import tc_loadgen as lg
import tc_model as tm

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

FILLER = "已完成上一问。"


async def one_turn(client, base_url, model, history, gen, backend):
    payload = {
        "model": model, "messages": history,
        "max_tokens": gen, "temperature": 0.0, "stream": True,
        "stream_options": {"include_usage": True}, "ignore_eos": True,
    }
    if backend != "vllm":
        payload["cache_prompt"] = False
    t0 = time.perf_counter()
    first = last = None
    n_out = 0
    text = []
    usage = {}
    try:
        async with client.stream("POST", f"{base_url}/chat/completions",
                                 json=payload, timeout=300) as r:
            r.raise_for_status()
            async for line in r.aiter_lines():
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if not data or data == "[DONE]":
                    continue
                try:
                    obj = json.loads(data)
                except Exception:
                    continue
                if obj.get("usage"):
                    usage = obj["usage"]
                for ch in obj.get("choices") or []:
                    d = (ch.get("delta") or {}).get("content")
                    if d:
                        now = time.perf_counter()
                        if first is None:
                            first = now
                        last = now
                        n_out += 1
                        text.append(d)
    except Exception as e:
        return None
    if first is None or last is None or n_out < 3:
        return None
    return dict(ttft_ms=(first - t0) * 1e3,
                tpot_ms=(last - first) * 1e3 / (n_out - 1),
                n_out=n_out, reply="".join(text)[:400],
                prompt_tokens=int(usage.get("prompt_tokens") or 0))


async def run_turns(base_url, model, turns, gen, backend, seed_len=200):
    history = [{"role": "user",
                "content": lg.make_prompt(seed_len, nonce=int(time.time()) % 10**9)}]
    rows = []
    async with httpx.AsyncClient() as client:
        for t in range(1, turns + 1):
            r = await one_turn(client, base_url, model, history, gen, backend)
            if not r:
                print(f"  第 {t} 轮失败")
                break
            r["turn"] = t
            rows.append(r)
            print(f"  第 {t} 轮：prompt={r['prompt_tokens']:>5} tok  "
                  f"TTFT={r['ttft_ms']:>8.1f} ms  TPOT={r['tpot_ms']:>6.2f} ms  "
                  f"({r['n_out']} tok)")
            # 把本轮问答追加进历史（助手回复用占位文本，长度≈gen）
            history.append({"role": "assistant",
                            "content": r["reply"] or FILLER * max(1, gen // 8)})
            history.append({"role": "user",
                            "content": f"继续，第 {t+1} 问：" +
                                       lg.make_prompt(120, nonce=t) })
    return rows


def predict(calib, L, conc):
    params = dict(bw_gbs=calib["bw"], peak_tflops=calib["peak"],
                  peak_prefill_tflops=calib["peak_prefill"],
                  c_srv_ms=calib["c_srv"], b_sat=calib["b_sat"],
                  ovh_ttft_ms=calib["ovh_ttft"])
    row = dict(model="3B", quant="fp16", ctx=2304, conc=conc,
               prompt_tokens=L, ctx_avg_tokens=L + cfg.GEN_TOKENS / 2)
    pr = tm.predict(params, row)
    return pr["pred_ttft_ms"], pr["pred_tpot_ms"]


def slope_pct(xs, ys):
    """对 (x, y) 做线性拟合，返回 y 相对初值每单位 x 的变化（%/单位）。"""
    xs = np.asarray(xs, float)
    ys = np.asarray(ys, float)
    if len(xs) < 2 or ys[0] <= 0:
        return float("nan")
    k = float(np.polyfit(xs, ys, 1)[0])
    return k / ys[0] * 100


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--wire-model", default=None)
    ap.add_argument("--backend", choices=["llamacpp", "vllm"], default="vllm")
    ap.add_argument("--calib", required=True)
    ap.add_argument("--turns", type=int, default=6)
    ap.add_argument("--gen", type=int, default=128)
    ap.add_argument("--conc", type=int, default=1)
    ap.add_argument("--hardware", default="（未填）")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    calib = json.loads(Path(args.calib).read_text(encoding="utf-8"))
    spec = calib["model"]
    cfg.MODEL_SPECS[("3B", "fp16")] = dict(
        weight_bytes=spec["weight_bytes"], n_params=spec["n_params"],
        n_layers=spec.get("n_layers", 36), n_kv_heads=spec.get("n_kv_heads", 2),
        head_dim=spec.get("head_dim", 128))

    model = args.wire_model
    if not model:
        d = httpx.get(f"{args.base_url}/models", timeout=10).json()
        entries = d.get("data") or d.get("models") or []
        model = entries[0].get("id") or entries[0].get("model")
    print(f"目标 {args.base_url}｜模型 {model}｜后端 {args.backend}｜{args.turns} 轮")

    rows = asyncio.run(run_turns(args.base_url, model, args.turns,
                                 args.gen, args.backend))
    if len(rows) < 2:
        print("轮数不足，无法算斜率")
        return

    # 预测 + 对比
    for r in rows:
        L = r["prompt_tokens"] or 200
        p_ttft, p_tpot = predict(calib, L, args.conc)
        r["pred_ttft_ms"] = round(p_ttft, 1)
        r["pred_tpot_ms"] = round(p_tpot, 2)
        r["ttft_err_pct"] = round((p_ttft / r["ttft_ms"] - 1) * 100, 1)
        r["tpot_err_pct"] = round((p_tpot / r["tpot_ms"] - 1) * 100, 1)

    turns_x = [r["turn"] for r in rows]
    meas_tpot = [r["tpot_ms"] for r in rows]
    pred_tpot = [r["pred_tpot_ms"] for r in rows]
    s_meas = slope_pct(turns_x, meas_tpot)
    s_pred = slope_pct(turns_x, pred_tpot)

    print(f"\n  {'轮':>2}{'prompt':>8}{'TTFT实测':>10}{'TTFT预测':>10}{'误差':>7}"
          f"{'TPOT实测':>10}{'TPOT预测':>10}{'误差':>7}")
    for r in rows:
        print(f"  {r['turn']:>2}{r['prompt_tokens']:>8}{r['ttft_ms']:>10.1f}"
              f"{r['pred_ttft_ms']:>10.1f}{r['ttft_err_pct']:>6.0f}%"
              f"{r['tpot_ms']:>10.2f}{r['pred_tpot_ms']:>10.2f}{r['tpot_err_pct']:>6.0f}%")

    print(f"\n  TPOT 随轮次的斜率：实测 {s_meas:+.2f}%/轮 vs 预测 {s_pred:+.2f}%/轮"
          f"  → 比值 {s_meas/s_pred:.2f}×" if s_pred else "")
    print("  （本机 E3 结果：实测 +2.44%/轮 vs 预测 +0.53%/轮 = 4.6×）")

    out = dict(hardware=args.hardware, backend=args.backend, model=model,
               turns=args.turns, gen=args.gen, conc=args.conc,
               rows=rows,
               tpot_slope_meas_pct_per_turn=round(s_meas, 3),
               tpot_slope_pred_pct_per_turn=round(s_pred, 3),
               slope_ratio=round(s_meas / s_pred, 2) if s_pred else None,
               note="多轮 KV 增长斜率；与本机 E3 的 4.6× 对比")
    fp = Path(args.out) if args.out else (cfg.RESULTS_DIR / "turns_probe.json")
    fp.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n→ {fp}")


if __name__ == "__main__":
    main()
