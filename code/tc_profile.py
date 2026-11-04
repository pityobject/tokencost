# -*- coding: utf-8 -*-
"""
tc_profile.py —— 一条命令，在你的服务上标定六参数模型

## 它解决什么

`tc_calibrate.py` 要求你手工准备 8 个测量点的 JSON。这个脚本把前面那一步也自动化：
**指向任何 OpenAI 兼容端点**（llama.cpp / vLLM / SGLang / TGI / Ollama），
它自己发请求、测 TTFT 与 TPOT、拟合、写出可直接给推荐器用的标定文件。

    python tc_profile.py --base-url http://127.0.0.1:8080/v1 --spec 3B:q4_k_m
    python tc_recommend.py --calib ../results/calib_profile.json --ttft 600 --tpot 60

## 测量计划（默认 8 点，约 5–10 分钟）

| 并发 | prompt 长度 | 用途 |
|---|---|---|
| 1 / 4 / 16 / 32 | 512 | 定 `c_srv`、`B_SAT`（并发维度） |
| 1 / 4 | 2048 | 定 `PEAK_pre`（长度维度） |
| 1 | 128 | 定 `ovh`（首字固定开销，越短越干净） |
| 8 | 512 | 中间并发（本项目的弱区，特意留一格） |

**为什么必须跨至少 4 档并发**：并发只有 1 和 8 时，`c_srv` 与 `ovh` 在数学上无法分辨
（见《自标定指南》里 20.0% vs 8.1% 的对比）。

## 模型规格从哪来

`predict()` 需要 `weight_bytes / n_params / n_layers / n_kv_heads / head_dim`：

- `--spec 3B:q4_k_m`：复用仓库里已核对的规格（Qwen2.5 各尺寸 / Llama-3.2-1B）
- `--spec-json my.json`：自己填（格式同 `tc_calibrate.py` 的 `model` 段）

## 自检

    python tc_profile.py --selftest

用仓库标定值合成带噪测量点，再拟合回去，验证整条链路（不需要服务器）。
"""

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

import httpx
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import tc_config as cfg
import tc_model as tm

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

DEFAULT_PLAN = [
    (1, 512), (4, 512), (16, 512), (32, 512),
    (1, 2048), (4, 2048), (1, 128), (8, 512),
]


# ---------------------------------------------------------------- 规格
def resolve_spec(spec_key=None, spec_json=None):
    """返回 (model_name, quant, spec_dict)。"""
    if spec_json:
        d = json.loads(Path(spec_json).read_text(encoding="utf-8"))
        m = d.get("model", d)
        need = ["name", "quant", "weight_bytes", "n_params",
                "n_layers", "n_kv_heads", "head_dim"]
        miss = [k for k in need if k not in m]
        if miss:
            raise SystemExit(f"spec 缺字段：{miss}")
        return m["name"], m["quant"], m
    if not spec_key or ":" not in spec_key:
        raise SystemExit("需要 --spec MODEL:QUANT（如 3B:q4_k_m）或 --spec-json FILE")
    name, quant = spec_key.split(":", 1)
    key = (name, quant)
    if key not in cfg.MODEL_SPECS:
        raise SystemExit(
            f"仓库里没有 {key} 的规格。可选：{sorted(cfg.MODEL_SPECS)}\n"
            f"或改用 --spec-json 自己填（weight_bytes 用 GGUF 文件实际字节数）")
    spec = cfg.MODEL_SPECS[key]
    return name, quant, dict(name=name, quant=quant, **spec)


# ---------------------------------------------------------------- 测一格的并发请求
def make_prompt(target_tokens):
    """构造约 target_tokens 个 token 的提示（真实 token 数由服务端 usage 回填）。"""
    unit = "请仔细阅读下面这段材料，记住其中的细节。"
    # 中文约 1 字 ≈ 1 token，留一点余量
    n = max(1, int(target_tokens * 0.95 / max(len(unit) * 0.9, 1)))
    body = unit * n
    return (f"{body}\n\n以上材料的最后一句是「记住其中的细节」。"
            f"请只回复 4 个字：我已阅读。")


async def one_request(client, base_url, model, prompt, max_tokens, timeout=180):
    """返回 dict(ttft_ms, total_ms, n_out, prompt_tokens)。流式取首字时间。"""
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    t0 = time.perf_counter()
    first = None
    n_out = 0
    usage = {}
    async with client.stream("POST", f"{base_url}/chat/completions",
                             json=payload, timeout=timeout) as r:
        r.raise_for_status()
        async for line in r.aiter_lines():
            if not line or not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            try:
                obj = json.loads(data)
            except json.JSONDecodeError:
                continue
            if obj.get("usage"):
                usage = obj["usage"]
            for ch in obj.get("choices", []):
                txt = (ch.get("delta") or {}).get("content")
                if txt:
                    if first is None:
                        first = time.perf_counter()
                    n_out += 1
    t_end = time.perf_counter()
    if first is None:
        first = t_end
    n_out = int(usage.get("completion_tokens") or n_out or 1)
    return dict(ttft_ms=(first - t0) * 1e3,
                total_ms=(t_end - t0) * 1e3,
                n_out=n_out,
                prompt_tokens=int(usage.get("prompt_tokens") or 0))


async def measure_cell(client, base_url, model, conc, target_L, n_requests, gen,
                       warmup=2):
    """在并发 conc 下打 n_requests 个请求（信号量限流，保持 b 路在飞）。"""
    prompt = make_prompt(target_L)
    for _ in range(warmup):
        try:
            await one_request(client, base_url, model, prompt, min(gen, 16))
        except Exception:
            pass
    sem = asyncio.Semaphore(conc)
    results = []

    async def task():
        async with sem:
            try:
                return await one_request(client, base_url, model, prompt, gen)
            except Exception as e:
                return dict(error=str(e)[:120])

    out = await asyncio.gather(*[task() for _ in range(n_requests)])
    results = [r for r in out if "error" not in r]
    if not results:
        raise SystemExit(f"并发 {conc} 的所有请求都失败：{out[0].get('error')}")
    ttfts = [r["ttft_ms"] for r in results]
    tpots = [(r["total_ms"] - r["ttft_ms"]) / max(r["n_out"] - 1, 1) for r in results]
    Ls = [r["prompt_tokens"] for r in results if r["prompt_tokens"] > 0]
    return dict(conc=conc, target_L=target_L,
                L=float(np.median(Ls)) if Ls else float(target_L),
                ttft=float(np.median(ttfts)),
                tpot=float(np.median(tpots)),
                n_ok=len(results), n_fail=len(out) - len(results))


# ---------------------------------------------------------------- 拟合
def fit_from_measurements(model_name, quant, spec, cells):
    cfg.MODEL_SPECS[(model_name, quant)] = dict(
        files=spec.get("files", [f"{model_name}.gguf"]),
        weight_bytes=int(spec["weight_bytes"]),
        n_params=float(spec["n_params"]),
        n_layers=int(spec["n_layers"]),
        n_kv_heads=int(spec["n_kv_heads"]),
        head_dim=int(spec["head_dim"]),
    )
    rows = []
    for i, c in enumerate(cells):
        rows.append(dict(
            cell_id=f"profile_{i}", model=model_name, quant=quant,
            ctx=int(c.get("ctx", 4096)), conc=int(c["conc"]),
            prompt_tokens=float(c["L"]),
            ctx_avg_tokens=float(c["L"]) + cfg.GEN_TOKENS / 2,
            ttft_p50_ms=float(c["ttft"]), tpot_p50_ms=float(c["tpot"]),
            cost_gpu_s_per_1m=np.nan,
        ))
    df = pd.DataFrame(rows)
    params = tm.fit(df)
    return params, df


def identifiability(params, df, model_name, quant):
    """decode 算力是否可识别：有没有任何一格的算力时间超过访存时间。"""
    spec = cfg.MODEL_SPECS[(model_name, quant)]
    bound = 0
    for _, r in df.iterrows():
        kv = cfg.kv_bytes_per_token(spec) * r["ctx_avg_tokens"] * r["conc"]
        t_mem = (spec["weight_bytes"] + kv) / (params["bw_gbs"] * 1e9)
        t_cmp = 2 * spec["n_params"] * r["conc"] / (params["peak_tflops"] * 1e12)
        if t_cmp > t_mem:
            bound += 1
    return bound > 0


# ---------------------------------------------------------------- 主流程
def cmd_measure(args):
    model_name, quant, spec = resolve_spec(args.spec, args.spec_json)
    plan = DEFAULT_PLAN
    if args.concs and args.lengths:
        concs = [int(x) for x in args.concs.split(",")]
        lens = [int(x) for x in args.lengths.split(",")]
        plan = [(c, L) for c in concs for L in lens]

    print("=" * 74)
    print(f"自标定：{args.base_url}  模型 {model_name}:{quant}")
    print("=" * 74)
    print(f"  测量计划 {len(plan)} 格：{plan}")
    print(f"  每格 {args.requests} 个请求，生成长度 {args.gen}\n")

    # 【服务端配置校验】llama.cpp 的 --ctx-size 是按 --parallel 平分的，
    # 所以"每槽够用"要求：--ctx-size >= per_slot_ctx × slots。
    max_conc = max(c for c, _ in plan)
    max_L = max(L for _, L in plan)
    need_per_slot = max_L + args.gen
    problems = []
    if max_conc > args.slots:
        problems.append(f"计划里有并发 {max_conc}，但服务端 --parallel 只有 {args.slots}")
    if need_per_slot > args.per_slot_ctx:
        problems.append(f"最长 prompt {max_L}+生成 {args.gen} = {need_per_slot} token，"
                        f"超过每槽 {args.per_slot_ctx}")
    if problems:
        print("  ⚠️ 服务端配置与测量计划不匹配：")
        for x in problems:
            print(f"     · {x}")
        print(f"     推荐启动命令：python tc_server.py <model> <quant> "
              f"{need_per_slot} {max_conc}")
        print(f"     （该命令会传 --ctx-size {need_per_slot * max_conc}、--parallel {max_conc}）")
        print(f"      总 KV ≈ {need_per_slot * max_conc * 36.9e3 / 1e9:.1f} GB（3B 量级），"
              f"8 GB 卡需留意是否进悬崖）\n")
    else:
        print(f"  ✅ 服务端配置匹配：每槽 {args.per_slot_ctx} ≥ 需要 {need_per_slot}，"
              f"槽位 {args.slots} ≥ 最大并发 {max_conc}\n")

    async def run():
        async with httpx.AsyncClient(timeout=300) as client:
            # 探测服务：先确认连得上，再（可选）问模型名。
            # /v1/models 不是所有服务都实现（llama.cpp 有、部分 mock 没有），
            # 所以它失败不该中断——只影响请求里的 model 字段。
            wire_model = args.wire_model or "local"
            try:
                r = await client.get(f"{args.base_url}/models", timeout=10)
                if r.status_code == 200:
                    ids = [m.get("id") for m in r.json().get("data", [])]
                    if ids and not args.wire_model:
                        wire_model = ids[0]
                    print(f"  服务可用，模型列表：{ids[:3]}")
                else:
                    print(f"  /models 返回 {r.status_code}，用 model='{wire_model}' 继续")
            except httpx.ConnectError as e:
                raise SystemExit(f"连不上 {args.base_url}：{e}")
            except Exception as e:
                print(f"  跳过 /models（{type(e).__name__}），用 model='{wire_model}' 继续")
            cells = []
            for conc, L in plan:
                t0 = time.time()
                c = await measure_cell(client, args.base_url, wire_model, conc, L,
                                       args.requests, args.gen)
                c["ctx"] = c.get("ctx", args.ctx)
                cells.append(c)
                print(f"  c{conc:<3} L目标{L:<5} → 实测 L={c['L']:.0f}  "
                      f"TTFT={c['ttft']:7.1f} ms  TPOT={c['tpot']:6.2f} ms  "
                      f"({c['n_ok']}/{c['n_ok']+c['n_fail']} 成功, {time.time()-t0:.0f}s)")
            return cells

    cells = asyncio.run(run())
    return model_name, quant, spec, cells, args


def cmd_fit(model_name, quant, spec, cells, args):
    params, df = fit_from_measurements(model_name, quant, spec, cells)

    print("\n  残差：")
    print(f"    {'配置':<14}{'TTFT实测':>10}{'预测':>9}{'误差':>8}"
          f"{'TPOT实测':>10}{'预测':>9}{'误差':>8}")
    et, ep = [], []
    for _, r in df.iterrows():
        pr = tm.predict(params, r)
        e1 = pr["pred_ttft_ms"] / r["ttft_p50_ms"] - 1
        e2 = pr["pred_tpot_ms"] / r["tpot_p50_ms"] - 1
        et.append(abs(e1))
        ep.append(abs(e2))
        print(f"    c{int(r['conc'])}/L{int(r['prompt_tokens']):<8}"
              f"{r['ttft_p50_ms']:>9.1f}ms{pr['pred_ttft_ms']:>8.1f}ms{e1*100:>7.0f}%"
              f"{r['tpot_p50_ms']:>9.2f}ms{pr['pred_tpot_ms']:>8.2f}ms{e2*100:>7.0f}%")

    ok = identifiability(params, df, model_name, quant)
    print(f"\n  标定结果（TTFT MAPE {np.mean(et)*100:.1f}% / TPOT MAPE {np.mean(ep)*100:.1f}%）：")
    print(f"    有效带宽 BW      {params['bw_gbs']:8.1f} GB/s")
    print(f"    decode 算力      {params['peak_tflops']:8.2f} TFLOP/s"
          + ("" if ok else "   ⚠️ 不可识别（本计划内 decode 始终访存受限，勿引用）"))
    print(f"    prefill 算力     {params['peak_prefill_tflops']:8.2f} TFLOP/s")
    print(f"    服务开销 c_srv   {params['c_srv_ms']:8.3f} ms/流/步")
    print(f"    prefill 饱和点   {params['b_sat']:8.1f}")
    print(f"    首字固定开销     {params['ovh_ttft_ms']:8.2f} ms")

    out = Path(args.out) if args.out else (cfg.RESULTS_DIR / "calib_profile.json")
    out.write_text(json.dumps(dict(
        hardware=args.hardware, backend=args.base_url,
        model=dict(spec), n_points=len(cells), measurements=cells,
        bw=params["bw_gbs"], peak=params["peak_tflops"],
        peak_prefill=params["peak_prefill_tflops"], c_srv=params["c_srv_ms"],
        b_sat=params["b_sat"], ovh_ttft=params["ovh_ttft_ms"],
        residual=dict(ttft_mape=float(np.mean(et) * 100),
                      tpot_mape=float(np.mean(ep) * 100)),
        decode_peak_identifiable=bool(ok),
    ), ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n已写入 {out}")
    print(f"下一步： python tc_recommend.py --calib {out} --ttft 600 --tpot 60")


# ---------------------------------------------------------------- 自检（不需要服务器）
def cmd_selftest(args):
    """用仓库标定值合成带噪测量点，再拟合回去：验证整条链路正确。"""
    fit = json.loads((cfg.RESULTS_DIR / "fit.json").read_text(encoding="utf-8"))
    truth = dict(bw_gbs=fit["bw"], peak_tflops=fit["peak"],
                 peak_prefill_tflops=fit.get("peak_prefill", fit["peak"]),
                 c_srv_ms=fit["c_srv"], b_sat=fit["b_sat"],
                 ovh_ttft_ms=fit["ovh_ttft"])
    model_name, quant, spec = resolve_spec("3B:q4_k_m")
    print("=" * 74)
    print("tc_profile 自检：合成数据 → 拟合 → 比对真值")
    print("=" * 74)
    rng = np.random.default_rng(20261105)
    cells = []
    for conc, L in DEFAULT_PLAN:
        row = dict(model=model_name, quant=quant, ctx=8192, conc=conc,
                   prompt_tokens=float(L), ctx_avg_tokens=L + cfg.GEN_TOKENS / 2)
        pr = tm.predict(truth, row)
        noise = lambda: float(np.exp(rng.normal(0, 0.02)))   # 2% 乘性噪声
        cells.append(dict(conc=conc, L=float(L), ctx=8192,
                          ttft=pr["pred_ttft_ms"] * noise(),
                          tpot=pr["pred_tpot_ms"] * noise(),
                          n_ok=16, n_fail=0))

    params, df = fit_from_measurements(model_name, quant, spec, cells)
    print(f"\n  {'参数':<20}{'真值':>12}{'拟合':>12}{'相对差':>10}")
    worst = 0.0
    for k, v in truth.items():
        got = params[k]
        rel = (got / v - 1) * 100
        if k in ("bw_gbs", "c_srv_ms", "b_sat", "peak_prefill_tflops"):
            worst = max(worst, abs(rel))
        print(f"  {k:<20}{v:>12.3f}{got:>12.3f}{rel:>+9.1f}%")
    print(f"\n  最大相对差（关键 4 参数）= {worst:.1f}%")
    if worst < 15:
        print("  ✅ 自检通过：链路正确，能在 8 个测量点上把参数找回来。")
        return 0
    print("  ⚠️ 自检未通过：链路或拟合有问题，先别去测真机。")
    return 1


def main():
    ap = argparse.ArgumentParser(description="一条命令在任意 OpenAI 兼容服务上标定")
    ap.add_argument("--base-url", default=None, help="如 http://127.0.0.1:8080/v1")
    ap.add_argument("--spec", default=None, help="复用仓库规格，如 3B:q4_k_m")
    ap.add_argument("--spec-json", default=None, help="自定义规格 JSON")
    ap.add_argument("--wire-model", default=None, help="请求里带的模型名（默认取 /models 第一个）")
    ap.add_argument("--concs", default=None, help="自定义并发，如 1,4,16,32")
    ap.add_argument("--lengths", default=None, help="自定义 prompt 长度，如 512,2048")
    ap.add_argument("--requests", type=int, default=16, help="每格请求数")
    ap.add_argument("--gen", type=int, default=64, help="每请求生成长度")
    ap.add_argument("--ctx", type=int, default=8192, help="每槽上下文（记录用）")
    ap.add_argument("--slots", type=int, default=16,
                    help="服务端 --parallel（必须 >= 计划里的最大并发）")
    ap.add_argument("--per-slot-ctx", type=int, default=2304,
                    help="服务端每槽上下文（必须 >= 最大 prompt + 生成长度）")
    ap.add_argument("--hardware", default="（未填）")
    ap.add_argument("--out", default=None)
    ap.add_argument("--selftest", action="store_true", help="不连服务器，验证链路")
    args = ap.parse_args()

    if args.selftest:
        return cmd_selftest(args)
    if not args.base_url:
        ap.error("需要 --base-url，或用 --selftest")
    if not args.spec and not args.spec_json:
        ap.error("需要 --spec 或 --spec-json")
    model_name, quant, spec, cells, args = cmd_measure(args)
    cmd_fit(model_name, quant, spec, cells, args)
    return 0


if __name__ == "__main__":
    sys.exit(main())


