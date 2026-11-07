#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A10 + vLLM 全网格（自包含版，不依赖仓库任何模块）

用法（在云端 Web 终端里）：
  python3 vllm_grid.py --smoke      # 先冒烟：1 格
  python3 vllm_grid.py              # 24 格

只做三件事：起 vLLM（已在跑就跳过）→ 测 24 格 → 写出 grid.json
拟合与判定在本地做（tc.py vllmgrid）。
"""
import argparse
import asyncio
import json
import os
import statistics as st
import subprocess
import sys
import time
from pathlib import Path

CONCS = [1, 2, 4, 8, 12, 16, 24, 32]
CTXS = [(512, 297), (2048, 297), (8192, 1973)]
GEN = 128
PORT = 8000
BASE = f"http://127.0.0.1:{PORT}/v1"


def find_model():
    import glob
    for pat in ("/mnt/workspace/tc_a10/models/*3B*", "/mnt/workspace/models/*3B*",
                "/root/models/*3B*", "/mnt/workspace/*/models/*3B*", "/models/*3B*"):
        for d in glob.glob(pat):
            if os.path.isdir(d) and any(f.endswith(".safetensors") or
                                        f == "config.json" for f in os.listdir(d)):
                return d
    return None


def ensure_server(model_path):
    import httpx
    try:
        r = httpx.get(f"{BASE}/models", timeout=3)
        if r.status_code == 200:
            print(f"  vLLM 已在运行：{r.json()['data'][0]['id']}")
            return True
    except Exception:
        pass
    print(f"  启动 vLLM（模型 {model_path}）…")
    subprocess.run(["pkill", "-f", "vllm.entrypoints"], check=False)
    time.sleep(3)
    log = open("/tmp/vllm_grid.log", "w")
    subprocess.Popen([sys.executable, "-m", "vllm.entrypoints.openai.api_server",
                      "--model", model_path, "--served-model-name", "local",
                      "--port", str(PORT), "--host", "127.0.0.1",
                      "--max-model-len", "16384", "--max-num-seqs", "32",
                      "--gpu-memory-utilization", "0.85", "--disable-log-requests"],
                     stdout=log, stderr=log)
    for i in range(60):
        try:
            if httpx.get(f"{BASE}/models", timeout=3).status_code == 200:
                print(f"  就绪（约 {i*5}s）")
                return True
        except Exception:
            pass
        time.sleep(5)
    print("  启动超时：tail -50 /tmp/vllm_grid.log")
    return False


async def one_request(cli, model, prompt_len, gen):
    """发一个请求，返回 (ttft_ms, tpot_ms)。TTFT 用首 token 时间戳。"""
    body = {"model": model, "stream": True,
            "messages": [{"role": "user", "content": "The quick brown fox jumps over the lazy dog. " * (prompt_len // 9 + 1)}],
            "max_tokens": gen, "temperature": 0.0, "ignore_eos": True,
            "stream_options": {"include_usage": True}}
    t0 = time.perf_counter()
    ttft = None
    n = 0
    usage_pt = 0
    async with cli.stream("POST", f"{BASE}/chat/completions", json=body) as r:
        async for line in r.aiter_lines():
            if not line.startswith("data: "):
                continue
            payload = line[6:]
            if payload.strip() == "[DONE]":
                break
            try:
                d = json.loads(payload)
            except Exception:
                continue
            if d.get("usage"):
                usage_pt = int(d["usage"].get("prompt_tokens") or 0)
            ch = (d.get("choices") or [{}])[0].get("delta") or {}
            if ch.get("content"):
                if ttft is None:
                    ttft = (time.perf_counter() - t0) * 1e3
                n += 1
    total = (time.perf_counter() - t0) * 1e3
    if ttft is None or n < 2:
        return None
    tpot = (total - ttft) / max(n - 1, 1)
    return ttft, tpot, usage_pt


async def measure(model, conc, prompt_len, rounds=2):
    import httpx
    async with httpx.AsyncClient(timeout=300, limits=httpx.Limits(
            max_connections=conc + 4, max_keepalive_connections=conc + 4)) as cli:
        ttfts, tpots = [], []
        prompt_len_actual = prompt_len
        for _ in range(rounds):
            res = await asyncio.gather(*[
                one_request(cli, model, prompt_len, GEN) for _ in range(conc)])
            ok = [r for r in res if r]
            if not ok:
                continue
            ttfts += [r[0] for r in ok]
            tpots += [r[1] for r in ok]
            if ok[0][2]:
                prompt_len_actual = ok[0][2]
        if not ttfts:
            return None
        return dict(conc=conc, L=float(prompt_len_actual),
                    ttft=round(st.median(ttfts), 3),
                    tpot=round(st.median(tpots), 4),
                    n_ok=len(ttfts), gen=GEN)


async def main_async(args):
    model = args.wire_model or "local"
    if not ensure_server(args.model_path):
        return 1
    cells = []
    todo = [(512, 297, [1])] if args.smoke else [(c, L, CONCS) for c, L in CTXS]
    for ctx, L, concs in todo:
        for conc in concs:
            print(f"  ctx={ctx:<5} conc={conc:<3} L={L} …", end=" ", flush=True)
            r = await measure(model, conc, L)
            if r:
                cells.append(r)
                print(f"ttft={r['ttft']:.0f}ms tpot={r['tpot']:.1f}ms")
            else:
                print("失败")
    out = Path(args.out)
    out.write_text(json.dumps(dict(hardware="A10 24GB / vLLM", cells=cells),
                              ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n→ {out}（{len(cells)} 格）")
    print("把下面这行 JSON 整段复制回来给我：")
    print(json.dumps(cells, ensure_ascii=False))
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--model-path", default=None)
    ap.add_argument("--wire-model", default="local")
    ap.add_argument("--out", default="grid.json")
    args = ap.parse_args()
    if not args.model_path:
        args.model_path = find_model()
        if not args.model_path:
            print("没找到 3B 模型目录。用 --model-path 指定，例如：")
            print("  python3 vllm_grid.py --model-path /mnt/workspace/tc_a10/models/Qwen2.5-3B-Instruct")
            return 1
    print(f"模型目录：{args.model_path}")
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    sys.exit(main())
