"""
tc_backend_ref.py —— 测"纯后端上限"，用来量化"实现效率"。

为什么需要它：
  我们的网格测的是 llama-server 这条完整路径（HTTP → 采样 → SSE 推流）。
  它测出来的吞吐，是"部署后的真实成本"，但不等于"显卡的能力"。
  两者之差，就是工程实现的效率损失——这恰恰是面试官最想听的那部分。

  这个脚本直接调用 llama.cpp 自带的 llama-batched-bench（不经过 HTTP），
  测出并发 1/4/16/32 下的纯后端吞吐，存进 data/backend_ref.csv。
  之后 tc_analyze 可以拿它和 server 实测对比，算出"服务路径达成率"。

实测参考（1.5B-Q4，本机 2026-10-31）：
  纯后端 178 / 550 / 1682 / 2389 t/s（并发 1/4/16/32）
  server 165 / 420 /  753 /  940 t/s  → 达成率 93% / 76% / 45% / 39%

用法：
  & "D:\\llm\\venv\\Scripts\\python.exe" tc_backend_ref.py
  & "D:\\llm\\venv\\Scripts\\python.exe" tc_backend_ref.py --models 1.5B,3B
"""

import argparse
import csv
import re
import subprocess
import sys

import tc_config as cfg

# llama-batched-bench 的输出行形如：
# |   296 |    128 |   32 |  13568 |    1.000 |  9472.10 |    1.715 |  2388.97 |    2.715 |  4998.28 |
ROW = re.compile(r"\|\s*(\d+)\s*\|\s*(\d+)\s*\|\s*(\d+)\s*\|")


def run_one(key, npl_list="1,4,16,32", npp=296, ntg=128, ctx=None):
    # 总 KV 至少要能装下 max(npl) x (npp+ntg) 个 token，否则 llama_decode 会失败
    if ctx is None:
        ctx = max(int(x) for x in npl_list.split(',')) * (npp + ntg) + 512
        ctx = min(ctx, 32768)

    model = cfg.model_path(key)
    if not model.exists():
        print(f"  跳过 {key}：模型文件不存在")
        return []

    batched = cfg.LLAMA_BIN / "llama-batched-bench.exe"
    if not batched.exists():
        print(f"  找不到 {batched}，跳过")
        return []
    cmd = [str(batched),
           "-m", str(model), "-ngl", "99",
           "-c", str(ctx), "-npp", str(npp), "-ntg", str(ntg), "-npl", npl_list]

    print(f"\n=== {key[0]} {key[1]} ===")
    print("  命令：" + " ".join(str(c) for c in cmd))
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=900,
                             cwd=str(cfg.LLAMA_BIN))
    except Exception as e:
        print(f"  失败：{type(e).__name__}: {e}")
        return []

    text = (out.stdout or "") + (out.stderr or "")
    rows = []
    for line in text.splitlines():
        m = ROW.match(line.strip())
        if not m:
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) < 10:
            continue
        try:
            b = int(cells[2]); s_pp = float(cells[5]); s_tg = float(cells[7])
        except ValueError:
            continue
        rows.append(dict(model=key[0], quant=key[1],
                         batch=b, pp_tokens=int(cells[0]), tg_tokens=int(cells[1]),
                         backend_pp_tps=s_pp, backend_tg_tps=s_tg))
        print(f"  并发 {b:>2}: prefill {s_pp:7.0f} t/s   decode {s_tg:7.1f} t/s")
    if not rows:
        print("  没解析到数据。原始输出尾部：")
        print("\n".join(text.splitlines()[-6:]))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="", help="只测指定模型，逗号分隔，如 1.5B,3B")
    ap.add_argument("--npl", default="1,4,16,32")
    args = ap.parse_args()

    cfg.ensure_dirs()
    keys = [k for k in cfg.MODEL_SPECS
            if not args.models or k[0] in {m.strip() for m in args.models.split(",")}]

    all_rows = []
    for key in keys:
        all_rows += run_one(key, npl_list=args.npl)

    if not all_rows:
        print("\n没有任何结果。先确认 llama-batched-bench.exe 在 D:\\llm\\bin 里。")
        sys.exit(1)

    out = cfg.DATA_DIR / "backend_ref.csv"
    old = []
    if out.exists():
        with out.open(encoding="utf-8-sig") as f:
            old = [r for r in csv.DictReader(f)
                   if (r["model"], r["quant"]) not in {(x["model"], x["quant"]) for x in all_rows}]

    with out.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        w.writeheader()
        for r in old:
            w.writerow(r)
        for r in all_rows:
            w.writerow(r)

    print(f"\n已写入 {out}")
    print("下一步：跑 tc_analyze.py，它会自动把 server 实测与这个上限对比，算出达成率。")


if __name__ == "__main__":
    main()

