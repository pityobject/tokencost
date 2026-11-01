"""
tc_probe.py —— 开跑实验前的"体检"：实测每个槽位到底能用多少上下文。

为什么需要它（这是本项目最容易翻车的地方）：
  llama-server 的 --ctx-size 是**整个服务端的总上下文**，服务端会按 --parallel 平分给各槽位。
  如果你只传 --ctx-size 512 却开 32 路并发，每槽位实际只有 16 个 token，
  所有请求都会被截断（truncated=true），测出来的 TTFT/TPOT 全是废数据。

  不同版本的 llama.cpp 在"统一 KV / 分槽 KV"上的行为有过变化，所以**不能靠猜**。
  本脚本的做法是：启动服务 → 发一条接近 ctx 长度的提示 → 看服务端自己回报的
  tokens_evaluated / tokens_cached / truncated，用实测把这件事钉死。

用法：
  & "D:\\llm\\venv\\Scripts\\python.exe" tc_probe.py                 # 探测若干代表性格
  & "D:\\llm\\venv\\Scripts\\python.exe" tc_probe.py --only 3B q4_k_m 2048 16
输出：data/probe.csv
"""

import argparse
import time

import httpx

import tc_config as cfg
import tc_server as srv

FILLER = ("上下文测试句子，用于把提示撑到指定长度，请不要真正回答问题，只需输出一个字符。")


def make_long_prompt(target_tokens: int, nonce: int = 0) -> str:
    """造一条指定长度的提示。

    加 nonce 的原因：探针会先后发"短提示"和"长提示"。如果长提示以短提示为前缀，
    服务端会把前半段当成缓存命中（tokens_cached ≈ 全长），证据就不干净了。
    加一个唯一前缀，保证两次都是全新内容。
    """
    chars = int(target_tokens / 0.65)          # 中文约 0.65 token/字
    body = (FILLER * (chars // len(FILLER) + 1))[:chars]
    return f"[{nonce}] {body}"


def probe_once(client, target_tokens, nonce=0):
    """打一次 /completion，读服务端回报的真实 token 数与是否被截断。"""
    r = client.post(f"{cfg.BASE_URL}/completion", json={
        "prompt": make_long_prompt(target_tokens, nonce),
        "n_predict": 1,
        "temperature": 0.0,
        "cache_prompt": False,       # 关掉缓存，让 tokens_cached 接近 0，证据才干净
    }, timeout=300)
    r.raise_for_status()
    j = r.json()
    return dict(
        requested_tokens=target_tokens,
        tokens_evaluated=j.get("tokens_evaluated"),
        tokens_cached=j.get("tokens_cached"),
        truncated=j.get("truncated"),
    )


def probe_cell(key, ctx, conc):
    proc = None
    try:
        proc, log = srv.start(key, ctx, conc, tag=f"probe_{key[0]}_{key[1]}_c{conc}")
        with httpx.Client(timeout=300) as c:
            props = {}
            try:
                props = c.get(f"{cfg.BASE_URL}/props", timeout=10).json()
            except Exception:
                pass
            time.sleep(1.0)
            short = probe_once(c, 64, nonce=1)
            long_ = probe_once(c, int(ctx * 0.7), nonce=2)
        row = dict(
            cell_id=cfg.cell_id(key[0], key[1], ctx, conc),
            model=key[0], quant=key[1], ctx_per_slot=ctx, conc=conc,
            ctx_total_requested=cfg.total_ctx(ctx, conc),
            server_n_ctx=(props.get("default_generation_settings") or {}).get("n_ctx"),
            server_total_slots=props.get("total_slots"),
            short_evaluated=short["tokens_evaluated"],
            long_evaluated=long_["tokens_evaluated"],
            long_truncated=long_["truncated"],
            long_cached=long_["tokens_cached"],
            ok=bool(long_["truncated"] is False and (long_["tokens_evaluated"] or 0) >= int(ctx * 0.5)),
        )
        return row
    except Exception as e:
        return dict(cell_id=cfg.cell_id(key[0], key[1], ctx, conc),
                    model=key[0], quant=key[1], ctx_per_slot=ctx, conc=conc,
                    ctx_total_requested=cfg.total_ctx(ctx, conc),
                    ok=False, error=f"{type(e).__name__}: {e}"[:200])
    finally:
        if proc is not None:
            srv.stop(proc)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs=4, metavar=("MODEL", "QUANT", "CTX", "CONC"))
    ap.add_argument("--all", action="store_true", help="对每个可行格都探一遍（很慢）")
    args = ap.parse_args()

    cfg.ensure_dirs()
    if args.only:
        m, q, c, n = args.only
        cells = [(m, q, int(c), int(n))]
    elif args.all:
        cells = [(c["model"], c["quant"], c["ctx"], c["conc"])
                 for c in cfg.all_cells() if c["feasible"]]
    else:
        # 默认只探最关键的 4 格：最高并发、最长上下文、以及两个极端组合
        cells = [("3B", "q4_k_m", 2048, 1),
                 ("3B", "q4_k_m", 2048, 16),
                 ("7B", "q4_k_m", 512, 32),
                 ("3B", "q4_k_m", 8192, 4)]

    rows = []
    for (m, q, ctx, conc) in cells:
        print(f"\n=== 探测 {m} {q} 每槽位 ctx={ctx} 并发={conc} "
              f"(总上下文 {cfg.total_ctx(ctx, conc)}) ===")
        row = probe_cell((m, q), ctx, conc)
        rows.append(row)
        if row.get("error"):
            print(f"  失败：{row['error']}")
        else:
            print(f"  服务端 n_ctx={row['server_n_ctx']}  槽位数={row['server_total_slots']}")
            print(f"  服务端 n_ctx = 每槽位上下文（= 传入的总上下文 ÷ 槽位数）")
            print(f"  短提示评估 {row['short_evaluated']} tokens")
            print(f"  长提示评估 {row['long_evaluated']} tokens  "
                  f"truncated={row['long_truncated']}")
            print("  （tokens_cached 的服务端口径不稳定，只进 CSV、不作判据；"
                  "判断只看 truncated 和实际评估到的 token 数）")
            print(f"  结论：{'每槽位上下文够用' if row['ok'] else '[注意] 每槽位上下文不足，这一格的数据不可用'}")

    import pandas as pd
    out = cfg.DATA_DIR / "probe.csv"
    pd.DataFrame(rows).to_csv(out, index=False, encoding="utf-8-sig")
    print(f"\n结果已写入 {out}")
    if all(r.get("ok") for r in rows):
        print("全部通过：可以放心跑 tc_run_grid.py")
    else:
        print("有格没通过。先把 tc_server.py 里的 --ctx-size 检查一遍（必须是 ctx × 并发）。")


if __name__ == "__main__":
    main()
