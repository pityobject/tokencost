"""
tc_eval.py —— 质量评估：不同模型/量化到底差多少？（项目的第三根轴）

为什么必须有这一步？
  没有质量约束，"最便宜"的答案永远是"最小的模型"。那这个推荐器就是个废话机器。
  真实工程里，成本、延迟、质量是一个三角：
    想省钱 → 换小模型 / 降量化 → 质量掉
    想保质 → 换大模型 → 更慢更贵
  所以必须把质量测出来，才能做真正有意义的取舍。

做法：
  用 60 道**可自动判分**的题目（算术、比较、排序、倒序、计数），
  对每个 模型×量化 组合跑一遍，算准确率。
  题目由代码按固定随机种子生成，期望答案由 Python 直接算出，所以判分绝对客观。

用法：
  & "D:\\llm\\venv\\Scripts\\python.exe" tc_eval.py            # 跑全部 5 个组合
  & "D:\\llm\\venv\\Scripts\\python.exe" tc_eval.py --only 3B q4_k_m
"""

import argparse
import asyncio
import json
import random
import re
import time

import httpx

import tc_config as cfg
import tc_server as srv

EVAL_CTX = 2048          # 评测用的上下文（题目很短，不需要大上下文）
EVAL_MAX_TOKENS = 64
# 60 题时单个准确率的 95% 置信区间半宽约 ±12 个百分点；把题目加到 120–180 题可降到 ±8。
WILSON_NOISE = 0.12


def wilson_ci(k, n, z=1.96):
    """Wilson 区间：小样本下比正态近似更靠得住，不会给出负的下界。

    为什么必须给区间：60 题里 55 题对（91.7%）和 57 题对（95%）的差距，
    很可能只是"这 6 道题恰好蒙对/蒙错"，不足以支撑"3B-Q8 比 3B-Q4 好"的结论。
    """
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    center = (p + z * z / (2 * n)) / d
    half = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / d
    return max(0.0, center - half), min(1.0, center + half)

_WORDS = ["machine", "strawberry", "language", "computer", "learning",
          "network", "protocol", "harbin", "student", "quantify"]


def build_items(n_per_kind=10, seed=20261019):
    """生成评测题。期望答案全部由 Python 算出，不靠人写。"""
    rng = random.Random(seed)
    items = []

    # 【提示词纪律】不要用"最后的数字"——大模型会理解成"最后一位数字"（1173 → 3）。
    # 统一改成"只输出一个阿拉伯数字作为答案"，并把格式要求写死在句子里。
    for _ in range(n_per_kind):
        a, b = rng.randint(120, 980), rng.randint(120, 980)
        items.append(dict(kind="add",
                          prompt=f"计算 {a} + {b} 的精确结果，只输出一个阿拉伯数字，不要写算式或解释。",
                          expected=str(a + b)))

    for _ in range(n_per_kind):
        a, b = rng.randint(12, 99), rng.randint(12, 99)
        items.append(dict(kind="mul",
                          prompt=f"计算 {a} × {b} 的精确结果，只输出一个阿拉伯数字，不要写算式或解释。",
                          expected=str(a * b)))

    for _ in range(n_per_kind):
        a, b = rng.randint(100, 999), rng.randint(100, 999)
        items.append(dict(kind="cmp",
                          prompt=f"在 {a} 和 {b} 这两个数里，哪一个更大？只输出那个更大的数字，不要解释。",
                          expected=str(max(a, b))))

    for w in _WORDS:
        items.append(dict(kind="reverse",
                          prompt=f"把英文单词 {w} 的字母顺序完全颠倒后输出。"
                                 f"只输出颠倒后的单词本身，不要解释、不要加引号。",
                          expected=w[::-1]))

    for w in _WORDS:
        ch = rng.choice([c for c in "rstn"])
        items.append(dict(kind="count",
                          prompt=f"在英文单词 {w} 中，字母 {ch} 出现了几次？"
                                 f"只输出一个阿拉伯数字，不要解释。",
                          expected=str(w.count(ch))))

    for _ in range(n_per_kind):
        nums = rng.sample(range(10, 100), 4)
        srt = sorted(nums)
        items.append(dict(kind="sort",
                          prompt=f"把这四个数从小到大排序：{', '.join(map(str, nums))}。"
                                 f"只输出用英文逗号分隔的结果，不要空格、不要解释。",
                          expected=",".join(map(str, srt))))

    # 给每题一个稳定编号，供配对检验使用
    for i, it in enumerate(items):
        it["idx"] = i
    return items


def grade(item, answer):
    """判分。宽容一点：只看关键内容，不管格式。"""
    if not answer:
        return False
    text = answer.strip().lower()
    kind, exp = item["kind"], item["expected"]

    if kind in ("add", "mul", "cmp", "count"):
        # 取回答里的第一个整数
        m = re.search(r"-?\d+", text.replace(",", "").replace("，", ""))
        return bool(m) and m.group(0) == exp

    if kind == "reverse":
        return exp.lower() in text

    if kind == "sort":
        got = re.findall(r"\d+", text)
        return got == exp.split(",")

    return False


async def ask(client, model, prompt):
    payload = {
        "model": "local",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": EVAL_MAX_TOKENS,
        "temperature": 0.0,
        "stream": False,          # 评测不需要时间戳，用非流式最省事
    }
    r = await client.post(f"{cfg.BASE_URL}/v1/chat/completions", json=payload, timeout=120)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


async def eval_model(key, items, verbose=True):
    url = f"{cfg.BASE_URL}"
    async with httpx.AsyncClient(timeout=180) as client:
        # 预热一次，避免第一题被 kernel 编译拖慢（不影响正确性，只是快）
        await ask(client, key, "你好")
        correct, details = 0, []
        for i, it in enumerate(items):
            if verbose and (i + 1) % 20 == 0:
                print(f"    {i+1}/{len(items)}")
            try:
                ans = await ask(client, key, it["prompt"])
            except Exception as e:
                ans = ""
                if verbose:
                    print(f"    第{i+1}题请求失败：{e}")
            ok = grade(it, ans)
            correct += ok
            details.append(dict(iid=f"{it['kind']}#{it.get('idx', i)}",
                                kind=it["kind"], prompt=it["prompt"],
                                expected=it["expected"], answer=ans, ok=ok))
    return correct / len(items), details


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs=2, metavar=("MODEL", "QUANT"))
    ap.add_argument("--n-per-kind", type=int, default=10)
    ap.add_argument("--models", default="0.5B,1.5B,3B,7B",
                    help="只评测这些模型（逗号分隔）")
    args = ap.parse_args()

    cfg.ensure_dirs()
    items = build_items(n_per_kind=args.n_per_kind)
    print(f"评测集：{len(items)} 题（{sorted(set(i['kind'] for i in items))}）")

    if args.only:
        keys = [(args.only[0], args.only[1])]
    else:
        want = {m.strip() for m in args.models.split(",") if m.strip()}
        keys = [k for k in cfg.MODEL_SPECS if k[0] in want]
    rows, all_details = [], []

    for key in keys:
        print(f"\n=== 评测 {key[0]} {key[1]} ===")
        t0 = time.time()
        proc = None
        try:
            proc, _ = srv.start(key, EVAL_CTX, 1, tag=f"eval_{key[0]}_{key[1]}")
            acc, details = asyncio.run(eval_model(key, items))
            k = int(round(acc * len(items)))
            lo, hi = wilson_ci(k, len(items))
            # 同时存点估计与 Wilson 下界：推荐器用下界做质量约束，
            # 因为"高估一个模型的质量"比"低估"危险得多。
            rows.append(dict(model=key[0], quant=key[1], n=len(items),
                             correct=k, acc=round(acc, 4),
                             acc_lo=round(lo, 4), acc_hi=round(hi, 4)))
            for d in details:
                all_details.append(dict(model=key[0], quant=key[1], **d))
            print(f"  准确率 {acc*100:.1f}%（耗时 {time.time()-t0:.0f}s）")
        except Exception as e:
            rows.append(dict(model=key[0], quant=key[1], n=len(items),
                             correct=0, acc=float("nan"), error=str(e)[:200]))
            print(f"  失败：{e}")
        finally:
            if proc is not None:
                srv.stop(proc)

    # 合并写入（支持分次跑）
    out = cfg.DATA_DIR / "quality.csv"
    old = []
    if out.exists():
        import pandas as pd
        old = pd.read_csv(out).to_dict("records")
    import pandas as pd
    merged = pd.DataFrame(old + rows).drop_duplicates(subset=["model", "quant"], keep="last")
    order = {m: i for i, m in enumerate(["1.5B", "3B", "7B"])}
    merged = merged.sort_values(by=["model", "quant"], key=lambda s: s.map(order) if s.name == "model" else s)
    merged.to_csv(out, index=False, encoding="utf-8-sig")

    det = cfg.DATA_DIR / "quality_details.jsonl"
    with det.open("a", encoding="utf-8") as f:
        for d in all_details:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")

    print("\n" + "=" * 60)
    print(merged.to_string(index=False))
    print("=" * 60)
    print(f"质量表：{out}")
    print("逐题明细：", det, "（写报告时可以从中挑几个错例）")
    if len(merged) > 1:
        best = merged.loc[merged["acc"].idxmax()]
        b_lo, b_hi = wilson_ci(int(best["correct"]), int(best["n"]))
        print(f"\n最高准确率：{best['model']}-{best['quant']} {best['acc']*100:.1f}%  "
              f"95%CI [{b_lo*100:.0f}%, {b_hi*100:.0f}%]")
        for _, r in merged.iterrows():
            lo, hi = wilson_ci(int(r["correct"]), int(r["n"]))
            drop = (best["acc"] - r["acc"]) * 100
            # 【单位纪律】drop 是"百分点"，wilson_ci 返回的是"比例"（0–1）。
            # 判"是否可分辨"要用区间是否重叠，而不是拿百分点去比小数——
            # 早先这里写成 drop < WILSON_NOISE(0.12)，导致 5 个百分点就被误判成"超出噪声"。
            overlap = (hi >= b_lo)                      # 与最高分的区间是否重叠
            flag = "" if overlap else "  ← 与最高分区间不重叠（可分辨）"
            print(f"  {r['model']:>4} {r['quant']:<7} {r['acc']*100:5.1f}%  "
                  f"95%CI [{lo*100:.0f}%, {hi*100:.0f}%]  "
                  f"比最好低 {drop:4.1f} 个百分点{flag}")

        n_items = int(merged["n"].max())
        half = (b_hi - b_lo) / 2 * 100
        print(f"\n判读纪律：{n_items} 题下最高分的区间半宽约 ±{half:.0f} 个百分点。")
        print("  两个组合的 95% 区间重叠时，不能下「A 比 B 好」的结论，")
        print("  只能说「在这批题上没测出差异」。要缩窄噪声就把 --n-per-kind 提到 20–30。")


if __name__ == "__main__":
    main()
