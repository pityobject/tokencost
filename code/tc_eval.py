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



# ---------------------------------------------------------------- 第二任务族
# 【为什么要加】原来的 6 种题型全是"算术/字符串操作"，任务族单一。
# 加这两族是为了回答"质量轴是不是只对算术有效"这个质疑——
# 格式遵循测的是"听不听话"，应用题测的是"多步推理"，都不是纯计算。
def build_format_items(n=30, seed=20261103):
    """格式遵循：答案对不够，还要守格式。这是线上服务最常见的失败模式。"""
    rng = random.Random(seed)
    items = []
    for i in range(n):
        variant = i % 5
        a, b = rng.randint(20, 89), rng.randint(11, 59)
        if variant == 0:
            items.append(dict(
                kind="format", check="json_n",
                prompt=f"计算 {a} + {b}。只输出 JSON，形如 {{\"n\": 数字}}，不要任何解释。",
                expected=str(a + b)))
        elif variant == 1:
            items.append(dict(
                kind="format", check="maxlen",
                prompt=f"用不超过 8 个汉字回答：{a} 和 {b} 哪个大？只给答案。",
                expected="8"))
        elif variant == 2:
            items.append(dict(
                kind="format", check="upper",
                prompt="把单词 token 全部大写输出，不要标点。",
                expected="TOKEN"))
        elif variant == 3:
            items.append(dict(
                kind="format", check="bullets",
                prompt="列出 3 个可以用 GPU 加速的场景，每条以 - 开头，一行一条。",
                expected="3"))
        else:
            items.append(dict(
                kind="format", check="no_explain",
                prompt=f"{a} 乘以 {b} 等于多少？只输出数字，一个字都不要多说。",
                expected=str(a * b)))
    return items


def build_word_items(n=30, seed=20261103):
    """多步应用题：需要两到三步推理，答案唯一。比纯算术更接近真实使用。"""
    rng = random.Random(seed)
    items = []
    for i in range(n):
        variant = i % 3
        a, b, c = rng.randint(12, 40), rng.randint(3, 9), rng.randint(2, 7)
        if variant == 0:
            ans = a * b - c
            prompt = (f"一个仓库每天入库 {a} 箱货物，连续 {b} 天，"
                      f"期间出库了 {c} 箱。现在仓库里增加了多少箱？只输出数字。")
        elif variant == 1:
            # 【必须与语义一致】"还剩多少" = 原有的 − 花掉的。
            # 原来写成 a + b*c，题干和答案对不上（17 元买 4 本 6 元 → 负数），
            # 导致所有模型几乎全错——那是**题错**，不是模型差。
            spent = b * c
            a = spent + rng.randint(5, 20)      # 保证钱够花，答案是正数
            ans = a - spent
            prompt = (f"小红有 {a} 元，她买了 {b} 本笔记本，每本 {c} 元。"
                      f"还剩多少元？只输出数字。")
        else:
            ans = (a - c) * b
            prompt = (f"一条流水线原来每小时生产 {a} 件，改造后每小时少 {c} 件，"
                      f"运行 {b} 小时共生产多少件？只输出数字。")
        items.append(dict(kind="word", check="number",
                          prompt=prompt, expected=str(ans)))
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

    # ---- 第二任务族 ----
    if kind == "word":
        m = re.search(r"-?\d+", text.replace(",", ""))
        return bool(m) and m.group(0) == exp

    if kind == "format":
        chk = item.get("check")
        if chk == "json_n":
            m = re.search(r'"n"\s*:\s*(-?\d+)', text)
            return bool(m) and m.group(1) == exp
        if chk == "number":
            m = re.search(r"-?\d+", text.replace(",", ""))
            return bool(m) and m.group(0) == exp
        if chk == "maxlen":
            body = re.sub(r"[\s，。.,!！?？]", "", answer.strip())
            return len(body) <= int(exp)
        if chk == "upper":
            letters = re.sub(r"[^A-Za-z]", "", answer)
            return letters == exp and answer.strip().upper().startswith("TOKEN")
        if chk == "bullets":
            lines = [ln for ln in answer.splitlines() if ln.strip().startswith("-")]
            return len(lines) >= int(exp)
        if chk == "no_explain":
            body = re.sub(r"[\s，。.,!！?？]", "", answer.strip())
            return body == exp

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
    ap.add_argument("--kinds", default="default",
                    help="default=原 6 种题型；或 format,word（第二任务族）")
    ap.add_argument("--out", default=None,
                    help="输出 CSV；默认 quality.csv（第二任务族务必另指定）")
    ap.add_argument("--only", nargs=2, metavar=("MODEL", "QUANT"))
    ap.add_argument("--n-per-kind", type=int, default=10)
    ap.add_argument("--models", default="0.5B,1.5B,3B,7B",
                    help="只评测这些模型（逗号分隔）")
    args = ap.parse_args()

    cfg.ensure_dirs()
    if args.kinds == "default":
        items = build_items(n_per_kind=args.n_per_kind)
    else:
        # 第二任务族：只生成被点名的题型，题量与原来同量级（每题独立、确定性生成）
        want = [k.strip() for k in args.kinds.split(",") if k.strip()]
        items = []
        if "format" in want:
            items += build_format_items(n=args.n_per_kind * 3)
        if "word" in want:
            items += build_word_items(n=args.n_per_kind * 3)
        if not items:
            raise SystemExit(f"--kinds 里没有可用题型：{want}")
        # 稳定编号（配对检验要用）
        for i, it in enumerate(items):
            it["idx"] = i
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
    out = (cfg.DATA_DIR / args.out) if args.out else (cfg.DATA_DIR / "quality.csv")
    old = []
    if out.exists():
        import pandas as pd
        old = pd.read_csv(out).to_dict("records")
    import pandas as pd
    merged = pd.DataFrame(old + rows).drop_duplicates(subset=["model", "quant"], keep="last")
    order = {m: i for i, m in enumerate(["1.5B", "3B", "7B"])}
    merged = merged.sort_values(by=["model", "quant"], key=lambda s: s.map(order) if s.name == "model" else s)
    merged.to_csv(out, index=False, encoding="utf-8-sig")

    # 【不要污染主基准】指定了 --out（第二任务族）时，明细写到同名 _details.jsonl；
    # 否则新题型的明细会混进 400 题基准的 details 里，把配对检验的数据源搞乱。
    if args.out:
        det = (cfg.DATA_DIR / args.out).with_name(
            (cfg.DATA_DIR / args.out).stem + "_details.jsonl")
    else:
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


