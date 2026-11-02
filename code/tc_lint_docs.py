# -*- coding: utf-8 -*-
"""
tc_lint_docs.py —— "已作废数字"扫描器

## 设计说明（第一版是错的）

第一版想"验证文档里每一个数字都能在产物里找到"，结果 45 条输出里全是误报
（92%、4.3 倍、238% 都是合法的派生量）。**全量校验在文档里不可行。**

于是改成精确定位：维护一份"已作废数字"清单（DEPRECATED），
每个条目记录**它当年错在哪、现在应该是多少**。文档里只要再出现这些数字就报警。

这份清单恰好就是本项目三次翻车的教训（口径混用 / 题集扩容 / 采样不足 / 公式不一致）：

| 作废值 | 为什么作废 | 现在应为 |
|---|---|---|
| 6.13% | 旧口径单人独占用墙钟 | 8.70% / 11.28% |
| 140 题 | 质量基准扩容 | 400 题 |
| 9.9 倍 / 24.9 ms / +25% | 泊松 30 秒采样噪声 | 13.0 倍 / 25.2 ms / +27% |
| 268% / 813% | 后悔值半宽公式与推荐器不一致（36.5 倍） | 274% / 740% |
| 259% / 772% | 候选集未统一到 rank() | 274% / 740% |
| 6pp → 7pp | 阈值精度不足 | 7.43 pp / 9.59 pp |
| 1.1 pp | bootstrap 独立重采样（错） | 配对重采样后宽度≈0 |

## 定位

**warning 级**：只提示，不失败。历史留档（审计记录、复核意见、附录）本来就要保留旧值，
所以只看 `TARGETS` 里那几个"对外文档"。

用法：
    python tc_lint_docs.py
    python tc_lint_docs.py --strict   # 命中就 exit 1
"""

import argparse
import sys

import tc_config as cfg

ROOT = cfg.PROJECT_ROOT

# 对外文档（历史留档不在其中）
TARGETS = [
    "results/CLAIMS.md",      # 自称"唯一数字源"，更要扫（第六轮复核指出）
    "README.md",
    "报告/技术报告_定稿_两页.md",
    "报告/简历条目与答辩稿_实测版.md",
    "报告/英文摘要与一页报告.md",
]

# 已作废的数字 → (当年为什么错, 现在应该是多少)
DEPRECATED = {
    "6.13%":         ("旧口径（单人独占用墙钟）", "8.70% 稳态 / 11.28% 墙钟"),
    "19.46%":        ("早于 6 参数模型", "8.70%"),
    "100 格":        ("数据规模变化", "108 格"),
    "76 格":         ("留出口径变化", "68 格（core）"),
    "140 题":        ("质量基准扩容", "400 题"),
    "9.9 倍":        ("泊松 30 秒采样噪声", "13.0 倍（λ=2）"),
    "24.9 ms":       ("同上", "25.2 ms"),
    "+25%":          ("同上", "+27%"),
    "268%":          ("后悔值半宽公式不一致（差 36.5 倍）", "274%"),
    "813%":          ("同上", "740%"),
    "259%":          ("候选集未统一到 rank()", "274%"),
    "772%":          ("同上", "740%"),
    "6pp → 7pp":     ("阈值精度不足", "7.43 pp / 9.59 pp"),
    "成本差 36%":     ("同上", "20847→4919→2182"),
    "1.1 pp":        ("bootstrap 独立重采样（方法错误）", "配对重采样后宽度≈0"),
}


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("--strict", action="store_true")
    args = ap.parse_args()

    hits = []
    for rel in TARGETS:
        p = ROOT / rel
        if not p.exists():
            continue
        for i, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
            # 历史叙述里的旧值是合法的（"上一版 140 题……"这种对比句）
            if any(h in line for h in
                   ("上一版", "曾经", "旧值", "历史", "此前", "早先", "→", "改为")):
                continue
            for bad, (why, now) in DEPRECATED.items():
                if bad in line:
                    hits.append((rel, i, bad, why, now, line.strip()[:58]))

    if not hits:
        print(f"[LINT][OK] 对外文档里没有出现已作废数字（检查了 {len(TARGETS)} 份）。")
        return

    print(f"[LINT][WARN] 命中 {len(hits)} 处已作废数字：")
    for rel, i, bad, why, now, ctx in hits:
        print(f"  {rel}:{i}")
        print(f"    发现「{bad}」—— {why}；现在应为「{now}」")
        print(f"    上下文：{ctx}")
    print("\n  这是 warning 级检查，不影响 CI 通过。")
    if args.strict:
        sys.exit(1)


if __name__ == "__main__":
    main()
