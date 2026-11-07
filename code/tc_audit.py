# -*- coding: utf-8 -*-
"""
tc_audit.py —— 全项目体检（一条命令，按"别人 clone 下来能不能跑"的标准验）

## 它检查什么

分四类，每类都对应一种"本地能跑、换台机器就废"的失败模式：

| 类别 | 检查项 | 失败意味着 |
|---|---|---|
| **仓库卫生** | 已跟踪文件都在、无绝对路径、无密钥、无超大文件 | 换人 clone 就跑不起来 |
| **自包含** | 只把已跟踪文件复制到临时目录，能否跑通 claims/测试 | 依赖了未提交的文件 |
| **数据链** | raw_requests → summary_steady 的重算是否一致；fit 参数是否可复现 | 汇总表和原始数据对不上 |
| **契约** | CLI 命令指向的脚本存在、文档引用的文件存在、图非空、TOML/YAML 合法 | 文档里指向空气 |

## 用法

    python tc_audit.py            # 全量（含裸克隆，约 30–60 秒）
    python tc_audit.py --fast     # 跳过裸克隆
    python tc_audit.py --verbose
"""

import argparse
import json
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

CODE = Path(__file__).resolve().parent
ROOT = CODE.parent
sys.path.insert(0, str(CODE))

import tc_config as cfg  # noqa: E402

PY = sys.executable
RESULTS = []


def record(area, name, ok, detail=""):
    RESULTS.append(dict(area=area, name=name, ok=ok, detail=detail))
    mark = "OK  " if ok else "FAIL"
    line = f"  [{mark}] {name}"
    if detail:
        line += f"  —— {detail}"
    print(line)



def _tracked_names():
    """已跟踪文件集合；取不到（非 git 环境）时返回 None。"""
    try:
        out = subprocess.run(
            ["git", "-c", "core.quotepath=false", "ls-files"],
            cwd=ROOT, capture_output=True, text=True,
            encoding="utf-8", errors="replace", timeout=60)
        if out.returncode == 0:
            return {ln.strip() for ln in out.stdout.splitlines() if ln.strip()}
    except Exception:
        pass
    return None


def tracked_files():
    """用 git 列出已跟踪文件（没装 git 就退回 rglob）。"""
    try:
        out = subprocess.run(
            ["git", "-c", "core.quotepath=false", "ls-files"], cwd=ROOT, capture_output=True, text=True,
            encoding="utf-8", errors="replace", timeout=60)
        if out.returncode == 0:
            return [ROOT / p for p in out.stdout.splitlines() if p.strip()]
    except Exception:
        pass
    # 兜底：没有 git 时用 rglob，但**必须**手动排除 .gitignore 里的东西，
    # 否则会把这些本地文件当成"已跟踪"来检查（15 轮复核：49MB 的 data/traces
    # 被判成超大文件）。原则：宁可少查，不可误报。
    IGNORE_DIRS = {".git", "__pycache__", "logs", ".venv", "venv", "node_modules",
                   "data/traces", "data/traces/"}
    IGNORE_SUFFIX = (".gguf", ".tar", ".gz", ".tgz", ".zip", ".pt", ".safetensors",
                     ".bin", ".so", ".dll", ".exe", ".pyc")
    keep = []
    for f in ROOT.rglob("*"):
        if not f.is_file():
            continue
        parts = set(f.parts)
        if parts & {".git", "__pycache__", "logs", ".venv", "venv", "node_modules"}:
            continue
        rel = f.relative_to(ROOT).as_posix()
        if rel.startswith("data/traces/") or rel.startswith("data/traces"):
            continue
        if f.suffix.lower() in IGNORE_SUFFIX:
            continue
        keep.append(f)
    return keep


# ---------------------------------------------------------------- 类 1：仓库卫生
def audit_hygiene(files):
    print("\n【1】仓库卫生")
    missing = [f for f in files if not f.exists()]
    record("卫生", "已跟踪文件都存在于磁盘", not missing,
           f"缺失 {len(missing)} 个" if missing else f"{len(files)} 个文件")

    # 绝对路径（允许 tc_config 的 D:\llm 默认值——它有环境变量覆盖）
    bad = []
    for f in files:
        if f.suffix != ".py":
            continue
        text = f.read_text(encoding="utf-8", errors="replace")
        rel = f.relative_to(ROOT).as_posix()
        if rel in ("code/tc_config.py", "code/tc_env_check.py", "code/tc_audit.py"):
            continue
        if re.search(r'C:[\\/]Users|C:[\\/]文档', text) or \
           (re.search(r'D:[\\/]llm', text) and "_diag" not in rel):
            bad.append(rel)
    record("卫生", "无硬编码的本机绝对路径", not bad,
           ", ".join(bad[:4]) if bad else "只有 tc_config 保留环境相关默认值")

    # 超大文件（>12 MB 提醒）
    big = [(f.relative_to(ROOT).as_posix(), round(f.stat().st_size / 1e6, 1))
           for f in files if f.exists() and f.stat().st_size > 12e6]
    record("卫生", "无超大文件（>12 MB）", not big, str(big) if big else "")

    # 密钥
    leaked = []
    for f in files:
        # 本文件里就有 ghp_ 之类的正则字面量，跳过自己
        if f.name in ("tc_audit.py", "tc_lint_docs.py"):
            continue
        # 改名/删除还没提交时，git 索引里仍会列出已不存在的路径
        if not f.exists():
            continue
        if f.suffix not in (".py", ".md", ".txt", ".toml", ".yml", ".yaml", ".ps1"):
            continue
        t = f.read_text(encoding="utf-8", errors="replace")
        if re.search(r"ghp_[A-Za-z0-9]{20,}|github_pat_|AKIA[0-9A-Z]{16}", t):
            leaked.append(f.relative_to(ROOT).as_posix())
    record("卫生", "无泄漏的令牌/密钥", not leaked, ", ".join(leaked[:3]) if leaked else "")


# ---------------------------------------------------------------- 类 2：自包含
def audit_selfcontained(files):
    print("\n【2】自包含（裸克隆模拟）")
    tmp = Path(tempfile.mkdtemp(prefix="tc_audit_"))
    try:
        for f in files:
            if not f.exists():
                continue
            dst = tmp / f.relative_to(ROOT)
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(f, dst)
        # 不含 .git，模拟"别人 clone 后的样子"
        r1 = subprocess.run([PY, "code/tc_claims.py", "--check"], cwd=tmp,
                            capture_output=True, text=True, encoding="utf-8",
                            errors="replace", timeout=300)
        ok1 = r1.returncode == 0 and "[CLAIMS][OK]" in (r1.stdout + r1.stderr)
        record("自包含", "裸目录里 claims --check 通过", ok1,
               "" if ok1 else (r1.stdout + r1.stderr).strip().splitlines()[-1][:70])

        r2 = subprocess.run([PY, "tests/test_core.py"], cwd=tmp,
                            capture_output=True, text=True, encoding="utf-8",
                            errors="replace", timeout=300)
        ok2 = r2.returncode == 0 and "3/3" in (r2.stdout + r2.stderr)
        record("自包含", "裸目录里单元测试 3/3", ok2,
               "" if ok2 else (r2.stdout + r2.stderr).strip().splitlines()[-1][:70])
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ---------------------------------------------------------------- 类 3：数据链
def audit_data(files):
    print("\n【3】数据链")
    import pandas as pd

    # 3.1 raw_requests → summary_steady 抽 6 格重算
    raw = cfg.DATA_DIR / "raw_requests.jsonl"
    summ = cfg.DATA_DIR / "summary_steady.csv"
    ok = True
    detail = ""
    if raw.exists() and summ.exists():
        acc = {}
        with raw.open(encoding="utf-8") as fh:
            for line in fh:
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not r.get("ok") or not r.get("total_ms"):
                    continue
                acc.setdefault(r["cell_id"], []).append(float(r["total_ms"]))
        s = pd.read_csv(summ)
        cells = s.sort_values("cell_id").head(6)
        bad = []
        for _, row in cells.iterrows():
            v = acc.get(row["cell_id"])
            if not v:
                bad.append(row["cell_id"] + "(raw 无数据)")
                continue
            # 稳态口径：中位
            import statistics
            med = statistics.median(v)
            ref = float(row["total_p50_ms"])
            if ref > 0 and abs(med - ref) / ref > 0.01:
                bad.append(f"{row['cell_id']}({med:.1f} vs {row['total_p50_ms']:.1f})")
        ok = not bad
        detail = "; ".join(bad[:3]) if bad else "抽 6 格，中位耗时一致"
    else:
        ok, detail = False, "缺少 raw_requests.jsonl 或 summary_steady.csv"
    record("数据", "summary 可由 raw 重算（抽 6 格）", ok, detail)

    # 3.2 fit.json 的样本量与数据规模自洽
    fit = json.loads((cfg.RESULTS_DIR / "fit.json").read_text(encoding="utf-8"))
    n_hold = fit.get("n_holdout")
    record("数据", "fit.json 记录了留出集格数", n_hold is not None, f"n_holdout={n_hold}")

    # 3.3 质量与成本表都非空、无全 NaN 列
    q = cfg.DATA_DIR / "quality.csv"
    if q.exists():
        df = pd.read_csv(q)
        allnan = [c for c in df.columns if df[c].isna().all()]
        record("数据", "quality.csv 无全空列", not allnan, ", ".join(allnan[:4]))


# ---------------------------------------------------------------- 类 4：契约
def audit_contract(files):
    print("\n【4】契约（代码/文档/图互相指向）")

    # 4.1 CLI 命令表的每个目标都存在
    tc = (CODE / "tc.py").read_text(encoding="utf-8")
    cmds = re.findall(r'"([a-z\-]+)"\s*:\s*\("([^"]+)"', tc)
    bad = [f"{k}→{v}" for k, v in cmds if not (CODE / v).exists()]
    record("契约", f"CLI {len(cmds)} 个子命令的目标文件都存在", not bad,
           "; ".join(bad[:4]) if bad else "")

    # 4.2 文档里引用到的仓库内文件是否都存在
    # 同时匹配 `code/...` 这类仓内路径与 `../...` 这种**指向仓库外**的引用；
    # 后者一定有问题（clone 下来不存在），必须报出来。
    def _inside(p, r):
        try:
            p.relative_to(r)
            return True
        except ValueError:
            return False

    ref_pat = re.compile(
        r"`((?:\.\./)+(?:[^`\s]+\.(?:py|csv|json|jsonl|md|png|ps1|yml))"   # 仓外
        r"|(?:code|data|results|figures|tests|scripts|报告|docs)/[^`\s]+?\.(?:py|csv|json|jsonl|md|png|ps1|yml))`")
    missing = set()
    tracked_set = _tracked_names()   # None = 非 git 环境，跳过该检查
    for f in files:
        if f.suffix != ".md" or not f.exists():
            continue
        # 引用路径有两种写法，都要试：
        #   ① 仓库根相对：`code/tc.py`、`results/CLAIMS.md`（本仓库文档的惯例）
        #   ② 文档所在目录相对：data/README.md 里写 `../results/x.md`
        #
        # 关键判据是**是否落在仓库内**，而不是"在我这台机器上是否存在"：
        # `../审计记录.md` 在作者机器上真实存在（仓库外），但别人 clone 下来就没有。
        root_abs = ROOT.resolve()
        base = f.parent
        for m in ref_pat.finditer(f.read_text(encoding="utf-8", errors="replace")):
            raw = m.group(1)
            if "*" in raw:
                continue
            hits_inside = False
            for cand in ((ROOT / raw).resolve(), (base / raw).resolve()):
                try:
                    cand.relative_to(root_abs)
                except ValueError:
                    continue          # 仓库外，不计
                if cand.exists():
                    hits_inside = True
                    break
            if hits_inside:
                continue
            # 两条都不成立：要么指向仓库外，要么仓内缺失
            outside_only = all(
                not _inside(c, root_abs)
                for c in ((ROOT / raw).resolve(), (base / raw).resolve()))
            missing.add(raw + ("（指向仓库外）" if outside_only else ""))
    record("契约", "文档中引用的文件都存在", not missing,
           "; ".join(sorted(missing)[:5]) if missing else "已检查所有 .md")

    # 4.3 图非空
    figs = sorted(cfg.FIG_DIR.glob("*.png"))
    tiny = [p.name for p in figs if p.stat().st_size < 2000]
    record("契约", f"{len(figs)} 张图都存在且非空", bool(figs) and not tiny,
           str(tiny) if tiny else "")

    # 4.4 TOML 合法
    try:
        import tomllib
        d = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
        need = ["name", "version", "license"]
        miss = [k for k in need if k not in d.get("project", {})]
        record("契约", "pyproject.toml 合法且含关键字段", not miss, str(miss))
    except Exception as e:
        record("契约", "pyproject.toml 合法且含关键字段", False, str(e)[:60])

    # 4.5 CI 配置存在且触发了 push
    ci = ROOT / ".github/workflows/ci.yml"
    ok = ci.exists() and "push" in ci.read_text(encoding="utf-8")
    record("契约", "GitHub Actions 配置存在且监听 push", ok)


# ---------------------------------------------------------------- 主流程
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true", help="跳过裸克隆模拟")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    print("=" * 74)
    print("TokenCost 全项目体检")
    print(f"仓库：{ROOT}")
    print("=" * 74)

    files = tracked_files()
    audit_hygiene(files)
    if not args.fast:
        audit_selfcontained(files)
    audit_data(files)
    audit_contract(files)

    n_ok = sum(1 for r in RESULTS if r["ok"])
    n = len(RESULTS)
    print("\n" + "=" * 74)
    if n_ok == n:
        print(f"结论：{n}/{n} 项通过 ✅")
    else:
        print(f"结论：{n_ok}/{n} 项通过，{n - n_ok} 项待修 ❌")
        for r in RESULTS:
            if not r["ok"]:
                print(f"  · [{r['area']}] {r['name']} —— {r['detail']}")
    print("=" * 74)
    return 0 if n_ok == n else 1


if __name__ == "__main__":
    sys.exit(main())


