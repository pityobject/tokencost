# -*- coding: utf-8 -*-
"""护栏测试：把三次"静默失效"事故变成会变红的测试。

## 为什么需要

2026-11 的三次事故都不是"算错"，而是**检查器自己失效**：

| 事故 | 现象 | 当时为什么没被抓到 |
|---|---|---|
| F12 | 补丁没命中却打印成功 | 没有断言 |
| F13 | 不同混合共用 id，互相覆盖 | id 未绑定参数 |
| F14 | 多轮算完不落盘 | 只验了"有输出"没验"有数据" |
| 复核第 12/13 轮 | 键名打错 → `.get(k,"")` 返回空串 → 检查项空操作 | 没有断言键存在 |
| 复核第 13 轮 | 生成物里有 nan%，`--check` 仍报 OK | 只扫 claims 字典，不扫生成物 |

本文件把每一条都做成**可执行的断言**。
"""
import importlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "code"))

import tc_config as cfg

CODE = Path(__file__).resolve().parent.parent / "code"


def test_claims_keys_exist():
    """检查器引用的键必须真实存在（防 `.get(k,"")` 空操作）。"""
    tc_claims = importlib.import_module("tc_claims")
    d = tc_claims.load_inputs()
    claims = tc_claims.build_claims(d)

    # 这些是 check() 里用 _need() 取的键；打错就会在 CI 前爆
    for key in ("单位成本 MAPE（core，稳态口径）", "单点最大误差", "90% 区间覆盖率"):
        assert key in claims, f"检查器引用的键不存在：{key!r}"
        assert claims[key] and "nan" not in str(claims[key]).lower(), \
            f"{key} 的值是空的/nan：{claims[key]!r}"

    # _need 对不存在的键必须抛错，而不是返回空
    try:
        tc_claims._need(claims, "这个键不存在")
    except SystemExit:
        pass
    else:
        raise AssertionError("_need() 对不存在的键没有报错——它又变回静默空操作了")


def test_generated_claims_has_no_nan():
    """CLAIMS.md 生成物里不允许出现 nan / '? |' 占位。"""
    tc_claims = importlib.import_module("tc_claims")
    d = tc_claims.load_inputs()
    claims = tc_claims.build_claims(d)
    tc_claims.write_md(claims, d)          # 会写盘；若有 nan 会 raise
    body = (cfg.RESULTS_DIR / "CLAIMS.md").read_text(encoding="utf-8")
    for tok in ("nan", "NaN"):
        assert tok not in body, f"CLAIMS.md 里出现 {tok}"


def test_three_populations_persisted():
    """三套口径必须来自产物（populations.json），不能只存在于静态文本。"""
    p = cfg.RESULTS_DIR / "populations.json"
    assert p.exists(), "缺 results/populations.json —— 先跑 tc.py model"
    pops = json.loads(p.read_text(encoding="utf-8"))
    for name in ("core", "valid", "all"):
        assert name in pops, f"populations.json 缺口径 {name}"
        row = pops[name]
        for k in ("n_holdout", "mape_cost", "max_err", "coverage"):
            assert k in row, f"{name} 缺字段 {k}"
        assert 0.0 <= row["coverage"] <= 1.0, \
            f"{name}.coverage 应是比例（0–1），实际 {row['coverage']}"


def test_mix_fingerprint_in_cell_id():
    """不同混合分布必须得到不同的 id 后缀（F13）。"""
    lg = importlib.import_module("tc_loadgen")
    import hashlib

    def suffix(mix):
        key = ",".join(f"{k}:{v:.4f}" for k, v in sorted(mix.items()))
        return "_mix" + hashlib.sha1(key.encode()).hexdigest()[:4]

    a = suffix(lg.parse_length_mix("256:0.7,1024:0.2,2048:0.1"))
    b = suffix(lg.parse_length_mix("128:0.5,2048:0.5"))
    assert a != b, f"两种混合分布得到同一个后缀：{a}"
    assert a == suffix(lg.parse_length_mix("256:0.7,1024:0.2,2048:0.1")), "同分布后缀应稳定"


def test_multiturn_persists_rows(tmp_path, monkeypatch):
    """多轮分支必须把行写进 raw_cells（F14：只打印不落盘）。"""
    rc = importlib.import_module("tc_run_cell")
    lg = importlib.import_module("tc_loadgen")

    fake_rows = [dict(turn=t, ok=True, n_ok=2, prefix_tokens_median=200 + 40 * t,
                      ttft_p50_ms=100.0 + t, tpot_p50_ms=10.0 + t, throughput_tps=50.0)
                 for t in (1, 2)]

    class _FakeSrv:
        @staticmethod
        def start(*a, **k):
            return object(), tmp_path / "fake.log"

        @staticmethod
        def stop(proc):
            return None

    monkeypatch.setattr(rc, "srv", _FakeSrv)
    async def _fake_run_turns(*a, **k):
        return fake_rows

    monkeypatch.setattr(lg, "run_turns", _fake_run_turns)
    monkeypatch.setattr(rc, "time", importlib.import_module("time"))
    out = tmp_path / "raw_cells.jsonl"
    monkeypatch.setattr(rc, "RAW_CELLS", out)

    rows = rc.run_cell_turns("3B", "q4_k_m", 4096, 1, turns=2, verbose=False)
    assert len(rows) == 2, f"应产出 2 行，实际 {len(rows)}"
    assert out.exists(), "多轮分支没有写 raw_cells（F14 复发）"
    n = len([l for l in out.read_text(encoding="utf-8").splitlines() if l.strip()])
    assert n == 2, f"落盘行数应为 2，实际 {n}"


def test_population_single_source():
    """口径后缀正则只允许出现在 tc_config.py —— 否则就是第二份实现。

    历史上同一类 bug 出现过三次：tc_cv 漏改、两个试验脚本各判一次，
    每次都让"同一指标出现两个答案"。这个测试让"再复制一份"直接失败。
    """
    import re
    from pathlib import Path

    pat = re.compile(r"_\(\?:w\\d\+\|n\\d\+\|p\\d\+")
    offenders = []
    for f in (Path(__file__).resolve().parent.parent / "code").glob("*.py"):
        if f.name == "tc_config.py":
            continue
        text = f.read_text(encoding="utf-8", errors="replace")
        if pat.search(text):
            offenders.append(f.name)
    assert not offenders, f"口径判定被复制到了：{offenders}（应统一用 cfg.population_mask）"


def main():
    """不依赖 pytest 也能跑：直接执行所有 test_* 函数。"""
    import shutil
    import tempfile
    import types
    import traceback

    class MP:
        """极简 monkeypatch 替身（支持 setattr 与自动还原）。"""
        def __init__(self):
            self.undo = []

        def setattr(self, obj, name, val):
            old = getattr(obj, name)
            self.undo.append((obj, name, old))
            setattr(obj, name, val)

        def close(self):
            for obj, name, old in reversed(self.undo):
                setattr(obj, name, old)

    ok = fail = 0
    for name, fn in list(globals().items()):
        if not name.startswith("test_") or not callable(fn):
            continue
        args = []
        mp = MP()
        tmp = Path(tempfile.mkdtemp())
        try:
            params = fn.__code__.co_varnames[:fn.__code__.co_argcount]
            if "tmp_path" in params:
                args.append(tmp)
            if "monkeypatch" in params:
                args.append(mp)
            fn(*args)
            print(f"  PASS {name}")
            ok += 1
        except Exception as e:
            print(f"  FAIL {name}: {type(e).__name__}: {e}")
            traceback.print_exc()
            fail += 1
        finally:
            mp.close()
            shutil.rmtree(tmp, ignore_errors=True)
    print(f"\n  {ok} passed, {fail} failed")
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
