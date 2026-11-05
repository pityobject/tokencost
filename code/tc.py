"""
tc.py —— 统一入口（P2-11）。

原来每个脚本都要 `cd code && python tc_xxx.py`，对外人来说门槛太高。
这里给一个单入口，把常用流程串起来：

    python code/tc.py check              # 环境自检
    python code/tc.py preflight          # 跑一格真实测量，确认机器满频
    python code/tc.py run                # 跑核心网格（45 格）
    python code/tc.py run dense          # 跑扩展预设（dense/high/ctx/quant/step/wave/knee/xfamily）
    python code/tc.py analyze            # 汇总 + 出图
    python code/tc.py steady             # 稳态口径重算
    python code/tc.py model              # 六参数拟合 + 留出验证
    python code/tc.py eval               # 质量基准（400 题）
    python code/tc.py quality-stats      # 配对检验 + 分层 + 阈值敏感性
    python code/tc.py recommend          # 配置推荐
    python code/tc.py sweep              # 质量—成本权衡表
    python code/tc.py verify             # 决策后悔值（16 场景）
    python code/tc.py threshold          # 阈值敏感性（fig6）
    python code/tc.py regret-controls    # 后悔值负对照
    python code/tc.py queue              # 排队层（WIP）
    python code/tc.py claims             # 生成对外数字表
    python code/tc.py claims --check     # 检查文档数字是否漂移（CI 用）

实现上只是转发给对应脚本，不改变任何既有行为。
"""

import subprocess
import sys
from pathlib import Path

def _locate_code_dir():
    """定位脚本目录。

    用 pip 安装后 `__file__` 在 site-packages 里，同级并没有 tc_*.py，
    所以先找当前目录下的 code/，再退回 __file__ 的父目录。
    """
    here = Path(__file__).resolve().parent
    if (here / "tc_recommend.py").exists():
        return here
    cwd_code = Path.cwd() / "code"
    if (cwd_code / "tc_recommend.py").exists():
        return cwd_code
    return here


CODE_DIR = _locate_code_dir()

COMMANDS = {
    "check":         ("tc_env_check.py", []),
    "preflight":     ("tc_preflight.py", []),
    "run":           ("tc_run_grid.py", []),
    "analyze":       ("tc_analyze.py", []),
    "steady":        ("tc_recompute_steady.py", []),
    "model":         ("tc_model.py", []),
    "eval":          ("tc_eval.py", []),
    "quality-stats": ("tc_eval_stats.py", []),
    "recommend":     ("tc_recommend.py", []),
    "sweep":         ("tc_recommend.py", ["--quality-sweep"]),
    "verify":        ("tc_verify.py", []),
    "claims":        ("tc_claims.py", []),
    "decompose":     ("tc_decompose_csrv.py", []),
    "threshold":     ("tc_threshold_sensitivity.py", []),
    "regret-controls": ("tc_regret_controls.py", []),
    "queue":         ("tc_queue.py", []),
    "lint":          ("tc_lint_docs.py", []),
    "calibrate":     ("tc_calibrate.py", []),
    "profile":       ("tc_profile.py", []),
    "tails":         ("tc_analyze_tails.py", []),
    "proflik":       ("tc_param_profile.py", []),
    "byquant":       ("tc_analyze_by_quant.py", []),
    "decompose-all": ("tc_decompose_cells.py", []),
    "stop":          ("tc_killsrv.py", []),
    "variance":      ("tc_analyze_variance.py", []),
    "gpucheck":      ("tc_gpucheck.py", []),
    "contamination": ("tc_check_contamination.py", []),
    "clockaudit":    ("tc_clock_audit.py", []),
    "audit":         ("tc_audit.py", []),
    "ci":            ("../scripts/ci.ps1", []),
    "reset":         ("tc_reset_data.py", []),
}


def usage():
    print(__doc__)
    print("可用命令：")
    for k in COMMANDS:
        print(f"  {k}")
    sys.exit(2)


def main():
    if len(sys.argv) < 2:
        usage()
    cmd = sys.argv[1]
    if cmd not in COMMANDS:
        print(f"未知命令：{cmd}\n")
        usage()

    script, pre = COMMANDS[cmd]
    rest = sys.argv[2:]

    # run 的子预设：`tc.py run dense` → tc_run_grid.py --preset dense
    if cmd == "run" and rest and not rest[0].startswith("-"):
        rest = ["--preset", rest[0]] + rest[1:]

    target = CODE_DIR / script
    if target.suffix.lower() == ".ps1":
        # .ps1 不能用 sys.executable 跑（会当 Python 语法报错）
        argv = ["powershell", "-ExecutionPolicy", "Bypass", "-File", str(target)] + pre + rest
    else:
        argv = [sys.executable, str(target)] + pre + rest
    print(f"[tc.py] {' '.join([Path(argv[0]).name] + argv[1:])}\n", flush=True)
    sys.exit(subprocess.call(argv, cwd=str(CODE_DIR)))


if __name__ == "__main__":
    main()





