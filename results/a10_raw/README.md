# A10 云端标定原始产物（2026-11-06）

从 PAI-DSW 实例（A10 24GB）打包下载，未经修改。

| 文件 | 内容 | 关键参数 |
|---|---|---|
| `calib_a10_llamacpp.json` | 6 点标定（run 1） | c_srv 1.529 ms · bw 471.5 · B_SAT 7.2 |
| `calib_a10_llamacpp_rep2.json` | 同配置重复标定（run 2） | c_srv 1.963 ms（**+28%**） |
| `calib_a10_vllm.json` | 6 点标定（run 1） | c_srv 0.273 ms · bw 384.9 · B_SAT 21.9 |
| `calib_a10_vllm_rep2.json` | 同配置重复标定（run 2） | c_srv 0.290 ms（**+6%**） |
| `calib_a10_llamacpp_long.json` | 长 prompt（L≈1841）c1/4/8/16 | 参数退化，仅取实测值 |
| `calib_a10_vllm_long.json` | 同上 | 参数退化，仅取实测值 |

## 用途

- `python code/tc_a10_repeat.py` 直接读本目录 → 重复性表 + 失效区跨栈表
- `python code/tc_compare_backends.py` 读顶层 `calib_a10_llamacpp.json` / `calib_a10_vllm.json` → 栈间对比

## 已知限制

- 每栈只标定了**两次**：`c_srv` 的重复性区间来自这两次（llama.cpp ±28% / vLLM ±6%），
  不是统计意义上的置信区间。
- 长 prompt 的两份文件只有 4 个同长度测点，`c_srv` 与 `B_SAT` 不可分辨，
  **不要引用它们的拟合参数**，只用 `measurements`。
