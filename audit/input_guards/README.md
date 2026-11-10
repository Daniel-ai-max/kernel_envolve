# Native 只读输入校验诊断

本修改减少重复输入校验的设备到主机传输，保留逐位比较、验证函数前后检查，以及绑定、布局和别名检查。Native CUDA worker 为只读输入建立私有 uint8 副本，预算取 2 GiB 与当时可用显存的 1/8 中较小值；预算不足或副本相关 OOM 时回退原 CPU 精确比较。通用 Session 默认行为不变。

## 固定条件与结果

基线提交为 `b084e38b73bfb2f46a544f608c504930d1603d92`。单张 B300，PyTorch 2.12.1+cu130，单 CPU 线程、确定性模式；通过现有 manager 的 GPU 锁与预算入口执行。候选产品文件和每份原始证据的 SHA256 见 [manifest.json](manifest.json)。

使用真实 KL Teacher K 的形状 `(1, 16, 127996, 192)`、FP32，共 1,572,814,848 字节。固定 ABBA 顺序，每块一次 warmup、三次 `Session.observe`；未重采样。输入校验发生在算子计时区间外。

| 顺序 | 实现 | 建立 Session（秒） | 三次 observe（毫秒） |
| --- | --- | ---: | --- |
| A1 | 基线 | 1.661 | 3620.272、3536.499、3505.491 |
| B1 | 候选 | 1.705 | 3.233、3.172、3.163 |
| B2 | 候选 | 1.699 | 3.182、3.169、3.174 |
| A2 | 基线 | 1.520 | 3472.700、3475.023、3478.965 |

六次样本的中位数由 3492.228 ms 降为 3.173 ms；私有副本实际占用 1,572,864,000 字节。这是大输入 `Session.observe` 的局部开销测量。没有重跑完整 native evaluator，不能据此报告全流程加速或 Student 算子收益。跨进程输入/输出传输、正确性比较和算子计时方式保持原样。

## 正确性与峰值口径

首次诊断已通过七项检查：FP32/BF16 非连续输入及别名的 Triton 原始字节写入（不增加 `_version`）、signed zero、NaN payload、预算不足后的精确拒绝、验证函数写输入的拒绝、refresh 后旧值的拒绝。

首次诊断在最后的真实 worker 子测试遇到诊断脚本生成的 adapter 缩进错误，整体状态保留为 `failed`，详见 [result.json](evidence/result.json)。只修复诊断脚本并执行一次 `--peak-only` 补测，未重复 ABBA 和前述七项检查；[补测结果](evidence/peak-result.json)为 `pass`。

补测调用真实修改版 `native_worker.serve()`，在测量后校验阶段注入一次 comparison OOM，确认释放副本并回退 CPU。测量时记录私有占用 16,384 字节，raw peak 49,664 字节，调整后为 33,280 字节；即使随后副本变为 0，扣除量仍取测量当时值，真实输入与候选输出分配仍计入。worker 用时 0.235 秒，补测脚本 0.774 秒，管理器收据总时长 4.208 秒。

首次脚本、修复后的脚本、两次请求、管理器收据和日志均保存在 [evidence](evidence/)。首次运行总时长 40.131 秒，脚本 36.701 秒。首次诊断的失败记录没有覆盖。

## 回归验证

- 新增 CPU 语义回归：10 项通过；两项 CUDA pytest 未在本机执行，真实 CUDA 检查由上述受管诊断完成。
- 原 native/general/evaluator/manager-native 定向回归：90 项通过。
- `ruff check` 和 `git diff --check` 通过。
- 主任务独立全套回归：392 passed、10 skipped，116.16 秒；使用绝对 `PYTHONPATH`，运行前后 src/tests 哈希一致。

运行 pytest 时应将 `PYTHONPATH` 设置为 checkout 的绝对 `src` 路径，避免测试子进程切换目录后加载机器已有的旧版安装。初次全套本机测试使用相对路径，出现 8 项旧版本 CLI 污染失败（383 passed、8 skipped）；该次结果不作为本修改回归通过证据。

PR CI 要求的格式修正仅涉及四份 Python 文件；原 GPU 源码 SHA 保持不变，格式前后 AST 完全相同，校验证据见 [format-equivalence.json](format-equivalence.json)。未重复 GPU 诊断。
