# 代码组织与修改入口

运行时代码按职责划分，命令入口负责解析与路由，评测器负责验证和晋升，管理层负责持续执行与证据展示。没有引入新的插件系统、服务或配置格式。

## 目录职责

| 路径 | 职责与主要入口 |
| --- | --- |
| `src/kernel_envolve/cli.py` | 用户 CLI；通用评测函数保留兼容导出 |
| `src/kernel_envolve/evaluation/generic/` | 独立任务评测：`evaluator.py` 冻结、审计、晋升；`worker.py` 执行检查和计时 |
| `src/kernel_envolve/evaluation/native/` | 原生项目评测：`evaluator.py` 主控、`config.py` 契约、`harness.py` 可信检查源码、`checks.py` 原生门禁、`confirmation.py` 独立确认、`worker.py` 子进程、`workload.py` 训练状态与梯度 |
| `src/kernel_envolve/evaluation/shared/` | 两类评测共用的输入保护、协议、评分、快照、进程和运行环境检查 |
| `src/kernel_envolve/manager/` | 管理层入口、持续任务调度、配置、SQLite 状态、预算与研究策略 |
| `src/kernel_envolve/manager/execution/` | 进程身份与锁、GPU 准入、worker、远程桥接和恢复 |
| `src/kernel_envolve/manager/results/` | `index.py` 收集已验证证据；assessment、validation、research 模块处理各类收据；archive 保存可移植快照；publication 验证 PR 交付 |
| `src/kernel_envolve/manager/dashboard/` | HTTP 服务、展示模型、摘要生成；`static/` 随 Python 包安装 |
| `src/kernel_envolve/rsi/` | 可选的策略学习、试验、验证、经验复用与独立状态；保留已有包边界和默认关闭行为 |
| `src/kernel_envolve/updates/` | 更新缓存、版本选择及固定版本会话 |
| `src/kernel_envolve/research.py`、`validation.py` | 面向研究脚本的轻量公共 API |
| `src/kernel_envolve/_io.py`、`_paths.py` | 私有基础工具：持久化写入、安装位置；不依赖 manager |
| `acceptance/a7b/` | A7B P0–P2 验收设计、清单及来源；目前仍是 `design_only` |
| `examples/` | 可复制的任务、adapter 和管理配置 |
| `skills/kernel-envolve/` | Agent 研究流程与交付规范 |
| `tools/` | 仓库维护校验，包括 A7B 文档生成与一致性检查 |
| `tests/` | 按 `test_native_*`、`test_manager_*` 等前缀定位回归；相互复用的 fixture 保持现有收集方式 |
| `audit/` | 历史实验和验证证据，保留其原始路径与内容 |

## 验收代码从哪里读

1. [通用评测主控](../src/kernel_envolve/evaluation/generic/evaluator.py)：`evaluate → audit_result → promote`。
2. [共享评分](../src/kernel_envolve/evaluation/shared/scoring.py)：正确性、覆盖、收益与退化如何影响资格。
3. [原生评测主控](../src/kernel_envolve/evaluation/native/evaluator.py)、[可信 harness](../src/kernel_envolve/evaluation/native/harness.py)和[原生检查](../src/kernel_envolve/evaluation/native/checks.py)：冻结哪些源码、执行哪些 gate。
4. [独立确认](../src/kernel_envolve/evaluation/native/confirmation.py)：确认集的使用、收据和晋升前核验。
5. [管理层证据索引](../src/kernel_envolve/manager/results/index.py)与[外部验证收据](../src/kernel_envolve/manager/results/validation_import.py)：面板为什么认定某条结果可用。
6. [A7B 验收清单](../acceptance/a7b/README.md)：不同算子的验收要求。清单设计不等于已经实现或通过 GPU 验收。

## 依赖与新增代码约定

- CLI 只解析参数、调用实现、转换退出码。通用评测逻辑不再放回 CLI；manager 直接使用评测模块。
- 新实现直接导入上述规范路径。根目录的旧评测模块，以及 manager 下已搬迁的旧模块，只做兼容转发，不增加业务逻辑。
- 包的 `__init__.py` 保持轻量，避免导入包时启动子进程、初始化设备或加载整棵管理依赖。
- 纯文件写入使用 `_io.py`。需要任务状态、证据核验或展示投影时，再依赖对应 manager 模块。
- 两类 worker 由主控启动；管理 worker 和 bridge 归入 execution。启动参数统一使用规范模块路径，避免依赖源文件相对位置。
- 静态资源通过 `importlib.resources.files("kernel_envolve.manager.dashboard")` 读取；不要假设源码 checkout 存在。
- manager 仍有必要的协作依赖：归档复用展示投影，恢复与结果核验使用状态库；评测进程在受管理运行时延迟导入 manager 以登记子进程。这些边界保留显式导入，不增加抽象框架。
- 修改行为时在现有领域测试中补充回归；迁移入口、资源或安装布局时，补充 `test_package_layout.py`。沿用 Ruff 和现有 CI，不另建一套规范工具。

## 兼容与证据边界

CLI 命令、参数、退出码和 `kernel_envolve.load_best` 保持原有入口。adapter 继续使用 `kernel_envolve.training.Workload`、`forward_backward`、`Session`；research 和 validation 公共路径保持稳定。

旧模块用 `sys.modules` 转发到同一个规范模块对象，保证旧导入与新导入共享模块状态，已有 monkeypatch 不会悄悄作用到另一个副本。旧 worker 的 `python -m` 入口继续转发执行。没有复制两套实现，也没有全局 import hook。

评测器指纹现在对整个运行时包的 Python 源文件按相对路径递归计算，涵盖移动后的子目录。收据 schema、SQLite schema、任务契约字段和 `.kernel_envolve` / `.kernel_native` 目录格式未变；但本次重构会改变评测器指纹，不能把旧版本 full 结果直接当作新版本的晋升依据。旧 full → confirm → promote 应在原固定版本会话完成，或在新版本重新评测。不会重写历史 hash 或历史收据。

## 本地验证

```bash
python -m pip install -e '.[dev]'
export KERNEL_ENVOLVE_AUTO_UPDATE=0
ruff check .
ruff format --check .
pytest -q
python tools/check_a7b_acceptance.py
```

布局回归检查兼容导入、嵌套源码指纹、worker 的安装路径和看板资源。发布前还应构建 wheel，并在 checkout 以外安装验证 CLI、CPU 示例与资源加载；仅在源码目录通过测试不足以证明打包完整。
