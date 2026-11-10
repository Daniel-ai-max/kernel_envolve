# kernel_envolve

kernel_envolve 是面向 coding agent 的算子持续优化工具，主要用于 [apodex-core](https://github.com/ApodexAI/apodex-core) 与 [Megatron-LM](https://github.com/ApodexAI/Megatron-LM) 的训练算子，也支持独立的 NumPy、PyTorch 和 CUDA 算子实验。

目标是在保持调用语义、精度和输入覆盖的前提下提升性能，并让多个算子的优化进展可追溯、可恢复、可查看。项目包含三部分：

- **优化流程（Skill）**：指导 Agent 分析瓶颈、验证假设、记录实验，并在确认收益后维护优化 PR。
- **评测器（Evaluator）**：冻结源码，检查正确性和性能，保存证据并更新已验证的最佳版本。
- **管理层（Manager）**：调度长期任务和 GPU 队列，管理暂停、预算及故障恢复，通过本地面板展示进展。

Agent 使用已有的模型和执行环境；本仓库提供流程、评测与协调能力，不内置模型服务。

[快速开始](#独立算子快速开始) · [项目原生评测](#内部项目使用) · [持续优化与面板](#持续优化) · [任务配置](#任务配置) · [开发与文档](#仓库结构与开发)

## 独立算子快速开始

需要 Python 3.11+。先用 CPU 示例运行完整流程，无需 GPU 或模型 API：

```bash
git clone https://github.com/Daniel-ai-max/kernel_envolve.git
cd kernel_envolve
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .

kernel-envolve evaluate --task examples/vector_add --mode quick
kernel-envolve evaluate --task examples/vector_add --mode full
```

命令输出包含结果目录 `run` 和能否晋升的 `eligible_for_best`。只有该字段为 `true` 的完整评测结果才能更新 best；将下面路径替换为输出中的 `run`：

```bash
kernel-envolve promote /path/to/run
```

在具备 NVIDIA CUDA 环境的 Linux 主机上，可运行 Triton 示例：

```bash
python -m pip install -e '.[cuda]'
kernel-envolve evaluate --task examples/cuda_vector_add --mode quick
kernel-envolve evaluate --task examples/cuda_vector_add --mode full
```

性能测量需要独占目标 GPU。已有项目镜像应保留其 PyTorch、Triton 和扩展版本，安装方式见下一节。

### 配合 Agent 使用

让 coding agent 读取 [SKILL.md](skills/kernel-envolve/SKILL.md)。支持本地 Skill 的宿主可安装 `skills/kernel-envolve/`；例如 Codex 的个人 Skill 目录为 `~/.codex/skills/`。

> 使用 $kernel-envolve 持续优化 examples/cuda_vector_add，使用指定 GPU，累计预算 20 分钟。保持任务定义与精度要求不变，通过完整评测后更新 best，并记录下一步。

Skill 规定优化方法，跨会话自动续接由下方管理层或现有宿主执行器负责。

### 自动跟随 main

安装含此机制的版本后，CLI 在使用前自动检查远端 `main`；管理器在每轮 Agent 启动前检查。新版本通过依赖和启动检查后启用，Skill 与评测器使用同一个提交。更新保存到独立缓存，不改动本地仓库、算子代码或 Python/CUDA 依赖。

手动启动 Agent 或连续执行 full → promote 时，使用固定版本的会话：

```bash
kernel-envolve session -- <agent命令及参数>
# 也可以进入固定版本的交互式 shell
kernel-envolve session -- bash
```

会话运行期间保持版本不变，下次会话再检查更新。已有 Agent 可执行 `kernel-envolve update`，读取返回的 `skill`，并在本轮使用返回的 `executable`；复制安装的 Skill 也会按此入口读取最新流程。

断网或新版本检查失败时保留可用版本，原因输出到 stderr。`kernel-envolve update --status` 查看最近检查结果。开发本仓库时设置 `KERNEL_ENVOLVE_AUTO_UPDATE=0` 使用本地代码。缓存、兼容性与升级边界见[自动更新指南](docs/auto-update.md)。

## 内部项目使用

训练算子使用 `native-evaluate`，由 `native.json` 和 `adapter.py` 接入项目原生测试、数值 gate 和 benchmark。

- **apodex-core**：维护 kernel 实现、算子接口和算子测试。
- **Megatron-LM**：验证调用接线、模块梯度及受影响的训练路径。
- **验证重点**：前向与反向总调用成本、逐尺寸退化、布局、状态更新和显存；按改动范围补充分布式、重计算及训练验证。

[MoE probs_scale 示例](examples/apodex_probs_scale) 已定义 core 契约测试、原生数值 gate 和 Megatron 模块梯度测试。先在项目指定镜像中准备候选及固定基线源码，按 `native.json` 设置 `APODEX_CORE_CANDIDATE`、`APODEX_CORE_BASELINE`、`MEGATRON_CANDIDATE` 和 `MEGATRON_BASELINE`，再运行：

```bash
# 在本仓库目录安装工具；Python 3.11+，复用项目镜像已有依赖
python -m pip install --no-deps -e .
kernel-envolve native-evaluate --task examples/apodex_probs_scale --mode quick
kernel-envolve native-evaluate --task examples/apodex_probs_scale --mode full --timeout 1800
kernel-envolve native-promote /path/to/run
```

`native-promote` 的路径取自合格的 full 结果。数值 gate 或 CI 通过仅说明相应检查通过，性能收益仍需测量确认。配置与适配接口见 [内部项目评测](skills/kernel-envolve/SKILL.md#apodex-内部项目评测)及[内部 evaluator](skills/kernel-envolve/SKILL.md#使用内部-evaluator)。

原生评测的 CLI 与 `result.json` 提供 `phase_wall_seconds`：源码快照、设置、正确性、原生门禁、测量、后置检查、完整性校验和清理。执行阶段失败保留已耗时间及 `failed_phase`，未执行阶段不填零。各阶段是主控进程的连续墙钟耗时，总和等于 `seconds`（至 worker 清理完成，不含评分和收据写入）；首次导入/编译可能计入首个正确性请求，测量阶段还含参考计算、传输和结果核验，不能作为纯 GPU kernel 时间。

## 持续优化

一个算子对应一个长期任务，执行轮次结束时保存 checkpoint，后续轮次继续已有工作。一次晋升或某个方向没有收益，都不自动结束任务；每轮在 `notes.md` 记录假设、结论、版本、资源消耗与下一步。

确认收益后，Agent 按 [PR 交付流程](skills/kernel-envolve/SKILL.md#收益确认后自动维护-pr)创建或更新目标仓库的优化 PR，然后继续探索。PR 发布由 Agent 完成；评测器负责验证与晋升。

### 多算子管理与面板

管理层面向单台 POSIX 主机：每个算子使用独立工作区和 Agent 命令，每张物理 GPU 配置 1–2 个会话。会话可并行分析和修改代码，受管理的 GPU 作业按队列串行执行。

| 能力 | 行为 |
|---|---|
| 任务续接 | 保存 checkpoint、轮次和事件；协调器重启后核对进程身份与锁，恢复观察已有作业 |
| 资源控制 | 支持暂停、恢复、累计会话/GPU 预算和失败退避；无法确认 GPU 清理完成时隔离资源 |
| 基线溯源 | 展示仓库、版本、实现提出日期及来源；单独记录本任务的冻结日期，未知日期保持未知 |
| 进展展示 | 提供算子总览、任务详情、best 历史、逐用例延迟、验证范围及资源队列 |
| 证据来源 | 区分候选、已验证 best 与报告的已集成版本；性能数据来自 evaluator 校验结果 |
| 自动更新 | 页面可见时约每 3 秒读取一致快照，后台约每 5 秒检查证据；保留编辑状态，断线退避重连 |

**先体验面板。** 完成基础安装后，在尚不存在的目录创建 CPU 演示：

```bash
python examples/manager/demo.py /tmp/kernel-envolve-demo

# 终端 1：运行协调器
kernel-envolve manage --state /tmp/kernel-envolve-demo/state run

# 终端 2：运行面板服务
kernel-envolve manage --state /tmp/kernel-envolve-demo/state serve --port 8787
```

打开 [http://127.0.0.1:8787](http://127.0.0.1:8787)。演示使用模拟 Agent、CPU 和合成等待，展示调度及评测流程；其中的加速比不代表真实训练算子收益。重新查看时直接运行 `run` / `serve`，无需重新生成目录。停止任务时先在面板暂停；关闭页面或退出协调器不会立即终止已启动轮次。

**接入真实任务。** 按[管理层配置指南](docs/optimization-manager.md#启动)准备独立工作区、物理 GPU UUID、固定 baseline、评测契约及 Agent 命令，然后启动：

```bash
kernel-envolve manage --state /work/manager-state init deployment.json
kernel-envolve manage --state /work/manager-state run
# 另一个终端
kernel-envolve manage --state /work/manager-state serve --port 8787
```

Agent 适配器从 stdin 接收恢复提示，结束前写入结构化 checkpoint；受管理会话中的所有 GPU 命令必须经 `kernel-envolve manage gpu-run -- <命令>` 执行。管理层依赖同一用户下的协作工作流，跨用户资源分配和多 GPU 分布式作业由外部系统安排。远程面板通过 SSH 转发访问。

部署、控制命令和恢复规则见[管理层文档](docs/optimization-manager.md)，数据一致性与刷新策略见[自动更新契约](docs/optimization-manager.md#自动更新契约)。

## 任务配置

两类任务使用不同的接入方式：

| 类型 | 文件 | 适用场景 |
|---|---|---|
| 通用任务 | `task.json`、`task.py`、`candidate.py` | 独立数组或张量算子；候选导出 `run(*inputs)` |
| 原生任务 | `native.json`、`adapter.py` | 前反向、训练状态和项目原生验证 |

通用任务由 `task.py` 定义输入生成、正确性参考和原始基线，`task.json` 定义输入覆盖、精度、权重与计时规则。完整接口、配置示例和默认值见[独立算子评测指南](docs/generic-evaluator.md)。

始终区分四种角色：**reference** 是正确性参考，**baseline** 是固定原始实现，**candidate** 是当前候选，**best** 是通过完整验证并晋升的版本。输入覆盖、精度和测量规则在任务建立时固定；改变契约时创建新任务版本。

## 评测规则

1. **quick 筛选**：用代表用例验证正确性并比较性能，适合日常迭代。
2. **full 确认**：验证全部声明用例，进行两批配对测量，检查总体收益、波动和逐用例退化；原生任务还要求声明的原生检查通过。
3. **promote 晋升**：复核证据、源码、契约与当前 best 后更新最佳版本。quick 结果不能晋升。

评测退出码 `0` 表示正确完成，`1` 表示正确性失败，`2` 表示配置或执行错误；能否晋升以 `eligible_for_best` 为准。有 best 时与当前 best 比较，full 同时检查固定原始基线，从而区分增量与累计收益。

通用结果保存在任务的 `.kernel_envolve/`，原生结果保存在 `.kernel_native/`，包含源码快照、配置、原始测量和日志。通用任务可通过 `from kernel_envolve import load_best` 加载最佳实现。诊断参数、计时边界与加载示例见[评测规则](docs/generic-evaluator.md#评测规则)。

## 输入覆盖与交付

用例应来自实际调用分布，覆盖尺寸、batch、上下文长度、dtype，以及各输入独立变化的 stride、padding 和转置；例如同时检查连续输入、非对齐长度和不同输入布局组合。既关注加权总体收益，也检查逐用例退化和 fallback。

交付前冻结实现，再验证未参与调参的尺寸、布局组合及数值种子，并保存源码哈希和复现证据。具体要求见[输入覆盖与交付](docs/generic-evaluator.md#输入覆盖与交付)。

## 仓库结构与开发

```text
src/kernel_envolve/          通用/原生评测、证据保存与版本晋升
src/kernel_envolve/manager/  持续任务、进程恢复、GPU 队列与面板
skills/kernel-envolve/       Agent 优化流程、PR 规范和参考入口
examples/                   CPU、CUDA、原生任务及管理层演示
docs/                       配置、使用和设计说明
tests/                      正常路径、故障注入与回归测试
audit/                      按实验或版本保存的验证证据
```

```bash
python -m pip install -e '.[dev]'
export KERNEL_ENVOLVE_AUTO_UPDATE=0
ruff check .
ruff format --check .
pytest -q
```

PyTorch 路径需要安装兼容的 PyTorch；CUDA 测试需在空闲 GPU 上设置 `KERNEL_ENVOLVE_TEST_CUDA=1`。浏览器轮询逻辑的测试使用 Node.js，缺少时跳过对应测试。

提交 PR 时使用[中文模板](.github/PULL_REQUEST_TEMPLATE.md)，说明修改前后、影响范围、实际验证结果和回滚方式。不适用项注明原因；性能收益附版本、硬件、覆盖范围和评测证据。发布前可运行 `python3 tools/check_pr_body.py --body-file /path/to/pr.md`。`Validate PR template` CI 会在创建、编辑正文和推送后检查；校验器只从默认分支加载，不执行 PR 分支代码。

更多资料：

- [Agent 优化流程与 PR 规范](skills/kernel-envolve/SKILL.md)
- [Super-agent 设计对比与管理层验证范围](docs/super-agent-comparison.md)
- [硬件文档与参考实现](skills/kernel-envolve/references/hardware.md)
- [参考架构来源](audit/reference_sources.json)与[评测设计来源](audit/evaluator_sources.json)
- [原生 evaluator B300 验证](audit/native_gpu_061/summary.json)、[通用 evaluator B300 验证](audit/general_contract_gpu/verified/summary.json)与[输入泛化验证](audit/generality_gpu_050/summary.json)
