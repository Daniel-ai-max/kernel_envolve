# kernel_envolve

kernel_envolve 是面向 coding agent 的算子开发与评测工具，提供优化工作流、正确性验证、性能测量和已验证实现的版本管理。

项目以算子的正确性、性能和输入适配能力为优化目标。Agent 负责分析与实现，评测器负责执行固定的验证规则并记录结果。支持 NumPy、PyTorch tensor 及 NVIDIA CUDA 当前流上的 eager 执行，可通过 Python 入口接入 Triton、CUDA 等实现。

## 安装

要求 Python 3.11+。GPU 开发直接在目标机器上进行，并安装与驱动兼容的 PyTorch 和 Triton。

```bash
git clone git@github.com:Daniel-ai-max/kernel_envolve.git
cd kernel_envolve
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[cuda]'
```

仅使用 NumPy/CPU 功能时，安装核心包：`python -m pip install -e .`。

## 使用方法

### 命令行

```bash
# 快速评测
kernel-envolve evaluate --task examples/cuda_vector_add --mode quick

# 完整评测
kernel-envolve evaluate --task examples/cuda_vector_add --mode full

# 将符合条件的完整评测结果登记为 best
kernel-envolve promote <run目录>
```

| 操作 | 用途 | 可更新 best |
|---|---|---|
| `evaluate --mode quick` | 在迭代子集上验证正确性并比较性能 | 否 |
| `evaluate --signal-only` | 验证正确性，仅测候选延迟 | 否 |
| `evaluate --case <id> --seed <n> --check-only` | 定位指定输入的正确性问题 | 否 |
| `evaluate --mode full` | 完整正确性覆盖与两批性能确认 | 结果合格后可执行 `promote` |
| `promote <run目录>` | 复核证据并更新 best | 是 |

定点选择和 `--check-only`、`--signal-only` 仅用于 quick 模式。各操作按需要选择，不要求每次修改都依次执行。

### Agent 工作流

让 agent 读取 [SKILL.md](skills/kernel-envolve/SKILL.md)。在 Codex 中使用时，可将 `skills/kernel-envolve` 复制到 `~/.codex/skills/`。

任务指令示例：

> 使用 $kernel-envolve 优化 examples/cuda_vector_add。目标设备为 B200，开发预算为 20 分钟。保持任务定义不变，依据完整评测结果更新 best。

工作流包括：

1. 确定计算语义、输入范围、精度要求、目标设备与性能目标，验证原始实现。
2. 根据瓶颈提出改动假设，修改候选实现。
3. 选择定点检查、quick 或 full 获取反馈；按需使用 profiler。
4. 对具备交付条件的候选执行 full，检查收益、逐项退化和测量波动。
5. 登记通过验证的实现，保存实验结论；交付前补充未参与调参的输入组合验证。

探索中的原型与已验证 best 分开保存。编译、初始化和测量开销可从评测记录中查看，用于选择后续实验的范围。硬件文档和参考实现见[资源索引](skills/kernel-envolve/references/hardware.md)。

## 任务定义

```text
my_task/
  task.json       输入覆盖、精度、性能目标与测量配置
  task.py         输入工厂、正确性参考与原始性能基线
  candidate.py    候选实现，导出 run(*inputs)
```

`task.py` 提供以下接口：

| 接口 | 约定 |
|---|---|
| `make_inputs(case, seed)` | 返回 NumPy 数组或实数 PyTorch tensor 组成的扁平 tuple；各次调用使用独立存储，同一 case/seed 可复现，shape/dtype/layout 不随 seed 改变 |
| `reference(*inputs)` | 正确性参考实现（oracle）；返回数组、tensor 或嵌套 tuple/list/dict |
| `baseline(*inputs)` | 固定的原始性能基线，不随 best 更新 |
| `validate(inputs, output)`，可选 | 验证精度、alias、布局或其他语义；成功返回 `None`，失败抛出 `AssertionError` |

候选通过实际输入的 shape、stride 和 dtype 获取元数据。评测器同时对 oracle 和被测实现执行自定义 `validate`。独立高精度参考的示例见 [reduction/task.py](examples/reduction/task.py)。

配置示例：

```json
{
  "name": "my-operator",
  "objective": "按调用频次加权的 eager 调用延迟",
  "device": "cuda:0",
  "timer": "wall",
  "cases": [
    {"id": "common", "args": {"n": 4096, "stride": 1}, "weight": 9},
    {"id": "tail", "args": {"n": 4099, "stride": 2}, "weight": 1, "quick": false},
    {"id": "neighbor", "args": {"n": 4097, "stride": 1}, "measure": false, "quick": false}
  ]
}
```

`cases[].args` 传入输入工厂。`quick: false` 的 case 默认只参加 full，也可通过 `--case` 单独检查。`measure: false` 的 case 只做正确性验证，不预热、不计时、不参与性能得分，且不能设置 `weight` 或 `max_us`。任务至少包含一个参与 quick 的计时 case。

### 主要配置

| 配置 | 默认值 | 说明 |
|---|---|---|
| `device` | `cpu` | `cpu` 或 `cuda:索引` |
| `timer` | `auto` | CPU 使用 wall time；CUDA 使用 event；也可显式选择 `wall` 或 `cuda_event` |
| `rtol` / `atol` | `1e-5` / `1e-6` | 浮点比较容差 |
| `equal_nan` | `false` | 是否允许同位置 NaN 相等 |
| `deterministic` | `false` | 相同输入复位后重跑，要求结果逐元素精确一致 |
| `mutable_inputs` | `[]` | 允许原地修改的输入索引；修改结果仍需与 oracle 一致 |
| `quick_seeds` / `full_seeds` | `[0]` / `[0,1,2]` | 正确性检查种子 |
| `quick_trials` / `full_trials` | `3` / `10` | 每个计时 case、每批的配对测量轮数 |
| `warmup` | `3` | 每个实现、每个计时 case、每批的预热次数 |
| `min_gain` | `0.01` | 最低加权延迟下降比例 |
| `noise_multiplier` | `2.0` | 配对差值的离散度门槛倍数 |
| `max_regression` | `0.05` | 单个计时 case 相对当前对照的最大允许退化 |
| `max_baseline_regression` | `0.05` | 单个计时 case 相对原始基线的最大允许累计退化 |
| `cases[].weight` | 必填，计时 case | 调用频次权重；无业务分布时可声明等权 benchmark |
| `cases[].max_us` | 无 | 单个计时 case 的延迟上限，单位微秒 |
| `reference_files` | `["task.py"]` | 参考实现及其本地依赖 |
| `candidate_files` | `["candidate.py"]` | 候选实现及其本地依赖 |

本地依赖使用相对导入，并列入相应文件清单。源码快照总大小上限为 32 MiB；编译产物和缓存存放在快照之外。覆盖、精度和测量配置在任务建立时固定；修改任务定义或评测协议后，应在新任务目录重新建立 best。

## 评测规则

### 正确性验证

评测器分别验证候选、当前对照及需要参与 full 的原始基线：

- 与 oracle 比较输出结构、shape、dtype、device 和数值；整数、bool 精确比较，浮点按配置容差比较。
- 检查未声明的输入修改、存储绑定和 alias 变化；冻结 oracle 输出，避免共享缓冲覆盖结果。
- 在原输入对象上换入符合任务取值域的新数据，检查按地址缓存旧结果等状态错误。
- 对计时 case 抽查最后一次实际计时调用的输出和输入状态，随后进行换值检查。
- 检查精度、线程等公共运行设置的持久变化，执行任务定义的自定义验证。

每轮评测使用独立子进程。超时会终止其进程组，失败的 CUDA context 不复用。结果保留失败 case、seed、检查阶段和异常日志。

### 性能测量

默认对照为当前 best；尚无 best 时使用原始 baseline。full 在已有 best 时同时重新测量原始 baseline，检查相对 best 的收益和相对原始实现的累计退化。

quick 测量一批样本，full 测量两批，第二批逆序遍历 case。两个实现采用交错的 A–B–B–A 顺序；三个实现采用轮换起始角色的 A–B–C–C–B–A 顺序。各角色使用匹配的输入值，每个测量样本更新输入。

| 计时方式 | 测量范围 |
|---|---|
| CPU wall | 完整函数调用 |
| CUDA event | 当前流上的 eager 事件区间，可能包含主机发射间隙 |
| CUDA wall | 调用前后同步设备，包含主机调用与设备完成等待 |

输入构造、复位和预热位于计时区间之外；`run` 内部的分发、布局转换和临时缓冲操作计入延迟。性能测量应独占目标 GPU，避免与其他 benchmark 或 profiler 并发。

完整评测按 case 权重计算总延迟：

```text
T = Σ weight(case) × median(latency_samples)
gain = 1 − T_candidate / T_comparator
```

相对当前对照和原始基线，两批结果均需达到最低总体收益，且配对收益超过最低收益和噪声门槛；所有计时 case 同时满足逐项退化限制和绝对延迟上限。噪声尺度使用配对差值的 `max(MAD, IQR/2)`，用于筛选测量波动，不作为统计置信区间。具体计算见 [scoring.py](src/kernel_envolve/scoring.py)。

CUDA Graph、多流、通信、backward 或显存约束任务使用覆盖相应执行语义的专用评测器。

### 结果与版本管理

评测结果保存在任务目录的 `.kernel_envolve/runs/<run-id>/`，包含源码快照、请求配置、原始计时、环境信息、判定结果和日志。`phase_wall_seconds` 记录初始化、正确性、测量及结果核查的耗时。

`promote` 仅接受合格的 full 结果，并重新核对源码、任务定义、评测器版本、覆盖完整性及当前 best 的身份。`best.json` 指向已验证源码快照，工作目录中的后续编辑不会改变已登记版本。

```python
from kernel_envolve import load_best

run = load_best("my_task")
output = run(*inputs)
```

退出码 `0` 表示评测正确完成，`1` 表示正确性失败，`2` 表示配置或执行错误。是否允许更新 best 由 `eligible_for_best` 决定。

## 输入覆盖与交付

任务覆盖应同时考虑尺寸、数据类型、数值分布和各输入的独立布局。重点包括非整倍数、分块阈值邻域、行间 padding、转置、空输入，以及实际调用入口中的 dispatch 和 fallback。

| 算子类型 | 主要覆盖维度 |
|---|---|
| elementwise / 融合 | 元素数、dtype、独立 stride、尾部与对齐 |
| 归约 / normalization | 行数、归约长度、数值抵消、动态范围 |
| GEMM | 独立变化的 M/N/K、dtype、转置和 leading dimension |
| attention | batch、query/KV 长度、head dimension、GQA、mask 和布局 |

迭代时使用代表子集获取反馈，full 验证完整开发覆盖；正确性补充 case 可关闭计时。性能集合应包含各主要输入区间和分支，逐项报告退化情况。

交付前冻结候选，在支持范围内验证未参与调参的尺寸、布局组合和数值种子，并保存输入定义、源码哈希与结果。full 只覆盖 `task.json` 声明的 case；未见输入由独立验证任务或项目测试补充。

## 参考设计

| 项目 | 参考机制 | 本项目的采用方式 |
|---|---|---|
| [kernel-lab][kernel-lab] | 独立 oracle、固定 baseline、累计预算、实际入口验证 | 原始基线与 best 分开维护，full 检查两套退化限制 |
| [AKO4ALL][ako4all] | signal/verdict 分离、换值检查、计时后检查 | 区分探索反馈与正式确认，验证输入状态和实际计时输出 |
| [KDA][kda] | 任务约定、实现、验证、证据与版本登记 | 用任务文件和 CLI 连接开发与评测 |
| [Atrex][atrex] | 配对测量、源码绑定、覆盖与接纳审计 | 校验交错顺序、完整记录及晋升条件 |
| [ForgeKernel][forgekernel] | 精度验证、保留重构原型、复核表面收益 | 分离探索原型与 best，支持自定义数值验证 |
| [AKO4X][ako4x] | 候选归档、父版本与失败经验 | 保存源码和实验记录，区分归档与晋升 |
| [JustGoal][justgoal] | 目标驱动的自主搜索 | Agent 自主选择实验路线 |
| [KernelAgent][kernelagent] | Profiling 与瓶颈分析 | 按需分析，测量结果绑定具体实现 |
| [GEAK][geak] | 独立验证、组合优化与原始基线对照 | 保留反馈、验证和复测环节 |
| [BackendBench][backendbench] | 分离正确性和性能测试集 | 通过 `measure` 配置两类覆盖 |

固定提交、源码路径和哈希见[架构来源清单](audit/reference_sources.json)与[评测来源清单](audit/evaluator_sources.json)。

## 验证记录

B300 上最近一次验证通过 122 项测试；归约、GEMM、融合任务的 6 组正确实现与 24 个故障反例在开发评测器和独立验收中均得到预期结果。每项任务包含 24 个正确性 case、12 个计时 case，覆盖 FP32、FP16、BF16；full 单次总耗时分别约为 3.98、7.69、7.14 秒。配置、环境和原始记录见[验证摘要](audit/general_contract_gpu/verified/summary.json)。

专项记录包括[输入泛化验证](audit/generality_gpu_050/summary.json)、[AKO4ALL 组件对照](audit/ako4all_comparison.json)、[BackendBench 组件对照](audit/backendbench_component_comparison.json)及[收益判定验证](audit/paired_policy_gpu.json)。实验脚本和历史数据保存在 `audit/`，不随运行包安装。

## 开发

```bash
python -m pip install -e '.[dev]'
ruff check .
ruff format --check .
pytest -q
```

CI 覆盖 Linux Python 3.11/3.12/3.13。CUDA 测试需在空闲 GPU 上设置 `KERNEL_ENVOLVE_TEST_CUDA=1`。

[kernel-lab]: https://github.com/guts-ssss/kernel-lab/blob/c7f576b5917491d790dd091a20dcbc6c4d0c6d61/references/evaluation.md
[ako4all]: https://github.com/TongmingLAIC/AKO4ALL/blob/bbd0e1cf1ce2fb19d4322a932680f4c3d175d80e/bench/kernelbench/bench.py
[kda]: https://github.com/NVlabs/kda/blob/acdd79343c5e9d0c7cb96289f26fd3f1051ce785/docs/agent-flow.md
[atrex]: https://github.com/alibaba/atrex-kernel-agent/blob/bab3218768baea30ba56a87851705d60d64ebbf3/src/aka/light/acceptance/recorded_verifier.py
[forgekernel]: https://github.com/OpenBMB/ForgeKernel/blob/6664d6080c37803605d6c6bc25f87656423ff114/src/forgekernel/loop/driver.py
[ako4x]: https://github.com/TongmingLAIC/AKO4X/blob/ce030ebd855dffb7ba6bdaef7ba7504a92cb9393/master/master.py
[justgoal]: https://github.com/OpenBMB/JustGoal/blob/b0a2b582e39daf3344faac1414ec7c26660633e3/README.md
[kernelagent]: https://github.com/meta-pytorch/KernelAgent/blob/e0647170da36ef9b059ac0bd3d60103aa4ed378b/triton_kernel_agent/opt_worker_component/orchestrator/optimization_orchestrator.py
[geak]: https://github.com/AMD-AGI/GEAK/blob/f189eff7c4036d9e9bd1d4fe99ec5c25ab6b8b20/kernel_workflow/README.md
[backendbench]: https://github.com/meta-pytorch/BackendBench/blob/2a8d7e19e2c13c789ff2dceb44883ea8ea04dab4/BackendBench/eval.py
