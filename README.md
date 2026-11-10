# kernel_envolve

kernel_envolve 用于辅助 coding agent 优化 Apodex 内部训练算子，主要面向 [apodex-core](https://github.com/ApodexAI/apodex-core) 和 [Megatron-LM](https://github.com/ApodexAI/Megatron-LM)。优化目标是在保持训练语义和精度的前提下，提高实际工作负载的性能，并适配不同尺寸、上下文长度和内存布局。

## 内部项目使用

让 agent 读取 [SKILL.md](skills/kernel-envolve/SKILL.md)，在目标 GPU 和项目指定的开发镜像中工作。内部评测使用 `native-evaluate`，任务由 `native.json` 和 `adapter.py` 定义。

> 使用 $kernel-envolve 持续优化 apodex-core 中的目标算子，在已授权资源范围内自主推进；一个方向没有收益时换方向继续。根据 Megatron-LM 的实际调用配置确定输入覆盖，复用项目数值 gate、单元测试和性能基准，验证前向、反向及模块接入。

- **实现位置**：kernel 和算子测试放在 apodex-core；Megatron-LM 负责调用接线、模块测试和训练验证。
- **迭代流程**：定向正确性测试与代表尺寸测速 → 有收益后完整验证相关算子 → 验证实际模块接入。分布式、重计算和训练验证按改动影响选择。
- **评测标准**：沿用项目精度要求，比较原始实现与候选的完整调用成本；训练算子重点看前向与反向总耗时、逐尺寸退化和显存。

具体流程、验证入口及证据要求见 [内部项目评测](skills/kernel-envolve/SKILL.md#apodex-内部项目评测)。内部任务复用原生测试和 benchmark，并按目标算子补齐输入覆盖、接口检查和性能收益判定。数值 gate 或 CI 通过本身不代表优化成功；下面的通用 evaluate 用于独立算子实验。

[MoE probs_scale 示例](examples/apodex_probs_scale) 接入 core 契约测试、原生数值 gate 和 Megatron 的分发及模块梯度测试。先在 `native.json` 配置候选和原始基线的源码目录；示例使用 `APODEX_CORE_CANDIDATE`、`APODEX_CORE_BASELINE`、`MEGATRON_CANDIDATE`、`MEGATRON_BASELINE` 环境变量。

```bash
# 在项目镜像内安装本仓工具，保留镜像已有依赖版本
python -m pip install --no-deps -e .
kernel-envolve native-evaluate --task examples/apodex_probs_scale --mode quick
kernel-envolve native-evaluate --task examples/apodex_probs_scale --mode full --timeout 1800
kernel-envolve native-promote <run目录>
```

原生评测冻结源码，分进程加载候选、原始基线、best 和独立参考，检查前反向输出、输入布局及状态更新。`full` 还要求声明的原生测试通过、报告齐全且无 skip，并复用两批配对计时、收益及逐用例退化判定。验证、重置和数据传输不计入调用耗时；结果保存在任务的 `.kernel_native/`。适配接口见 [内部 evaluator](skills/kernel-envolve/SKILL.md#使用内部-evaluator)。

## 持续优化

持续维护优化假设，围绕计算组织、访存、融合、反向、输入分布和模块调用寻找改进。一次晋升或暂时没有收益都不结束任务；已验证版本保持可用，探索继续推进。

每轮把实验结论、代码版本、资源消耗和下一步写入 `notes.md`，中断后从记录恢复。评测先做定向筛选，再对有收益的候选完整确认，避免重复实验和重复全量验证。具体规则见 [持续优化流程](skills/kernel-envolve/SKILL.md#持续优化)。

每次确认明确收益后，agent 自动向目标仓库提交优化 PR；同一算子的 PR 未合入时更新原分支和正文，合入后再开启下一阶段 PR。PR 使用规范的中文描述，列出改动、累计及增量收益、逐场景耗时、验证结果和复现证据。提交后继续优化，详见 [PR 交付流程与格式](skills/kernel-envolve/SKILL.md#收益确认后自动维护-pr)。

跨会话自动续跑需要宿主执行器唤起 agent；当前 CLI 提供单次评测和版本晋升，skill 提供持续探索与恢复规则。累计额度用尽或用户暂停时保存状态，满足恢复条件后再继续。

## 独立算子快速开始

通用评测器要求 Python 3.11+，支持 NumPy、PyTorch 和 NVIDIA CUDA。内部项目沿用自身 Python、PyTorch、Triton 及扩展版本，不使用此处安装命令替换项目环境。

```bash
git clone git@github.com:Daniel-ai-max/kernel_envolve.git
cd kernel_envolve
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[cuda]'

# 快速评测
kernel-envolve evaluate --task examples/cuda_vector_add --mode quick

# 完整评测
kernel-envolve evaluate --task examples/cuda_vector_add --mode full

# 保存符合条件的完整评测结果
kernel-envolve promote <run目录>
```

### 使用 Agent

让 agent 读取 [SKILL.md](skills/kernel-envolve/SKILL.md)。在 Codex 中使用时，可将 `skills/kernel-envolve` 复制到 `~/.codex/skills/`。

> 使用 $kernel-envolve 优化 examples/cuda_vector_add，目标设备为 B200，预算为 20 分钟。保持任务定义不变，通过完整评测后更新 best。

每轮围绕一个改动假设进行实现和评测。有错误时定位失败输入，有优化收益时执行完整确认。根据实际耗时选择 quick 或 full，不要求每次修改都走完所有步骤。

## 任务配置

```text
my_task/
  task.json       输入用例、精度要求和测量配置
  task.py         输入生成、正确性参考和原始基线
  candidate.py    候选实现，导出 run(*inputs)
```

`task.py` 的接口如下：

| 接口 | 作用 |
|---|---|
| `make_inputs(case, seed)` | 生成可复现的输入，返回数组或 tensor 组成的 tuple |
| `reference(*inputs)` | 提供正确性参考结果 |
| `baseline(*inputs)` | 提供固定的原始性能基线 |
| `validate(inputs, output)`，可选 | 补充精度、布局等检查；成功返回 `None`，失败抛出 `AssertionError` |

各次输入生成应使用独立存储；同一 case 的 shape、dtype 和布局不随 seed 改变。本地依赖使用相对导入，并分别列入 `reference_files` 或 `candidate_files`。接口示例见 [task.py](examples/cuda_vector_add/task.py)。

`task.json` 示例：

```json
{
  "name": "my-operator",
  "objective": "按调用频次加权的调用延迟",
  "device": "cuda:0",
  "timer": "wall",
  "cases": [
    {"id": "common", "args": {"n": 4096, "stride": 1}, "weight": 9},
    {"id": "tail", "args": {"n": 4099, "stride": 2}, "weight": 1, "quick": false},
    {"id": "neighbor", "args": {"n": 4097, "stride": 1}, "measure": false, "quick": false}
  ]
}
```

- `args`：传给输入生成函数的参数。
- `weight`：该用例的调用频次权重；无业务分布时可使用等权重。
- `quick: false`：默认只参加 full，也可通过 `--case` 单独检查。
- `measure: false`：只验证正确性，不计时；不能设置 `weight` 或 `max_us`。

任务至少包含一个参与 quick 的计时用例。常用配置如下，完整配置见 [storage.py](src/kernel_envolve/storage.py)。

| 配置 | 默认值 | 含义 |
|---|---|---|
| `rtol` / `atol` | `1e-5` / `1e-6` | 浮点比较容差 |
| `min_gain` | `0.01` | 最低总体延迟下降比例 |
| `max_regression` | `0.05` | 单个用例相对当前对照的最大允许退化 |
| `max_baseline_regression` | `0.05` | 单个用例相对原始基线的最大允许退化 |
| `mutable_inputs` | `[]` | 允许原地修改的输入索引 |

输入覆盖、精度和测量规则在任务建立时固定。修改这些规则后，应在新任务目录重新评测。

## 评测规则

### 评测模式

| 模式 | 用途 |
|---|---|
| `quick` | 在代表用例上验证正确性并比较性能，用于日常迭代 |
| `full` | 验证全部声明用例，测量两批样本，用于确认改进 |
| `--check-only` | 只检查正确性，可结合 `--case`、`--seed` 定位错误 |
| `--signal-only` | 验证正确性，仅测候选延迟，用于筛选参数 |

后两项用于 quick 模式。只有合格的 full 结果可以通过 `promote` 更新 best。

### 正确性与性能

评测器比较输出的数值、结构、shape、dtype 和 device，同时检查输入修改、共享存储及运行设置变化。它还会在原输入对象上换值重测，并抽查实际计时调用的结果。每轮使用独立子进程，保留失败输入信息和日志。

性能对照每轮重新测量：有 best 时比较 best，full 还会同时检查原始基线。候选和对照交错执行，按用例权重汇总延迟。两批结果都需满足收益和测量波动要求，各计时用例也需满足退化限制。具体判定见 [scoring.py](src/kernel_envolve/scoring.py)。

`timer: wall` 测量完整调用，CUDA 调用前后同步设备。默认 `auto` 在 CUDA 上使用 event，其区间可能包含主机发射间隙。输入生成和预热不计入延迟，候选内部的分发、布局转换和临时缓冲操作计入。性能测量应独占 GPU。内置评测采用 eager 执行，输入须为不带 autograd 的实数张量；训练梯度、CUDA Graph 和分布式任务使用项目评测入口。

### 结果使用

结果保存在 `.kernel_envolve/runs/<run-id>/`，包含源码快照、配置、原始测量和日志。`promote` 复核源码、任务配置、评测器版本和当前 best 后更新记录。

```python
from kernel_envolve import load_best

run = load_best("my_task")
output = run(*inputs)
```

退出码 `0` 表示评测正确完成，`1` 表示正确性失败，`2` 表示配置或执行错误。能否更新 best 以 `eligible_for_best` 为准。

## 输入覆盖与交付

用例应覆盖实际使用中的 size、batch、上下文长度、dtype，以及各输入独立变化的 stride、padding 和转置。性能集合应包含主要尺寸区间、分支切换位置和 fallback，避免总体收益掩盖局部退化。

迭代阶段使用代表用例，完整验证时扩大正确性覆盖；补充用例可设置 `measure: false` 以减少计时开销。交付前冻结实现，再验证未参与调参的尺寸、布局组合和数值种子，并保存结果与源码哈希。

## 资料与开发

- [硬件文档与参考实现](skills/kernel-envolve/references/hardware.md)
- [参考架构来源](audit/reference_sources.json)与[评测设计来源](audit/evaluator_sources.json)
- [内部 evaluator 的 B300 集成验证](audit/native_gpu_060/summary.json)
- [通用 evaluator 的 B300 验证](audit/general_contract_gpu/verified/summary.json)与[输入泛化验证](audit/generality_gpu_050/summary.json)

```bash
python -m pip install -e '.[dev]'
ruff check .
ruff format --check .
pytest -q
```

CUDA 测试需在空闲 GPU 上设置 `KERNEL_ENVOLVE_TEST_CUDA=1`。历史实验和对照数据保存在 `audit/`。
