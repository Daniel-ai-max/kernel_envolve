# kernel_envolve

kernel_envolve 是面向 coding agent 的算子开发与评测工具。Agent 负责分析和编写实现，评测器负责验证正确性、比较性能，并保存已验证的最佳版本（best）。

支持 NumPy、PyTorch 和 NVIDIA CUDA 算子，优化目标是正确、快速，并适配实际使用中的尺寸、数据类型和内存布局。

## 快速开始

要求 Python 3.11+。GPU 开发在目标机器上进行，PyTorch 和 Triton 版本需与驱动兼容。

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

仅使用 CPU 时，安装 `python -m pip install -e .`，示例位于 `examples/vector_add`。

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

`timer: wall` 测量完整调用，CUDA 调用前后同步设备。默认 `auto` 在 CUDA 上使用 event，其区间可能包含主机发射间隙。输入生成和预热不计入延迟，候选内部的分发、布局转换和临时缓冲操作计入。性能测量应独占 GPU。内置评测采用 eager 执行；CUDA Graph 等任务需使用对应评测器。

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
- [B300 验证记录](audit/general_contract_gpu/verified/summary.json)与[输入泛化验证](audit/generality_gpu_050/summary.json)

```bash
python -m pip install -e '.[dev]'
ruff check .
ruff format --check .
pytest -q
```

CUDA 测试需在空闲 GPU 上设置 `KERNEL_ENVOLVE_TEST_CUDA=1`。历史实验和对照数据保存在 `audit/`。
