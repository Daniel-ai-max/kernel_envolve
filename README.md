# kernel_envolve

**一个 skill，加一个可靠的 evaluate。让 coding agent 直接迭代算子。**

Agent 负责读硬件文档、研究参考实现、提出假设、写代码和诊断；evaluate 负责独立验证、实测和保存证据。没有模型 API 封装、调度服务或数据库。Codex、Claude 等能编辑文件和运行命令的 agent 都可以使用。

## 先跑通

要求 Python 3.11+、Linux/macOS。CPU 示例只需要 NumPy。

```bash
git clone git@github.com:Daniel-ai-max/kernel_envolve.git
cd kernel_envolve
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'

kernel-envolve evaluate --task examples/vector_add --mode quick
kernel-envolve evaluate --task examples/vector_add --mode full
# 将 full 输出的 run 路径填入下行；只有 eligible_for_best=true 才能晋升。
kernel-envolve promote <run路径>
```

CPU 示例从 Python 标量循环优化成 NumPy 加法，用于真实跑通验证、计时、晋升闭环；这个加速比不代表 GPU 优化能力。

在 NVIDIA GPU 机器上安装与驱动匹配的 PyTorch 和 Triton，然后运行：

```bash
python -m pip install -e '.[cuda]'
kernel-envolve evaluate --task examples/cuda_vector_add --mode quick --timeout 600
kernel-envolve evaluate --task examples/cuda_vector_add --mode full --timeout 600
```

CUDA 示例包含尾部 mask、非连续输入和多个大小，比较 Triton 与 `torch.add`。它是可修改的起点，不预设 Triton 一定更快。

## 交给 agent

直接让 agent 读取 [SKILL.md](skills/kernel-envolve/SKILL.md) 即可。需要安装到 Codex 时，把 `skills/kernel-envolve` 复制到 `~/.codex/skills/`。示例任务：

> 使用 $kernel-envolve 优化 examples/cuda_vector_add。设备是 B200，先确认实际环境与任务语义，预算 20 分钟。保留可解释的实验记录，只有 full 确认收益才更新 best。

skill 中的循环很短：

1. 确认语义、shape/layout、精度、设备、计时范围和目标权重；先跑通基线。
2. 提出一个可检验的瓶颈假设，选择最便宜的有效实验。
3. 修复正确性；quick 筛掉明显失败/退化的候选。
4. 对有希望的候选做 full；根据原始测量和噪声判断，而非相信候选输出的分数。
5. full 合格后 promote；失败的结构性原型仍可保留为下一轮素材。

每轮无需固定重读全部文档、全量 profiling 或重复 full。结构尚未写完时先做编译/局部检查；调 tile 等参数可以成批探索。代码父版本可以来自失败原型，性能比较对象始终是已验证 best。

## 一个任务，三个文件

```text
my_task/
  task.json       固定的覆盖、精度、目标和测量设置
  task.py         输入工厂、正确性 oracle、原始 baseline
  candidate.py    本轮候选，导出 run(*inputs)
```

`task.py` 接口：

```python
def make_inputs(case: dict, seed: int) -> tuple:
    # 每次返回新的数组/张量；同一 case/seed 的值、dtype、layout、alias 关系一致。
    ...


def reference(*inputs):
    # 正确性 oracle；返回可观察的数组/张量，或其嵌套 tuple/list/dict。
    ...


def baseline(*inputs):
    # 原始实现，独立于随迭代变化的 best。
    ...
```

输入为扁平 tuple，元素是 NumPy 数组或 PyTorch tensor。shape、stride、标量参数等可放到 case args，让工厂构造输入；候选通过实际输入的 shape/stride 获取它们。参考 [CPU 示例](examples/vector_add/task.py) 和 [Triton 示例](examples/cuda_vector_add/candidate.py)。

最小配置：

```json
{
  "name": "my-operator",
  "objective": "按线上调用频次加权的延迟",
  "device": "cuda:0",
  "cases": [
    {"id": "common", "args": {"n": 4096}, "weight": 9},
    {"id": "tail", "args": {"n": 4099}, "weight": 1, "quick": false}
  ]
}
```

case 权重由任务明确给出。没有真实调用分布时，可以声明等权 benchmark 目标。`quick: false` 表示该 case 只在 full 中运行；正确性规则在两个模式中相同。

可选设置：

| 设置 | 默认 | 含义 |
|---|---|---|
| `reference_files` | `["task.py"]` | 固定 oracle/baseline/输入工厂的本地依赖 |
| `candidate_files` | `["candidate.py"]` | 候选及其本地依赖，全部进入快照 |
| `device` | `cpu` | `cpu` 或 `cuda:索引` |
| `rtol` / `atol` | `1e-5` / `1e-6` | 浮点逐元素容差，按任务数值要求配置 |
| `equal_nan` | `false` | 是否允许同位置 NaN 相等 |
| `mutable_inputs` | `[]` | 允许原地更新的输入索引；更新后的值仍须与 oracle 一致 |
| `quick_seeds` / `full_seeds` | `[0]` / `[0,1,2]` | fresh 检查种子；每个种子还有一次同对象换值检查 |
| `quick_trials` / `full_trials` | `3` / `10` | 每个 case、每批的配对轮数 |
| `warmup` | `3` | 每个实现、每批、每个 case 的预热次数 |
| `min_gain` | `0.01` | 加权延迟下降比例必须超过该值 |
| `noise_multiplier` | `2.0` | MAD 噪声门槛倍数 |
| `max_regression` | `0.05` | 任一 case 相对当前 comparator 的最大允许延迟增长 |
| case 的 `max_us` | 无 | 该 case 候选延迟的绝对上限，微秒 |

这些默认值用于起步，不是所有算子的精度或收益标准。任务建立时确定它们，迭代候选时保持不变。修改固定任务契约后，使用新的任务目录重新建立 best。

本地 helper 用相对导入，例如 `from .helper import launch`，并写入对应 files 列表。配置、构建代码、小型常量文件也要列入。快照只收集显式文件，拒绝隐藏文件和符号链接，总大小上限 32 MiB。输入采用工厂生成；外部依赖使用当前 Python 环境，结果记录 Python、平台和 NumPy/PyTorch/Triton 版本。

## evaluate 如何判定

### 正确性先于计时

每个 case、seed 都分别检查 candidate 和 comparator：

- 新建匹配的输入，与固定 oracle 比较输出结构、shape、dtype、device 和逐元素值。整数/bool 精确比较。
- 检查未声明的输入修改、输入布局改变和 alias 关系改变；声明的原地更新须与 oracle 更新后的输入一致。
- 保留原输入对象，用 `seed + 1000003` 生成的新值覆盖其内容，再执行并验证，捕获按对象地址缓存旧输出的问题。

任何数值失败都会停止该轮计时，得到 `execution=completed, correctness=fail`。导入、编译、运行异常或超时得到 `execution=error`。每轮使用新子进程，超时终止其进程组；失败的 CUDA context 不复用。失败记录包含 case、seed、fresh/reuse、实现角色及误差，日志保留异常栈。

### 测量和确认

默认 comparator 是任务当前 best，没有 best 时使用固定 baseline。每次都重新测 comparator，不把历史 timing 当分母。`--baseline` 可专门测原始 baseline，但已有 best 时，这种结果不能替换 best。

quick 测所选 case，一批样本；full 测全部 case，两批样本，第二批倒序遍历 case。每轮执行 A–B–B–A，下一轮反转顺序，A/B 分别代表 comparator/candidate。保留所有样本及 case、seed、batch、trial、role、repeat 身份。

CPU 使用 `perf_counter_ns`，测完整函数调用；CUDA 使用当前流上的 event。输入构造/复位、预热和同步等待不计入延迟；实际 `run` 内的布局转换、临时缓冲和 kernel 工作保留在调用中。每次测量前重新生成输入，适用于声明的原地更新；测得的是该输入准备方式下的延迟。

full 的每一批分别计算：

```text
T_candidate  = Σ weight(case) × median(candidate_samples)
T_comparator = Σ weight(case) × median(comparator_samples)
gain         = 1 - T_candidate / T_comparator
noise_guard  = noise_multiplier × Σ weight(case) × (MAD_candidate + MAD_comparator)
               / T_comparator
```

两批的 gain 都必须超过 `max(min_gain, noise_guard)`，且所有 case 都满足回退限制和绝对延迟上限。MAD 门槛是保守筛选规则，不是统计置信区间。结果接近门槛时，检查环境竞争、提高测量轮数并重新评测；profiling 用于解释瓶颈，不与正式计时混跑。

`--signal-only` 仅限 quick：仍检查候选正确性，只测候选延迟，适合快速调参。不产生 speedup，也不能晋升。

### 结果与 best

```text
my_task/.kernel_envolve/
  runs/<run-id>/
    source/          本轮显式源码文件快照
    request.json     模式、源码/契约/evaluator 哈希、comparator 身份
    result.json      正确性覆盖、原始计时、环境、判定及原因
    receipt.json     结果哈希
    stdout.log
    stderr.log
  best.json          指向已验证 run；同时保留 previous run 指针
```

`promote` 会校验结果和源码、重算覆盖及收益、检查 evaluator 和任务契约是否变化，并在锁内核对 best 是否仍是当时比较的版本。quick、缺失/重复样本、无效耗时、噪声过大、旧 comparator、事后修改的证据均不能晋升。best 引用保存的源码及 helper，工作目录继续编辑不会改变 best。

文件快照和哈希用于复现与防止误用；候选作为本机代码运行，这不是隔离不可信程序的安全沙箱。任务内有运行锁和晋升锁；GPU 性能实验仍需由使用者在目标设备上串行执行，其他任务目录不会自动共享设备锁。

命令退出码：`0` 表示评测正确完成（包含慢或收益不确定），`1` 表示正确性失败，`2` 表示配置/执行错误。是否值得晋升看 `eligible_for_best`，不能只看退出码。

## 第一版范围

已实现普通 CPU 函数、NumPy、PyTorch tensor 和 NVIDIA CUDA 当前流上的 kernel 评测，支持非连续输入、结构化输出和显式原地更新。更换实现语言不需要改 agent 流程：只要 `candidate.run` 能调用实际入口，并将本地依赖列入快照即可。

本版不内置 CUDA Graph、多流事件汇合、分布式通信、autograd 图、显存上限判定或任意原生工程构建缓存。遇到这些任务时，保留该 skill 的迭代方法，接入任务已有的专用 evaluator。CUDA event 指标不能直接替代 CPU launch/端到端应用延迟，输出 alias/存储布局等特殊契约也需要专用验证。GPU 示例需在目标机器上完成验证后再用于性能结论。

## 从参考设计中吸收了什么

| 参考 | 吸收的价值 | 本仓库的落地 |
|---|---|---|
| [kernel-lab](https://github.com/guts-ssss/kernel-lab/tree/c7f576b5917491d790dd091a20dcbc6c4d0c6d61) | 简短 skill、失败机制重用、实际入口验证 | 假设驱动的迭代流程；允许从未晋升原型继续探索；交付入口核对 |
| [AKO4ALL](https://github.com/TongmingLAIC/AKO4ALL/tree/bbd0e1cf1ce2fb19d4322a932680f4c3d175d80e) | 多输入验证、同对象换值哨兵 | fresh + reuse 检查；逐项覆盖记录 |
| [ForgeKernel](https://github.com/OpenBMB/ForgeKernel/tree/6664d6080c37803605d6c6bc25f87656423ff114) | 验证后晋升、保留最好实现、硬件知识 | source bundle + 原始证据 + best 指针；硬件资料按需查阅 |
| [AKO4X](https://github.com/TongmingLAIC/AKO4X/tree/ce030ebd855dffb7ba6bdaef7ba7504a92cb9393) | 候选归档、跨轮继续优化 | 普通文件、Git 和 run 目录，不另设长期调度框架 |
| [KernelAgent](https://github.com/meta-pytorch/KernelAgent) / [GEAK](https://github.com/AMD-AGI/GEAK) | 从硬件反馈解释瓶颈 | skill 中按需 profiling，资料与参考实现入口 |
| [JustGoal](https://github.com/OpenBMB/JustGoal/tree/b0a2b582e39daf3344faac1414ec7c26660633e3) | 通用 coding agent 自主探索 | 不绑定模型 SDK，探索策略交给 agent |

这版针对“已有强 coding agent，需要快速迭代单个算子”的目标更轻：只安装一个 Python 包，agent 直接用命令行工作。相较完整自动化平台，它没有批量任务调度、自动组装多 agent、分布式资源管理等能力。实现结构更适合当前需求，不等于已证明搜索效果或最终性能优于参考仓库；比较性能需要相同硬件、任务和预算上的实测。

硬件资料和算子库入口集中在 [hardware.md](skills/kernel-envolve/references/hardware.md)，只维护链接及简短用途，不复制整套 wiki。以上为设计借鉴，本仓库代码独立实现。

## 开发检查

```bash
python -m pip install -e '.[dev]'
ruff check .
ruff format --check .
pytest -q
```

测试覆盖真实 CPU 子进程评测、正确性拒绝、缓存旧输出、输入修改、超时、覆盖完整性、噪声与权重判定、快照/helper 完整性、旧 best 晋升冲突等。CI 在 Linux 的 Python 3.11/3.12/3.13 上运行 CPU 测试。PyTorch CPU 测试在装有 torch 时额外执行；CUDA 测试需显式设置 `KERNEL_ENVOLVE_TEST_CUDA=1` 并使用空闲 GPU。
