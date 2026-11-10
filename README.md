# kernel_envolve

**一个 skill，加一个可靠的 evaluate。让 coding agent 直接迭代算子。**

Agent 负责读硬件文档、研究参考实现、提出假设、写代码和诊断；evaluate 负责独立验证、实测和保存证据。没有模型 API 封装、调度服务或数据库。Codex、Claude 等能编辑文件和运行命令的 agent 都可以使用。

## 什么才算好的算子

**目标是正确、快速，并适配实际工作负载的整个输入范围。** 不能把范围缩成几个容易赢的 benchmark 点。对外应提供稳定的调用入口，size、batch、上下文长度等变化由实现处理，不需要用户手动改代码、换 tile 或重新调参。

- **正确：** 保持计算语义、精度、输出结构、输入修改规则；改变归约顺序或使用低精度时，验证数值敏感输入。涉及训练时还要验证所需的 backward。只在随机正常值上 allclose 通过不够。
- **快速：** 根据真实调用分布评估总体收益，同时检查各尺寸区间、布局和 dtype 的表现。大尺寸的收益不能掩盖常见小尺寸明显变慢；既检查相对当前 best 的退化，也检查相对原始实现的累计退化。
- **泛用：** 测常用值，也测开发时未用于调参的合法尺寸、分块/dispatch 阈值附近、非整倍数和不同布局。仅更换随机种子只验证输入数值变化，不验证尺寸泛化。
- **可用：** 最终交付实际调用入口，包括 dispatch、布局转换、临时内存和 fallback；按部署所需的 eager/Graph 方式测量。首轮编译、稳态延迟和显存开销分别观察，避免用一个稳态数字掩盖实际接入成本。

泛用可以通过少量有依据的分区实现，例如小尺寸与大尺寸采用不同 kernel；正常支持范围内未命中优化分支时，使用正确的通用实现。验证完整 dispatch 的所有分支和切换邻域，测量 fallback 含分发开销的性能，并报告哪些区域获得加速。新增尺寸应自动工作，不能通过输入地址、种子、评测状态识别 benchmark。

### 怎样选覆盖，兼顾泛用性和迭代效率

从实际调用范围建立 case 集合，按影响实现策略的维度取代表点和交叉组合，不机械枚举所有维度的笛卡尔积。下列值按任务实际支持范围选择；例如非因果 attention 未支持时，不把它混入同一个语义契约。

| 算子 | 应变化的主要维度 | 特别容易遗漏的位置 |
|---|---|---|
| elementwise / 融合 | 元素数、dtype、连续/步长布局 | 单元素、小张量、大张量、tile 前后一个元素、非对齐尾部 |
| 归约 / normalization | 行数、归约长度、dtype、数值分布 | 很短/很长的归约、非 2 的幂、抵消、大动态范围、不同归约分块 |
| GEMM | M/N/K 独立变化、dtype、转置/stride | 瘦长矩阵、各维非对齐、K 跨分块阈值、小 batch |
| attention | batch、query/KV 长度、head dimension、head 数与 GQA 比例、mask/layout | decode 的短 query、不同 prefill 长度、长上下文、ragged/mask、不同算法切换处 |

一次任务建立以下覆盖，并始终使用同一对外入口：

1. **迭代子集：** 少量代表不同瓶颈和分支的 case，用于 quick 或定点检查。每轮不必全量扫描。
2. **完整开发覆盖：** 各输入区间和边缘情况进入 full；总体权重按真实调用分布设置，逐 case 回退门槛始终生效。按区域汇总，报告最差退化和 fallback 表现，不能只报平均 speedup。
3. **交付泛化验证：** 冻结候选后，在支持范围内选取未用于调参的尺寸、维度组合和数值种子，同时验证正确性与性能。从已有范围采样并记录采样规则和实际 case，便于复现；不添加新的任务语义。失败就修复并重新验收最终源码，不能删除失败 case 来宣布成功。

这些是覆盖的用途，不是每次修改都要串行执行的三个阶段。最终泛化验证提供新的覆盖，不是重复跑相同 full。若真实业务只使用少数固定 shape，可以针对它们优化并如实交付固定 workload 的结果；默认通用任务不能自行缩成这种目标。

**当前实现对应什么：** `task.json` 的 `cases` 声明尺寸、dtype 和布局参数，`quick: false` 的 case 只参加 full；full 对所有声明 case 执行正确性与计时，并限制逐项退化。evaluator 不会根据三个 shape 自动推断整个范围，也不会自动生成未见尺寸。需要在任务开始时构造广域 case，并在交付时用独立验证任务或项目已有测试补充未见尺寸；保存其结果与最终源码哈希。只通过内置 full 的 best，表示它通过了该固定任务的覆盖，不自动表示完成泛化验收。

下面的 vector-add 示例以及此前三任务预实验仍是小型接口/可行性示例，未完成上述广域和未见尺寸验收；已有结果不作为通用性证明。

## 先跑通

默认直接在目标 NVIDIA GPU 机器上运行 agent 的文件/命令工具和 evaluator；模型推理仍由模型服务提供。代码、依赖和评测结果留在同一个工作目录，正常迭代直接调用 CLI。无需单独申请 CPU 实例，也没有先 CPU 后 GPU 的准备阶段。

要求 Python 3.11+，GPU 环境安装与驱动匹配的 PyTorch 和 Triton：

```bash
git clone git@github.com:Daniel-ai-max/kernel_envolve.git
cd kernel_envolve
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[cuda]'

# 下面是不同反馈方式，按需要选择，不要求依次执行。
kernel-envolve evaluate --task examples/cuda_vector_add --mode quick --timeout 600
kernel-envolve evaluate --task examples/cuda_vector_add --mode full --timeout 600
# 将合格 full 输出的 run 路径填入下行。
kernel-envolve promote <run路径>
```

CUDA 示例包含尾部 mask、非连续输入和多个大小，比较 Triton 与 `torch.add`。第一次实际 evaluate 即可发现设备、依赖和编译问题；无需额外增加一套固定自检阶段。新增 evaluator 的正确/错误对照验证属于验证器开发，已有验证器无需每轮重跑。

CPU 算子也受支持：安装核心包 `pip install -e .`，评测 `examples/vector_add` 即可。这个 NumPy 示例用于无 GPU 时了解接口，不是 GPU 开发的前置步骤。[独立安装验收](audit/core_package_validation.json)已验证 full → promote → load_best。

## 交给 agent

直接让 agent 读取 [SKILL.md](skills/kernel-envolve/SKILL.md) 即可。需要安装到 Codex 时，把 `skills/kernel-envolve` 复制到 `~/.codex/skills/`。示例任务：

> 使用 $kernel-envolve 优化 examples/cuda_vector_add。设备是 B200，先确认实际环境与任务语义，预算 20 分钟。保留可解释的实验记录，只有 full 确认收益才更新 best。

skill 中的循环很短：

1. 确认语义、尺寸范围、dtype/layout、精度、设备和真实调用分布；构造代表 case 与广域覆盖，跑通基线。
2. 提出一个可检验的瓶颈假设，选择最便宜的有效实验。
3. 选择反馈范围：quick 自带正确性检查；若 quick 成本接近 full 且候选大概率进入确认，直接 full。
4. 对有希望的候选做 full；根据原始测量和噪声判断，而非相信候选输出的分数。
5. full 合格后 promote；失败的结构性原型仍可保留为下一轮素材。

正常迭代无需固定执行 check-only → quick → full 三连：check-only 用于修复已知失败，quick/signal 用于筛选，full 可直接运行。每轮无需固定重读全部文档、全量 profiling 或重复 full。结构尚未写完时先做编译/局部检查；调 tile 等参数可以成批探索。代码父版本可以来自失败原型，性能比较对象始终是已验证 best。

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
    # 返回独立存储且符合输入取值域的数据；同一 case/seed 值一致，dtype/layout/shape/alias 不随 seed 变。
    ...


def reference(*inputs):
    # 正确性 oracle；返回可观察的数组/张量，或其嵌套 tuple/list/dict。
    ...


def baseline(*inputs):
    # 原始实现，独立于随迭代变化的 best。
    ...
```

可运行的 [reduction 示例](examples/reduction/task.py)演示了普通 allclose 以外的精度检查：用 `math.fsum` 作为独立求和方法，限制相对 golden 的误差。测试验证了“容差内的小偏置被拒绝、精度改善的实现被接受”。示例门槛只适用于该任务，不能套到所有算子。

任务还可以定义 `validate(inputs, output)`，用来检查输出 alias、布局或数学不变量：成功返回 `None`，失败抛出 `AssertionError`。evaluator 对 oracle 和每个实现都执行该检查；oracle 不满足它时报告任务错误，候选不满足时报告正确性失败。它的源码同样属于固定契约。复杂统计精度/梯度验证仍使用任务已有的专用 evaluator。

输入为扁平 tuple，元素是 NumPy 数组或不带 autograd 的实数 PyTorch tensor。shape、stride、标量参数等可放到 case args，让工厂构造输入；候选通过实际输入的 shape/stride 获取它们。参考 [CPU 示例](examples/vector_add/task.py) 和 [Triton 示例](examples/cuda_vector_add/candidate.py)。

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
| `timer` | `auto` | CPU wall time / CUDA eager event 区间（可能含 host 发射间隙）；`wall` 在 CUDA 上测含设备同步的完整调用 |
| `deterministic` | `false` | 同值复位后重跑，要求逐元素精确一致；不是通用 bitwise 规则 |
| `rtol` / `atol` | `1e-5` / `1e-6` | 浮点逐元素容差，按任务数值要求配置 |
| `equal_nan` | `false` | 是否允许同位置 NaN 相等 |
| `mutable_inputs` | `[]` | 允许原地更新的输入索引；更新后的值仍须与 oracle 一致 |
| `quick_seeds` / `full_seeds` | `[0]` / `[0,1,2]` | fresh 检查种子；每个种子还有一次同对象换值检查 |
| `quick_trials` / `full_trials` | `3` / `10` | 每个 case、每批的配对轮数 |
| `warmup` | `3` | 每个实现、每批、每个 case 的预热次数 |
| `min_gain` | `0.01` | 加权延迟下降比例必须超过该值 |
| `noise_multiplier` | `2.0` | 配对差值的 MAD/IQR 噪声门槛倍数 |
| `max_regression` | `0.05` | 任一 case 相对当前 comparator 的最大允许延迟增长 |
| `max_baseline_regression` | `0.05` | 任一 case 相对固定起始 baseline 的累计延迟增长上限 |
| case 的 `max_us` | 无 | 该 case 候选延迟的绝对上限，微秒 |

这些默认值用于起步，不是所有算子的精度或收益标准。任务建立时确定它们，迭代候选时保持不变。修改固定任务契约后，使用新的任务目录重新建立 best。

本地 helper 用相对导入，例如 `from .helper import launch`，并写入对应 files 列表。配置、构建代码、小型常量文件也要列入。快照只收集显式文件，拒绝隐藏文件和符号链接，总大小上限 32 MiB。快照中不允许未列出的新文件；编译产物、缓存写到快照外。Python 入口与 helper 直接从已校验源码编译，忽略时间戳匹配但内容过期的 `.pyc`。输入采用工厂生成；外部依赖使用当前 Python 环境，结果记录 Python、平台和 NumPy/PyTorch/Triton 版本。

## evaluate 如何判定

### 正确性先于计时

每个 case、seed 都分别检查 candidate 和 comparator；full 比较 best 时还检查固定 baseline：

- 拒绝跨工厂调用共用输入存储，以及同一 case 随 seed 改变 shape/layout 的情况。否则 oracle 与候选可能互相污染，或把新 shape 的 JIT 编译计入延迟。
- 新建匹配的输入，与固定 oracle 比较输出结构、shape、dtype、device 和逐元素值。整数/bool 精确比较。浮点比较提升计算精度，避免比较过程自身在 FP16 溢出；不改变被检查输出的 dtype 契约。
- 在候选运行前冻结 oracle 的输出，在后续实现/oracle 运行前冻结抽查的真实计时输出，防止共享返回缓冲被覆盖后产生假 PASS。
- `deterministic=true` 时，在相同值、相同对象上复位并重跑，要求结果逐元素一致；普通误差容差仍单独检查。
- 检查未声明的输入修改、输入布局/存储绑定改变和 alias 关系改变（含零 stride 广播视图）；声明的原地更新须与 oracle 更新后的输入一致。
- 保留原输入对象，用任务工厂按 `seed + 1000003` 生成的新值覆盖其内容，再执行并验证，捕获按对象地址缓存旧输出的问题。换值仍遵守任务取值域，例如 sqrt 的正数输入、合法索引；不统一换成正态随机数。
- 对每个 case/实现，直接检查最后一个实际计时调用的返回值及输入状态（`timed_output`），而非重新调用候选来代替它；这是抽查，不是所有计时输出的全量验证。随后在同一输入对象上换值验证，捕获预热/计时后才出现的错误。`--check-only` 没有计时阶段，只做前面的检查。

任何数值失败都会停止该轮计时，得到 `execution=completed, correctness=fail`。导入、编译、运行异常或超时得到 `execution=error`。每轮使用新子进程，超时终止其进程组；失败的 CUDA context 不复用。失败记录包含 case、seed、fresh/reuse/timed_output/post、实现角色及误差，日志保留异常栈。

### 测量和确认

默认 comparator 是任务当前 best，没有 best 时使用固定 baseline。full 有 best 时还在同一轮重新测固定 baseline，分别检查增量收益和累计回退。没有 best 时两个对照是同一个实现，只测一次。每次都重新测对照，不把历史 timing 当分母。`--baseline` 可专门测原始 baseline，但已有 best 时，这种结果不能替换 best。

quick 测所选 case，一批样本；full 测全部 case，两批样本，第二批倒序遍历 case。两个实现采用 A–B–B–A 并轮换起始角色；三个实现采用 A–B–C–C–B–A，并在各轮循环改变起始角色。每个角色的两次调用使用不同的值，同一 repeat 的各角色使用匹配的值。输入种子由 case、batch、trial、repeat 和任务 seeds 确定，每个测量样本都更新，可从记录复现。判定时同时核对样本身份、数量和实际执行顺序。

CPU 使用 `perf_counter_ns`，测完整函数调用；CUDA 默认使用当前流上的 event。**event 记录的是 eager 调用的事件区间：GPU 提前执行了 start event 时，等待 Python 发射 kernel 的空闲时间也可能计入，不能称为纯 kernel 时长。** 设置 `timer=wall` 时，CUDA 计时在调用前后做设备同步，包含 CPU launch 和完成等待；它与 event 指标是不同的任务契约，不能混比。输入构造/复位、预热和计时前同步在区间外；wall 的完成同步在区间内，event 的主机等待在区间外。实际 `run` 内的布局转换、临时缓冲和 kernel 工作保留在调用中。每次测量前重新生成输入，适用于声明的原地更新；测得的是该输入准备方式下的延迟。

full 的每一批分别计算：

```text
T_candidate  = Σ weight(case) × median(candidate_samples)
T_comparator = Σ weight(case) × median(comparator_samples)
gain         = 1 - T_candidate / T_comparator
D(case, trial) = median(本组两次 candidate 调用) - median(本组两次 comparator 调用)
paired_gain  = -Σ weight(case) × median(D(case, :)) / T_comparator
dispersion   = max(MAD(D), IQR(D) / 2)
noise_guard  = noise_multiplier × Σ weight(case) × dispersion(case) / T_comparator
```

两批相对 best 都须满足 **`gain > min_gain` 且 `paired_gain > max(min_gain, noise_guard)`**；相对固定 baseline 也单独计算并满足相同条件。前者保留声明的加权总延迟目标，后者要求配对结果也支持收益，不能拿配对差值的低波动为不一致的总体中位数收益背书。所有 case 仍须满足两套回退限制和绝对延迟上限。这避免逐轮 4% 的局部退化越积越大，也避免只赢过一个已经退化的 best。MAD/IQR 门槛仍是保守筛选规则，不是统计置信区间。结果接近门槛时，先检查环境竞争；确需调整协议时记录原因、保留旧结果，重新建立对应契约和对照。不要重复抽样直到出现有利结果；profiling 用于解释瓶颈，不与正式计时混跑。

`--case`、`--seed`、`--check-only` 仅限 quick，可定位 full 中才运行的 case，不修改固定任务契约，也不能产生晋升资格。`worker_process_wall_seconds` 记录子进程启动到退出的等待耗时；`worker_wall_seconds` 还包含父进程的结果核查。CLI 同时输出 `phase_wall_seconds`：setup（CUDA/模块初始化）、correctness、measurement（预热、输入生成、计时及后检查）、finalize（证据检查与评分）。阶段计时不进入算子计时区间，失败时保留已完成及当前阶段耗时；子进程耗时与阶段和的差值还包括 Python/NumPy 导入、退出及父进程唤醒，不能全部称为模型或 kernel 时间。兼容字段 `measurement_wall_seconds` 继续表示测量阶段总耗时。

0.4.1 起，执行异常直接在 CLI 的 `error` 中返回失败阶段；捕获的编译/子进程错误保留 stdout/stderr 各末尾 8192 个字符、截断标记及退出码（若有）。agent 可直接据此修复，不必为获取编译错误重复执行。完整 worker 日志仍保存在 run 目录。

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

父进程首先核对 worker 结果与原请求的模式、身份、选择范围等字段，并从原始证据重新计算判定；无效报告保留到 `rejected_result.json`，不发放合格 receipt。

`promote` 会校验结果和源码、重算覆盖及收益、检查 evaluator 和任务契约是否变化，并在锁内核对 best 是否仍是当时比较的版本。quick、缺失/重复/乱序样本、无效耗时、噪声过大、旧 comparator、事后修改的证据均不能晋升。与 best 完全相同的候选源码也不能靠噪声再次晋升；注释变更虽会改变哈希，但仍需 agent 核实实际生效的机制变化。best 引用保存的源码及 helper，工作目录继续编辑不会改变 best。

调用已验证版本时使用稳定入口：

```python
from kernel_envolve import load_best

run = load_best("my_task")  # 校验 best 的结果和源码哈希，加载保存的 helper
output = run(*inputs)
```

测试在新 Python 进程中验证该入口：即使工作目录的 candidate 已损坏，也仍调用保存的 best；快照损坏则拒绝加载。集成进已有模型的 dispatcher 时，仍需验证模型实际调用了这一入口。

0.4 将噪声门槛改为“总体收益 + 配对收益与配对噪声”共同确认，覆盖和调用次数不变。旧 run 保留为历史，不能直接沿用其晋升资格；已有 best 的源码仍保存在快照中。需要按新协议重新建立 best 时，将 task.json、task.py、candidate.py 及声明的依赖复制到新任务目录后重测，不修改旧证据。

文件快照和哈希用于复现与防止误用；候选作为本机代码运行，这不是隔离不可信程序的安全沙箱。任务内有运行锁和晋升锁；GPU 性能实验仍需由使用者在目标设备上串行执行，其他任务目录不会自动共享设备锁。

命令退出码：`0` 表示评测正确完成（包含慢或收益不确定），`1` 表示正确性失败，`2` 表示配置/执行错误。是否值得晋升看 `eligible_for_best`，不能只看退出码。

## 第一版范围

已实现普通 CPU 函数、NumPy、PyTorch tensor 和 NVIDIA CUDA 当前流上的 kernel 评测，支持非连续输入、结构化输出和显式原地更新。更换实现语言不需要改 agent 流程：只要 `candidate.run` 能调用实际入口，并将本地依赖列入快照即可。

本版不内置 CUDA Graph、多流事件汇合、分布式通信、autograd 图、显存上限判定或任意原生工程构建缓存。遇到这些任务时，保留该 skill 的迭代方法，接入任务已有的专用 evaluator。CUDA event 指标不能直接替代 CPU launch/端到端应用延迟，输出 alias/存储布局等特殊契约通过 task.validate 显式验证。B300 上的实际验证见下文；迁移硬件或运行方式时仍按对应契约重测。

## 详细设计对比与审查结论

**建议：对“已有强 coding agent，在现有机器上迭代少量算子”的需求，采用本仓库作为默认工作流。** 原因是探索约束少，而正确性、同期比较、累计预算和晋升证据由可测试的代码执行。这个选择不等于已经证明它在搜索收益、复杂精度验证、远端恢复、端到端系统优化上超过所有参考仓库。

本次以 9 个仓库的固定提交为参照，复核 17 份关键源码/规范，并逐文件与远端内容核对。[来源清单](audit/reference_sources.json)保存提交、路径和 SHA-256。下面区分实际代码、流程规范与未做的性能实验，不把“参考仓库没在这个文件出现某能力”写成“整个仓库没有该能力”。

### 先修复自己的差距

| 第一版的问题 | 对照依据 / 反例 | 本次实现 | 可重复验证 |
|---|---|---|---|
| 只限制相对 best 的回退 | kernel-lab 要求固定 baseline 累计预算；100→104→108 每步可低于 5%，累计却超过 | full 同轮测 best 和起始 baseline，逐 case 两套预算，两个目标的收益/噪声分别确认 | `test_cumulative_regression_cannot_hide_behind_better_aggregate`、`test_improving_degraded_best_is_not_original_baseline_win` |
| 新对象却循环使用相同数值 | AKO4ALL 的 fresh-values 模式防止值缓存影响计时 | 每个 timed sample 换值，各角色按 sample 匹配；计时输出抽查，再换值验证 | `test_timing_values_vary_and_match_across_roles`、`test_actual_timed_output_is_checked_without_candidate_rerun`、`test_post_check_detects_failure_after_initial_correctness` |
| 全量覆盖不等于按正确顺序测量 | Atrex 同时约束 ABBA 调度与完整记录 | 重算样本身份与顺序，拒绝先测完 A 再测 B 却冒充配对结果 | `test_wrong_temporal_order_rejected_even_with_complete_coverage` |
| Skill 写“先修失败 case”，CLI 却没有定点命令 | KDA/kernel-lab 要求选择最便宜的有效反馈 | quick 支持 `--case --seed --check-only`，无需改 task.json | `test_targeted_check_only_keeps_contract_and_covers_excluded_case` |
| 返回值正确仍可能违反接口语义 | kernel-lab 明确输入/输出 alias、layout 与实际入口 | 存储绑定检查、任务 validate、`load_best()` 正常入口 | `test_task_validator_checks_output_alias`、`test_public_entry_uses_verified_snapshot_in_fresh_process` |
| 字节相同代码可能碰巧测得更快 | ForgeKernel 的 false-win 路径记录过相同代码的环境假胜利 | 当前 best 的候选文件完全相同则不晋升；基线始终同期实测 | `test_identical_source_cannot_be_promoted_by_noise` |

反例测试集中在 [scoring](tests/test_scoring.py)、[integration](tests/test_integration.py) 和 [arrays](tests/test_arrays.py)。这证明的是本仓库在这些故障下的行为，不能把测试数量当成与其他系统的可靠性排名。

### 九个仓库逐项比较

| 参照和实际核对位置 | 它值得保留的设计 | 本仓库的选择与优势 | 参考仓库仍更强 / 判断 |
|---|---|---|---|
| **[kernel-lab][kl-eval]**：SKILL、evaluation、records | 探索与交付分离；独立 oracle、best、固定 baseline；代价预算；验证实际入口 | 将其中适合单算子延迟优化的规则落实为 evaluator、晋升检查和可调用的 best；任务无需试验/版本数量目标 | 它的多指标、梯度/漂移验证、批次不变性、版本 campaign 规范更完整。我们提供更直接的可执行实现，没有超越它全部语义覆盖 |
| **[AKO4ALL][akoall-bench]**：bench.py、[skill][akoall-skill] | signal/verdict 分离；fresh-values；计时后 mutation sentinel；可换 bench | 增加固定 baseline 累计预算、结构化原始证据、source/helper 快照、确定性晋升；定点诊断不要求每轮紧跟日志和 commit 两次工具调用 | 它已有 KernelBench 和多个实现 backend 的适配。我们的 task 接口需转换；迭代程序步骤更少，但尚无相同 agent 预算下的耗时对照 |
| **[KDA][kda-flow]**：agent-flow | 契约→实现→验证→证据→晋升；工作流与任务分开 | 常见数组/tensor 任务开箱具备 evaluate；一个 notes 足够，无须先准备 draft/plan 两个文档 | KDA 本身更通用、更轻，适合已有成熟 evaluator 的项目。我们减少评测接线工作，不是所有项目都需要本 evaluator |
| **[Atrex Light][atrex-record]**：recorded_verifier、[native_abba][atrex-abba]、verifier | Supervisor 管正式证据、源码绑定、配对测量、完整策略匹配、恢复与复用 | 本地直接执行，无 runtime/provider 接入；加权总延迟与逐 case 双预算贴合当前任务；0.4 同时检查总体与配对收益 | 它的远端执行、记录所有权、批次恢复与接纳审计更成熟，测量默认锁频。旧版过于保守的问题促成了 0.4 修订，但噪声模型下仍有取舍，不能将更严格直接等同于更优 |
| **[ForgeKernel][forge-driver]**：driver、[precision_gate][forge-precision]、恢复失效逻辑 | 允许性能回退作为重构脚手架；false-win 复测；precision gate；依赖恢复 | 保留探索与 best 两条线，去掉 teacher/meta 和依赖图组织；固定 oracle + 本地可复算判定 | 它对数值精度、复杂重构、跨阶段恢复更成熟。我们默认 allclose 不能替代它的高精度 gate；精度敏感任务应接入已有专用验证 |
| **[AKO4X][akox-archive]**：archive_variant、compute_score | 代码、配置、父版本、失败经验一起继承；长期归档 | 小任务普通目录即可；归档与晋升明确区分，只有完整正式证据能改 best | 它的 master/child、长周期候选谱系与会话归档更完整。`archive_variant` 允许无评测证据是保存原型能力，不能把这一点写成它的晋升漏洞 |
| **[JustGoal][justgoal]**：公开实现与消融目录 | 强 agent 自主构建搜索路线；少约束目标驱动是一条重要基线 | 预先提供可靠测量，避免每个任务重写同类 evaluator；保持路线自主 | 它有真实 B200 kernel 与实验产物，我们还没有同预算 GPU 对照。附加 skill 是否提升最终效果必须实测，不能从架构图推导 |
| **[KernelAgent][kernelagent]**：optimize_kernel | profiling→瓶颈→改动；复用同基线 NCU；runtime 与 SOL 各自绑定实现 | profiler 按需调用，结果中的每个耗时都绑定 source/case/role；使用者可直接用现有 agent | 它的 roofline/NCU、beam 多候选与子图生成组合更系统。不能把它描述为“每轮必定重复 baseline profile”来制造优势 |
| **[GEAK kernel_workflow][geak]**：工作流 README | AMD kernel/model 优化、分析、独立验证、组合与原始基线比较 | 当前 NVIDIA/CPU 任务不需要接入其多角色运行框架；借鉴其反馈与复测原则 | 它有 AMD 和 serving 工作流的针对性，我们不提供相同能力；不能跨硬件做速度排名 |

一个具体的目标函数差异：两个等权 workload 的 baseline 延迟是 `[1, 100]`，candidate 是 `[0.25, 150]`。几何平均延迟下降约 **38.76%**，但总延迟从 101 增至 150.25，增加 **48.76%**。Atrex 的几何平均目标和 AKO4X 的平均 speedup 都有 benchmark 用途；本仓库选择固定权重总延迟，是为了匹配声明的调用成本，并非宣称其他聚合公式本身错误。若用户目标就是排行榜几何平均，应直接使用该排行榜 evaluator。

### 流程合理性：哪些是强制，哪些交给 agent

| 决策点 | 采用的流程 | 原因与实际代价 |
|---|---|---|
| 起步 | 确认契约、跑通 oracle/baseline；按实际瓶颈查资料 | 不规定首先阅读所有文档，也不要求先写完整设计稿 |
| 修错 | 定点 check-only，成功后补 quick/full | 减少无关 case、预热和计时；诊断结果永不晋升 |
| 参数探索 | quick 或 signal，保留配置和原始结果 | signal 不产生 speedup；不将旧基线 timing 带入新候选 |
| 结构探索 | 可继续失败或退化原型；verified best 独立保存 | 不把研究过程强制改成单调爬山；代码父版本不等于性能对手 |
| 正式确认 | 一次 full 内两批，同期比较 best 与起始 baseline | 用多一些正式调用换取累计预算保障；不每个微调都 full |
| 波动 | inconclusive，保留全部结果；排查环境或明确修订协议 | 不自动重试直到偶然赢；两批一致也不冒充统计置信证明 |
| 收尾 | 复算晋升谓词，检查 source/helper，正常入口加载 best | 合格 full 不无条件再做第三次；接入模型后另查 dispatcher 是否生效 |
| 超时/恢复 | 新子进程、保留日志；远端 job 先查状态 | 本版超时后重跑该轮，尚未实现 Atrex 式批次断点恢复 |

**默认成本可直接计算。** C 个 case、S 个 correctness seeds、T 个 timing trials、R 个实现：完整 quick/full 分别是 1/2 批，每批 `2×C×T×R` 次测量；默认完整正确性含 `C×R×(2S+2)` 条检查记录，额外的 2 是实际计时输出抽查和计时后换值检查；输出抽查复用真实返回值，不额外运行候选。check-only 是 `2×C×S×R` 次检查、零计时。每次实现检查另运行 oracle，成本不能忽略。开启 deterministic 后，完整模式和 check-only 都再增加 `C×R×S` 条检查与对应调用。

full 已有 best 时 R 从 2 变为 3，因此计时调用数增加 50%；这是修复累计预算缺口的明确代价。没有 best 时复用 baseline 对照，不重复测两遍。预算紧时减少探索轮的范围或用 signal，而不删正式验收的必要对照。任务的 `worker_wall_seconds` 用于检查这些开销是否值得；本次 [本地 smoke 记录](audit/local_validation.json)只测 CPU evaluator 开销：定点检查是 4 次检查、零计时；quick 是 8 次检查、12 个计时样本；full 起步为 32/160，已有 best 为 48/240。本例只有亚毫秒函数，Python 启动耗时足以掩盖样本数差异，单次数据不用于 wall time 排名；不能由更少样本直接推断 wall time 更短，也不据此推断 GPU 或 agent 总搜索效率。

### Evaluate 的可信度与仍需验证的部分

| 维度 | 已由代码和测试保障 | 不混淆的结论 |
|---|---|---|
| 数值与状态 | fresh/reuse/timed_output/post；整数精确检查；dtype/shape/device；非有限数；原地结果；自定义语义校验 | allclose 不等于梯度、方向漂移、批次不变性、race 或数学证明；按任务接入专用验证 |
| 时间与目标 | 同期交错测量、原始顺序校验、正有限耗时、权重目标、双预算、两批噪声筛选 | MAD/IQR 是启发式门槛；case 预算针对声明的中位延迟，不是尾延迟 SLA 或置信上界 |
| 身份 | task/ref/candidate/helper/evaluator 哈希；结果 receipt；当前 best 重新核对 | 外部动态库、驱动与环境需部署固定；本地哈希不是对不可信候选的安全隔离 |
| 晋升与入口 | 只接受 full；拒绝改写、漏测、乱序、旧比较对象、相同源码；新进程 load_best 验证 | 哈希变化不证明机制有用；模型中的默认 dispatcher 仍须集成测试 |
| 迭代效率 | 诊断、signal、quick、full 各有用途；不强制多 agent、每轮 profiler 或额外稿件 | 减少固定步骤是已实现事实；节省多少模型/GPU 时间、最终 kernel 多快尚需实际任务对照 |

### 是否优于所有参考仓库

**当前建议：在单 agent、少量 NVIDIA/CPU 算子、已有执行机器的场景使用本方案。** 九个参考仓库的机制取舍已逐项比较；证据强度分为以下几类：

| 要回答的问题 | 已有证据 | 当前结论 |
|---|---|---|
| 是否保持 skill + evaluator 的简单结构 | 独立 wheel 只包含八个运行模块，干净环境仅安装本包与 NumPy；完整晋升、正常入口加载通过 | 已验证。容器/模型对照脚本位于 audit，不随运行包安装 |
| evaluate 是否更能识别具体错误 | AKO4ALL 原生 CLI、BackendBench 原生组件的同契约反例；本仓库回归测试与 B300 验收 | 在已测 alias、原地更新、输入域等场景有具体优势；不据此给整个系统排名 |
| evaluate 是否提高反馈效率 | 同 B300 的版本耗时对照；0.4 的配对收益模拟和实机探针 | 相等大张量探针的验证开销下降；部分噪声下收益检出改善，暂停噪声仍有取舍；0.4 总评测耗时没有证明下降 |
| skill 是否让同一 agent 更快找到更好的实现 | 九次预实验已执行；八次受接入故障影响，一次正常完成 | 本批不能做流程排名，不能从步骤更少推导搜索收益 |
| 是否全面超过九个参考系统 | 仍缺完整系统对照；复杂精度、远端恢复、AMD/serving 等能力并不相同 | 尚未证明；当前方案按既定使用场景取舍，不追求包含所有框架功能 |

下一项决定性实验是同模型、同任务、同预算的流程对照：先比较 **JustGoal 式目标提示、AKO4ALL 流程、本 skill**，统一最终验收，保留全部失败、逐 case/加权收益、模型成本、GPU 时间和恢复成本。当前三任务预实验只检验实验可行性；正式效果比较须重复独立运行。根据观察到的缺口改进流程，避免在没有证据时继续增加机制。

#### 实际对照怎么做

先区分两个问题：**换 skill 是否改善搜索**，以及**完整系统是否更有效**。前者给三个实验组相同的开发 evaluator、工具、资料和初始代码，只改变工作流指令；后者保留各仓库原生流程和开发 evaluator，只统一任务、资源预算和最终验收。前者只能得出 workflow 消融结论，不能据此给整个参考仓库排名。“JustGoal 式目标驱动”若只复现提示词，也必须标成提示词基线，不能称为复现了 JustGoal 的全部实验。

| 实验环节 | 开始前固定的操作 | 必须留下的证据 |
|---|---|---|
| 任务 | 归约、GEMM、带非连续/尾部形状的融合算子各至少一项；固定公开输入域、dtype、误差、原始实现和权重 | 任务源码、依赖版本、契约哈希；至少一个正确实现和一个错误实现通过/拒绝记录 |
| 模型与预算 | 固定实际可用的模型版本、推理设置、上下文、工具权限和起始提示词；明确 wall time、模型 token/费用及 GPU 时间上限 | 模型调用记录及实际消耗；编译、验证、profiler、恢复都计入开发预算；最终统一验收单独记录 |
| 独立搜索 | 使用干净工作目录和独立对话；各组不得读取其他组候选或本轮未公开结果；每项任务每组至少 5 次作为初步对照 | 每次运行 ID、随机化的运行次序、所有中间失败和最终源码；5 次是起步数量，不是统计充分性的保证 |
| 最终交付 | 到预算立即冻结该组声明的交付版本，不能事后从全部候选中挑最终验收分数最高者 | 源码和 helper 哈希、正常调用入口；没有交付则记失败，明确交付原 baseline 则记零改进 |
| 独立验收 | 独立进程检查所有冻结提交；使用事先固定、agent 无读取权限的取值种子及公开输入域内的测试 case；GPU 独占，交错测量候选与原始 baseline | 正确性失败、逐 case 延迟原始样本、测量顺序、硬件/软件/精度/时钟策略；验收反馈不得返回本轮搜索 |
| 分析 | 全部运行进入结果；先报告正确交付率，再报告收益及总成本；正确结果条件下的速度另列，不能删除失败后声称整体更快 | 每任务全部运行的散点/分布、失败数、成本和逐 case 回退；重采样以独立搜索运行作为单位，不能把同一 kernel 的 timing samples 当成独立 agent 实验 |

上述对照的公开开发任务现已落地：[行归约](audit/search_tasks/row_sum)、[FP32 GEMM](audit/search_tasks/matmul)、[带非连续输入的融合 ReLU](audit/search_tasks/fused_relu)。每项包含三种 shape/layout、固定 FP64 oracle、原始实现和候选入口，统一测同步的 eager wall latency。CPU 与 B300 自检均确认三项正确实现通过、六项错误数值/输入篡改探针被拒绝；B300 还分别跑通完整评测，每项 48 条正确性记录、240 个计时样本，未将相同原始实现晋升。三项完整评测单次总耗时分别约 2.17/2.47/2.63 秒，包含初始化；这不是 quick/full 成本对照或搜索效果排名。[原始记录与复现入口](audit/search_workload_preflight.json)已保存，任务已结束并释放 GPU。

[九次预实验方案](audit/search_pilot_plan.json)已执行：三种流程、三项任务，每次最多八分钟，统一 GPT-6 Astra／medium，串行使用一张 B300。[完整提示词](audit/search_pilot_prompts/)和隐藏最终种子的哈希在搜索前固定；实验使用冻结的 **0.4.0**，未混入后续 0.4.1 改动。结果见下节。正式比较仍需多次独立运行。

独立的[最终验收脚本](audit/search_final_acceptance.py)现已实现：不导入 kernel_envolve evaluator/scoring，以 NumPy FP64 计算期望值，检查数值、shape、dtype、设备、输入状态及同地址换值；独立生成交错计时，每个实际计时输出都在计时区间外校验，保留原始样本，不读取开发阶段的晋升状态。为控制预实验的适配成本，三组都只提交单个 candidate.py。CPU 和 B300 各通过三个正确对照、拒绝十五个错误数值/输入篡改/缓存/dtype/shape 探针；B300 额外完成三项完整计时，每项 36 条初始检查、240 个逐项验证的计时样本。[自检记录](audit/search_final_validation.json)保留实际结果；correct 只表示功能正确，不表示性能胜出。本次启动前另外完成了真实 GPU 超时清理和后续请求复用检查，见[本批证据](audit/search_pilot_20261110/)。

预实验另有一个审计用[单次容器执行器](audit/run_bounded_trial.py)，不进入产品运行时。静态探针验证：容器读不到 host 上的隐藏种子文件（含指向它的符号链接）；超时会删除本次容器、停止脱离父进程的子进程；随后冻结预先约定的 submission.py，工作目录再改动也不影响交付快照；未交付文件明确记失败，不自动选历史最佳。[记录与复现脚本](audit/bounded_trial_validation.json)保留了三项结果，三秒超时样本包含清理共约 3.10 秒。该项静态检查使用无网络 Python 容器；真实模型接入结果另见下节。

最终验收代码不调用任何实验组的晋升结论；对接现成外部 harness 时，先确认语义、计时范围和精度规则一致。若编写任务专用验收器，必须明确它是独立实现而非现成第三方认证，先用正确/错误实现自检。运行期间遇到 GPU 抢占或硬件故障，依据事先规定的环境判据判废并保留原记录；不能因为候选慢或失败而重跑。

“更合理”的证据要分开给：流程步骤和强制开销可从实现核对；evaluate 防错能力需要相同契约下的有效实现与故障实现对照；搜索效果需要上面的独立运行。比较参考 evaluator 时，仅把它承诺支持、且接口适配后语义相同的反例列入分母，不支持的功能单列，不能算作错误。7 项旧新版本探针是本仓库的回归对照；下文另有 BackendBench 原生正确性组件及 AKO4ALL 原生 CLI 的直接对照，尚未完成全部参考系统的统一实测。

成功判据和实际可接受的收益/成本差额应在实验开始前填写；如果正确率、收益和成本互有取舍，就报告取舍，不生成一个事后加权的“总冠军”。只有观察到的优势足够稳定、且覆盖声明的目标任务，才把结论从“建议采用”升级为“该任务和预算下更优”。

硬件资料和参考库入口仍集中在 [hardware.md](skills/kernel-envolve/references/hardware.md)。代码独立实现，链接用于设计来源与适用能力核对。

[kl-eval]: https://github.com/guts-ssss/kernel-lab/blob/c7f576b5917491d790dd091a20dcbc6c4d0c6d61/references/evaluation.md#L5
[akoall-bench]: https://github.com/TongmingLAIC/AKO4ALL/blob/bbd0e1cf1ce2fb19d4322a932680f4c3d175d80e/bench/kernelbench/bench.py#L554
[akoall-skill]: https://github.com/TongmingLAIC/AKO4ALL/blob/bbd0e1cf1ce2fb19d4322a932680f4c3d175d80e/SKILL.md#L88
[kda-flow]: https://github.com/NVlabs/kda/blob/acdd79343c5e9d0c7cb96289f26fd3f1051ce785/docs/agent-flow.md#L1
[atrex-record]: https://github.com/alibaba/atrex-kernel-agent/blob/bab3218768baea30ba56a87851705d60d64ebbf3/src/aka/light/acceptance/recorded_verifier.py#L25
[atrex-abba]: https://github.com/alibaba/atrex-kernel-agent/blob/bab3218768baea30ba56a87851705d60d64ebbf3/src/aka/light/evaluation/native_abba.py#L330
[forge-driver]: https://github.com/OpenBMB/ForgeKernel/blob/6664d6080c37803605d6c6bc25f87656423ff114/src/forgekernel/loop/driver.py#L7175
[forge-precision]: https://github.com/OpenBMB/ForgeKernel/blob/6664d6080c37803605d6c6bc25f87656423ff114/src/forgekernel/loop/precision_gate.py#L1
[akox-archive]: https://github.com/TongmingLAIC/AKO4X/blob/ce030ebd855dffb7ba6bdaef7ba7504a92cb9393/master/master.py#L674
[justgoal]: https://github.com/OpenBMB/JustGoal/blob/b0a2b582e39daf3344faac1414ec7c26660633e3/README.md#L26
[kernelagent]: https://github.com/meta-pytorch/KernelAgent/blob/e0647170da36ef9b059ac0bd3d60103aa4ed378b/triton_kernel_agent/opt_worker_component/orchestrator/optimization_orchestrator.py#L311
[geak]: https://github.com/AMD-AGI/GEAK/blob/f189eff7c4036d9e9bd1d4fe99ec5c25ab6b8b20/kernel_workflow/README.md#L1

### 九次预实验的实际结果与 0.4.1 修正

**这批不能判断哪种流程更优。** 实验接入先出现容器未传入 stdin 的启动错误（零模型请求），修复后才开始九次搜索。实际搜索 0–1 受代理串行锁问题影响，2–7 受流式结束分隔符被截断影响；其中 1、7 由控制器中止。只修复公共接入后继续剩余次数，没有重试这些搜索。第 8 次正常结束。两类故障、全部交付源码和独立验收原始样本均保存在[实验记录](audit/search_pilot_20261110/summary.json)中。

| 序号 / 任务 | 提示流程 | 搜索秒数 | 独立最终验收 | 两批加权延迟降低 |
|---|---|---:|---|---:|
| 0 / 行归约 | AKO4ALL 改编 | 35.3 | 正确，原始实现 | −0.7% / −3.6% |
| 1 / 行归约 | 本 skill | 94.3 | 正确，原始实现 | 0.7% / 0.4% |
| 2 / 行归约 | 目标提示 | 84.2 | 编译失败，缺少 CUDA 头文件 | — |
| 3 / GEMM | 本 skill | 26.8 | 正确 | 34.8% / 35.7% |
| 4 / GEMM | 目标提示 | 28.8 | 正确 | 35.3% / 34.4% |
| 5 / GEMM | AKO4ALL 改编 | 34.2 | 正确，原始实现 | −1.4% / −1.3% |
| 6 / 融合 ReLU | 目标提示 | 4.5 | 未交付 | — |
| 7 / 融合 ReLU | AKO4ALL 改编 | 9.3 | 未交付 | — |
| 8 / 融合 ReLU | 本 skill | 224.3 | 正确，正常完成搜索 | 65.2% / 65.5% |

表中 0–7 均属于受基础设施影响的部分结果；原始实现相对自身的正负变化是测量波动。第 8 次三个 case 的候选延迟约 15.8 / 23.2–23.6 / 18.9–19.1 微秒，baseline 约 45.3–45.4 / 64.5–64.8 / 57.8–58.1 微秒。这只证明该次交付在指定任务上有效，不能将其与未正常完成的另外两组作流程排名。

全部九次搜索共发起 61 个模型请求，收到 59 个完整回复；完整回复报告输入 1,045,635 tokens、输出 16,003 tokens，`cost` 合计 3.283895（提供方返回值）。两次中止请求可能有未返回的用量。GPU 从首次申请到最终确认释放共 **1707.1 秒 / 28.5 分钟**，包含重启和停机间隔，未重置预算；[释放凭据](audit/search_pilot_20261110/resource-release.json)确认已停止且原节点无本任务 Pod。

这批对迭代效率最有用的发现是：第 8 次的 6 次 evaluate 全为 full，worker 累计 78.8 秒，其中 setup 69.6 秒、正确性 4.59 秒、测量 0.31 秒；GPU 工具调用累计 95.3 秒，还包含远程转发及结果收集。主要成本来自编译与初始化，单纯缩小计时样本并不能解决它。运行中发现 Ninja 缺失，以及多次 nvcc 编译失败，而 0.4.0 丢失了捕获的 stderr，迫使 agent 猜测并重试。

因此 **0.4.1 只做有证据支持的小改动**：CLI 直接给出失败阶段和编译输出；skill 默认在 GPU 机器上直接调用本地 evaluator，按实际阶段耗时选择反馈范围，重复编译优先用工具链已有缓存。没有新增 CPU 阶段、模型代理或调度服务。实验代理的[源码快照](audit/search_pilot_20261110/experimental-runner-sources.json.gz)仅供审计；产品仍是 skill + evaluate。本地验证 99 项通过、8 项 CUDA 测试跳过，新增测试覆盖真实子进程编译错误反馈、截断、非 UTF-8 输出、失败不可晋升及超时部分输出；本批 GPU 结果仍只对应 0.4.0。

## Evaluate 深度审查：0.3 补齐的缺口

这轮继续阅读 [robust-kbench 的 forward 检查][robust-forward]、[BackendBench 的结构化比较][backend-eval]、[PyTorch benchmark 的计时][torch-timer]与[离散度判断][torch-common]、[ComputeEval 的正确性/性能范围][compute-eval]，并复核 ForgeKernel 的精度 gate。新增来源固定在 [evaluator_sources.json](audit/evaluator_sources.json)。吸收具体机制，而不复制其默认容差或宣称覆盖了其完整功能。

另外，用同一套 [故障探针](tests/probe_previous_release.py)实际运行旧版 0.2 和新版 0.3。[结果记录](audit/evaluator_regression_probes.json)复现了 7 项差异：旧版会放过共享 oracle/计时缓冲覆盖、公共数值设置泄漏、未登记源码、旧字节码、零 MAD 多峰计时和 FP16 比较溢出；新版分别拒绝错误证据或执行正确源码。这是同环境下对本仓库旧版的回归验证，不是对其他仓库的性能排名。

| 发现的缺口 | 修复及行为 | 对应反例测试 |
|---|---|---|
| oracle 与候选共享返回缓冲，后调用把前结果改写 | 及时冻结 oracle 和真实计时输出；值比较与实际对象的 alias 校验分开 | `test_oracle_output_cannot_be_overwritten_by_candidate`、`test_actual_timed_output_cannot_be_overwritten_by_later_oracle` |
| 工厂返回缓存输入，或 seed 改变 shape | 强制独立存储、固定每 case 的输入结构；参考计算与候选互不污染 | `test_cached_factory_inputs_are_rejected`、`test_factory_shape_drift_cannot_include_compilation_in_timing` |
| 换值检查误拒广播视图和空张量 | 零 stride 维只写底层代表切片；空 storage 不视作跨输入共享 | `test_torch_expanded_views_and_empty_storage_are_valid` |
| 容差内漂移掩盖不确定性，或比较算术自身溢出 | 可选同输入精确重跑；宽精度比较；正确处理 conjugate view | `test_optional_exact_value_determinism`、`test_float16_tolerance_math_must_not_overflow_to_infinity` |
| allclose 允许小但系统性的精度退化 | 任务可设独立 golden gate；提供 reduction 正反例，不套统一统计方法 | `test_golden_gate_accepts_improved_accuracy_and_rejects_bias_within_allclose` |
| 候选把全局 dtype/TF32/归约精度/线程等设置留给后续对照 | 冻结任务初始化后的运行设置；检查导入、验证、预热、计时调用后的持久变化 | `test_torch_default_dtype_cannot_leak_between_roles`、`test_numpy_global_error_policy_cannot_leak_between_roles` |
| MAD=0 却存在多峰长尾 | 噪声尺度使用 `max(MAD, IQR/2)`，记录两者；无自动重跑直到通过 | `test_bimodal_timing_cannot_hide_behind_zero_mad` |
| 有限原始耗时算出无限 speedup，或布尔值冒充耗时 | 同时验证原始和派生指标，拒绝非有限值与错误类型 | `test_derived_ratio_overflow_rejected_with_finite_raw_samples`、`test_boolean_is_not_a_timing_measurement` |
| CUDA event 被误当完整调用成本 | 显式选择 event/wall 并绑定契约；wall 调用前后同步 | `test_wall_clock_synchronizes_before_and_after_device_work`；CUDA 两种模式有单独硬件 opt-in 测试 |
| 源码哈希正确，实际却加载旧字节码或未列出文件 | 从源码编译快照模块、忽略 `.pyc`；核对完整文件清单 | `test_snapshot_import_ignores_timestamp_valid_stale_bytecode`、`test_added_source_file_invalidates_manifest` |
| Python `-O` / `PYTHONOPTIMIZE` 静默移除断言 | 评测拒绝优化模式；快照源码始终保留 assert | `test_python_optimization_cannot_disable_task_assertions`、`test_snapshot_loader_keeps_assertions_even_in_optimized_python` |
| 子进程报告串任务或自称 eligible | 父进程匹配请求、校验结构并重算；错误原报告保留供诊断 | `test_supervisor_does_not_trust_eligibility_or_wrong_identity`、`test_malformed_worker_envelope_is_rejected` |

全局设置检查防止持久污染，不代替误差检查，也不是拦截所有进程状态的沙箱。task.py 可以在初始化时明确设置所需精度/线程策略；候选不得把变更留给其他实现。Torch 的 dtype、matmul 精度、TF32、FP16/BF16 reduction、deterministic algorithms、cuDNN 策略、线程以及当前 CUDA device/stream 会记录并核对；已知线程环境变量也进入结果。

源码核对也发现不能照搬的细节：上述 BackendBench 固定提交中，`allclose(a, b, atol, rtol)` 调用 `_allclose(a, b)` 时没有转发容差。这里借鉴结构化输出与算子输入覆盖思路，保留显式传递任务容差，并用宽/严容差与 golden 正反例测试约束其行为。

### 与 BackendBench 的直接正确性对照

在固定提交 `2a8d7e1` 的干净 checkout 中直接导入**未修改的** `BackendBench.eval.eval_correctness`，用 CPU float32 输入运行 11 个场景、各 3 个种子；本仓库通过真实隔离子进程执行相同任务的 check-only。两边均设 `rtol=atol=0.01`、允许同位置 NaN。基础输入种子相同，本仓库另执行输入状态与换值检查，因此这不是等检查次数的效率比较。另直接调用公开 `allclose` 做 6 个容差参数探针。

| 对照场景 | BackendBench 固定版本 | kernel_envolve 0.3.1 | 说明 |
|---|---|---|---|
| 正确加法、非连续加法、tuple 输出、返回输入、声明的原地加法（5 项） | 全部接受 | 全部接受 | 防止只展示拒绝错误、却误伤有效实现 |
| 错误数值、shape、dtype（3 项） | 全部拒绝 | 全部拒绝 | 基础正确性能力一致 |
| reference 返回输入；candidate 原地修改并返回它 | 接受 | 拒绝 | 原生入口中 reference 的返回值也随输入改变；冻结 oracle 可避免假相等 |
| 输出数值正确，但修改了应只读的输入 | 接受 | 拒绝 | 本仓库额外落实输入不变契约；BackendBench 此入口没有独立 mutation gate |
| 原地加法候选错误地累加两倍 y | 接受 | 拒绝 | 同一输入被依次修改，reference 返回的别名也随之改变；独立输入与冻结结果避免污染 |
| 显式零容差，`1.0` 与 `1.005` | 接受 | 拒绝 | 公开 helper 未向内部转发零容差 |
| 显式 `atol=0.1, rtol=0`，`1.0` 与 `1.05` | 拒绝 | 接受 | 同一参数传递问题也会误拒有效候选 |
| 其余 4 个容差正反例 | 与声明容差一致 | 与声明容差一致 | 包括相等值、默认容差及超过宽松容差的错误 |

[复现脚本](audit/probe_backendbench.py)、[来源哈希与结果摘要](audit/backendbench_component_comparison.json)、[完整结果及逐场景代码](audit/backendbench_component_raw.json.gz)保留了适配方式、实际加载的上游文件哈希和环境。脚本只提供 `args/kwargs` 测试数据，不替换上游判定函数，也不调用模型 API。这证明本仓库在这些状态隔离与容差遵循场景中有具体优势；没有运行 BackendBench 的完整 backend 工作流，也没有据此生成仓库总排名。

这些补强保留四类操作：**定点检查、quick/signal、full、promote**。探索步骤按反馈成本选择，只有正式 full 是晋升的必经步骤。强制新增检查只针对证据失真；同输入确定性与计时口径由任务明确选择。Golden 示例展示如何扩展现有 `validate`，不另建验证框架。GPU race/sanitizer、复杂梯度统计、CUDA Graph 和真正未见过的隐藏测试仍需相应任务 evaluator；ComputeEval 的 held-out harness 不能由公开可见的固定 full seeds 冒充。

[robust-forward]: https://github.com/SakanaAI/robust-kbench/blob/078f5bab29934a822268d59a4e707d449abf9b4e/robust_kbench/sandbox/correct_forward_fn.py
[backend-eval]: https://github.com/meta-pytorch/BackendBench/blob/2a8d7e19e2c13c789ff2dceb44883ea8ea04dab4/BackendBench/eval.py
[torch-timer]: https://github.com/pytorch/pytorch/blob/8a9a9e737dba491a2eea270641bff86eea6e1ea2/torch/utils/benchmark/utils/timer.py
[torch-common]: https://github.com/pytorch/pytorch/blob/8a9a9e737dba491a2eea270641bff86eea6e1ea2/torch/utils/benchmark/utils/common.py
[compute-eval]: https://github.com/NVIDIA/compute-eval/blob/00fbad91b63957a002b501f33a06510cd25beaae/README.md

## B300 实机验证：0.3.0 初始验收

在 flexible 空间实际申请 1 张 B300（本会话上限 32 张），绑定工单 `WOC-ff0a397b3590121d` 和正确性课题。PyTorch 2.12.1 / CUDA 13.0 / Triton 3.7.1 环境中，**95 项测试通过、零跳过**；7 项 CUDA 测试覆盖 Triton 的 event/wall 完整评测、错误输出、输入篡改、旧输出缓存、精度设置污染，以及晋升后从正常入口加载 best。晋升测试故意让 baseline 等待，只验证晋升流程，不将它的加速比当作优化成果。任务已 Completed、Pods=0。[验收记录](audit/gpu_validation.json)与[完整原始结果](audit/gpu_validation_raw.json.gz)保留硬件、镜像、源码及测试哈希。

额外跑通两种计时口径下的 check、signal、quick、full 和错误输出场景，共 10 个场景。真实 vector-add 示例在 eager 口径下慢于 `torch.add`，因此没有晋升。该样本的 quick 进程耗时约 3.5–3.6 秒，full 约 4.5–4.8 秒；启动和输入检查是实际开销的一部分。这说明单次测量条数不能直接代表 agent 的总迭代成本。

### 计时校准发现的限制

[独立计时探针](audit/probe_gpu_timing.py)在相同输入上比较本 evaluator 的 event/wall、Triton `do_bench` 和 `do_bench_cudagraph`，执行三轮并轮换实现顺序。[结果](audit/gpu_timer_probe.json)保存逐轮摘要，[原始数据](audit/gpu_timer_probe_raw.json.gz)还包含全部直接计时样本和安装版本的 Triton 计时源码。三轮中位数如下，单位为 μs：

| 输入与实现 | eager event | 同步 wall | Triton do_bench | CUDA Graph 回放 |
|---|---:|---:|---:|---:|
| 257 元素，torch | 12.18 | 13.66 | 7.20 | 1.18 |
| 257 元素，Triton | 24.18 | 23.88 | 7.14 | 1.02 |
| 257 元素，torch 前增加 200 μs host sleep | 273.74 | 274.38 | 241.95 | 1.30 |
| 非连续 65,539 元素，torch | 12.77 | 14.70 | 7.23 | 1.69 |
| 非连续 65,539 元素，Triton | 23.50 | 23.33 | 7.20 | 1.38 |

host sleep 不改变加法 kernel，却明显增加普通 event 区间；Graph 录制后的回放不重复执行这段 Python。非连续输入的快慢排序也从 eager 下 torch 更快变为该 Graph 样本中 Triton 更快。因此，**目前的内置 evaluator 适用于声明的 eager 调用目标，不能拿它决定 Graph 中的 kernel 优劣**。Triton 普通 bench 的缓存处理与 Graph 的固定地址、缓存复用、批量回放也不同，不能仅把三列差值都归因为 Python 开销；Graph 耗时也不等于 profiler 中孤立的单条 kernel 时长。

这次实测补强了硬件验证，同时确认参考计时工具有当前内置 evaluator 未覆盖的运行方式。保持 skill 的轻量流程；先明确实际部署是 eager 还是 Graph，再选择覆盖该语义的任务 evaluator。这里没有把两种口径合成一个 speedup，也没有把硬件验收当作模型搜索效果排名。

## 0.3.1：评测效率与迭代成本

这轮重点减少验证中的重复数据搬运。Torch 整数/bool 直接精确比较；浮点/复数先判断是否完全相等，相等时省去整张量复制到 CPU，不相等时仍走原有宽精度容差比较。NaN 规则保持不变。实测发现小 CUDA 张量增加比较 kernel 的初始化反而可能更慢，因此浮点/复数的 CUDA 快速路径仅用于 **至少 1,048,576 个元素**；小张量保留原路径。CPU tensor 使用相等快速路径。

在同一张 B300 上，针对三个尺寸、check/signal/quick/full 四种模式，对 0.3.0 与 0.3.1 各运行三轮，共 **72 次评测**。交替旧新版本次序、轮换模式次序，两次初始冷启动单独保留。oracle、baseline 和 candidate 均为相同的 `torch.add`，旧新版本检查数与计时样本数完全一致。下表是包含子进程启动及父进程核查的 **evaluate API 总耗时中位数**，单位秒：

| 每个输入的元素数 | 模式 | 0.3.0 | 0.3.1 | 耗时变化 |
|---|---|---:|---:|---:|
| 257 | full | 1.872 | 1.923 | 增加 2.7% |
| 1,048,577 | full | 3.975 | 1.923 | 减少 51.6% |
| 4,194,305 | quick | 4.925 | 2.071 | 减少 58.0% |
| 4,194,305 | full | 7.830 | 1.973 | 减少 74.8% |

大张量相等检查的收益明确；小张量仍受约 2 秒的进程启动与初始化成本影响，quick/full 中位数增加约 50 毫秒，没有证明小张量加速。相等快速路径也不代表所有近似数值输出都能获得同样收益。此次测量针对 evaluator 开销，未测 agent 搜索效果或 kernel 加速比。

**流程应根据实际成本选择反馈。** 新增的阶段耗时让 agent 看清时间花在哪里；本例 quick 与 full 成本接近，对已准备正式确认的候选直接 full，可省掉额外一次 quick 启动。已知正确性错误优先定点 check-only，大量候选筛选仍可用 quick/signal。无需强制三阶段串行，也不跨候选复用 PASS、不复用失败 CUDA context、不减少正式 full 的覆盖。

最终版本在 B300 上 **100 项测试全部通过、零跳过**，包括大小张量、容差、NaN/Inf、复数和别名 NaN 的比较规则；本地为 92 通过、8 项 CUDA 测试跳过。[实测记录与复现脚本](audit/evaluator_efficiency.json)、[72 次完整结果](audit/evaluator_efficiency_raw.json.gz)和[调整阈值前的探索结果](audit/evaluator_efficiency_exploratory_raw.json.gz)均已保存。此次最多同时使用 1 张 GPU，验证任务已完成并释放资源；上述 0.3.0 初始验收与计时探针保留原版本身份。

## 与 AKO4ALL 的 B300 原生 CLI 对照

在同一张 B300 上直接执行固定提交 `bbd0e1c` 的**未修改 bench.py CLI**与本仓库 CLI。12 类正确性场景各运行两边，共 24 次；另对 257 与 1,048,577 元素的加法做两档配置、各三轮成本测试，共 24 次，交替运行次序。两次初始冷启动单列。正确性任务使用相同基础种子、float32 和 `rtol=atol=1e-4`，这些有针对性的探针不用于估计通用任务成功率。

| 场景 | AKO4ALL 固定版本 | kernel_envolve 0.3.1 |
|---|---|---|
| 正确加法、非连续加法、返回输入、合法原地加法（4 项） | 全部通过 | 全部通过 |
| 错误数值、错误 shape（2 项） | 全部拒绝 | 全部拒绝 |
| 输出正确但篡改输入、按输入对象缓存旧输出（2 项） | 计时后的换值哨兵拒绝 | 计时前的输入状态/换值检查拒绝 |
| reference 返回输入，candidate 修改并返回它 | 通过，reference 输出也被覆盖 | 拒绝，oracle 输出已冻结 |
| 原地加法错误地累加两倍 y | 通过，共享输入和返回别名掩盖差异 | 拒绝，独立输入并检查原地结果 |
| 输入工厂仅生成正数的正确 sqrt | 初始检查 3/3 通过，哨兵换成 randn 后因 NaN 拒绝 | 通过，换值仍来自任务工厂 |
| 初始正确、随后运行抛异常 | `CORRECT=True`、退出码 0，但 `RUNTIME=-1` 且有错误记录 | `execution=error`、退出码 2，不产生晋升资格 |

最后一行两边都没有有效性能结果，区别是故障状态表达；不能将上游 `CORRECT=True` 或退出码 0 单独理解为完整验收成功。sqrt 案例说明取值域是验证契约的一部分，盲目增加随机扰动反而会误拒正确候选。skill 已将这一点纳入任务准备，运行时仍使用现有输入工厂，无需新框架。

反馈成本以完整 CLI 进程启动到退出计时，三轮中位数如下，单位秒：

| 输入元素数 | 本探针配置 | AKO4ALL | kernel_envolve |
|---|---|---:|---:|
| 257 | quick 对齐样本数 | 2.107 | 1.987 |
| 257 | full 对齐样本数 | 2.127 | 2.088 |
| 1,048,577 | quick 对齐样本数 | 2.138 | 2.037 |
| 1,048,577 | full 对齐样本数 | 2.150 | 2.040 |

这里 quick/full 是本仓库模式及探针档位：AKO4ALL 分别配置 **1/6、3/40** 次 correctness/performance trials，使两边分别记录 12/80 个双实现计时样本；上游默认是 **10/50**，本表不是其默认 final 成本。AKO4ALL 每角色预热 10 次、丢弃首个计时样本并在每次计时前清空 L2；本仓库每批每角色预热 3 次、采用交错测量并增加状态与覆盖检查，未清空 L2。因此这组数据表明两套配置反馈成本同为约 2 秒，不能据此给 kernel 延迟或完整搜索系统排名。缓存策略也属于计时口径，不应为获得更好分数临时切换。

[复现脚本](audit/probe_ako4all.py)、[摘要、配置及资源记录](audit/ako4all_comparison.json)、[完整 CLI 输出与逐任务源码](audit/ako4all_comparison_raw.json.gz)均保留。任务 `b51d88b2ac8342ada2008ca925d464b7` 已 Completed、Pods=0，本次使用 1 张 B300。

## 0.3.1 与 Atrex 的接纳规则对照：更严格也有代价

用固定提交 `bab3218` 的原生 `score_verification_payload` 与本仓库 `score` 做了 **24,000 组固定条件的模拟**：四类噪声、真实收益 0%/5%/20%，每种组合 2,000 次。单个 shape 排除了跨 shape 聚合目标的差异；每方 80 个观察值。Atrex 按原生 ABBA 顺序分成四次运行，每次 20 个样本取中位数；本仓库按两批、每批 10 组交错测量组织。两边使用同一条模拟环境变化序列，最低收益都设为 1%。Atrex 默认 0% 门槛的结果另存原始记录。

下表是模拟中候选被接受的比例；真实收益为 0% 时，该比例表示误接受，有真实收益时表示检出能力：

| 模拟环境 | 真实收益 | kernel_envolve | Atrex，1% 门槛 |
|---|---:|---:|---:|
| 约 0.5% 独立抖动 | 0% | 0% | 0% |
| 约 0.5% 独立抖动 | 5% | 100% | 100% |
| 约 0.5% 独立抖动 | 20% | 100% | 100% |
| 约 10% 独立抖动 | 0% | 0% | 34.15% |
| 约 10% 独立抖动 | 5% | 0% | 93.55% |
| 约 10% 独立抖动 | 20% | 4.85% | 100% |
| 每四个观察值共同变快或变慢 | 0% | 0% | 43.15% |
| 每四个观察值共同变快或变慢 | 5% | 0% | 68.70% |
| 每四个观察值共同变快或变慢 | 20% | 0.65% | 99.50% |
| 平滑周期漂移，随机初相位 | 0% | 0% | 47.00% |
| 平滑周期漂移，随机初相位 | 5% | 0% | 54.15% |
| 平滑周期漂移，随机初相位 | 20% | 40.90% | 99.65% |

**0.3.1 减少误接受的同时，可能漏掉明显收益。** 旧 guard 用两边各自的 MAD/IQR，没有充分利用它们共同变化的信息；它衡量的是样本离散程度，不是收益估计的标准误或置信区间。正常使用时，噪声导致的 inconclusive 应保留候选、检查环境，不能直接解释为“更慢”或“已到性能极限”。skill 已明确这个区别。

这些比例属于上述合成模型，不是 GPU 实测误晋升率。Atrex 默认锁频，本模拟没有执行其运行环境控制、远端服务或正确性检查；80 个观察值也不表示两套系统占用相同 wall time。完整记录包含参数、种子、逐轮判定及 Wilson 区间；区间只描述模拟抽样误差，0/2,000 不等于真实概率为零。

随后试验了一个小改动：仅将噪声估计换成每组配对差值的 MAD/IQR，保留原来的总体中位延迟收益。另用固定新种子检查九类噪声，包括重尾、偶发暂停、相关漂移、阶跃和交替抖动，共 **54,000 组模拟**。它提高了部分收益检出率，却在周期漂移的零收益场景接受了 **198/2,000（9.9%）**；收益统计量与噪声统计量没有同步对齐。因此**否决该原型，未将它用于正式晋升**。0.4 的修订同时确认配对收益，见下文。

一个已保存的反例中，两批总体中位延迟分别显示约 1.31%/1.13% 的收益，但配对差值的中位数对应约 −0.034%/−0.035%，真实收益为零。用后一种统计量的低波动去支持前一种统计量的收益，就会制造假确认；仅有 ABBA 顺序并不足以让任意统计公式成立。

[原生规则对照脚本](audit/probe_acceptance_policy.py)、[被否决原型的复现脚本](audit/probe_paired_noise_proposal.py)、[完整摘要与原型结果](audit/acceptance_policy_comparison.json)、[逐轮判定及示例原始测量](audit/acceptance_policy_raw.json.gz)均已保存。本轮使用 CPU 模拟，未申请 GPU 或调用模型。

## 0.4：改善收益判断，不增加评测调用

0.3.1 会把两边共同的时钟漂移都计入噪声，导致明显收益也无法确认。0.4 同时检查**总体收益、每组配对收益及其波动**：前者守住加权总延迟目标，后两者使用同一组差值，避免上述被否决原型的统计错配。原有双批确认、逐 case 回退限制、固定 baseline、正确性与证据核查保持不变。评分时一次分组原始记录，避免逐 case 重复扫描。

修订后换用新种子，在原先探索过的九类噪声之外增加 Gamma、连续暂停、双阶跃和指数抖动，共 **13 类环境 × 3 档收益 × 2,000 次 = 78,000 组对照**。旧、新评分接收完全相同的 80 个观察值。全部 **26,000 个零收益场景，两版都未接受**。部分有收益场景的接受率如下：

| 模拟环境 | 真实收益 | 0.3.1 | 0.4 |
|---|---:|---:|---:|
| 约 10% 独立抖动 | 20% | 4.95% | 87.45% |
| 每四个观察值共同漂移 | 5% | 0% | 100% |
| 平滑周期漂移 | 5% | 0% | 100% |
| 相关时钟漂移（AR1） | 20% | 31.30% | 100% |
| 偶发暂停 | 5% | 99.95% | 94.50% |
| 连续暂停 | 5% | 100% | 89.35% |

它提高了多种噪声下的检出能力，但暂停场景仍有取舍；这些模拟结果不等于真实任务中的错误概率保证。[复现脚本](audit/probe_paired_policy.py)、[全部条件与评分成本](audit/paired_policy_validation.json)、[逐轮判定和原始示例](audit/paired_policy_validation_raw.json.gz)保留完整结果。单 case 的纯评分 CPU 中位耗时为 0.209 → 0.217 毫秒，16 case 为 4.139 → 3.483 毫秒；这不包含执行候选。

随后在 **1 张 B300** 上运行最终版本，**104 项测试全部通过、零跳过**，并完成 42 次完整评测探针：

| 实机探针 | 次数 | 0.3.1 在同一原始记录上重算后合格 | 0.4 实际合格 |
|---|---:|---:|---:|
| baseline/candidate 均为相同 torch.add，event 计时 | 30 | 0 | 0 |
| 两边加入相同幅度的周期变化等待，wall 计时 | 6 | 0 | 0 |
| candidate 等待幅度减少 20%，wall 计时 | 6 | 0 | 6 |

第三项实际总体延迟下降约 18.4%–20.6%，用于检验有漂移时的收益判断，并非算子优化成果。旧版列是离线重算，未另跑一批计时，也未替换新版的实际结果。

另外交替执行旧、新 evaluator，在两种尺寸、quick/full 下各测三次，共 **24 次 evaluate API 总耗时**，初始冷启动另列。两版 quick 均为 8 项正确性检查/12 个计时样本，full 均为 16/80；下表为包含子进程启动与父进程核查的中位数，单位秒：

| 输入元素数 | 模式 | 0.3.1 | 0.4 |
|---|---|---:|---:|
| 257 | quick | 1.821 | 1.821 |
| 257 | full | 1.873 | 1.822 |
| 1,048,577 | quick | 1.921 | 1.971 |
| 1,048,577 | full | 1.922 | 1.973 |

总耗时仍约 2 秒，大输入中位数增加约 50 毫秒，本轮没有证明 evaluate 更快。实际流程继续根据成本选择：已接近交付且 quick/full 成本相近时直接 full；有已知错误时先定点检查；不要强制每个候选依次走完 check、quick、full。0.4 的价值主要是减少噪声下漏掉有效候选的情况，不是增加一个评测阶段。

[实机摘要与资源记录](audit/paired_policy_gpu.json)、[完整结果、旧规则重算及复现脚本](audit/paired_policy_gpu_raw.json.gz)已保存。任务 `4eae86bad7144349a0f86925bd36b6de` 已 Completed，最终实时查询 Pods=0。本地 96 项测试通过、8 项 CUDA 测试跳过；[实际 smoke 记录](audit/local_validation.json)验证了定点检查、signal、quick、首次 full 晋升，以及相同源码不能再次晋升。参考仓库对照已证明若干具体检查和判定差异；完整 agent 搜索的速度与最终优化效果，仍需前述同模型、同预算对照。

## 开发检查

```bash
python -m pip install -e '.[dev]'
ruff check .
ruff format --check .
pytest -q
```

测试覆盖真实 CPU 子进程评测、正确性拒绝、缓存旧输出、输入修改、超时、覆盖完整性、噪声与权重判定、快照/helper 完整性、旧 best 晋升冲突等。CI 在 Linux 的 Python 3.11/3.12/3.13 上运行 CPU 测试，3.12 额外安装 PyTorch CPU，覆盖 tensor 路径；CUDA 测试需显式设置 `KERNEL_ENVOLVE_TEST_CUDA=1` 并使用空闲 GPU。
