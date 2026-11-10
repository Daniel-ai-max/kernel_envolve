# kernel_envolve

**一个 skill，加一个可靠的 evaluate。让 coding agent 直接迭代算子。**

Agent 负责读硬件文档、研究参考实现、提出假设、写代码和诊断；evaluate 负责独立验证、实测和保存证据。没有模型 API 封装、调度服务或数据库。Codex、Claude 等能编辑文件和运行命令的 agent 都可以使用。

## 什么才算好的算子

**目标是正确、快速，并适配实际工作负载的整个输入范围。** 不能把范围缩成几个容易赢的 benchmark 点。对外应提供稳定的调用入口，size、batch、上下文长度等变化由实现处理，不需要用户手动改代码、换 tile 或重新调参。

- **正确：** 保持计算语义、精度、输出结构、输入修改规则；改变归约顺序或使用低精度时，验证数值敏感输入。涉及训练时还要验证所需的 backward。只在随机正常值上 allclose 通过不够。
- **快速：** 根据真实调用分布评估总体收益，同时检查各尺寸区间、布局和 dtype 的表现。大尺寸的收益不能掩盖常见小尺寸明显变慢；既检查相对当前 best 的退化，也检查相对原始实现的累计退化。
- **泛用：** 测常用值，也测开发时未用于调参的合法尺寸、分块/dispatch 阈值附近、非整倍数和不同布局。各输入的列 stride、行 stride/leading dimension 应独立覆盖，包含实际会出现的行间 padding；所有输入同时按一个 stride 切片容易遗漏此类问题。仅更换随机种子只验证输入数值变化，不验证尺寸泛化。
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

**当前实现对应什么：** `task.json` 的 `cases` 声明尺寸、dtype 和布局参数，`quick: false` 的 case 只参加 full；full 对所有声明 case 执行正确性检查；默认还计时并限制逐项退化，`measure: false` 可增加只做正确性检查的补充 case。evaluator 不会根据三个 shape 自动推断整个范围，也不会自动生成未见尺寸。需要在任务开始时构造广域 case，并在交付时用独立验证任务或项目已有测试补充未见尺寸；保存其结果与最终源码哈希。只通过内置 full 的 best，表示它通过了该固定任务的覆盖，不自动表示完成泛化验收。

下面的 vector-add 是接口示例。三任务预实验的冻结候选后来接受了新尺寸和行间 padding 验证：新尺寸通过，padding 失败，详见实测结果。

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

计时 case 的权重由任务明确给出。没有真实调用分布时，可以声明等权 benchmark 目标。`quick: false` 表示正常 quick 不选该 case；可通过 `--case` 定点诊断。正确性规则在两个模式中相同。

同一个任务中可以加入只做正确性检查的 case：

```json
{"id": "tail-neighbor", "args": {"n": 4097}, "measure": false, "quick": false}
```

它仍是 full 的硬门槛，执行 fresh、同地址换值及任务要求的确定性检查，但不预热、不反复计时，也不参与加速比。此类 case 不填 `weight` 或 `max_us`；任务至少保留一个会计时的 quick case。CLI 的 `coverage.correctness_cases` 与 `coverage.timed_cases` 分开列出实际验证范围。单独诊断它时使用 `--case tail-neighbor --check-only`。

这是扩大正确性覆盖的低成本方式。性能覆盖仍需包含各关键尺寸区间、dispatch 边界和 fallback；不能把可能变慢的区域全部设为 `measure: false`，再声称全域更快。计时选择属于固定任务契约，不能在观察到结果后删除慢 case 或取消其计时。

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
| `quick_trials` / `full_trials` | `3` / `10` | 每个计时 case、每批的配对轮数 |
| `warmup` | `3` | 每个实现、每批、每个计时 case 的预热次数 |
| `min_gain` | `0.01` | 加权延迟下降比例必须超过该值 |
| `noise_multiplier` | `2.0` | 配对差值的 MAD/IQR 噪声门槛倍数 |
| `max_regression` | `0.05` | 任一计时 case 相对当前 comparator 的最大允许延迟增长 |
| `max_baseline_regression` | `0.05` | 任一计时 case 相对固定起始 baseline 的累计延迟增长上限 |
| case 的 `measure` | `true` | `false` 时只验证正确性；不得带权重或延迟上限 |
| case 的 `max_us` | 无 | 该计时 case 候选延迟的绝对上限，微秒 |

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
- 对每个计时 case/实现，直接检查最后一个实际计时调用的返回值及输入状态（`timed_output`），而非重新调用候选来代替它；这是抽查，不是所有计时输出的全量验证。随后在同一输入对象上换值验证，捕获预热/计时后才出现的错误。`--check-only` 和 `measure: false` 的 case 没有计时阶段，只做前面的检查。

任何数值失败都会停止该轮计时，得到 `execution=completed, correctness=fail`。导入、编译、运行异常或超时得到 `execution=error`。每轮使用新子进程，超时终止其进程组；失败的 CUDA context 不复用。失败记录包含 case、seed、fresh/reuse/timed_output/post、实现角色及误差，日志保留异常栈。

### 测量和确认

默认 comparator 是任务当前 best，没有 best 时使用固定 baseline。full 有 best 时还在同一轮重新测固定 baseline，分别检查增量收益和累计回退。没有 best 时两个对照是同一个实现，只测一次。每次都重新测对照，不把历史 timing 当分母。`--baseline` 可专门测原始 baseline，但已有 best 时，这种结果不能替换 best。

quick 对所选且 `measure: true` 的 case 计时，一批样本；full 对全部 `measure: true` 的 case 计时，两批样本，第二批倒序遍历 case。两个实现采用 A–B–B–A 并轮换起始角色；三个实现采用 A–B–C–C–B–A，并在各轮循环改变起始角色。每个角色的两次调用使用不同的值，同一 repeat 的各角色使用匹配的值。输入种子由 case、batch、trial、repeat 和任务 seeds 确定，每个测量样本都更新，可从记录复现。判定时同时核对样本身份、数量和实际执行顺序。

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

两批相对 best 都须满足 **`gain > min_gain` 且 `paired_gain > max(min_gain, noise_guard)`**；相对固定 baseline 也单独计算并满足相同条件。前者保留声明的加权总延迟目标，后者要求配对结果也支持收益，不能拿配对差值的低波动为不一致的总体中位数收益背书。所有计时 case 仍须满足两套回退限制和绝对延迟上限。这避免逐轮 4% 的局部退化越积越大，也避免只赢过一个已经退化的 best。MAD/IQR 门槛仍是保守筛选规则，不是统计置信区间。结果接近门槛时，先检查环境竞争；确需调整协议时记录原因、保留旧结果，重新建立对应契约和对照。不要重复抽样直到出现有利结果；profiling 用于解释瓶颈，不与正式计时混跑。

`--case`、`--seed`、`--check-only` 仅限 quick，可定位 full 中才运行的 case，不修改固定任务契约，也不能产生晋升资格。`worker_process_wall_seconds` 记录子进程启动到退出的等待耗时；`worker_wall_seconds` 还包含父进程的结果核查。CLI 同时输出 `phase_wall_seconds`：setup（CUDA/模块初始化）、correctness、measurement（预热、输入生成、计时及后检查）、finalize（证据检查与评分）。阶段计时不进入算子计时区间，失败时保留已完成及当前阶段耗时；子进程耗时与阶段和的差值还包括 Python/NumPy 导入、退出及父进程唤醒，不能全部称为模型或 kernel 时间。兼容字段 `measurement_wall_seconds` 继续表示测量阶段总耗时。

执行异常直接在 CLI 的 `error` 中返回失败阶段；捕获的编译/子进程错误保留 stdout/stderr 各末尾 8192 个字符、截断标记及退出码（若有）。agent 可直接据此修复，不必为获取编译错误重复执行。完整 worker 日志仍保存在 run 目录。

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

evaluator 或任务契约变化后，旧 run 保留为历史，不能沿用其晋升资格。将任务及声明的依赖复制到新任务目录后重测；保留旧证据和 best 源码。

文件快照和哈希用于复现与防止误用；候选作为本机代码运行，这不是隔离不可信程序的安全沙箱。任务内有运行锁和晋升锁；GPU 性能实验仍需由使用者在目标设备上串行执行，其他任务目录不会自动共享设备锁。

命令退出码：`0` 表示评测正确完成（包含慢或收益不确定），`1` 表示正确性失败，`2` 表示配置/执行错误。是否值得晋升看 `eligible_for_best`，不能只看退出码。

## 适用任务

已实现普通 CPU 函数、NumPy、PyTorch tensor 和 NVIDIA CUDA 当前流上的 kernel 评测，支持非连续输入、结构化输出和显式原地更新。更换实现语言不需要改 agent 流程：只要 `candidate.run` 能调用实际入口，并将本地依赖列入快照即可。

本版不内置 CUDA Graph、多流事件汇合、分布式通信、autograd 图、显存上限判定或任意原生工程构建缓存。遇到这些任务时，保留该 skill 的迭代方法，接入任务已有的专用 evaluator。CUDA event 指标不能直接替代 CPU launch/端到端应用延迟，输出 alias/存储布局等特殊契约通过 task.validate 显式验证。B300 上的实际验证见下文；迁移硬件或运行方式时仍按对应契约重测。

## 从参考设计吸收了什么

对已有强 coding agent、在目标 GPU 上迭代算子的需求，本仓库采用 **skill + evaluate**：探索路线由 agent 决定，正确性、同期性能比较和晋升由代码执行。审查固定了九个仓库的提交，并逐文件核对 17 份关键源码/规范；[来源清单](audit/reference_sources.json)保留路径与 SHA-256。

| 参照和实际核对位置 | 它值得保留的设计 | 本仓库如何采用 | 参考仓库仍更强 / 判断 |
|---|---|---|---|
| **[kernel-lab][kl-eval]**：SKILL、evaluation、records | 探索与交付分离；独立 oracle、best、固定 baseline；代价预算；验证实际入口 | 将其中适合单算子延迟优化的规则落实为 evaluator、晋升检查和可调用的 best；任务无需试验/版本数量目标 | 它的多指标、梯度/漂移验证、批次不变性、版本 campaign 规范更完整。我们提供更直接的可执行实现，没有超越它全部语义覆盖 |
| **[AKO4ALL][akoall-bench]**：bench.py、[skill][akoall-skill] | signal/verdict 分离；fresh-values；计时后 mutation sentinel；可换 bench | 增加固定 baseline 累计预算、结构化原始证据、source/helper 快照、确定性晋升；定点诊断不要求每轮紧跟日志和 commit 两次工具调用 | 它已有 KernelBench 和多个实现 backend 的适配。我们的 task 接口需转换；迭代程序步骤更少，但尚无相同 agent 预算下的耗时对照 |
| **[KDA][kda-flow]**：agent-flow | 契约→实现→验证→证据→晋升；工作流与任务分开 | 常见数组/tensor 任务开箱具备 evaluate；一个 notes 足够，无须先准备 draft/plan 两个文档 | KDA 本身更通用、更轻，适合已有成熟 evaluator 的项目。我们减少评测接线工作，不是所有项目都需要本 evaluator |
| **[Atrex Light][atrex-record]**：recorded_verifier、[native_abba][atrex-abba]、verifier | Supervisor 管正式证据、源码绑定、配对测量、完整策略匹配、恢复与复用 | 本地直接执行，无 runtime/provider 接入；加权总延迟与逐 case 双预算贴合当前任务；0.4 同时检查总体与配对收益 | 它的远端执行、记录所有权、批次恢复与接纳审计更成熟，测量默认锁频。旧版过于保守的问题促成了 0.4 修订，但噪声模型下仍有取舍，不能将更严格直接等同于更优 |
| **[ForgeKernel][forge-driver]**：driver、[precision_gate][forge-precision]、恢复失效逻辑 | 允许性能回退作为重构脚手架；false-win 复测；precision gate；依赖恢复 | 保留探索与 best 两条线，去掉 teacher/meta 和依赖图组织；固定 oracle + 本地可复算判定 | 它对数值精度、复杂重构、跨阶段恢复更成熟。我们默认 allclose 不能替代它的高精度 gate；精度敏感任务应接入已有专用验证 |
| **[AKO4X][akox-archive]**：archive_variant、compute_score | 代码、配置、父版本、失败经验一起继承；长期归档 | 小任务普通目录即可；归档与晋升明确区分，只有完整正式证据能改 best | 它的 master/child、长周期候选谱系与会话归档更完整。`archive_variant` 允许无评测证据是保存原型能力，不能把这一点写成它的晋升漏洞 |
| **[JustGoal][justgoal]**：公开实现与消融目录 | 强 agent 自主构建搜索路线；少约束目标驱动是一条重要基线 | 预先提供可靠测量，避免每个任务重写同类 evaluator；保持路线自主 | 它有真实 B200 kernel 与实验产物；我们的九次预实验受接入故障影响，未形成有效的同预算对照。附加 skill 是否提升最终效果必须实测，不能从架构图推导 |
| **[KernelAgent][kernelagent]**：optimize_kernel | profiling→瓶颈→改动；复用同基线 NCU；runtime 与 SOL 各自绑定实现 | profiler 按需调用，结果中的每个耗时都绑定 source/case/role；使用者可直接用现有 agent | 它的 roofline/NCU、beam 多候选与子图生成组合更系统。不能把它描述为“每轮必定重复 baseline profile”来制造优势 |
| **[GEAK kernel_workflow][geak]**：工作流 README | AMD kernel/model 优化、分析、独立验证、组合与原始基线比较 | 当前 NVIDIA/CPU 任务不需要接入其多角色运行框架；借鉴其反馈与复测原则 | 它有 AMD 和 serving 工作流的针对性，我们不提供相同能力；不能跨硬件做速度排名 |

此外，evaluate 参考了 [robust-kbench 的 forward 检查][robust-forward]、[BackendBench][backend-eval]、[PyTorch benchmark 的计时][torch-timer]和[离散度判断][torch-common]、[ComputeEval][compute-eval]；固定来源见 [evaluator_sources.json](audit/evaluator_sources.json)。BackendBench 将正确性集与性能集分开的接口直接启发了 `measure: false`，这项设计有明确来源。

以下改动由参考规范和可复现反例推动，均已落实：

| 第一版的问题 | 对照依据 / 反例 | 本次实现 | 可重复验证 |
|---|---|---|---|
| 只限制相对 best 的回退 | kernel-lab 要求固定 baseline 累计预算；100→104→108 每步可低于 5%，累计却超过 | full 同轮测 best 和起始 baseline，逐 case 两套预算，两个目标的收益/噪声分别确认 | `test_cumulative_regression_cannot_hide_behind_better_aggregate`、`test_improving_degraded_best_is_not_original_baseline_win` |
| 新对象却循环使用相同数值 | AKO4ALL 的 fresh-values 模式防止值缓存影响计时 | 每个 timed sample 换值，各角色按 sample 匹配；计时输出抽查，再换值验证 | `test_timing_values_vary_and_match_across_roles`、`test_actual_timed_output_is_checked_without_candidate_rerun`、`test_post_check_detects_failure_after_initial_correctness` |
| 全量覆盖不等于按正确顺序测量 | Atrex 同时约束 ABBA 调度与完整记录 | 重算样本身份与顺序，拒绝先测完 A 再测 B 却冒充配对结果 | `test_wrong_temporal_order_rejected_even_with_complete_coverage` |
| Skill 写“先修失败 case”，CLI 却没有定点命令 | KDA/kernel-lab 要求选择最便宜的有效反馈 | quick 支持 `--case --seed --check-only`，无需改 task.json | `test_targeted_check_only_keeps_contract_and_covers_excluded_case` |
| 返回值正确仍可能违反接口语义 | kernel-lab 明确输入/输出 alias、layout 与实际入口 | 存储绑定检查、任务 validate、`load_best()` 正常入口 | `test_task_validator_checks_output_alias`、`test_public_entry_uses_verified_snapshot_in_fresh_process` |
| 字节相同代码可能碰巧测得更快 | ForgeKernel 的 false-win 路径记录过相同代码的环境假胜利 | 当前 best 的候选文件完全相同则不晋升；基线始终同期实测 | `test_identical_source_cannot_be_promoted_by_noise` |
| 扩大正确性覆盖会附带全部计时成本 | BackendBench 分开 correctness/performance 集 | 补充 case 可只验证正确性；计时 case 继续同时接受正确性检查 | [覆盖回归测试](tests/test_coverage.py)、[成本探针](audit/coverage_scope_validation.json) |
| 新 size 通过仍可能不支持新布局 | 冻结候选在 B300 的行间 padding 验证失败 | skill 要求独立变化各输入 stride；新增 packed 通过、padded 拒绝的真实子进程回归 | [B300 记录](audit/generality_gpu_050/summary.json)、[覆盖回归测试](tests/test_coverage.py) |

目标函数也必须匹配使用场景。两个等权 workload 的 baseline 为 `[1,100]`，candidate 为 `[0.25,150]`，几何平均延迟降低约 38.76%，总延迟却从 101 增到 150.25。本仓库按声明的调用权重计算总延迟，并限制逐 case 退化；排行榜任务则应使用排行榜规定的目标，不能把不同分数直接排名。

## 验证结果与迭代成本

### 先看泛用性：尺寸通过，布局失败

0.5 在一张 B300 上通过 **117 项测试、零跳过**。随后对预实验冻结的三个优化候选做新覆盖验收，未修改候选、调用模型或沿用开发时的晋升结论。使用独立 NumPy FP64 oracle，并抽查实际计时输出。

| 冻结候选 | 新覆盖 | 正确性 | 两批加权延迟降低 | 超过 5% 的逐 case 退化 |
|---|---|---|---:|---|
| GEMM / skill 产出 | 10 个 M/N/K/stride 组合 | 通过 | 33.83% / 33.75% | 无 |
| GEMM / 目标提示产出 | 同上 | 通过 | 34.45% / 34.27% | 无 |
| 融合 ReLU / skill 产出 | 11 个 rows/columns/stride 组合 | 通过 | 65.21% / 65.15% | 无 |
| 上述三个候选 | 另外增加行间 padding | 全部在首个 case 失败 | 未计时 | — |

新尺寸覆盖瘦长矩阵、非整倍数、tile 邻域；融合长度包括 1、255/256/257、1023/1024/1025、8191/8192/8193、32769。三个候选在原 FP32 布局族内适配了这些新组合，但两个 GEMM 用 `row * K` 访问 A，融合 kernel 用 `i * column_stride` 线性访问输入，都隐含行紧密排列假设。padding baseline 控制全部通过，候选失败。**尺寸泛化通过还不足以交付布局通用的算子。** padding 扩大了原预实验的输入契约，原始结果仍保留原判定。

[摘要与资源记录](audit/generality_gpu_050/summary.json)、[原始结果](audit/generality_gpu_050/raw.json.gz)、[源码快照](audit/generality_gpu_050/sources.tar.gz)和[环境清单](audit/generality_gpu_050/manifest.json)可复核。源快照在匹配 GPU 环境中以 `PYTHONPATH=src python validate_gpu.py` 复现。任务 Completed、Pods=0；平台运行 158 秒，从提交到确认释放约 274 秒。

### 成本按实际耗时判断

| 对照 | 测得结果 | 对流程的影响 |
|---|---|---|
| 0.5，12 个 case 全计时 vs 4 个代表 case 计时，各三次、交替次序 | fresh/reuse 检查均为 144；计时样本 288→96；计时输出及后检查 48→16；API 耗时中位数 4.466→3.955 秒，减少 11.4% | 广域正确性与代表性性能覆盖分开；收益包含减少预热和后检查，未计时区域不能宣称加速 |
| 0.3.0→0.3.1，相同 torch.add，72 次 B300 评测 | 4,194,305 元素 full 中位数 7.830→1.973 秒；257 元素 1.872→1.923 秒 | 大张量完全相等时省去 CPU 搬运；小张量保留原路径，不能笼统宣称所有情况加速 |
| 0.3.1→0.4，同 B300，24 次交替评测 | quick/full 总耗时仍约 2 秒，大输入增加约 50 毫秒 | 新配对判定改善收益识别，并未证明反馈更快；候选接近交付且 quick/full 成本相近时直接 full |
| 正常完成的第 8 次 agent 预实验 | 6 次 full 的 worker 共 78.8 秒，setup 占 69.6 秒，正确性 4.59 秒，测量 0.31 秒 | 主要成本在编译/初始化；CLI 保留编译错误输出，复用工具链缓存，减少无用启动 |

对应证据：[0.5 覆盖成本](audit/generality_gpu_050/summary.json)、[大张量优化及 72 次结果](audit/evaluator_efficiency.json)、[0.4 实机对照](audit/paired_policy_gpu.json)、[agent 预实验](audit/search_pilot_20261110/summary.json)。这些测量分别回答 evaluator 成本与接入成本，不混成完整搜索系统排名。

默认调用成本也可直接计算：C 个正确性 case、M 个计时 case、S 个种子、T 个 trial、R 个实现，quick/full 分别有 B=1/2 批。计时样本为 `2×B×M×T×R`；fresh/reuse 正确性记录为 `2×C×S×R`，有计时时另加 `2×M×R` 条 timed_output/post 记录。开启 deterministic 再加 `C×S×R` 条。oracle、输入生成、预热和初始化也占 wall time；样本更少不必然总耗时更短。full 已有 best 时 R 从 2 变为 3，多出的固定 baseline 用于守住累计退化预算。

### 直接对照验证了哪些优势

| 对照与方法 | 已确认的结果 | 证据 |
|---|---|---|
| AKO4ALL 固定 `bbd0e1c` 的原生 CLI，同 B300，12 类正确性场景 | 双方接受四类合法实现、拒绝基础错误；共享 oracle 别名、错误原地累加两倍的反例，上游接受、本仓库拒绝；正数 sqrt 被其 randn 哨兵误拒，本仓库保留输入域 | [配置与摘要](audit/ako4all_comparison.json)、[完整输出及源码](audit/ako4all_comparison_raw.json.gz) |
| 同一 AKO4ALL 对照，两种尺寸/两档样本、每档三轮 | 两边总 CLI 反馈成本都约 2 秒；本仓库约 1.99–2.09 秒，上游约 2.11–2.15 秒 | 同上；上游预热、L2 清理和默认样本数不同，这不是默认配置或 kernel 延迟排名 |
| BackendBench 固定 `2a8d7e1` 原生 `eval_correctness`，11 场景×3 seeds，加 6 个容差探针 | 双方接受五类合法实现、拒绝基本错误；共享返回别名、只读输入修改、错误原地更新由本仓库额外拒绝；上游公开 helper 未转发容差，既有误接受也有误拒绝 | [复现脚本](audit/probe_backendbench.py)、[摘要](audit/backendbench_component_comparison.json)、[原始记录](audit/backendbench_component_raw.json.gz)；CPU 组件对照，非完整 backend 运行 |
| Atrex 固定 `bab3218` 原生接纳规则与旧 0.3.1，24,000 组模拟 | 旧规则降低误接受，同时漏掉明显收益；更严格不自动等于更好 | [原生规则对照](audit/acceptance_policy_comparison.json) |
| 0.4 修订，13 类噪声×3 档收益×2,000 次，共 78,000 组配对模拟 | 零收益 26,000 组两版均未接受；10% 独立抖动、20% 真收益的检出率 4.95%→87.45%；部分暂停场景检出下降 | [完整模拟](audit/paired_policy_validation.json)，是旧新版对照，非 GPU 错误概率保证 |
| 0.4 B300 的 42 次探针，同一记录用旧规则重算 | 相同实现及相同周期等待均零晋升；候选等待减少 20% 时旧规则 0/6、新规则 6/6 | [实机摘要](audit/paired_policy_gpu.json)，人为等待用于测判定机制，不算 kernel 优化成果 |

曾尝试只更换噪声公式而保留不匹配的收益统计量，该原型在零收益周期漂移场景误接受 198/2,000，已否决；[原型和原始反例](audit/acceptance_policy_comparison.json)仍保留。当前同时要求总体收益与配对收益，以避免用一个统计量的低噪声支持另一个统计量的表面收益。

计时工具对照也改变了设计判断：[B300 校准](audit/gpu_timer_probe.json)中，257 元素 torch 的 eager event/wall/Graph 回放分别约 12.18/13.66/1.18 微秒；在 host 增加 200 微秒 sleep 后分别变为 273.74/274.38/1.30 微秒。实际部署用 Graph 时必须采用 Graph evaluator；本库 eager event 的间隙不能解释成纯 kernel 时间。

状态隔离、精度与证据检查的其他回归包括冻结 oracle/计时返回值、FP16 比较溢出、全局数值设置污染、零 MAD 多峰、遗漏 helper、过期 `.pyc`、Python `-O` 和伪造 worker 结果。详见[旧新版本反例](audit/evaluator_regression_probes.json)、[测试](tests)。这些保障已进入运行时，无需用户每轮手动执行额外流程。

### 搜索流程是否更好：现有结果与下一步

九次 GPT-6 Astra 预实验已经执行并保留全部结果：六次正确交付、一次编译失败、两次未交付。其中 0–1 受模型代理串行锁影响，2–7 受 SSE 终止分隔符截断影响，1、7 被控制器中止；只有第 8 次正常完成。该次融合 ReLU 两批收益为 65.2%/65.5%，但不能与异常结束的组比较来排名流程。[全部运行及成本](audit/search_pilot_20261110/summary.json)、[释放记录](audit/search_pilot_20261110/resource-release.json)保留；GPU 保守计费范围约 28.5 分钟，九次额度已用完。

实验接入现在改用 Codex 原生 Responses provider，取消自建模型 HTTP/SSE 代理。凭证仅按环境变量名传入容器；[本地合成验证](audit/direct_provider_validation.json)检查多轮工具调用、分段终止事件、HTTP 延迟关闭、超时清理和源码冻结，**零真实推理、零 GPU**。它验证传输与生命周期机制，真实模型和远端 GPU 联通仍需在正式实验时确认。相关代码只在 `audit/`，不进入产品运行包，也不增加 CPU 开发阶段。

后续对照采用以下具体方案；尚未启动，另需模型/GPU 预算授权：

- **比较对象：** 目标提示、AKO4ALL 改编流程、本 skill；同模型 GPT-6 Astra/medium，同开发 evaluator、工具、资料和起始实现，只改变工作流指令。因此结果回答 skill 的价值，不能冒充各仓库完整系统排名。
- **规模：** 归约、GEMM、融合三任务×三组×五次独立搜索，共 45 次，每次最多 8 分钟。串行 1 张 B300，搜索最多 6 小时，环境检查和独立验收最多 1 小时，总 GPU 上限 7 小时。模型以 45 次独立会话和逐次 wall deadline 为上限，记录全部实际 tokens/费用，不复用此前九次授权。
- **覆盖：** 在新任务目录固定公开尺寸范围、dtype、各输入独立 stride/padding、数值要求和实际入口；公开代表开发点，独立验收从同域保留未见尺寸/布局。所有区域都正确，代表性能点覆盖小/大尺寸、分支邻域和 fallback。冻结任务/提示词/运行次序/验收样本后再开始，不改历史三-case 预实验。
- **执行：** 干净工作目录和独立对话，按任务/重复次数分组并随机化三组顺序；预算到立即冻结声明交付。共享编译环境预热策略固定，不允许前组候选或编译缓存污染后组。接入故障触发暂停排查，不自动消耗重试次数。
- **验收：** 独立进程、GPU 串行、同期对照；验收不反馈到同次搜索。保留失败与无交付；先报告正确交付率，再报告全部运行的收益、区域最差退化及模型/GPU/评测总成本。修改候选后重做最终验收；已揭示的失败 case 进入回归集，另取新的同域样本检验泛化。

当前可以确认的是：简单工作流已实现，一些具体正确性检查和 evaluator 成本有可复现改进；原生搜索流程的总体胜负仍未得到有效对照。下一步用真实搜索结果决定是否还需加机制。

## 开发检查

```bash
python -m pip install -e '.[dev]'
ruff check .
ruff format --check .
pytest -q
```

CI 在 Linux Python 3.11/3.12/3.13 上运行 CPU 测试，3.12 额外安装 PyTorch CPU。CUDA 测试设置 `KERNEL_ENVOLVE_TEST_CUDA=1` 并使用空闲 GPU。这是维护 evaluator 的开发检查，不是 GPU 算子开发的 CPU 前置阶段。

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
[robust-forward]: https://github.com/SakanaAI/robust-kbench/blob/078f5bab29934a822268d59a4e707d449abf9b4e/robust_kbench/sandbox/correct_forward_fn.py
[backend-eval]: https://github.com/meta-pytorch/BackendBench/blob/2a8d7e19e2c13c789ff2dceb44883ea8ea04dab4/BackendBench/eval.py
[torch-timer]: https://github.com/pytorch/pytorch/blob/8a9a9e737dba491a2eea270641bff86eea6e1ea2/torch/utils/benchmark/utils/timer.py
[torch-common]: https://github.com/pytorch/pytorch/blob/8a9a9e737dba491a2eea270641bff86eea6e1ea2/torch/utils/benchmark/utils/common.py
[compute-eval]: https://github.com/NVIDIA/compute-eval/blob/00fbad91b63957a002b501f33a06510cd25beaae/README.md
