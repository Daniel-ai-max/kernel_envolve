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
# 修单个失败 case 时，只做定点正确性检查，无需改任务配置。
kernel-envolve evaluate --task examples/vector_add --mode quick --case strided --seed 17 --check-only
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
| `rtol` / `atol` | `1e-5` / `1e-6` | 浮点逐元素容差，按任务数值要求配置 |
| `equal_nan` | `false` | 是否允许同位置 NaN 相等 |
| `mutable_inputs` | `[]` | 允许原地更新的输入索引；更新后的值仍须与 oracle 一致 |
| `quick_seeds` / `full_seeds` | `[0]` / `[0,1,2]` | fresh 检查种子；每个种子还有一次同对象换值检查 |
| `quick_trials` / `full_trials` | `3` / `10` | 每个 case、每批的配对轮数 |
| `warmup` | `3` | 每个实现、每批、每个 case 的预热次数 |
| `min_gain` | `0.01` | 加权延迟下降比例必须超过该值 |
| `noise_multiplier` | `2.0` | MAD 噪声门槛倍数 |
| `max_regression` | `0.05` | 任一 case 相对当前 comparator 的最大允许延迟增长 |
| `max_baseline_regression` | `0.05` | 任一 case 相对固定起始 baseline 的累计延迟增长上限 |
| case 的 `max_us` | 无 | 该 case 候选延迟的绝对上限，微秒 |

这些默认值用于起步，不是所有算子的精度或收益标准。任务建立时确定它们，迭代候选时保持不变。修改固定任务契约后，使用新的任务目录重新建立 best。

本地 helper 用相对导入，例如 `from .helper import launch`，并写入对应 files 列表。配置、构建代码、小型常量文件也要列入。快照只收集显式文件，拒绝隐藏文件和符号链接，总大小上限 32 MiB。输入采用工厂生成；外部依赖使用当前 Python 环境，结果记录 Python、平台和 NumPy/PyTorch/Triton 版本。

## evaluate 如何判定

### 正确性先于计时

每个 case、seed 都分别检查 candidate 和 comparator；full 比较 best 时还检查固定 baseline：

- 新建匹配的输入，与固定 oracle 比较输出结构、shape、dtype、device 和逐元素值。整数/bool 精确比较。
- 检查未声明的输入修改、输入布局/存储绑定改变和 alias 关系改变；声明的原地更新须与 oracle 更新后的输入一致。
- 保留原输入对象，用 `seed + 1000003` 生成的新值覆盖其内容，再执行并验证，捕获按对象地址缓存旧输出的问题。
- 对每个 case/实现，直接检查最后一个实际计时调用的返回值及输入状态（`timed_output`），而非重新调用候选来代替它；这是抽查，不是所有计时输出的全量验证。随后在同一输入对象上换值验证，捕获预热/计时后才出现的错误。`--check-only` 没有计时阶段，只做前面的检查。

任何数值失败都会停止该轮计时，得到 `execution=completed, correctness=fail`。导入、编译、运行异常或超时得到 `execution=error`。每轮使用新子进程，超时终止其进程组；失败的 CUDA context 不复用。失败记录包含 case、seed、fresh/reuse/timed_output/post、实现角色及误差，日志保留异常栈。

### 测量和确认

默认 comparator 是任务当前 best，没有 best 时使用固定 baseline。full 有 best 时还在同一轮重新测固定 baseline，分别检查增量收益和累计回退。没有 best 时两个对照是同一个实现，只测一次。每次都重新测对照，不把历史 timing 当分母。`--baseline` 可专门测原始 baseline，但已有 best 时，这种结果不能替换 best。

quick 测所选 case，一批样本；full 测全部 case，两批样本，第二批倒序遍历 case。两个实现采用 A–B–B–A 并轮换起始角色；三个实现采用 A–B–C–C–B–A，并在各轮循环改变起始角色。每个角色的两次调用使用不同的值，同一 repeat 的各角色使用匹配的值。输入种子由 case、batch、trial、repeat 和任务 seeds 确定，每个测量样本都更新，可从记录复现。判定时同时核对样本身份、数量和实际执行顺序。

CPU 使用 `perf_counter_ns`，测完整函数调用；CUDA 使用当前流上的 event。输入构造/复位、预热和同步等待不计入延迟；实际 `run` 内的布局转换、临时缓冲和 kernel 工作保留在调用中。每次测量前重新生成输入，适用于声明的原地更新；测得的是该输入准备方式下的延迟。

full 的每一批分别计算：

```text
T_candidate  = Σ weight(case) × median(candidate_samples)
T_comparator = Σ weight(case) × median(comparator_samples)
gain         = 1 - T_candidate / T_comparator
noise_guard  = noise_multiplier × Σ weight(case) × (MAD_candidate + MAD_comparator)
               / T_comparator
```

两批相对 best 的 gain 都必须超过 `max(min_gain, noise_guard)`；相对固定 baseline 也单独计算相同的收益/噪声门槛。所有 case 必须同时满足两套回退限制和绝对延迟上限。这避免逐轮 4% 的局部退化越积越大，也避免只赢过一个已经退化的 best。MAD 门槛是保守筛选规则，不是统计置信区间。结果接近门槛时，先检查环境竞争；确需调整协议时记录原因、保留旧结果，重新建立对应契约和对照。不要重复抽样直到出现有利结果；profiling 用于解释瓶颈，不与正式计时混跑。

`--case`、`--seed`、`--check-only` 仅限 quick，可定位 full 中才运行的 case，不修改固定任务契约，也不能产生晋升资格。`worker_wall_seconds` 记录整个子进程耗时；`measurement_wall_seconds` 记录预热、输入生成、计时、后检查所在阶段的总耗时，两者都不是纯 GPU 执行时间。

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

`promote` 会校验结果和源码、重算覆盖及收益、检查 evaluator 和任务契约是否变化，并在锁内核对 best 是否仍是当时比较的版本。quick、缺失/重复/乱序样本、无效耗时、噪声过大、旧 comparator、事后修改的证据均不能晋升。与 best 完全相同的候选源码也不能靠噪声再次晋升；注释变更虽会改变哈希，但仍需 agent 核实实际生效的机制变化。best 引用保存的源码及 helper，工作目录继续编辑不会改变 best。

调用已验证版本时使用稳定入口：

```python
from kernel_envolve import load_best

run = load_best("my_task")  # 校验 best 的结果和源码哈希，加载保存的 helper
output = run(*inputs)
```

测试在新 Python 进程中验证该入口：即使工作目录的 candidate 已损坏，也仍调用保存的 best；快照损坏则拒绝加载。集成进已有模型的 dispatcher 时，仍需验证模型实际调用了这一入口。

0.2 更新了验证/计时协议与累计预算。旧 run 保留为历史，不能沿用其晋升资格；已有 0.1 best 的任务请将 task.json、task.py、candidate.py 及声明的依赖复制到新任务目录后重测。

文件快照和哈希用于复现与防止误用；候选作为本机代码运行，这不是隔离不可信程序的安全沙箱。任务内有运行锁和晋升锁；GPU 性能实验仍需由使用者在目标设备上串行执行，其他任务目录不会自动共享设备锁。

命令退出码：`0` 表示评测正确完成（包含慢或收益不确定），`1` 表示正确性失败，`2` 表示配置/执行错误。是否值得晋升看 `eligible_for_best`，不能只看退出码。

## 第一版范围

已实现普通 CPU 函数、NumPy、PyTorch tensor 和 NVIDIA CUDA 当前流上的 kernel 评测，支持非连续输入、结构化输出和显式原地更新。更换实现语言不需要改 agent 流程：只要 `candidate.run` 能调用实际入口，并将本地依赖列入快照即可。

本版不内置 CUDA Graph、多流事件汇合、分布式通信、autograd 图、显存上限判定或任意原生工程构建缓存。遇到这些任务时，保留该 skill 的迭代方法，接入任务已有的专用 evaluator。CUDA event 指标不能直接替代 CPU launch/端到端应用延迟，输出 alias/存储布局等特殊契约通过 task.validate 显式验证。GPU 示例需在目标机器上完成验证后再用于性能结论。

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
| **[Atrex Light][atrex-record]**：recorded_verifier、[native_abba][atrex-abba]、verifier | Supervisor 管正式证据、源码绑定、配对测量、完整策略匹配、恢复与复用 | 本地直接执行，无 runtime/provider 接入；加权总延迟与逐 case 双预算贴合当前任务 | 它的远端执行、记录所有权、批次断点恢复、接纳审计更成熟。我们的进程隔离不等于它的运行基础设施或安全边界 |
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

**默认成本可直接计算。** C 个 case、S 个 correctness seeds、T 个 timing trials、R 个实现：完整 quick/full 分别是 1/2 批，每批 `2×C×T×R` 次测量；完整正确性含 `C×R×(2S+2)` 条检查记录，额外的 2 是实际计时输出抽查和计时后换值检查；输出抽查复用真实返回值，不额外运行候选。check-only 是 `2×C×S×R` 次检查、零计时。每次实现检查另运行 oracle，成本不能忽略。

full 已有 best 时 R 从 2 变为 3，因此计时调用数增加 50%；这是修复累计预算缺口的明确代价。没有 best 时复用 baseline 对照，不重复测两遍。预算紧时减少探索轮的范围或用 signal，而不删正式验收的必要对照。任务的 `worker_wall_seconds` 用于检查这些开销是否值得；本次 [本地 smoke 记录](audit/local_validation.json)只测 CPU evaluator 开销：定点检查是 4 次检查、零计时；quick 是 8 次检查、12 个计时样本；full 起步为 32/160，已有 best 为 48/240。本例只有亚毫秒函数，Python 启动耗时足以掩盖样本数差异，单次数据不用于 wall time 排名；不能由更少样本直接推断 wall time 更短，也不据此推断 GPU 或 agent 总搜索效率。

### Evaluate 的可信度与仍需验证的部分

| 维度 | 已由代码和测试保障 | 不混淆的结论 |
|---|---|---|
| 数值与状态 | fresh/reuse/timed_output/post；整数精确检查；dtype/shape/device；非有限数；原地结果；自定义语义校验 | allclose 不等于梯度、方向漂移、批次不变性、race 或数学证明；按任务接入专用验证 |
| 时间与目标 | 同期交错测量、原始顺序校验、正有限耗时、权重目标、双预算、两批噪声筛选 | MAD 是启发式门槛；case 预算针对声明的中位延迟，不是尾延迟 SLA 或置信上界 |
| 身份 | task/ref/candidate/helper/evaluator 哈希；结果 receipt；当前 best 重新核对 | 外部动态库、驱动与环境需部署固定；本地哈希不是对不可信候选的安全隔离 |
| 晋升与入口 | 只接受 full；拒绝改写、漏测、乱序、旧比较对象、相同源码；新进程 load_best 验证 | 哈希变化不证明机制有用；模型中的默认 dispatcher 仍须集成测试 |
| 迭代效率 | 诊断、signal、quick、full 各有用途；不强制多 agent、每轮 profiler 或额外稿件 | 减少固定步骤是已实现事实；节省多少模型/GPU 时间、最终 kernel 多快尚需实际任务对照 |

### 是否优于所有参考仓库

本次可以确认：**相对自己的 0.1 版本，评测完整性和迭代操作有具体改善，并有对应反例测试；相对纯流程规范，本仓库增加了可执行、可复算的保障。** 对目前的单 agent、少量算子、现有机器场景，这是建议使用的组合。

目前不能确认“全面优于九个仓库”。我们继承了它们许多核心方法，也尚缺 GPU 实验、专用精度 gate 和远端恢复等证据。强行给所有行打胜出会掩盖实际缺口。

要验证优化效果的优势，下一步应在同一 B200/B300、同一模型、同一任务集和预算下，先比较最相关的 **JustGoal 式目标驱动、AKO4ALL、本 skill**：共同使用独立的最终 evaluator，记录最终正确率、相对原始实现的逐 case/加权收益、总 wall time、模型成本、GPU 时间和失败恢复成本，保留多次独立运行。任务至少覆盖归约、GEMM 和含非连续/尾部形状的融合算子。先验证这组对照有收益，再决定是否引入更重的编排能力。

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

## 开发检查

```bash
python -m pip install -e '.[dev]'
ruff check .
ruff format --check .
pytest -q
```

测试覆盖真实 CPU 子进程评测、正确性拒绝、缓存旧输出、输入修改、超时、覆盖完整性、噪声与权重判定、快照/helper 完整性、旧 best 晋升冲突等。CI 在 Linux 的 Python 3.11/3.12/3.13 上运行 CPU 测试，3.12 额外安装 PyTorch CPU，覆盖 tensor 路径；CUDA 测试需显式设置 `KERNEL_ENVOLVE_TEST_CUDA=1` 并使用空闲 GPU。
