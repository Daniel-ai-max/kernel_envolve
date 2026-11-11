# 有界的跨任务自进化

RSI 的目标是减少重复错误、改进假设选择和实验安排，在相同预算下更快得到通过完整验证的算子。它是可选的后台学习流程；现有 evaluator、kernel 晋升规则和生产 Manager 继续负责原有职责。

实现起点为 Manager 的 `a3210b0`；其内容已由独立操作通过 PR #1 合入 main（`8cb7665`），两个基线源码树相同。`feature/rsi-evolution` 已整理为 main 之上的 RSI 独立增量，本次没有向 main 提交或合并。试用本分支必须设置 `KERNEL_ENVOLVE_AUTO_UPDATE=0`，避免现有自动更新入口切回尚无 RSI 的远端 main。

最新实证状态（2026-11-11）：当前余量分配原型的 64 次新对照已完成，但未通过验收：相比原四步搜索，优化耗时减少 37.32%、tokens 减少 49.64%，质量分数下降 5.85%，逐任务与逐重复保护拒绝；o/p 确认关闭未运行。固定两步复用此前也未通过原四步系统的质量保护，旧 k/l 仍关闭。真实 Manager GPU 接入工程验证已通过，但不等于学习收益。RSI 未生产激活，所有负结果及失败成本保留，见 [完整结果与成本](rsi-design-review.md)。

## 实现范围

闭环是：Manager 自动保存实验结果和研究 checkpoint → RSI 增量收集 → 学习适配器生成带证据的局部修改 → 独立任务上配对验证 → 发布有范围的策略快照 → 后续 Agent 轮次使用 → 审计和回退。

第一版进化的是条件化的经验和研究指引，包括失败诊断、方向选择、quick/full 安排与工具使用建议。候选中不允许可执行插件、任意代码补丁、任务契约或资源配置。它不能修改 oracle、精度、full 测量协议、kernel 晋升门槛或发布权限。更合理的门槛可以成为研究建议，但评测方法变更仍走独立的软件评审和校准流程。

它没有训练模型权重，也不宣称解决了通用递归智能增长。学习者可使用与任务执行者相同的已有模型服务；不内置新的模型供应商。

## 两种晋升相互独立

- kernel 晋升：仍由 `promote` / `native-promote` 决定，RSI 不调用它们。
- 策略晋升：验证新策略能否在完整研究任务中稳定改善结果或减少 solver 耗时，然后更新下一轮的研究建议。

验证不会改动生产 candidate、best、基线或 Manager 数据库。策略被撤回时，已经独立验收的 kernel 无需删除。

## 开销与运行隔离

默认不开启 RSI。普通评测不导入 RSI、不启动学习进程、不扫描实验历史。Manager 的 checkpoint 入口每轮只尝试读取一个小文件：最多 8 KiB，其中建议文本最多 2400 UTF-8 字节、最多六条规则。运行中的 prompt 已经冻结，不会热更新。

无文件、契约不匹配、损坏、超限或非普通文件都会回退原有流程。读取不访问 RSI SQLite，不等后台锁，不访问网络，不调用模型。`audit/rsi_overhead.py` 测量这条读取路径；它不测模型额外 token 的延迟。

独立 sidecar 每次最多读取 128 条 Manager 事件和 128 条已索引 run。学习窗口最多 128 条，模型输入最多 128 KiB，超出时缩小本批窗口而不丢弃剩余记录；有新证据且达到阈值、冷却期结束、累计预算足够时才调用学习器。失败批次也标记已消费，避免无限重试相同日志。一次 cycle 最多验证一个候选。

学习调用不暴露 CUDA；GPU 验证要求显式配置与生产 Manager 资源不同的物理 GPU，复用宿主设备锁并检查空闲，不在生产 GPU 队列插入长验证任务。多个 Manager/OS 用户仍需外部资源分配器保证设备独占。CPU 学习/验证可能消耗 CPU、内存、磁盘和模型服务配额；需要严格资源隔离时，在独立主机或有 CPU/内存限制的服务中运行 sidecar。进程 wall-time 预算不等于模型 token/金额预算，后者应由模型适配器或供应商配额限制。

RSI 的进程预算也不等于云资源租用期限。临时分配 GPU 时，部署必须使用平台执行的任务期限和禁止自动重试的策略；本机计时器只能作为额外保护，不能成为唯一的资源释放机制。网络断开或本机暂停会使外部 stop 请求无法执行。平台期限应覆盖学习、验证和清理的完整预算，结束后仍需核实实际 Pod 已退出。2026-11-10 的一次试跑准备暴露了本地 watchdog 的 DNS 失败，资源超时记录见 [试跑事故收据](../audit/rsi_pilot_incident_20261110.json)；该次没有完成 RSI GPU 对比实验。

## 分工与证据

generic 评测的后台证据包含有界 `failed_checks`：最多八项实际失败检查，以及冻结 case 的参数、总数和截断标记；每项参数最多 512 UTF-8 字节，超限保留未知。它只描述已观察到的失败。评测可能提前停止，未列出的形状、精度或布局不能被推断为通过。旧索引和 native 契约不适用时该字段为未知，不自动回写旧学习证据。

执行 Agent 沿用原有 `notes.md` 和结构化 report，记录假设、结果解释与下一步。不要要求每个工具调用都生成长反思，也不需要第二套平行工作日志。

RSI 收集两类数据并标注来源：

- evaluator 的已索引结果：运行身份、契约、正确性、结果 hash、性能判定、失败原因；涵盖成功与失败。
- report 与生命周期事件：Agent 的解释、超时和中断等。它们不是性能事实。

经验必须引用存在且属于当前 scope 的证据。提案、采用和定期审计都会核对引用；run 引用会重新调用现有证据校验器。事件摘要有长度限制，仍保存原事件哈希以便核验。没有 LLM 凭空填写 speedup 的路径。

新规则包含 `when`、`action`、`unless`，按任务 ID 显式确定推广范围。未提及的旧规则保留；删除必须给理由。候选只有通过验证才影响生产提示。相同父版本下，相同建议内容即使更换说明、证据或条目 ID，也不会得到新的验证机会。

### 条件筛选与可用能力

可选的 `profiles` 为生产任务声明操作者确认的 `tags`（工作负载标签）和 `capabilities`（适配器已有能力）。验证模板各自声明 `profile`。标签不是从任务名、模型解释或成绩猜测出来的；更新它们需要新配置，学习器不能修改。没有 profile 时视为事实未知，不视为拥有所有能力。

在声明了 profile 的 scope 内，每条新增/更新规则必须显式给出 `applies_to` 和 `requires`：前者的全部标签、后者的全部能力必须同时存在于当前任务的 profile 才会选中。空列表表示不要求该条件。例：`applies_to: ["row-reduction"]`、`requires: ["full-evaluate"]`；需要 `host-device-profiler` 的建议不能发给只声明了 full evaluator 的任务。规则至少要匹配一个声明的生产 profile，否则提案拒绝。旧配置和不带条件的历史规则保持原来的 scope 行为；若希望全面采用条件筛选，应初始化明确声明 profile 的新实验。

筛选在后台生成每任务快照时完成；生产依然只读一次文件。验证 solver 使用完全相同的筛选函数，收据所绑定的 suite 包含 profile。每一种实际发布的非空规则组合必须在至少两项冻结验证模板中出现；分别测过规则 A 与 B 不足以授权同时发布 A+B。组合覆盖不代表任务语义、设备和输入分布已泛化，仍需要独立审计。

这些声明限制建议的发送范围，不授予执行权限，也不能证明自然语言建议如实声明了全部依赖。建议文本仍不可信，工具约束、进程隔离和独立 evaluator 继续负责执行边界。六条规则和 2400 字节的整体候选上限保持不变；本版不是无限记忆库或语义检索器。

## 安装和接入

在已准备好依赖的环境中使用本分支：

```bash
export KERNEL_ENVOLVE_AUTO_UPDATE=0
python -m pip install --no-deps -e /path/to/kernel_envolve-rsi
```

先按 `optimization-manager.md` 注册生产任务。为策略验证另建任务模板目录，不能与生产工作区重叠；这些模板不应来自本轮用于生成经验的任务。操作者负责任务划分，系统只能检查路径分离，不能识别语义重复。

示例配置（命令和路径需要替换）：

```json
{
  "manager": "/work/manager-state",
  "scopes": {"reduction-b300": ["reduce-a", "reduce-b"]},
  "profiles": {
    "reduce-a": {"tags": ["row-reduction"], "capabilities": ["full-evaluate"]},
    "reduce-b": {"tags": ["row-reduction"], "capabilities": ["full-evaluate"]}
  },
  "learner_command": ["/opt/adapters/learn"],
  "learn_timeout_seconds": 120,
  "learn_budget_seconds": 1200,
  "validation_budget_seconds": 7200,
  "min_new_evidence": 8,
  "max_evidence": 48,
  "cooldown_seconds": 1800,
  "validation_device": "GPU-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee",
  "auto_validate": true,
  "suites": {
    "reduction-b300": {
      "model": "provider/model-pinned-version",
      "command": ["/opt/adapters/solve"],
      "repeats": 2,
      "solver_seconds": 120,
      "evaluation_seconds": 300,
      "max_seconds": 3600,
      "min_quality_gain": 0.01,
      "max_quality_regression": 0.02,
      "min_cost_gain": 0.1,
      "max_cost_regression": 0.05,
      "tasks": [
        {"path": "/work/validation/reduce-a", "kind": "generic", "profile": {"tags": ["row-reduction"], "capabilities": ["full-evaluate"]}},
        {"path": "/work/validation/reduce-b", "kind": "generic", "profile": {"tags": ["row-reduction"], "capabilities": ["full-evaluate"]}}
      ]
    }
  }
}
```

原生任务使用 `kind: "native"`，并声明允许修改的具体文件，例如：

```json
{
  "path": "/work/validation/native-probs-scale",
  "kind": "native",
  "editable": ["candidate/core/apodex_core/ops/example_kernel.py"]
}
```

路径格式是 `candidate/<native.json 中的源码根名>/<文件>`。配置中的 source 根先按原生 snapshot 规则冻结，转换为独立 Git 源码副本。adapter、测试、baseline 和其他未声明文件必须保持不变。第一版要求可编辑文件已存在，不允许 solver 新增或删除文件。

```bash
kernel-envolve rsi --state /work/rsi-state init /work/rsi-config.json
kernel-envolve rsi --state /work/rsi-state cycle
# 独立终端或服务；不用改生产 manage run
kernel-envolve rsi --state /work/rsi-state run --interval 60
```

配置、验证模板和预算在 init 后固定；改变模型、模板、推广范围或门槛应建立新的实验配置，不能编辑 config.json 后继续沿用旧证据。一个 Manager 只允许一个 RSI 发布者。更换配置时先停止旧 sidecar、执行 `disable` 和 `detach`，再用新目录初始化；旧状态保留证据，但不能再次发布。初始化失败可能留下未完成目录供诊断，不会静默覆盖；换新目录重试。没有配置 suite 的 scope 可以生成候选，但不能采用；`auto_validate` 要求所有 scope 都有 suite。

预算是累计值，重启不重置。启动外部工作前先持久化预留最大额度；正常结束按实际 wall time记账，崩溃未结算保留全部预留。到额度后停止相应工作，不影响生产任务。模板准备和验证 I/O 也计入验证耗时；慢文件系统与进程清理可能使实际时间略超预留，后续按实际消耗扣账。

## 适配器协议

可选的 `research.run` 研究循环若在真实 Manager agent round 中使用，必须显式传入
`managed_evaluations=True`。此时每次 initial/quick/full 都通过现有 `manage gpu-run`
的同一准入实现排队，保留父轮次、累计预算、物理设备锁和子进程清理；决策回调不持有
GPU 锁。不能把整段研究循环包在 `gpu-run` 内。默认独立实验仍沿用直接评测路径，
Manager round 中误用直接路径会被拒绝。

这个适配层只支持单文件 generic `candidate.py`，不自动收集或发布历史代码。
返回前重新核对任务契约、实际源码、模式和 evaluator 收据，不从 stdout 读取成绩。
排队和执行共用有界期限；超时保留失败并清理进程，不能变成有效观察或触发决策超时恢复。
它使用 POSIX 主线程定时器，拒绝覆盖调用者已有的 `ITIMER_REAL`；其他超时框架应放在
独立外层进程中。清理可能在期限后增加少量时间，父轮次与累计资源预算始终保留。

搜索结束仍需要独立的最终 full：可信适配器可调用
`rsi.managed.evaluate(task, "full", timeout, new_evidence_directory)`，然后按原流程检查及
决定是否 promote。每次调用要求任务外的新证据目录。这个工程桥接没有替代最终 gate，
也不是自动成果发布或已验证的生产收益闭环。

RSI sidecar 的学习器和验证 solver 都是操作者配置的 argv，不通过 shell 拼接，也不会从候选内容生成可执行命令。脚本路径建议使用绝对路径。这两个隔离验证子进程固定使用当前 kernel_envolve 安装，禁用自动跟随 main，不继承生产 Manager 的任务/轮次环境变量；上面的 Manager 内评测桥接则保留当前轮次身份。

学习器在独立目录运行，从 stdin 接收 JSON：说明、当前 scope/策略和一批证据。它写出 `proposals.json`（最多 64 KiB、三项，空列表合法）。每项格式：

```json
{
  "scope": "reduction-b300",
  "parent": "base",
  "hypothesis": "提前诊断特定布局错误，减少无效 full，但保留有效候选",
  "upsert": [{
    "id": "check-layout",
    "when": "明确满足的输入和硬件条件",
    "action": "建议的诊断或实验安排",
    "unless": "不适用条件；不能替代强制验证",
    "evidence": ["输入 evidence 中的完整 ID"],
    "applies_to": ["row-reduction"],
    "requires": ["full-evaluate"]
  }],
  "remove": []
}
```

solver 从 stdin 接收冻结任务副本路径、允许修改的文件、策略文本、固定模型标识、repeat seed 和时间额度。它应使用相同模型配置完成优化，退出前保存候选。不要调用 promote，不要离开允许目录或启动脱离父进程的任务。seed 是请求的重复身份；对不支持确定性种子的供应商不宣称可精确复现。

验证收据绑定命令可执行文件，以及 argv 中通过绝对路径直接传入的文件内容和解析路径；每次配对执行前、整轮结束及采用/audit 时检查。脚本使用绝对路径。间接导入的本地模块、提示词与配置文件应在 suite 的 `adapter_files` 中显式声明（最多 128 个文件，配置相对路径在 init 时解析）。文件变化会拒绝旧收据，后台 audit 撤回依赖它的活动策略。这个检查仅在后台执行；它不证明远端模型版本、环境变量或整个安装环境未变，部署仍需固定镜像与供应商配置。旧版缺少适配器身份的收据不可继续采用；升级遵循 disable → audit → enable。

solver 的 stdout 不是成绩来源。退出后控制器检查文件范围，启动独立 full evaluator，复核原始结果和收据。通用与 native 流程都支持；原生流程保留原项目数值/梯度/集成 gate。

外部适配器必须在前台运行。控制器限制运行时间和日志大小，终止其进程组，并在 GPU 使用后检查清理。生产宿主原本是同用户协作模型；这不是 OS 安全沙箱。若适配器驱动可任意执行命令的非受信任 Agent，应由部署方提供容器/独立用户、只挂载 trial 工作区、不暴露状态目录及发布凭据的隔离。文件哈希检测改动，不能抵抗同权限进程同时篡改程序和证据。

## 验证与采用规则

每个 suite 至少两项任务、两轮重复。每一对从相同冻结起点开始，交替运行 control/candidate 的顺序；control 是当前稳定策略，candidate 是局部更新后的完整策略。模板和 solver 模型配置相同。验证任务的源码、路径和逐次结果不提供给学习器的输入窗口，学习器只见生产轨迹。

当前保守规则：

1. candidate 每次 full 都正确完成；失败不能由收益或成本补偿。
2. 每个任务对的质量不低于 control 超过声明的退化额度。
3. 每轮重复分别计算任务等权平均质量，不能整体下降；正确性失败的 `-1` 标记按零收益计入平均，正确性修复单独计数，不能用负数差值制造性能收益。
4. 每轮的 solver 总耗时和 trial 总耗时均不得超过 control 的 `1 + max_cost_regression` 倍，默认允许 5% 测量波动。trial 包含副本准备、solver 和独立 full evaluator；质量提升也不能绕过这个成本上限。
5. 每轮还须满足至少一项：平均质量提升超过 `min_quality_gain`；修复 control 的正确性失败；或者保留已有整体正收益且 solver 总耗时降低超过 `min_cost_gain`。不能跨轮抵消退化，也不能用一项任务的改善掩盖整体退化。收据记录每轮汇总供审计。

质量是 full evaluator 允许晋升的结果相对固定 baseline 的延迟下降；不合格候选不计正收益。未产出可晋升结果时用保留 baseline 的零收益计算，正确性失败另记失败。策略验证不会实际晋升这些实验 kernel。

这是有限 suite 上的保守工程准入，不是统计显著性保证，也不是广泛泛化证明。等权平均是声明的验证目标，不代表生产任务频率；各任务的耗时仍需逐项检查，整体成本约束不能保证每项都更快。5% 不是校准过的置信区间；应先测环境抖动再固定门槛。成本测 solver 和 trial wall time，未测 token、实际 GPU 活跃时间和端到端训练吞吐，也未将生成与验证策略的一次性成本计入收益。验证集反复使用仍可能产生适应性过拟合，需由操作者周期性更换独立审计任务；`min_*` 不是宣称通用最优阈值。

验证器升级后，旧 receipt 的源码身份不再匹配，不能继续用于采用；后台 audit 会撤回使用旧验证器材料的活动策略。升级时先 disable，执行 audit 后再 enable。旧配置未声明 `max_cost_regression` 时采用 0.05；如需其他值，建立新配置，不修改已有配置或重试旧候选。

候选只有一次验证机会。失败或中断记录为 rejected；重启后未完成的 validating 不会自动重跑。完整 receipt 与状态原子保存，采用前复查证据、模板、evaluator 与策略验证器源码身份及当前父版本。其他候选先被采用时，旧父版本候选变为 stale，不能沿用之前的比较。

学习器调用失败不阻止已有候选的验证。没有新证据不调用模型；GPU 忙时不等待、不扣验证预算。验证失败保留稳定策略。额外的独立审计测试应由部署方在采用范围之外执行；当前自动准入不替代生产训练验收。

## 查看、停用与回退

```bash
kernel-envolve rsi --state /work/rsi-state status
kernel-envolve rsi --state /work/rsi-state collect
kernel-envolve rsi --state /work/rsi-state propose proposal.json
kernel-envolve rsi --state /work/rsi-state validate <candidate-id>
kernel-envolve rsi --state /work/rsi-state adopt <validation-receipt>
kernel-envolve rsi --state /work/rsi-state audit
kernel-envolve rsi --state /work/rsi-state rollback reduction-b300 --reason '发现跨任务反例'
kernel-envolve rsi --state /work/rsi-state disable
kernel-envolve rsi --state /work/rsi-state enable
# 更换发布者前先停止 sidecar 并 disable，然后 detach
kernel-envolve rsi --state /work/rsi-state detach
```

`disable` 可在学习或验证运行中执行，立即让新轮次回退原流程，并阻止采用；已经运行的有界调用会结束，已启动任务继续其固定 prompt。要立即停止 sidecar，发送 SIGTERM。`enable` 恢复当前稳定策略，不清空预算或证据。

回退撤销当前策略，恢复未撤销的父版本，必要时恢复空策略。周期性 audit 核验支持证据与采用时的验证材料，失效时回退。SQLite 保存采用谱系，策略文件只是可重建视图；中断发布后下一次 cycle / publish 重建。

不要手改 `rsi-context` 文件。生产正常路径不会再次检查全部历史，只做小快照的格式、hash 和任务身份检查；后台停止时无法继续发现证据后来失效的情况。运维应监控 sidecar 的 `last_cycle`、`last_error`，必要时 disable。状态文件位于本地可信文件系统。

## CPU 演示与测试

```bash
python examples/rsi/demo.py /tmp/kernel-rsi-demo
kernel-envolve rsi --state /tmp/kernel-rsi-demo/rsi init /tmp/kernel-rsi-demo/rsi.json
kernel-envolve rsi --state /tmp/kernel-rsi-demo/rsi cycle
kernel-envolve rsi --state /tmp/kernel-rsi-demo/rsi status
python audit/rsi_overhead.py
pytest -q tests/test_rsi.py tests/test_rsi_native.py
```

演示的学习器和 solver 都是确定性脚本，优化对象含人工 sleep。它检验学习、验证、采用、回退和数据隔离的工程闭环，不证明真实 LLM 的学习效果或 GPU 优化收益。

## 研究依据与设计取舍

- [ACE，2025-10-06](https://arxiv.org/abs/2510.04618)：执行/反思/整理分工、局部条目更新。本实现保留原始事件，不用反思文本替代事实。
- [Recursa，2026-11-02](https://arxiv.org/abs/2611.04035)：局部改进、合并及回归验证。本实现用完整合并策略进行匹配任务验证。
- [Meta-Harness，2026-03-30](https://arxiv.org/abs/2603.28052)：从历史代码与轨迹诊断。本实现保留 source/run 引用，窗口只提供有界摘要。
- [Dream-RSI，2026-09-14](https://arxiv.org/abs/2609.14858)：真实探索与离线改进分离。本实现尚未实现反事实搜索树回放，直接用隔离任务验证，不把缺失分支的结果当成已知。
- [EvoAudit，2026-11-05](https://arxiv.org/abs/2611.07624)：检查旧能力保留、成本与适用范围。这里区分研究候选和稳定策略，保留失败记录。

后续研究方向包括：基于实际轨迹校准筛选门槛、按问题簇检索经验、独立审计任务池、token 成本记录、受隔离的诊断工具进化。它们应在验证现有闭环的真实收益后逐项引入。

针对参考方法的逐项取舍、准入反例和实证比较方案见 [RSI 设计复核](rsi-design-review.md)。工程边界更明确不等于已证明方法优于所有参考工作。

初版工程验证记录见 [RSI 验证收据](../audit/rsi_20261110.json)：全仓 236 passed / 8 GPU-only skipped，33 项 RSI 专项测试，独立 wheel 安装闭环，以及每轮建议读取的本机微基准。该记录不包含真实模型学习收益或真实 GPU 策略实验。

准入复核后的验证见 [RSI 复核收据](../audit/rsi_review_20261110.json)：全仓 246 passed / 8 GPU-only skipped，43 项 RSI 专项测试；保留初次测试启动路径问题与复验说明。旧安装收据不代表新版验证器已复验。


### 最新收益验收状态（2026-11-10）

两轮扩展开发对照均未通过。最新轮每组 8/8 正确且可晋升，RSI trial 时间减少约 14.3%，但性能收益分数降低约 3.9%；加上学习时间后只节省约 8.9%，尚未计训练与选择。错误候选与 control 同为 6 次。未采用策略，未启动最终留出审计，PR 保持草稿。原始收据和成本见 [最新设计复核](rsi-design-review.md) 与 [第二轮收据](../audit/rsi_value_development_round2_20261110.json)。以下较早 pilot 为历史记录，不代表当前版本的收益结论。

### 真实 GPU pilot 状态（2026-11-10）

最新四组多步开发比较已完成：control/fixed/raw 各 4/4 个可晋升结果，learned 为 2/4（一次模型超时、一次性能不确定）；三种非空策略均未通过准入，active 保持 base。61 份 evaluator 收据离线复核通过；所有尝试成本含失败，不删除负面结果。具体版本、资源账及分析见 [设计复核最新比较](rsi-design-review.md) 和 [四组收据](../audit/rsi_multistep_gpu_20261110.json)。本地后续适配器身份检查通过全仓 265 passed / 8 skipped，尚未进入 GPU 对照。下面保留早期单次生成试跑的历史，不能混作同一个实验。

已完成 gpt-6-astra / 单张 B300 的两学习任务、两验证任务、两轮配对试跑。验证 8/8 正确性通过，但两组均无可晋升 kernel；候选策略每轮 solver 总耗时增加约 12–13%，被准入门槛拒绝，未发布到生产。详见 [设计复核](rsi-design-review.md) 和 [结构化收据](../audit/rsi_gpu_pilot_20261110.json)。这不是学习有效性或优于参考方法的证明。

随后改进的后台证据摘要包含有界的数值收益、晋升资格与截断标记；旧条目不回写。建议头部不再将待验证策略预先称作“已验证”，因此验证器身份改变，升级前先 disable，再 audit，最后 enable。新摘要与措辞通过 CPU 测试，尚未做新的 GPU 收益比较。

### 实验性多步 solver

`examples/rsi/research_solver.py` 是可选的通用任务实验适配器，不替换生产 Manager。操作者配置的 decision 命令从 stdin 读取任务、候选和真实评测历史，写出 `decision.json`。例如 solver argv 为：

```text
python /checkout/examples/rsi/research_solver.py --actions 2 --model-seconds 55 --evaluation-seconds 30 --model-command /opt/adapters/decision
```

只支持单个可编辑 `candidate.py`。固定动作是 `propose`（完整 source、quick/full、hypothesis）、`evaluate`（已有 candidate 哈希、quick/full、hypothesis）、`stop`（hypothesis）。命令由操作者配置；学习建议不能增加动作、改预算或改 evaluator。先 full 测量原始候选，每个动作后提供真实结果；相同源码/模式不重复测量。只有 full 正确且质量严格改善的候选才能替换保留的起点，quick 仅供诊断。最终仍由独立 RSI 控制器执行自己的 full，不以 solver 的成绩决定策略采用。

`search-trace.json` 保存每个假设、动作、源码身份、真实结果路径、成本和最终选择，保留失败与停止，不覆盖已有实验目录。模型输入最多 128 KiB；动作数、源码大小、每步与总预算受限。超时或中断恢复已确认候选并清理 evaluator 子进程；硬杀仍依赖外层进程监督和平台资源期限。这不是 OS 沙箱，模型传输回调也须由外层强制限时。

这是对 GEPA 等轨迹反馈机制的局部借鉴，同时为后续检验搜索记录能否用于回放建立可追溯数据；目前没有实现 GEPA 搜索或 Dream-RSI 的反事实回放。它能观察停止、无效重复、quick/full 安排，但仅两次动作不足以代表长期自主研究能力。与单次生成相比，它增加了测量成本，必须让所有对照使用相同工具与预算，不能把执行器本身的改进归因于 RSI。

多步适配器额外生成 `research-reports.json`：每条假设从对应 evaluator 的冻结源码取样，校验结果收据、任务契约、源码哈希与归档候选的一致性，分别记录 `attempted_candidate_id` 和 `final_selected_candidate_id`。停止动作不伪造评测或源码归属。不要把工作目录中已经恢复的最佳候选内容，作为某次失败尝试的源码上传；这会产生不真实的诊断线索。收集器保留 evaluator 来源的 `source_digest`，以区分同一源码的重复结果与真正不同的尝试。

## 实验性：把历史成功实现作为一次候选实验

完整成功代码提示和完整历史提示的真实 GPU 对照均未通过质量/成本验收，不能自动启用。后续原型优先检验直接成果复用：`rsi.reuse.select` 只从操作者固定的已批准内容 ID 中，按完整 `task.py` 字节（包含参考实现和输入生成器）、精度/可变输入配置、任务 profile 与当前实测运行环境匹配。仅支持 `reference_files=["task.py"]`、`candidate_files=["candidate.py"]`。不做模糊语义匹配，不用历史成绩给不同任务排序，也不保证新 shape/stride/seed 会通过。环境应来自当前运行环境观察，不能直接复制历史记录；物理 GPU UUID 和逻辑 stream/device 编号不作为兼容性判断。

返回值只是待测源码。受信任的研究适配器需要显式传入同一候选源码检查器，才可使用：

```python
from kernel_envolve.rsi.reuse import select
from kernel_envolve.rsi.research import run

proposal = select(library, approved_ids, task, task_profile, observed_environment)
trace = run(
    packet,
    output,
    decide,
    max_actions=4,
    artifact_proposal=proposal,
    source_validator=validate_candidate_source,
)
```

`validate_candidate_source` 是操作者提供的候选权限/语法检查器，同时检查历史候选和模型提出的候选；这不是内置 OS 沙箱。批准的库 manifest 和适配器依赖必须绑定进 `adapter_files`。内容哈希证明内容身份，不证明发布者可信。默认不提供 proposal，不扫描历史、不执行历史源码，生产 Manager 与既有策略发布器均不自动使用此原型。

有 proposal 时，执行器先完整评测原候选，再把历史实现作为第一个普通 `propose/full` 动作。它消耗四个动作中的一个，剩下最多三个模型决策，不额外赠送实验预算。旧成绩不进入当前选优；只有本次 full 正确且严格改善的结果才能替换原候选。历史实现失败、变慢或不具备晋升收益时，保留原来已经完整验证的候选。末尾控制器仍另做独立 full 评测，未增加自动 promote 路径。

proposal 记录源码、参考源码和来源 ID，保存有界快照并检测篡改；同一动作在 trace 中标记为 `artifact_proposal`，与适配器决策分开记录。此步没有调用 `decide`，模型决策耗时记录为零；实际 API 成本仍需外部调用账本，不能据此推断整个任务零 tokens。没有匹配返回 None；损坏或不完整的已批准记录明确报错，不能悄悄换一个来源继续冒用旧验证。已有收据因研究执行器/选择器身份改变不能直接沿用。

这是候选级复用的工程原型，尚无该机制的 GPU 收益结论；没有训练权重、开放自改系统代码或降低评测门槛。测试可证明动作计账、失败保留、输入身份、运行环境约束及独立最终评测，不证明实际节省成本。

### 有界的模型超时恢复（可选）

`research.run(..., stop_on_decision_timeout=True)` 允许一次决策回调抛出
`TimeoutError` 后停止搜索，保留本次已经完整评测且正确的 incumbent。默认仍为
`False`，既有适配器行为不变。超时占用本次动作，不重试、不追加动作、不延长预算；
trace 将其明确记为控制器停止和 `decision_error`，保留实际等待时间，token 用量为
未知，不能按零成本计算。report 和完整 episode 同样保留这项失败。

这个选项只捕获决策回调的超时。没有正确的 full 候选、评测器或源码校验器超时、
取消信号、其他适配器错误、经验快照或任务保护文件被改动，仍会失败。quick 候选
不能因此成为最终候选。外层 trial 仍必须在原有预算内独立完成 full 验收，才能报告
有效结果；该选项不调用 promote，也不修改晋升规则。

适配器必须明确区分传输超时和其他失败，并提供自身的调用期限。这个选项不是中断
阻塞回调的计时器，也不能挽救外层进程被硬终止的情形。使用它需要新的适配器身份和
验证；不追溯修改此前封存实验的失败记录。CPU 恢复测试不能证明模型超时率、RSI
质量或净收益已经改善。

### 完整事实的去重输入（可选）

`research.run(..., compact_history=True)` 仅改变发给决策模型的视图：完整源码和
全部候选观察保留在 `candidates`，`history` 中同一观察改为包含 candidate、run、
mode、status 的 `observation_ref`。失败条件、逐 case 数据、重复跳过状态、超时
记录和假设均保留；原始 trace 仍保存完整观察。缺少来源、出现歧义或事实不一致
会报错，不静默丢数据。默认关闭，前台 Manager 不使用此功能。

在已结束试验的 129 个真实请求上，独立解码逐项还原原数据通过；包含引用说明后，
control / 成功源码提示 / 直接复用的数据包字节减少约 22.7% / 24.7% / 31.4%。
本地视图构造中位数约 0.28ms、最大约 0.64ms。该检查没有调用模型或 GPU，不证明
推理速度、token 费用或质量改善。复现工具是 `audit/rsi_linked_history_replay.py`，
输入是原始 model-calls 目录；结果见 `audit/rsi_linked_history_replay_20261111.json`。
后续对照需对控制组和相应复用组一致应用，并单列完整表示的消融，才能区分记忆
作用和表示开销。新表示需要新的适配器/验证器身份，旧收据不能直接沿用。


### 2026-11-11：最新收益验证状态

最新固定动作预算试验完成 64 次独立 full，全部正确。复用两步相对无记忆两步：质量分数 +40.25%、trial 时间 −25.19%、tokens −31.28%；相对原始无记忆四步：质量分数 −4.59%、时间 −48.01%，未通过事前要求的双对照验收。质量分数变化不是 kernel 速度比。RMS 的一次独立最终性能退化被原门槛拒绝，不能用先前搜索成绩替代。预留 k/l 确认集关闭。后续 m/n 自适应预算对照也已结束：主候选耗时 −37.32%、tokens −49.64%，但质量分数 −5.85%，被原保护拒绝，o/p 确认关闭未运行；固定复用消融组未改选为胜者。详细数据和冻结协议见 [设计复核](rsi-design-review.md)。


### 可选的当前余量驱动动作预算

`research.run(..., max_actions=4, adaptive_artifact_budget=True)` 是默认关闭的研究选项。
历史实现仍占用第一个普通 full 动作；只有这次新测量完整、正确、可晋升，且所有已计时的 objective 和 guard 用例在两个批次都保留足够余量时，后续总动作数才缩短为两次。固定 baseline 和当前 comparator 的允许退化比例各取一半作为搜索预算启发式，**不修改原 evaluator 的任何门槛**。模型输入中的 `remaining_actions` 是缩短后的真实剩余数，不会隐含增加免费试验。

无历史实现、重复观察、运行环境缺失、未测全、计时无效、固定 baseline 不明或声明绝对延迟上限时，保留四步。比较对象为 best 时，必须使用另存的固定 baseline 数据，不能拿 best 充当 baseline。最多支持 256 个配置用例、128 个计时用例；超过范围保留四步。判断使用当前未截断的 evaluator 结果，不使用模型视图的前 16 行或历史成绩。

trace 保存有效动作上限、判定原因、源码和任务绑定、原始结果哈希及最大逐 case 退化值。正确性、严格选优、原累计时长、GPU 准入与独立最终 full 均保留。余量是一项工程假设，不保证最终性能，也不等于置信区间；已完成 CPU 边界和 Manager 集成测试；该策略已在预登记 GPU 对照中被拒绝，未启用。新增分配器已纳入验证器身份，旧策略收据不能冒用新版本。

### 有界历史成果快照（实验接口，默认关闭）

后台可信适配器可调用 `rsi.artifact_snapshot.materialize(task, artifacts, profile, directory, confirmation_run=...)`。它重验完整研究轨迹、绑定源码的 evaluator 收据及不同于搜索的最终 full，保存完整 episode 和最多 32 KiB 的内容寻址候选快照。最终不正确或不可晋升时不生成快照；生成快照不代表它对新任务有收益。

调用者显式批准返回的内容身份后，可把 `{path, identity, profile}` 作为 `research.run(..., artifact_snapshot=..., source_validator=...)` 的参数。它与直接 `artifact_proposal` 互斥。路径必须在候选工作区之外；前台只读取这一个有界普通文件，不扫描 episode 库。兼容检查使用本轮本来就必须执行的 initial full，核对实际环境、evaluator 身份、参考/输入生成器源码、精度与可变输入契约、操作者声明的 profile；无需另占一次 GPU 环境探测。

缺失、损坏、超限、非普通文件或不兼容的可选快照回退原搜索路径。成功读取只提供一个尚未在当前任务验证的候选，仍占用普通 full 动作，经过原严格选优和独立最终 full；运行中不热更新。trace 绑定读取原因、初始 full 和内容身份。此接口没有自动发布、收益准入、撤回或跨代选优能力：调用者不能仅凭 `materialize` 成功就将其当成已通过验证的生产策略，内容哈希也不代表可信发布者或 OS 隔离。

这项工程接口不属于已关闭的 `6e97cdd` 自适应对照，该对照未通过收益验收。不能用其结果证明新快照路径、自动发布或两代演化有效。


### 自动形成有证据谱系的成果候选（后台实验接口）

可信实验编排器在研究轮次与独立最终 full 完成后，可调用：

```python
from kernel_envolve.rsi.artifact_lineage import collect, audit, validation_snapshot

record_id = collect(
    library,
    task,
    research_artifacts,
    task_profile,
    confirmation_run=independent_final_run,
    parent=parent_record_id,  # 首个种子用 None；不能把观察记录当父版本
)
record = audit(library, record_id)
```

收集器从原始收据重建完整 episode，不接收 agent 声称的成绩或操作者手工选择的源码。若指定父版本，必须核实本轮实际读取了该父快照、绑定当前 initial full、将其中源码作为第一个普通动作进行评测。只有本轮 agent 提出的不同源码被选中，并通过独立最终 full，才形成 `candidate`。初始与最终的 evaluator 或运行环境不一致时仅保留观察。没有变化、快照回退、最终失败或视图超限也仅形成 `observation`，不伪造新一代。

完整失败事实留在 episode，`parent_observations` 指向父代码在本轮契约下的实际观察；失败只约束已测契约，不能推断该实现对所有未来输入无效。相同父版本、scope 和源码对应同一个 `proposal_key`，即使另一次测量产生新的 episode ID。采用台账必须按此键消耗一次验证机会；形成器只提供身份，下面的 `Registry` 实现持久化防重。

文件均按内容寻址，重复收集同一轮幂等；中断可能留下无引用数据，不会留下活动指针。`audit` 最多回查 32 层祖先的原始完整收据、源码和真实父快照使用，并绑定收集器所在的验证器代码身份。版本漂移、原始证据被改动或缺失时拒绝，即使物化文件自身哈希仍然匹配。后一次元数据读取不能取代已经验证的首个动作。

在**另行授权的后台实验**中，`validation_snapshot(library, record_id)` 重验该记录后返回 `{path, identity, profile}`，可供 `research.run` 的显式 `artifact_snapshot` 输入。它不是采用批准。不要把祖先审计放在生产前台；前台仍只有原有单文件读取，没有历史遍历、额外模型调用或 GPU 探测。收集触发由可信编排器负责，没有生产 Manager 自动挂钩；下面的实验 `Registry` 负责活动版本发布、撤回和收益准入。

CPU 工程测试形成种子与两个连续子代，并覆盖失败保留、父版本未实际使用、环境变化、旧收据被篡改、无变化与重复收集。它只验证链路，不是两代学习收益。已关闭的真实 GPU 归档也经新收集器离线验证；61 份种子视图、3 份最终不合格观察与 28 次错误候选全部保留，没有新 GPU/model 请求，没有将历史数据冒充为新代际试验。

### 成果目录的晋升、撤回与后台发布（实验接口，默认不接入生产）

`rsi.artifact_registry.Registry` 在上述形成器之上增加了持久化生命周期。
这是一项新工程机制，**不是已关闭 GPU 试验的获胜版本，也尚未证明跨代收益**。
`Registry.create(root, archive, seeds, max_attempts=..., max_seconds=..., max_tokens=...)`
显式建立一个实验目录：每个精确 reference/profile/precision/runtime scope 只有一个
经原始收据核实的种子。SQLite 中的一次事务更新整个目录与活动指针。

可信控制器在学习开始前调用 `declare_learning(entries)`，按固定顺序登记未来收据的
`{scope, receipt}`；这些路径必须尚不存在。完成后写入
`{provenance: {task, artifacts, task_profile, confirmation_run, origin_task?}}`，失败则
写入明确的 `error`。`collect_learning(phase)` 检查全部登记记录，并选择每个 scope
在登记顺序中**第一个**合格且改变了源码的子代。其余槽位保留父版本；其他成功、失败
和未变化的记录均保留，不通过跨任务成绩排名挑一个有利结果。没有变化就不形成新目录。

`reserve(candidate, suite)` 必须先于验证调用。它把父版本、模板、solver 文件身份、
验证器身份和预算封存；每个改变的 `proposal_key` 只获得一次验证机会。中断保留已占用
预算和已消耗机会。换一个解释、收据或学习 phase 不会给相同父版本/源码重新抽签。
验证配置不能弱化原先 0.02 的逐任务质量保护、5% 耗时保护、0.01 的质量收益或 10%
成本收益门槛。另有不能低于 10% 的实用门槛：每次重复相对每个对照，必须达到
质量分数相对提升至少 10%，或在质量不下降时总 trial 耗时减少至少 10%；同时仍须通过
原保护和 token 成本限制。分数变化不是 kernel 加速比。所有槽位需要 retention/fresh 两种配置与至少两次匹配重复；配置仅改名称
不能伪装成新模板。fresh 验证配置与形成数据隔离；retention 必须覆盖种子及父版本准入中全部已测配置，
可以重测已知旧配置来检查能力保留，不能将其称为盲测泛化。已登记验证配置不能再用于形成新代。
这只检查实际配置身份，不能证明语义层面的无限泛化或阻止不可信控制器泄露信息。

每个配置分别运行 `control`（无经验）、`seed`（冻结初始目录）、`parent` 和 `candidate`。
`artifact_validation.trial_order(suite)` 交替反转组内顺序；各组的模型、指令前缀、传输选项、
动作上限和原始候选相同。经验源码占普通动作，保留 initial full 与独立 final full。
`finish(attempt, evidence)` 在资源释放、原始工作区与全部 transport 收据收回后进行离线核验：

- 从原始 evaluator 收据重新计算正确性与质量，核对保护文件、初始源码、完整轨迹、独立
  final、实际读取和执行的各组成果；不能用 agent 自报分数或 `accepted` 字段替代。
- 把每个模型请求绑定到任务、重复编号、当时的候选/历史/剩余动作，并核对实际响应中
  的动作、源码、mode 与执行记录；模型和传输设置漂移、遗漏调用、重复响应、缺失 usage
  都不能通过。验证前发生的调用不能冒充本次验证。
- 对三个对照逐一执行原门槛，同时检查 token 成本与 retention/fresh 分区质量保护。
  所有重复都必须合格；并不把两个重复合并，以一个幸运结果抵消另一次退化。
- 保存全部结果和后台核验耗时。无法完整核实时至少保留预算上限的扣账；已知超支仍扣入
  台账，未知 token 从不记成零。任何验证失败都不会自动重试或启用新版本。

`suite` 的任务来自 `trials.freeze_template`，另带 `kind="generic"`、规范化 `profile`、
`scope`、`partition`；当前仅支持单文件 generic `candidate.py`。它还需要 `model`、
`reasoning`、`prompt_prefix_hash`、`request_options`（除 model/reasoning/input 外的完整
请求选项）、`command`/`adapter_files`、上述阈值、`max_token_regression`、`repeats`、
`max_actions`、`max_seconds`、`max_tokens`、`min_relative_quality_gain` 和 `min_wall_saving`。

`evidence` 包含 `attempt`、`trials`、完整 `calls` 文件路径列表和 `release_receipt` 路径。
每个 trial 含 `repeat/task/arm`、`provenance`、可信控制器计量的 `seconds/solver_seconds`；
provenance 的 `origin_task` 允许已释放 Linux 工作区到本地归档的单目录路径映射。
模型收据采用当前研究 transport 的 `trial_id/started_at/status/request/response` 格式。
释放收据包含终态 `phase`、空 `pods`、可用的 `live_read_status.pods` 和 `at`。
这些是可信控制器提供的计量与平台采集记录，不是模块自己查询平台或第三方计费的证明。

`adopt(candidate)` 会再次审计原始数据，并在事务中拒绝 stale parent、禁用状态、已使用
或已撤回的版本。`snapshot(scope)` **只由后台编排器在轮次之间调用**；发现证据漂移时
撤回受影响的活动目录及后代，恢复仍可审计的祖先，全部无效则返回无经验。
`set_enabled(False)` 禁止后续解析和晋升；`rollback(reason)` 显式撤回。已经开始的轮次
仍使用原来固定的输入，不声称能够撤销远端正在执行的代码。

前台仍只接收 `{path, identity, profile}` 并读取一份最多 32 KiB 的快照；没有新增生产
Manager 自动挂钩、数据库查询、历史扫描、反思模型调用或 GPU 探测。该数据库、归档、
模板和模型 transport 日志必须由可信控制器持有，不能让候选 solver 写入。内容哈希不能
代替 OS 隔离或认证。远端容器、实际 solver/source-guard 与源代码包仍由外层冻结协议
核对；本地 adapter 文件身份不是远端二进制执行证明。

此台账只覆盖晋升验证。学习、失败、初始经验获取、GPU 租约及全局模型费用仍由外层
campaign 账本与平台硬期限限制。实际 ROI 必须计入这些成本及后台成本；一项验证门槛
通过不等于整个 campaign 已回本，也不是统计显著性、所有算子上的收益或论文优越性证明。

初始化还会拒绝评测器版本不同的种子，在创建数据库前返回明确错误。历史原始记录可被
审计，不代表其快照能在当前评测器下激活。迁移必须固定历史候选源码，在当前评测器下
重新执行完整评测和独立确认，并建立新的种子记录；不能只改快照中的评测器哈希。
实际 GPU 型号、软件版本和运行设置仍由前台已有的首次完整评测核对，不能仅凭此静态
预检宣称运行环境兼容。实验控制器也应核对经验动作确实执行，避免把回退路径当作 RSI。
