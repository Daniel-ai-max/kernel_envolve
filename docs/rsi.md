# 有界的跨任务自进化

RSI 的目标是减少重复错误、改进假设选择和实验安排，在相同预算下更快得到通过完整验证的算子。它是可选的后台学习流程；现有 evaluator、kernel 晋升规则和生产 Manager 继续负责原有职责。

实现起点为 Manager 的 `a3210b0`；其内容已由独立操作通过 PR #1 合入 main（`8cb7665`），两个基线源码树相同。`feature/rsi-evolution` 已整理为 main 之上的 RSI 独立增量，本次没有向 main 提交或合并。试用本分支必须设置 `KERNEL_ENVOLVE_AUTO_UPDATE=0`，避免现有自动更新入口切回尚无 RSI 的远端 main。

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

## 分工与证据

执行 Agent 沿用原有 `notes.md` 和结构化 report，记录假设、结果解释与下一步。不要要求每个工具调用都生成长反思，也不需要第二套平行工作日志。

RSI 收集两类数据并标注来源：

- evaluator 的已索引结果：运行身份、契约、正确性、结果 hash、性能判定、失败原因；涵盖成功与失败。
- report 与生命周期事件：Agent 的解释、超时和中断等。它们不是性能事实。

经验必须引用存在且属于当前 scope 的证据。提案、采用和定期审计都会核对引用；run 引用会重新调用现有证据校验器。事件摘要有长度限制，仍保存原事件哈希以便核验。没有 LLM 凭空填写 speedup 的路径。

新规则包含 `when`、`action`、`unless`，按任务 ID 显式确定推广范围。未提及的旧规则保留；删除必须给理由。候选只有通过验证才影响生产提示。相同父版本下，相同建议内容即使更换说明、证据或条目 ID，也不会得到新的验证机会。

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
      "tasks": [
        {"path": "/work/validation/reduce-a", "kind": "generic"},
        {"path": "/work/validation/reduce-b", "kind": "generic"}
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

两个适配器都是操作者配置的 argv，不通过 shell 拼接，也不会从候选内容生成可执行命令。脚本路径建议使用绝对路径。子进程固定使用当前 kernel_envolve 安装，禁用自动跟随 main，不继承生产 Manager 的任务/轮次环境变量。

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
    "evidence": ["输入 evidence 中的完整 ID"]
  }],
  "remove": []
}
```

solver 从 stdin 接收冻结任务副本路径、允许修改的文件、策略文本、固定模型标识、repeat seed 和时间额度。它应使用相同模型配置完成优化，退出前保存候选。不要调用 promote，不要离开允许目录或启动脱离父进程的任务。seed 是请求的重复身份；对不支持确定性种子的供应商不宣称可精确复现。

solver 的 stdout 不是成绩来源。退出后控制器检查文件范围，启动独立 full evaluator，复核原始结果和收据。通用与 native 流程都支持；原生流程保留原项目数值/梯度/集成 gate。

外部适配器必须在前台运行。控制器限制运行时间和日志大小，终止其进程组，并在 GPU 使用后检查清理。生产宿主原本是同用户协作模型；这不是 OS 安全沙箱。若适配器驱动可任意执行命令的非受信任 Agent，应由部署方提供容器/独立用户、只挂载 trial 工作区、不暴露状态目录及发布凭据的隔离。文件哈希检测改动，不能抵抗同权限进程同时篡改程序和证据。

## 验证与采用规则

每个 suite 至少两项任务、两轮重复。每一对从相同冻结起点开始，交替运行 control/candidate 的顺序；control 是当前稳定策略，candidate 是局部更新后的完整策略。模板和 solver 模型配置相同。验证任务的源码、路径和逐次结果不提供给学习器的输入窗口，学习器只见生产轨迹。

当前保守规则：

1. candidate 每次 full 都正确完成；失败不能由收益或成本补偿。
2. 每个任务对的质量不低于 control 超过声明的退化额度。
3. 每轮重复至少一项任务出现质量收益，或在不损失已有正收益时减少 solver wall time。

质量是 full evaluator 允许晋升的结果相对固定 baseline 的延迟下降；不合格候选不计正收益。未产出可晋升结果时用保留 baseline 的零收益计算，正确性失败另记失败。策略验证不会实际晋升这些实验 kernel。

这是有限 suite 上的保守工程准入，不是统计显著性保证，也不是广泛泛化证明。成本只测 solver wall time，未测 token、实际 GPU 活跃时间和端到端训练吞吐。验证集反复使用仍可能产生适应性过拟合，需由操作者周期性更换独立审计任务；`min_*` 不是宣称通用最优阈值。

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

本次工程验证记录见 [RSI 验证收据](../audit/rsi_20261110.json)：全仓 236 passed / 8 GPU-only skipped，33 项 RSI 专项测试，独立 wheel 安装闭环，以及每轮建议读取的本机微基准。该记录不包含真实模型学习收益或真实 GPU 策略实验。
