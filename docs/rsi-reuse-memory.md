# 经验复用反馈：先让历史工作帮助下一轮

本轮设计吸收了已完成实验的正面观察：固定四步复用在小样本上改善了质量和时间，自适应缩短搜索则付出了质量代价。因此优先完善“重新验证历史实现 → 保留正常探索 → 记录适用条件”的路径。新模块默认不接入生产，没有新 GPU 或模型实验；它的测试验证工程行为，不证明新增学习收益。

## 三个层次的权限

| 层次 | 可以做什么 | 不能据此做什么 |
| --- | --- | --- |
| 原始事实与源码档案 | 保存成功、失败、无变化和独立最终检查 | 把旧成绩当作当前任务成绩 |
| `ReuseMemory` 显式登记 | 提供一个值得重新测量的源码，使用本轮普通动作 | 减少探索上限、自动晋升、改精度和 evaluator |
| `Registry` 独立准入 | 完成预先冻结的对照后，原子采用或回退 | 用形成候选或一次局部有效替代独立收益验证 |

登记可复用源码无需先证明全局 10% 收益，因为它只授予一次普通实验的输入资格，当前 incumbent 仍受到既有选优和最终 full 保护。原 Registry 的采用门槛、失败结果和已关闭实验保持不变。两者不是共享预算的不同入口：新模块不运行或授权实验，真实执行预算仍属于外层控制器；不能借此在验证集上无限选优。

## 一轮怎样运行

可信后台在分配 worker 之前调用 `prepare`，固定当前任务、初始源码、证据目录和动作上限，最多返回一个已核验的快照。运行时只拿 `{path, identity, profile}`，仍通过 `research.run` 既有的单文件读取和首次 full 检查实际环境。相同 initial 源码直接跳过，不浪费一个无意义动作。

四动作研究中，历史源码占第一步，其余最多三步继续交给原 Agent；没有增加免费试验，也不启用自适应两步策略。所谓“保留预算”是保留四个普通动作的总上限，不是保留四次模型调用再额外送一次复用评测。Agent 仍可以按原协议主动停止，独立最终 full 仍由外层控制器执行。

研究完成后，可信后台调用 `observe`，重新核对原始测量和独立 final，并复用现有 lineage 收集器。它不会调用模型或新增评测。真实的新源码可以返回 `formed_candidate`，但不会自动登记或采用；可信控制器须明确决定它是否进入后续研究；这可以是预先配置的后台策略，不要求每轮人工审批，但当前组件不会替调用者扩大来源集合。生产全局默认版本的更新继续走 Registry。

## 反馈如何改变下一次推荐

反馈以“参考代码及完整任务配置（不含名称/说明）+ 精确 scope + 被复用源码”定位。正确性或原性能约束失败与当前起点无关；未改善和未完成记录则额外绑定起点，避免把相对某个强实现的劣势泛化到更弱的起点。scope 包含 profile、精度契约、软件/硬件运行信息及 evaluator 身份。未观察到的其他形状不继承失败判断。

| 状态 | 证据含义 | 后续推荐（见上述范围区分） |
| --- | --- | --- |
| `helpful` | 复用源码当前完整评测合格且质量高于初始实现，最后选中的结果也独立合格 | 允许再用；不是反事实省时或生产收益证明 |
| `unhelpful_on_target` | 源码合格，但在这个起点上没有观察到严格质量改善 | 停止推荐 |
| `failed_on_target` | 复用源码有当前正确性/性能不合格观察 | 停止推荐；后续修复不能抵消旧源码的失败 |
| `not_activated` / `context_changed` | 实际未加载，或评测上下文变化 | 停止自动重试，保留为接入/上下文问题 |
| `inconclusive_final` | 搜索阶段有用，但最终选中结果未独立合格 | 停止自动重试，不归因为历史源码的普遍失败 |
| `pending` / `incomplete` | 尚在等待，或中断、缺失完整证据 | 不发重复推荐；不虚构质量结论 |

一次噪声导致的非改善也会暂停这个精确条件上的推荐，这是当前偏保守的成本控制选择，不是统计上证明该源码永远无用。更换真实输入配置或运行 scope 形成不同条件；更换起点仅能重新考察未改善/未完成记录，不能绕过相同配置上的已知正确性或性能约束失败。同一源码仅换收据、目录或任务展示名称不会清除失败。确需重新研究旧条件时应另行制定协议，不能后台自动重试直到出现好结果。

有多个可用条目时按明确登记的先后顺序选择，没有根据不同任务的历史分数排名，也没有尚未验证的 bandit/UCB 调度器。失败条目被跳过后可以使用后面的条目。`set_entry_enabled` 可撤回指定条目；再次开启不会抹去失败反馈。新候选不自动挤掉旧经验，这限制了自主探索能力，但保留了可解释性和选择控制。

## 后台开销与证据边界

每个库最多 64 个登记来源、4096 次使用记录，达到上限返回无经验，继续原研究路径；这是元数据容量上限，不是 GPU/token 额度。失效、关闭、同源码和已有负反馈在重建原始证据前尽早跳过；相关有效来源在后台审计，不给前台增加 SQLite 查询、归档遍历或反思调用。完整祖先审计仍可能耗时，应在 worker 分配前准备，不能把它塞进 GPU 持锁区或声称生产延迟已经验证。

同一目标的 pending 预留与同一证据目录的使用具有唯一约束，同一源码在相同配置上的 pending 使用也会阻止换起点后并发重试；事务内再次检查禁用和反馈，避免并发重复推荐。中断使用需显式 `abandon`，保留消耗过的机会；没有自动超时重试。已经发给运行中 worker 的快照无法被这层撤回，禁用影响后续准备。

任务、研究输出、复用库和原始档案必须是互不包含的目录；已预留研究输出目录即使没有文件、没有新候选或已经停用，也不能重新使用。原始档案必须只允许可信控制器写入；哈希不是身份认证或 OS 沙箱。当前实现仅支持本地路径绑定，不提供远端路径映射、自动任务提交、生产 Manager 挂钩或日志扫描守护进程。创建新库不能重置外层研究预算。部署仍要隔离 CPU、磁盘、API 配额和 GPU，完整 ROI 还要计入后台审计与经验采集成本。

复用控制器有独立实现身份，绑定现有验证器。改变控制器或验证器后，旧库不能继续准备/收集；需保留旧库并按冻结代码审计。本次就绪审查还修复了 Registry 事务和版本边界，因此验证器身份会变化，旧 lineage/Registry 收据不能直接冒用新版本。原始 evaluator 身份和收据不修改，也不通过重写哈希迁移。

版本不兼容属于软件迁移问题，不是经验质量失败。Registry 的 snapshot/rollback/finish/enable 现在在改变状态前拒绝不兼容代码；disabled 状态仍可读取为无经验。复用库提供 `ReuseMemory(root, recovery=True)`，只用于状态查看、停用和明确关闭未完成记录；prepare/register/observe/重新启用仍必须通过版本检查。未知数据库 schema 或配置完整性损坏会明确拒绝，不强行修库。

## 显式接入示例

以下是可信研究控制器的调用顺序，不是自动提交实验的命令。`archive` 中的记录须已经由 `artifact_lineage.collect` 生成；`scope` 来自被明确批准用于本次研究的记录。运行 Agent 不能写 memory、archive 或原始测量收据。

```python
from kernel_envolve.rsi import artifact_lineage
from kernel_envolve.rsi.research import run
from kernel_envolve.rsi.reuse_memory import ReuseMemory

memory = ReuseMemory.create(memory_dir, archive, max_entries=64, max_trials=4096)
try:
    memory.register(approved_record)
    scope = artifact_lineage.audit(archive, approved_record)["scope"]
    prepared = memory.prepare(scope, task, task_profile, artifacts, max_actions=4)
    # 此后才分配已获授权的 worker；未返回快照时仍运行原四动作流程。
    trace = run(
        packet,
        artifacts,
        decide,
        max_actions=4,
        artifact_snapshot=prepared["snapshot"],
        source_validator=trusted_source_guard,
        adaptive_artifact_budget=False,
    )
    # 外层控制器执行原有独立最终 full，完成资源释放后再汇总。
    if prepared["ticket"] is not None:
        feedback = memory.observe(prepared["ticket"], confirmation_run=final_run)
finally:
    memory.close()
```

恢复已有库使用 `ReuseMemory(memory_dir)`，不要每轮新建。遇到未完成研究时，控制器应保存原日志，再显式调用 `abandon(ticket, reason)`；它不停止任何 GPU 作业，也不证明资源已释放。反馈中的耗时来自 trace，tokens 仍为未知，不推算节省调用或宣称回本；计量继续使用外部完整调用账本。


## 命令入口与故障恢复

源码环境或安装包均可使用 `python -m kernel_envolve.rsi.reuse_cli`。它只操作后台元数据与已有证据，不运行研究、模型或 evaluator，也不会申请资源。所有正常输出是 JSON；错误写入 stderr 并返回退出码 2。路径应使用独立的持久目录。

```sh
python -m kernel_envolve.rsi.reuse_cli --state /work/reuse-state init --archive /work/lineage-archive
python -m kernel_envolve.rsi.reuse_cli --state /work/reuse-state register RECORD_ID
python -m kernel_envolve.rsi.reuse_cli --state /work/reuse-state status --pending-limit 20
python -m kernel_envolve.rsi.reuse_cli --state /work/reuse-state prepare --scope SCOPE_ID --task /work/operator/task --artifacts /work/research/round-1 --actions 4
python -m kernel_envolve.rsi.reuse_cli --state /work/reuse-state observe TICKET_ID --final /work/operator/task/.kernel_envolve/runs/FINAL_ID
python -m kernel_envolve.rsi.reuse_cli --state /work/reuse-state disable
```

`prepare` 可通过 `--profile profile.json` 指定 profile；其输出中的 snapshot 交给原研究执行器，ticket 留在可信控制器。不要将整个数据库挂入 solver。没有候选时 snapshot/ticket 都为 null，继续原研究即可；配置和证据冲突会明确报错，不能把接入错误统计为 RSI 处理成功。

`status` 使用只读数据库连接，不审计原始源码，也不启动工作。它显示 code_compatible、启用状态、每类反馈数量和有上限的 pending 列表；pending 不证明 worker 仍在运行。版本不兼容时仍可 status/disable/entry-disable/abandon，不能 enable/register/prepare/observe。不存在的状态目录不会创建空数据库。

确认外部 worker 已终止并保存日志后，可显式关闭中断记录：

```sh
python -m kernel_envolve.rsi.reuse_cli --state /work/reuse-state abandon TICKET_ID --reason 'Worker stopped; independent final unavailable'
```

同一关闭请求可重放；未知 ticket、已消费 ticket 的不同结论都被拒绝，不留下额外反馈文件。不删除 pending、不改成质量成功，也不重新获得机会。`entry-disable ENTRY_ID` 仅撤回一个来源，`entry-enable ENTRY_ID` 和 `enable` 只在版本一致时允许；已有失败反馈始终保留。
