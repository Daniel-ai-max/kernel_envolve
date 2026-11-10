# 与 DeepSeek Super-agent 的对比

本实现借鉴 Super-agent 的长期任务、持久化执行意图和恢复机制，将它们收敛到固定算子
持续优化。对齐的是这些工程约束；当前 CPU 测试不能证明两者在真实生产负载下的
稳定性相同。

对照版本：[`deepseek-ai/dsh-superagent-prototype@8c32a9d`](https://github.com/deepseek-ai/dsh-superagent-prototype/tree/8c32a9dc5d72c033816c6a8110b37f40cf94fba4)。
其 [README](https://github.com/deepseek-ai/dsh-superagent-prototype/blob/8c32a9dc5d72c033816c6a8110b37f40cf94fba4/README.md)
明确区分公开研究原型、内部版本与文章中的未来设计。因此这里没有把文章里的完整
Watchdog、跨任务智能调度或自主组织能力当成已实现功能。

## 能力映射

| 关注点 | Super-agent 公开原型 | 本实现 |
|---|---|---|
| 长期任务与执行分离 | Ticket / Assignment / Executor Session | task / agent round / 可配置 argv 适配器；每轮结束保存 checkpoint，算子任务继续 |
| 提交后执行 | Engine 在事务中记录 Effects，提交后由适配器执行 | SQLite 事务预留名额并保存 job，提交后启动独立主管；不靠内存任务列表 |
| 持久化 | SQLite WAL / FULL、领域记录与待执行效果 | SQLite WAL / FULL；任务、进程身份、队列、累计用量、报告、事件与证据索引 |
| 重启恢复 | 保留 Session 等状态，记录中断的 Assignment，并带恢复说明继续 | 协调器重启不终止存活主管；身份匹配才接管；失联时先清理子进程、核对锁，必要时隔离资源 |
| 执行并发 | Executor 槽位与队列 | 每资源 1–2 个 Agent 会话；GPU 作业独立 FIFO；同一用户下按物理 UUID 加宿主锁 |
| 失败与暂停 | 失败持久化、有限重试、退避、人工继续 | 累计 wall/GPU 额度、轮次超时、失败退避、连续 3 次失败停止；暂停立即持久化取消意图 |
| 进展呈现 | Ticket 状态、需要人处理的问题、系统问题 | 总览、任务详情、资源队列；显示真实阶段、证据问题与资源阻塞，不伪造完成百分比 |
| 结果可信度 | Executor Report 与外部系统状态 | 研究报告与 evaluator 事实分开；只有完整验证且通过晋升链的 best 展示收益 |
| 基线溯源 | 非算子专项能力 | 固定仓库/版本/入口、实现提出日期与来源、冻结日期、硬件、评测契约 |
| 外部事件与授权 | Sources、Threads、Mandates、输入投递和消息 Outbox | 固定任务和明确资源配置；不引入消息连接器、提醒、隐式任务发现与授权推断 |
| 模型运行时 | DeepSeek Harness 接口 | 不新增推理循环；适配器消费 stdin 恢复提示，写结构化报告，可自行复用宿主 Session |

源代码依据：

- [领域模型](https://github.com/deepseek-ai/dsh-superagent-prototype/blob/8c32a9dc5d72c033816c6a8110b37f40cf94fba4/docs/001-domain.md)：Ticket、Assignment、Session 的边界。
- [执行与恢复](https://github.com/deepseek-ai/dsh-superagent-prototype/blob/8c32a9dc5d72c033816c6a8110b37f40cf94fba4/docs/003-execution.md)：失败、重试、重启与 Stored work。
- [架构](https://github.com/deepseek-ai/dsh-superagent-prototype/blob/8c32a9dc5d72c033816c6a8110b37f40cf94fba4/docs/007-architecture.md)：事务内入队、提交后执行及组件边界。
- [Store](https://github.com/deepseek-ai/dsh-superagent-prototype/blob/8c32a9dc5d72c033816c6a8110b37f40cf94fba4/src/app/store.ts)、[Effects](https://github.com/deepseek-ai/dsh-superagent-prototype/blob/8c32a9dc5d72c033816c6a8110b37f40cf94fba4/src/app/effects.ts)、[Executor](https://github.com/deepseek-ai/dsh-superagent-prototype/blob/8c32a9dc5d72c033816c6a8110b37f40cf94fba4/src/executor/dsh.ts)：对应实现。

## 为算子优化增加的约束

同一 GPU 上可以有两个 Agent 同时分析、改代码，但正确性检查、编译、benchmark 和
profiler 都必须经过 `manage gpu-run`。GPU 作业结束、拥有者进程身份与设备锁是释放
依据；心跳过期只代表需要检查。独立进程组的 evaluator 子进程也登记身份并继承锁，
覆盖尚未创建 CUDA context 的导入/编译阶段。无法确认清理完成时停止资源复用。

原始 baseline 固定。当前候选、已验证 best 和 Agent 报告的已集成版本分别存储；每次
晋升记录前一个 best、结果 hash 和晋升时间，面板只认可从当前 best 可达且经过校验的
链。收据删除、源码篡改、契约变化、结果目录丢失都会使对应证据失效。旧记录没有
晋升时间时保持未知；历史 GPU 用量只是索引时的观测值，不冒充晋升当时的精确消耗。

界面只保留可操作信息：总览展示 baseline 版本、提出日期、已验证加速、最近晋升、
覆盖与资源；详情展示每批每个 case 的延迟及历史。源码身份、笔记和事件默认折叠。
没有常驻教程、免责声明或消息提醒。收益是单算子固定契约下的结果，不推导框架收益。

实现集中在 [manager](../src/kernel_envolve/manager/)：`store` 负责事务、`service`
负责协调、`worker`/`gpu`/`process` 负责生命周期和资源、`evidence` 负责结果校验、
`views`/`web` 负责查询和本机面板。复用标准库 SQLite/HTTP，不增加运行时依赖。

## 验收证据（2026-11-10）

| 项目 | 结果 |
|---|---|
| 全仓测试 | `180 passed, 8 skipped in 66.14s`；8 项为 opt-in CUDA 测试 |
| 管理层测试 | 32 项；包括实际子进程、两个 GPU wrapper 在 CPU 资源上的互斥、主管 SIGKILL、独立进程组清理、暂停/快速恢复、预算、checkpoint、历史链、native autograd/gate 收据、收据删除及恢复 |
| 静态检查 | Ruff lint/format、`git diff --check` 通过 |
| 包构建 | wheel 和 sdist 构建通过，三份本地 UI 静态资源包含在 wheel 中 |
| 独立安装 | 全新 Python 3.11 venv 安装 wheel；确认从 site-packages 导入，已安装 CLI help/status、HTTP 页面/JS/CSS/状态及未授权控制拒绝通过 |
| 浏览器 | 总览、详情、搜索及空结果、暂停/恢复、资源会话数 2→1→2；窄屏单列/表格滚动；观测无控制台错误 |
| 演示 | 3 个显式标识的 CPU demo，使用真实 evaluator 收据与合成等待；不是实际算子性能数据 |

具体测试见 [test_manager.py](../tests/test_manager.py) 和
[test_manager_native.py](../tests/test_manager_native.py)。运行和配置见
[使用文档](optimization-manager.md)。

## 实际使用边界

- 本次未接入真实模型供应商，也没有 NVIDIA GPU；尚需在目标 Linux GPU 主机上验证
  UUID 探测、实际 CUDA 子进程清理、共享卡计时和真实 Agent 适配器。CPU 进程测试
  不能替代这一步。
- 单主机、同一操作系统用户、受信任的工作流。Agent 与用户同权，不提供 OS 沙箱；
  外部程序可以绕开 wrapper，空闲探测与不受管理作业之间存在竞争窗口。
- 一个任务固定一个硬件/评测契约。改变原始 baseline、精度或输入覆盖应建立新任务。
  多 GPU/分布式验证需要另行调度；跨实例只保证设备互斥，不保证全局 FIFO。
- GPU 用量计作业占卡时间，包括编译和正确性检查；不是 CUDA 活跃时间。崩溃未知
  区间按经过时间保守计账。没有无限自动重试或丢弃预算来恢复任务的路径。

这些边界保留在文档和实际错误状态中，不作为面板的常驻说明。
