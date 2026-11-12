# 持续优化管理

管理层将一个算子的长期任务与单轮 Agent 执行分开，复用现有 evaluator 和 best。
一个 POSIX 主机运行一个协调器，每个资源由一张或一组 GPU 组成，支持一到两个 Agent 会话；同一资源的 GPU 作业排队互斥。
Agent 自主选择优化路线，协调器只管理生命周期、资源和证据。没有消息推送或模型路由。

## 启动

在实际 GPU 主机的项目环境内安装本包。先准备每个算子的独立 worktree、task
目录和原始 baseline，再编写部署 JSON。管理状态目录必须位于所有任务工作区之外。
GPU 用 `nvidia-smi --query-gpu=uuid --format=csv,noheader` 得到的物理 UUID 标识，
不接受 `cuda:0` 或 MIG 切片作为调度身份。GPU 作业只暴露所分配的设备，主任务使用 `cuda:0`。

```sh
kernel-envolve manage contract --task /work/probs-scale/task --kind native
kernel-envolve manage --state /work/manager-state init deployment.json
kernel-envolve manage --state /work/manager-state run
# 另一个终端；远程主机可用 SSH -L 8787:127.0.0.1:8787 转发
kernel-envolve manage --state /work/manager-state serve --port 8787
```

启动后确认 Agent 收到恢复提示并提交真实工作，GPU 命令能进入队列且结果可读。
面板可选，不作为实验启动或继续的前置条件；启用时再检查显示与实际作业一致。远程节点的 `127.0.0.1:8787` 不能直接作为本机入口，先建立 SSH 转发：
`ssh -N -L 8787:127.0.0.1:8787 <GPU主机>`，再打开本机的 `http://127.0.0.1:8787`。
平台没有 SSH 时使用其认证端口转发，保持面板只监听 loopback。

部署配置（替换所有示例路径、命令和标识后使用）：

```json
{
  "version": 1,
  "resources": [{"id": "gpu-0", "device": "GPU-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee", "sessions": 2}],
  "tasks": [{
    "id": "probs-scale-b300",
    "name": "probs_scale",
    "workspace": "/work/probs-scale",
    "task_dir": "/work/probs-scale/task",
    "kind": "native",
    "resource": "gpu-0",
    "command": ["/opt/agent-adapter", "--workspace", "/work/probs-scale"],
    "round_timeout_seconds": 1800,
    "wall_budget_seconds": 28800,
    "gpu_budget_seconds": 14400,
    "cooldown_seconds": 10,
    "baseline": {
      "repository": "https://github.com/ApodexAI/apodex-core",
      "revision": "<固定源码 commit>",
      "entrypoint": "<实际算子入口>",
      "introduced_on": null,
      "introduction_url": null,
      "frozen_on": "2026-11-10",
      "hardware": "B300",
      "contract": "<manage contract 输出>"
    }
  }]
}
```

`introduced_on` 是基线实现被引入的日期，需要对应 PR/commit 的 HTTPS 证据链接。
未知时保留 null。`frozen_on` 是本任务选择该基线的日期，不能混用仓库最新提交日期。
`revision`、入口和硬件是操作人填写的溯源信息；`contract` 在 init 时与 evaluator
计算的契约核对，导入的每条结果也必须匹配。新增硬件/覆盖/精度或原始基线时建立新任务。
重跑相同 init 是幂等的；它不会重置暂停、用量、checkpoint 或 best。

wall 额度单位为墙钟秒，GPU 额度单位为卡秒（设备数 × 占用秒数）；null 表示不设上限。
两种额度都是累计值，不是每轮重置。session wall 时间包含该轮的等待；GPU 用量包含
导入、编译、正确性和测量，按整组独占时间记账，即使部分卡未计算。主管失联期间
采用保守的经过时间记账。单卡和 CPU 测试资源保持原来的秒数口径。

### 同主机多卡资源

需要 CP/TP 等分布式验收时，将资源的 `device` 改为 `devices`，按本地 rank 顺序列出
该主机全部目标物理 UUID。两个字段只能选一个，组内及同一部署的不同资源之间不得重叠。
例如：

```json
{"id": "gpu-group", "devices": ["GPU-aaaaaaaa-bbbb-cccc-dddd-000000000001", "GPU-aaaaaaaa-bbbb-cccc-dddd-000000000002"], "sessions": 1}
```

示例仅展示双卡配置格式；CP8 应列出八个真实 UUID。`CUDA_VISIBLE_DEVICES` 保留声明顺序，
分布式命令自行启动对应 rank，例如在 `manage gpu-run` 内执行 `torchrun --nproc-per-node 8`。
管理器不负责跨主机调度。作业入队时固定设备列表及数量；子 GPU 作业继承父轮次快照，
之后任务转移资源不会改变历史用量。

每张卡仍使用同一个逐 UUID 锁，与旧单卡作业互斥。锁按 UUID 排序获取，任一卡不可用就
立即释放已获得的锁并等待；全部锁获得后才逐卡核对身份和空闲状态。清理和恢复同样
检查每张卡，任一卡无法确认释放时隔离整个资源。

本版本将 SQLite 升级至 schema 2，旧作业按单卡迁移，不改历史累计值。升级会先获取
协调器锁，并在同一事务内确认没有排队/运行作业和存活的登记后代；不满足时拒绝迁移，
原 schema 与账本不变。部署前停止旧协调器，等待或正常终止现有轮次，备份状态后更新
协调器、主管及面板。旧版本拒绝打开
升级后的可写状态；需要回退时必须先结束新作业、保存新增证据，再恢复升级前备份，
不能直接降级运行。旧 schema 1 备份仍可由新版面板只读查看。

## Agent 适配接口

`command` 是 argv 数组，不经 shell 拼接。使用你现有 coding-agent 的非交互入口，或
一个转接它的可执行脚本。管理器不附带模型、API key 或新的推理循环，也不直接调用
桌面应用中的 chat。适配器必须在当前进程生命周期内运行，不得 daemonize。

若适配器通过文件或消息等待外部 coding agent，必须同时启动负责提交请求的宿主。
只启动等待进程不会自动产生优化工作；恢复时先确认宿主在线、已有请求及最后结果，
再续接下一项实验，不反复开启没有工作内容的轮次。

每轮 stdin 是恢复提示：任务范围、固定基线、上一轮结构化报告、最近事件、预算和
工作目录。Agent 仍需要读取项目指令、kernel-envolve skill、任务 notes.md 和已有 run。
如果宿主支持原生 session resume，适配器可以在自己的任务工作区保存宿主 session ID
并续接；否则每轮使用新 session，通过同一任务 checkpoint 恢复。两者都不改变任务身份。

环境变量：

| 变量 | 用途 |
|---|---|
| `KERNEL_ENVOLVE_STATE` | 管理状态目录 |
| `KERNEL_ENVOLVE_TASK` | 长期任务 ID |
| `KERNEL_ENVOLVE_ROUND` | 当前执行轮次 ID；用于拒绝过期报告 |
| `KERNEL_ENVOLVE_REPORT` | 本轮结束时的 JSON 报告文件 |

```sh
# GPU 上运行的所有命令均经该入口，包括局部正确性检查和 profiler。
kernel-envolve manage gpu-run --expected-seconds 300 -- kernel-envolve native-evaluate --task ./task --mode quick
kernel-envolve manage gpu-run --expected-seconds 1200 -- kernel-envolve native-evaluate --task ./task --mode full
kernel-envolve native-promote ./task/.kernel_native/runs/<run>
# 交付时主动索引当前算子，再 sync-pr；局部候选不需要这一步。
kernel-envolve manage refresh --task probs-scale-b300
kernel-envolve manage report --file progress.json
```

`--expected-seconds` 始终声明本次命令的预计墙钟时长，建议使用命令超时作为保守上界。
例如八卡作业声明 300 秒，GPU 额度须至少剩余 2400 卡秒；wall 额度仍需 300 秒。
管理器在入队前和获得设备锁后分别检查父轮次、累计 wall/GPU 剩余额度，排队时间也会扣除。
不足时返回退出码 `2` 和 `error_type: gpu_duration_admission`，不启动子命令、不自动重试或延长预算。
Agent 应保存检查点并重新规划；剩余累计预算不会因换轮恢复。额度按实际经过时间记账，
声明时长不预扣额度，也不替代命令自身超时。此参数可选，旧调用未声明时保留原行为，
不会提前判断作业是否来得及完成。

GPU wrapper 会转发命令输出的最后 64 KiB，完整日志在管理状态目录的 `jobs/<id>/`。
Agent 主进程默认看不到 CUDA 设备；GPU wrapper 给子进程设置固定顺序的目标 UUID。不能再嵌套
gpu-run。分布式验证必须使用已声明的资源组，不得自行扩展到组外 GPU。

GPU 命令可直接执行原生测试、benchmark 或 profiler；有依据的候选再进入 evaluator。
若适配脚本需要先检查条件，再转交 CLI，使用
`os.execvpe(argv[0], argv, os.environ)`；另起子进程时须传
`pass_fds=tuple(json.loads(os.environ["KERNEL_ENVOLVE_LOCK_FDS"]))` 并等待退出。
`KERNEL_ENVOLVE_LOCK_FDS` 包含全部锁描述符；旧的 `KERNEL_ENVOLVE_LOCK_FD` 保留第一项，
仅兼容旧单卡适配器。默认 `subprocess.run` 会关闭这些描述符。CLI 会在复制源码前拒绝
缺失或失效的描述符，并给出上述修复方式；创建 worker 时也再次检查。
检查不会重新获取锁，也不能通过删除 manager 环境变量来恢复执行。

报告示例：

```json
{
  "phase": "实现反向融合",
  "hypothesis": "复用前向统计量，减少一次读写",
  "outcome": "小尺寸正确，长序列仍需验证",
  "next_action": "检查 padded case 的反向梯度，然后执行 quick",
  "pr_urls": [],
  "session_url": "",
  "integrated_revision": ""
}
```

报告只描述研究活动。speedup、覆盖、best 都来自 evaluator 结果和晋升链。收到失效
轮次的报告会拒绝；无效报告将本轮记为失败。轮次成功退出后继续调度，而不是结束任务。
执行失败后保留任务，并按 10、20、40、80、160、300 秒退避，后续最多等待 300 秒。
下一轮会收到最近失败作业的退出码、日志路径和错误摘要，由 Agent 诊断原因、修复后
继续实验。管理器只负责调度配置的命令；外部 bridge 仍需宿主实际接收请求并运行
Agent，新的进程或心跳不代表模型已经工作。

清除连续失败计数需要成功结束的轮次、新的结论与下一步，以及实现内容变化或新的
实际 GPU 作业结果。仅修改报告措辞、重复相同命令和结果不会清除计数。数值检查失败
可以成为诊断依据，但不能作为放宽 oracle、容差或验收条件的理由；恢复记录也不等于
evaluator 验收通过。所有重试沿用原累计预算，预算耗尽、暂停及 GPU 资源隔离仍有效。

## 操作与恢复

```sh
kernel-envolve manage --state /work/manager-state status
kernel-envolve manage --state /work/manager-state pause probs-scale-b300
kernel-envolve manage --state /work/manager-state configure probs-scale-b300 \
  --changes '{"gpu_budget_seconds":21600,"resource":"gpu-0"}'
kernel-envolve manage --state /work/manager-state resume probs-scale-b300
kernel-envolve manage --state /work/manager-state resource gpu-0 --sessions 1
kernel-envolve manage --state /work/manager-state refresh --force
```

暂停会在下一次主管检查时终止当前轮次和 GPU 作业；必须等待作业退出后才能改额度或
资源绑定。notes.md 只包含最后完成的 checkpoint，运行中的本地编辑保留供恢复核对。
恢复不会清空累计用量。降低会话数只影响新调度，不强杀现有会话。
升级时，仅旧版以 `Three consecutive failed rounds; inspect and resume` 原因自动阻塞的
任务会转入诊断重试，并保留累计用量和失败次数；预算已耗尽则结束。其他阻塞、暂停
及已被新任务替代的任务不会自动恢复。

评测契约升级时，先以独立工作区注册替代任务，再将旧任务转入历史：

```sh
kernel-envolve manage --state /work/manager-state supersede probs-scale-b300 --by probs-scale-b300-v2
```

旧任务必须已结束全部作业。替代任务保留同一资源、算子入口和固定源码基线，允许更新
评测契约；其 wall/GPU 总额度分别不得超过旧任务未用额度，已消耗的额度不能用来抵扣
这项检查。命令原子记录替代关系和双方账本快照，不搬移用量、不修改额度、不启动
替代任务，也不解除资源隔离。已有 v2 迁移可直接补记关系，重复执行同一关系无副作用。
旧任务继续保留日志、评测和实际用量；暂停旧任务是空操作，恢复及修改旧任务均被拒绝。
替代链的累计用量是各版本实际用量之和；外部资源截止时间仍沿用原配置。

协调器退出时，独立的轮次主管继续执行并检查预算；重启协调器会核对进程身份和锁，
接管观察，而不重新运行活着的轮次。新主机不能仅复制 SQLite 后自动重放作业。
部署时可用 systemd/现有作业系统守护 `manage run`，配置失败重启与启动退避。
面板服务独立运行，关闭页面不影响任务，页面会显示协调器是否在线。

面板区分进程心跳与研究活动。会话在线但连续 5 分钟没有变化的研究报告或 GPU 作业，
将无进展信号留给 Agent 宿主核对和接续，不在用户界面展示“待核实”或无进展倒计时。
重复相同报告、主管心跳和自动开启下一轮不会清除内部信号。真实诊断或实验完成后
通过 `manage report` 更新结论和下一步；报告不能替代性能证据。
任务详情优先读取 `task_dir/notes.md`；未提供时读取 `workspace/notes.md`，
兼容评测配置与研究工作区分开的目录结构。

GPU 锁在本操作系统用户的 `~/.cache/kernel-envolve/locks` 下，按物理 UUID 命名，
能防止该用户的多个管理状态目录同时使用同一卡。FIFO 顺序属于单个管理实例；不
承诺跨实例公平性。多操作系统用户应由外部资源分配器保证独占。

主管崩溃后先停止登记的子进程，再检查锁和 NVIDIA 进程。继承设备锁的 evaluator
子进程即使尚未创建 CUDA context，也会阻止重新分配。无法确认 GPU 清理完成时将
资源隔离；检查实际占用后使用 `manage resource gpu-0 --clear`，该命令仍需通过锁和
设备空闲检查，不能强制绕过。时间戳过期本身不授权释放设备。

这里不提供 OS 安全沙箱。Agent 与本机用户同权，能够绕过 wrapper；外部不协作程序
也可能在空闲检查后启动。可靠计时要求独占的 GPU 主机/账户和受管理的工作流。

## 证据与面板

管理器定期增量索引 `.kernel_envolve/` 或 `.kernel_native/`，验证收据、源码和契约，
重新计算判定。只读索引失败不阻塞 GPU 调度，错误保留在对应任务。`refresh --force`
重新校验已导入记录。历史源码/收据被修改、结果目录丢失、best 指针损坏都显式标记。

晋升在 run 内保存 `promotion.json`，包含前一 best 指针和时间。历史只从当前 best
的可达链读取，写了收据但尚未切换 best 的中断晋升不会进入历史。旧版本没有晋升
时间时显示未知，不伪造日期。统计按固定 baseline 的两批延迟下降中位数换算加速比；
详情保留逐批、逐 case 绝对耗时，不把单算子加速平均成框架收益。

SQLite WAL/FULL 保存任务、轮次、GPU 请求、报告、事件与索引；写事务提交后才启动
进程。每个任务工作区有宿主锁，数据库另有活跃轮次唯一约束。终端日志保存在
`jobs/<id>/`，输入提示、报告和 stdout/stderr 都可追溯。原始实验文件不搬迁、不删除。

## 本地演示

```sh
python examples/manager/demo.py /tmp/kernel-manager-demo
# 执行脚本打印的 run 和 serve 命令
```

演示使用 CPU、合成等待和真正的 evaluator 收据，用于检查界面和进程调度。它不
代表任何 GPU 算子的真实收益。所有任务名称和硬件字段都带 demo/演示标识。

## 自动更新契约

本节描述面板数据刷新。工具代码跟随 main 的机制见[自动更新指南](auto-update.md)：每轮 Agent 启动前检查，轮次内固定 Skill、CLI 和评测器版本。

更新分为采集、发布、读取和显示四步。协调器在上次扫描结束至少 5 秒后尝试下一次扫描；
同一部署只允许一个扫描执行。源码/收据的读取、指纹检查和判定计算在写事务之外
完成，每个算子的 runs、best、晋升历史、错误和观测时间再一次性提交。新实验先落盘、
校验，再进入面板；面板展示最近一次已发布的证据，不直接拼接正在变化的结果文件。
扫描不重叠执行。存在排队或运行中的 GPU 作业时不开始历史文件扫描；扫描期间
每隔至少 0.25 秒在文件校验检查点查询一次队列，有新 GPU 作业则退出，保留上一份
完整发布的结果和原观测时间。单次文件读取不可中断，0.25 秒不是硬实时停止保证。
持续繁忙时证据索引可能延后，作业、研究报告和用量仍独立更新；实验不等面板刷新。
正式 evaluate/confirm/promote/sync-pr 的验证与此调度策略无关，仍完整执行。
晋升后 Agent 执行 `manage refresh --task <id>` 主动核对该算子的证据，再绑定 PR，
不依赖后台空闲窗口；此入口不会扫描其他算子的历史。不带 `--task` 时仍核对全部任务。

面板采用短轮询，适合当前单主机、少量观察者的工作方式：

| 数据/场景 | 更新规则 |
|---|---|
| 运行阶段、队列、用量 | 读 supervisor/协调器已经提交的状态；一次成功响应结束约 3 秒后再请求 |
| 总览与当前详情 | 同一次 HTTP 请求、同一个 SQLite WAL 只读快照；总览省略逐 case/source 大字段，只给选中算子传详情；历史只传曲线/实验摘要，逐 case 数据仅传当前 best |
| best、逐 case 延迟、验证范围 | 来自同一份已发布证据；后台索引独立于打开面板的数量运行 |
| 切换算子、手动刷新 | 立即读取新快照，取消旧读取并拒绝它迟到的结果 |
| 隐藏标签页 | 停止请求；再次可见时立即刷新 |
| 网络或服务故障 | 单次请求 8 秒超时；保留旧数据，以 6/12/24/30 秒退避重连；成功后恢复 3 秒间隔 |
| 编辑和阅读 | 复用 DOM 节点，仅更新变化内容；保留筛选、焦点、表格滚动和展开项 |

“同步时间”是面板读到状态快照的时间；“评测扫描时间”是该算子后台检查的观测时间，
放在验证范围和 best 标题的 tooltip 中。证据观测年龄与扫描状态分别显示；
协调器离线、浏览器断线分别显示，收到 HTTP 响应本身不代表评测或协调器仍在推进。
正常时不额外增加说明区。

只读请求不执行 schema 初始化或持久化写入；WAL 读快照允许调度器同时提交变化。
后台每轮仍需遍历历史文件的元数据以发现篡改，只有变化的内容重新校验；大量历史
文件时扫描可能变慢，延迟状态会如实显示。手动 `manage refresh --force` 在已有扫描
执行时明确拒绝并要求稍后重试，不排入无界重扫队列。

元数据遍历复用 `scandir` 的单次文件状态读取，跳过 Python 缓存子树；文件变更时间、
类型及身份也参与失效判断，等长改写后恢复 mtime 仍会触发重新校验。符号链接直接
拒绝，变化后的源码、收据和晋升链仍执行完整哈希验证。

自动更新回归覆盖见 `tests/test_manager.py` 与 `tests/manager_sync.mjs`：包含并发读取
一致性、发布失败回滚、读查询不刷新证据时间、单扫描互斥、迟到响应、隐藏页和退避。

### 面板状态

总览默认显示当前任务（包括已停止和预算结束的任务），勾选“显示历史任务”可查看被替代任务及证据。
总览显示 GPU 实际状态、最近执行、Agent 结论与下一步、记录数和当前契约的验收收益。
GPU 执行时间来自作业开始/结束记录；评测记录时间来自收据的 recorded_at（文件修改时间），不代表内核执行时刻。Agent 报告时间单独显示，维护更新不会变成新实验。
GPU 作业次数包含失败或仅编译的命令，不称为有效实验次数；会话连接诊断保留在详情。

历史记录只沿显式 `supersede` 链累计，区分当前与历史评测数，点击可查看旧任务与作业记录。
旧契约 best 不继承为当前收益；尚无当前 best 时明确显示“尚无完整验收收益”。Agent 文字中的提速不作为验证数据。
总览摘要保持简短，完整报告、配置、日志与证据保留在详情。面板为只读视图，控制操作由 CLI 执行。

“最近评测”展示最近一份已校验收据的结果、通过的检查数和失败原因，与“已晋升 best”分开。
quick 通过不代表完整验收通过，没有晋升也不代表没有做过评测。收据导入与 best 校验原子发布。

证据扫描显示“正在校验”“排队校验”及上次观测时间；超过 15 秒未更新不再直接标为故障。
扫描出错或单项扫描超过 5 分钟才提示检查。API 保留 `evidence_stale` 作为观测年龄信息，
不据此判断 GPU、Agent 或评测失败。升级时旧缓存会重新校验一次以补齐最近评测摘要。

## 消耗卡时

总览显示平台分配的 GPU 数量 × 占用时长（卡时），包含准备、诊断和空闲占卡。
GPU 命令运行时间仍在作业详情保留，不用于替代占卡消耗。缺少用量记录时显示“—”。

平台接入端在获取作业分配、运行或释放状态后，将带来源的观察结果导入：

```sh
kernel-envolve manage --state /work/manager-state import-usage --file allocation-usage.json
```

```json
{"version":1,"allocations":[{"id":"platform-job-id","started_at":1794304800,"observed_at":1794308400,"ended":true,"gpus":2,"tasks":{"operator-a":1,"operator-b":1},"source":"platform-receipt.json"}]}
```

时间为 UTC Unix 秒。`observed_at` 是已计费到的观察时间；释放后设 `ended=true`，
不会随着面板刷新继续累加。运行中由平台接入端更新观察，不根据模型报告推算。
共享 GPU 须显式分配份额（如各 0.5），份额之和等于总卡数；分配或归属变化时
结束旧区间并创建新的唯一 ID。重复导入同一观察不累加；已结束记录不可覆盖，
运行中记录不能减少消耗或改变归属。迁移后的任务沿 supersede 链继承卡时，
总量按唯一分配记录计算，避免重复计数。此账本用于展示，既有执行预算保持原口径。

## 最新实现

总览“最新实现”显示已验证 best 对应的实现 PR 和 GitHub `updatedAt`，
不再单列提出时间或晋升时间。Agent 每次发布或更新 PR 后执行
（`--run` 使用 Manager 状态中的 best.run）：

```sh
kernel-envolve manage --state /work/manager-state sync-pr operator-a --run /work/task/state/runs/id --source-root core --base-branch dsa-dense-warmup-integration --url https://github.com/org/repo/pull/123
```

命令重新验证完整评测及已声明的确认收据，再通过已认证的 `gh` 比对 PR 的不可变
Git tree 与冻结源码。`--source-root` 指定本次交付的 native 源码根；generic 任务使用
`task`。PR 仓库、目标分支、所有 `implementation_paths` 文件（含辅助文件、增删文件）
及固定生产依赖都必须匹配；范围外的 PR 改动会被拒绝。多仓评测中，其他源码根和
可信测试覆盖文件仍作为评测条件保存，不要求塞入本次实现 PR。本地 Git 镜像须能
沿最多 8 层、无环的 origin 链追溯到对应 GitHub 仓库。Git 元数据不作为交付源码。

绑定保存 result、确认收据和源码摘要。没有 best、证据失效或新 best 尚未关联 PR 时
显示“—”。现有协调器每 60 秒检查到期的关联，每轮最多检查一个 PR；页面只读本地
状态。PR head 或目标变化后须重新执行 `sync-pr`，重新核验源码才能视为当前实现。
GitHub 暂时不可用或观察超过 240 秒时，保留最后验证的 PR、更新时间和固定提交链接；
接口中的 `current_head_verified=false` 区分历史证明与当前 head，避免网络故障抹掉成果。

## 前台宿主接续

若当前 coding agent 通过命令工具操作远端工作区，使用内置 bridge，不再复制实验中的
转接脚本。配置该任务 `"host_driven": true`，并把 `command` 设为
`["kernel-envolve", "manage", "bridge", "--root", "/work/bridge"]`。bridge 产物目录必须位于
所有任务 workspace 之外，避免日志被计作实现变更。普通非交互
coding-agent 命令仍使用默认 `host_driven: false`。

前台宿主亲自执行一次接续循环：

1. `manage bridge-open TASK --owner HOST --claim-id UNIQUE` 声明准备启动一轮。
   协调器消费一次许可后才启动；相同 claim 重放只返回原收据，不产生新轮。
2. 从 `manage status` 获取当前 round，读取
   `/work/bridge/TASK/ROUND/prompt.txt` 和任务 checkpoint。
3. 写请求 JSON，例如
   `{"kind":"gpu","argv":["kernel-envolve","native-evaluate","--task","/work/task","--mode","quick"],"expected_seconds":300}`。
   调用 `manage bridge-request --root /work/bridge --task TASK --round ROUND --owner HOST
   --id REQUEST --file request.json --wait-seconds 30`。
4. CLI 默认最多等30秒，完成立即返回；`--wait-seconds 0` 保留非阻塞查询。返回 `pending` 时，用相同参数去掉 `--file` 继续等待，不插入固定 sleep。返回完成收据时，当前 coding agent
   阅读 stdout/stderr、核对产物并决定下一步，然后调用
   `manage bridge-consume --root /work/bridge --task TASK --round ROUND --owner HOST
   --id REQUEST --sha256 RECEIPT_SHA256 --outcome '本次发现' --next-action '具体下一步'`。
5. 完成消费后才能提交下一条请求。结束本轮时提交
   `{"kind":"finish","report":{"outcome":"本轮结论","next_action":"下一轮具体实验"}}`。
   新轮需要新 claim，不会因等待进程退出而自动重复开启。

所有命令使用相同 `manage --state /work/manager-state`。`gpu` 请求由本轮固定版本的
`manage gpu-run` 执行并声明时长；排队受 manager 的父轮次和累计预算控制，不把预计执行时长
当作排队超时。`cpu` 仅供源码检查等无 GPU 命令，`expected_seconds` 同时作为其超时上限。
CPU 请求在独立的已登记进程组中执行，正常退出也会清理派生进程，清理完成后才返回收据。
桥进程失联时由轮次主管回收；未获得完整收据的执行不会写成成功。单次等待最多 60 秒，
返回包含有限长度日志和收据哈希；完整日志留在本轮目录。每个请求 ID 只执行一次，
不同内容不可复用 ID，旧结果未消费不可提交新工作。进程崩溃后已领取但没有完成收据的
请求返回 `interrupted_without_receipt`；先核对产物，不自动重试。

宿主等待按 `min(5秒, idle_seconds/3)` 续租，消费时立即更新接续时间；读取已完成结果不续租。没有在途命令且宿主离开超过 120 秒（可配置 1–300 秒），
bridge 正常让出本轮（`yielded`），不增加失败次数或失败退避，保留已有真实失败、任务和预算，
不自动开启空轮。只有受信 bridge 的闲置收据和专用退出码能进入该状态，普通命令或 GPU
执行失败仍按失败保存。当前 coding agent 必须持续读取结果并
提交下一步；bridge 不提供模型或后台模型调度。用户停止、预算用尽和任务替换始终生效，
事后补记结果消费也不会恢复任务。关闭轮次的结果仍可消费并保存到该轮报告；只有任务最新
轮次的分析会更新当前研究记录，旧轮迟到的分析不会覆盖新轮。补记不会修改原执行结果、
失败次数、调度时间或预算；重复消费同一收据也不会刷新进展时间。旧版本已经消费但未进入
研究记录的结果，可重放原 `bridge-consume` 命令补齐投影，无需重跑实验。
