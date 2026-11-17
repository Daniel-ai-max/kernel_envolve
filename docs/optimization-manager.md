# 持续优化管理

管理层将一个算子的长期任务与单轮 Agent 执行分开，复用现有 evaluator 和 best。
一个 POSIX 主机运行一个协调器，每个资源由一张或一组 GPU 组成，支持一到两个 Agent 会话；同一资源的 GPU 作业排队互斥。
Agent 自主选择优化路线，协调器只管理生命周期、资源和证据。没有消息推送或模型路由。

## 单任务启动

```sh
# 仅展示已有实验；不启动模型，不探测或申请 GPU
kernel-envolve dashboard --task /work/kernel-task

# 同时启动真实模型宿主，沿已有工作区和记录在预算内接续
kernel-envolve dashboard --task /work/kernel-task --seconds 3600 -- /opt/model-host-adapter

# 查看已有单任务或多任务状态，保留 PR、best、用量和停止状态
kernel-envolve dashboard --task /work/kernel-task --state /work/manager-state
```

没有 evaluator 时，可直接从源码工作区启动：

```bash
kernel-envolve dashboard --research /work/operator-checkout --baseline HEAD --device GPU-... --seconds 3600 -- /opt/model-host-adapter
```

`--research` 选择独立 Git checkout，首次将 `--baseline`（默认 HEAD）解析为固定提交。源码已有修改保留，不自动 checkout、复制仓库或加载项目。只需工作区、baseline、已分配设备和真实宿主命令；不要求 `task.json`、`task.py`、`candidate.py`、`native.json` 或 adapter。CPU 管理回归可用 `--device cpu`。省略宿主命令仅观察记录，不探测或占用 GPU。

研究任务以 `kind=research` 注册到同一个 Store、bridge、GPU 队列和看板。源码身份摘要用于重开同一任务，**不是数值正确性契约**；执行退出码、原始日志和研究说明不会成为 verified best。源目录里即使存在 evaluator 收据或 best 指针，也不导入到研究任务。正式验收继续使用独立的 generic/native 任务；本入口不自动转换任务类型或继承验收资格。已有研究任务重开时不重新选 baseline、不续加预算、不恢复暂停；预算和控制仍用现有管理命令。可信外部 verifier 可通过现有 `watch-assessment` 延迟接入，展示有明确范围的已验证结果，但不改变任务类型或产生 best。

每个受管研究 GPU 命令自动保存执行前后源码记录：固定 Git baseline、tracked binary diff 与非 ignored 的新增源代码/脚本。内容按 SHA 去重，绑定作业 ID、argv、退出状态和有上限的日志哈希；前后变化会标明。Git 输出、单文件和总读取量均有运行中上限，超大、超额、读取失败或并发变化会记录缺口，探索仍可继续。ignored 文件、非源码的 untracked 文件、外部依赖及子模块内容不属于此记录，baseline Git 对象须单独保留；这不是正式 evaluator 快照。源码修改用于原有恢复判断，不自动复活暂停、预算耗尽或 blocked 任务。周期看板归档只保留引用，终态/显式归档自动带去重增量；完整日志仍可用 `--include-logs` 导出。

`task.json` 与 `native.json` 均可使用此入口。它复用同一个管理器、记录索引和 HTTP 看板，不创建另一套界面。
新任务默认状态保存到 `${XDG_DATA_HOME:-~/.local/share}/kernel-envolve/dashboards/<任务路径摘要>`；已有 `~/.cache` 状态会继续复用，不重置记录或预算。服务地址默认
`http://127.0.0.1:8787`，可用 `--port` 调整。原生任务的工作区默认包含任务目录和所有 candidate 源码根，必要时用 `--workspace` 指定专用父目录。

带命令时，宿主遵循下文的[自动接续协议](#预算内自动接续)：从 stdin 读取恢复上下文，真实模型会话结束后退出。默认预算一小时；`--seconds` 只在首次配置执行器时设置累计会话额度和绝对截止时间。
CUDA 任务使用 `--device GPU-...` 选择物理设备；仅一个可见设备时可自动选择。此入口不向平台申请卡。
重新打开既有执行任务时不重复传入预算或设备，原预算与截止保持不变。

仅查看时不调度、不恢复进程，也不启动状态库中的其他任务；已有协调器时复用其记录同步。
关闭浏览器不会结束任务；退出启动命令不会立刻杀死已启动轮次，但会停止后续接续，重新执行带原宿主命令的入口即可续接。用户明确停止时使用 `manage pause`，停止状态不会被看板自动恢复。

看板从作业与评测记录读取当前工作、最近活动及局部信号；正式收益只来自已验证 best。
心跳或研究文字不会刷新实验时间，持续五分钟没有新实验时不再显示普通会话在线。
GPU 隔离后宿主仍可做 CPU 诊断，GPU 准入保持关闭；无法恢复时显示恢复失败。
卡时沿用平台分配账本，缺少用量记录时显示 `—`，不会以 GPU 命令耗时冒充总占卡时间。

## 持久化与复盘

协调器与单任务看板在后台保存一致的 SQLite、收益与评测摘要、研究记录、PR、卡时和页面资源。GPU 作业进行中让路，归档最多每分钟更新一次，记录未变则跳过；退出时补存。自动归档保留最近两份完整快照，每份 SQLite 都包含全部历史记录；任务终态和显式导出的快照不会被自动清理。运行状态仍以实际作业与最近实验为准，历史快照不显示为正在执行。

```sh
# 原状态库仍在：只读查看，不启动协调器、模型或 GPU
kernel-envolve dashboard --state /persistent/manager-state

# 查看自动归档；也可拷贝到另一台机器后打开
kernel-envolve dashboard --archive /persistent/manager-state/dashboard-archive

# 交付归档：补存完整作业日志及所需原始证据文件
kernel-envolve manage --state /persistent/manager-state archive \
  --output /persistent/review --include-logs --attach /persistent/assessment.tar.gz
kernel-envolve dashboard --archive /persistent/review
# 必要时选择某份历史快照
kernel-envolve dashboard --archive /persistent/review/snapshot-<id>
```

临时 GPU 节点应将 `--state` 指向持久卷，或设置 `KERNEL_ENVOLVE_ARCHIVE=/persistent/review`。仅存在节点 `/tmp` 内的文件不能在节点销毁后恢复。归档带校验清单，完整写入后才原子切换；复制整个目录即可恢复，不需要原源码路径或执行主机身份。读取归档不会修改原始账本，也不提供执行控制接口。表格保留最近记录，随包 SQLite 保留完整账本；原始张量及源码产物仍由 evaluator 的产物目录保存，需要离线复现时用 `--attach` 一并带走。

## 独立验收接入

内置 evaluator 继续自动索引原有收据。外部验收只需由部署者一次性安装可信校验脚本，之后收据发布会被后台自动识别，不要求研究 Agent 再写报告或确认结果：

```sh
kernel-envolve manage --state /persistent/manager-state watch-assessment operator-a \
  --receipt /persistent/results/final-receipt.json \
  --verifier /opt/trusted/verify-assessment.py
```

校验脚本放在候选工作区外，注册时固定其 SHA；通过 stdin 接收 `task`、`baseline`、`receipt`，核验原始产物、输入覆盖、正确性及计时后，只向 stdout 输出 `scoped-assessment-v1`。脚本须自行固定并校验依赖与评测契约。管理器限制执行时间和输出大小，校验任务及基线绑定、有限数值与字段结构，持久化原始输入、结构化结果和校验器身份。它不会把未经核验的 Agent 文字当作收益。脚本失败、变更或结果错配会显示“验收同步失败”，保留旧成果及失败原因；相同失败输入不反复执行，修复证据后可用 `manage refresh --force` 明确重验。

`scoped-assessment-v1` 的字段沿用看板：`schema`、`status="verified"`、`promotion=false`、`task`、`baseline_revision`、`source_sha256`、`evidence_sha256`、`recorded_at`、`metrics`，以及可选的 `scope_label`、`qualification`、`scenarios`、`checkpoints`。每项 metric 包含 `label`、`baseline_speedup`、`cases`，可附 `slower_cases`、`worst_slowdown_percent`。场景包含 `name`、`mode`、`baseline_us`、`candidate_us`；时间点包含 `minutes`、`eager_ratio`、`graph_ratio`。

算子范围验收与全链晋升保持分离，正式 best 优先；导入不修改 best 或晋升门禁。任务已经结束也允许补入后到的独立验收，累计预算和停止状态不变。无人运行的进程不会继续同步：远端收尾应在销毁节点前完成最后一次 `manage refresh` 与 `manage archive`，由执行宿主完成，不增加 Agent 每轮操作。

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

### 预算内自动接续

一次会话退出、保存候选、晋升或发布 PR 不结束长期任务。普通 `command` 直接运行
模型适配器时，协调器已会沿原任务再次启动；前台 bridge 要接入实际宿主命令才能自动
启动下一次模型会话。仅发送 `bridge-open` 或维持 bridge 进程不能调用模型。

在任务配置中增加宿主入口，并将 `command` 设置为 bridge：

```json
"host_driven": true,
"command": ["kernel-envolve", "manage", "bridge", "--root", "/work/bridge"],
"host_command": ["/opt/model-host-adapter"],
"wall_budget_seconds": 3600
```

`host_command` 是实际启动或续接 coding agent 的 argv，不是空等脚本。适配器从 stdin
接收恢复提示，直到本次模型会话结束才退出；如果调用远程宿主，应等待其真实终态，
不能提交远程请求后立即返回成功。使用宿主已有的会话恢复能力即可，不引入第二套模型循环。
继承 `KERNEL_ENVOLVE_STATE`、`TASK`、`ROUND`（后两项同样带 `KERNEL_ENVOLVE_` 前缀），
并提供 `KERNEL_ENVOLVE_BRIDGE_ROOT` 和 `KERNEL_ENVOLVE_BRIDGE_OWNER`，用于原有
`bridge-request` 接口。若宿主只能通过桌面工具派单，仍需该宿主实际调用续接工具；
没有可执行适配器时不能宣称已经无人值守运行。

协调器自动为真实宿主建立一次轮次，保存进程身份与 stdout/stderr；宿主正常退出即
形成会话终态，不需要 `finish`、consume、报告或重复 notes。下一会话的提示包含
最近会话输出、实验目录、best 指针位置及原有可选研究摘要。失败请求及已否定方案
保留在原记录中，agent 按需读取；管理器不把请求数或文字更新判定为研究收益。

累计 wall/GPU 额度不重置。可额外设置 `deadline_at`（UTC Unix 秒）限制绝对结束时间，
覆盖会话间冷却、恢复和等待；调度、在途进程与 GPU 时长准入均检查它。对照实验应固定
共同开始与截止时间，并为收尾预留时间。`host_command` 至少需要有限 wall 额度或截止时间。

现有 bridge 任务可先暂停并等待在途作业清理，再通过 `manage configure TASK --changes`
设置 `host_command` 和 `deadline_at`，然后恢复；工作区、任务 ID、best 和累计用量不变。
这是部署切换，不是每轮研究步骤。启用后由管理器拥有宿主接续，不再手动 `bridge-open`。

宿主退出后经过现有冷却/故障退避继续；无需每轮注册任务。连续三次未观察到实际执行或
实现变化时停止自动空转，保存现场并明确原因。这是运行活动诊断，不判断科学价值；同一
命令配合不同实现的实验按不同活动记录；改变输入文件的实际实验也不会仅因命令相同被终止，
是否获得新信息仍须结合原始结果判断。适配器确认无法恢复的执行故障时，
在输出中保存原因并退出 `78`，管理器直接终止自动续接。普通候选被否定应正常返回，
不能使用该退出码。用户停止、预算结束和已替换任务均不会自动复活。

模型宿主不暴露 CUDA 设备、不持有 GPU 设备锁；实验仍经现有 `gpu-run`。清理覆盖宿主
及登记后代。此处控制本机进程与设备使用，云平台申请/释放仍由部署入口负责；部署入口
应根据待运行实验申请资源、在预算结束或不可恢复故障后释放，不能把无 GPU 命令误报为
零占卡费用。既有 allocation 账本保留准备与空闲占卡时间。

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

每轮 stdin 是恢复提示：任务范围、固定基线、已有研究摘要、最近事件、预算和
工作目录。Agent 按需读取项目指令、kernel-envolve skill 与已有源码和 run；notes 可用于交接，但不要求重复运行记录。
如果宿主支持原生 session resume，适配器可以在自己的任务工作区保存宿主 session ID
并续接；否则每轮使用新 session，通过同一任务 checkpoint 恢复。两者都不改变任务身份。

环境变量：

| 变量 | 用途 |
|---|---|
| `KERNEL_ENVOLVE_STATE` | 管理状态目录 |
| `KERNEL_ENVOLVE_TASK` | 长期任务 ID |
| `KERNEL_ENVOLVE_ROUND` | 当前执行轮次 ID；用于拒绝过期报告 |
| `KERNEL_ENVOLVE_REPORT` | 可选研究结论的 JSON 文件；执行状态另由主管记录 |

```sh
# GPU 上运行的所有命令均经该入口，包括局部正确性检查和 profiler。
kernel-envolve manage gpu-run --expected-seconds 300 -- kernel-envolve native-evaluate --task ./task --mode quick
kernel-envolve manage gpu-run --expected-seconds 1200 -- kernel-envolve native-evaluate --task ./task --mode full
kernel-envolve native-promote ./task/.kernel_native/runs/<run>
# 有需要交接的新结论时可保存；不是每次实验的必经步骤。
kernel-envolve manage report --file progress.json
```

`--expected-seconds` 始终声明本次命令的预计墙钟时长，建议使用命令超时作为保守上界。
例如八卡作业声明 300 秒，仅命令本身就需要 2400 卡秒 GPU 额度和 300 秒 wall 额度。
父轮次与累计 wall/GPU 预算还须包含实际启动和清理开销；GPU 开销也按设备数换算为卡秒。
管理器在入队前和获得设备锁、标记作业运行后分别检查父轮次、累计 wall/GPU 剩余额度。
排队消耗父轮次和 wall 时间；第二次检查也包含当前 GPU 作业 claim 后的实际经过时间。
因此把总预算恰好设为声明时长可能在启动子命令前被拒绝，应在原有额度内预留开销。
报错保留浮点数精度并显示缺口，避免微小差值被舍入成“声明与余额相等”；比较不设容差。
不足时返回退出码 `2` 和 `error_type: gpu_duration_admission`，不启动子命令、不自动重试或延长预算。
Agent 应保存检查点并重新规划；剩余累计预算不会因换轮恢复。额度按实际经过时间记账，
声明时长不预扣额度，也不替代命令自身超时。此参数可选，旧调用未声明时保留原行为，
不会提前判断作业是否来得及完成。

Linux 上，GPU 作业专用主管在启动命令时将其继承的 `RLIMIT_CORE` 软限额设为 **1 字节**，
并保留原硬限额；命令及编译子进程跨 `fork`/`exec` 继承该值。Linux 管道式 `core_pattern`
通常忽略限额 0，但内核将 1 作为跳过转储处理程序的特殊值。主管随后恢复自身限额，
不更改宿主配置、Agent 或其他作业，也不新增包装 CLI。若继承的硬限额已为 0 或设置被拒绝，
保留原限额并在该作业 `stderr.log` 中说明仍可能触发管道转储，不因此拒绝原命令。
这是自有命令的默认崩溃处理策略，不是隔离边界；子进程显式改写软限额后不再保证抑制转储。
仅覆盖受管理的 GPU 入口；需要 core 文件调试的命令可以自行调整允许范围内的软限额。

直接使用 gpu-run 时，wrapper 转发 stdout/stderr 各最后 64 KiB；截断时附上
`KERNEL_ENVOLVE_LOGS`，记录原始字节数和 `jobs/<id>/` 中的完整日志路径。
bridge 直接读取原始日志，默认各返回 4 KiB 尾部与完整路径，不再复制 wrapper 输出。
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

研究摘要按需保存到 `KERNEL_ENVOLVE_REPORT`，或通过 `manage report --file progress.json` 更新。
只记录需要交接的结论与未完成工作，无需重复源码、日志、用量或 PR 数据。例如：

```json
{"outcome":"小尺寸正确，长序列尚未确认","next_action":"检查长序列退化原因"}
```

执行状态、退出码和原始日志由主管记录；speedup、覆盖与 best 来自 evaluator 证据，
PR 状态来自已验证绑定。研究结论不能代替执行恢复或性能验收，省略分析不阻止下一实验。
提交的摘要仍需符合字段契约并属于有效轮次；只读查询不产生报告。

执行失败保留原任务与证据，按 10、20、40、80、160、300 秒退避；新轮收到实际退出码、
日志与错误摘要。Agent 自主诊断和选择下一实验，不靠改写报告或心跳宣称故障已修复。
数值 FAIL 是实验结果，不允许放宽 oracle、容差或覆盖；恢复不会延长预算、解除暂停或 GPU 隔离。
bridge 不提供模型，外部宿主仍须实际运行 Agent，进程在线不代表研究正在推进。

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
资源绑定。原始运行记录和运行中的本地编辑保留供恢复核对。
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
重复相同报告、主管心跳和自动开启下一轮不会成为新实验。研究结论可随下一请求
或在交接时保存，面板从已有记录投影；不要求另行填写报告来刷新页面。
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
PR 交付使用 `sync-pr` 完成证据核验与绑定，无需先额外 refresh。主动诊断索引时可用
`manage refresh --task <id>`，不带 `--task` 则核对全部任务。

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
旧契约 best 不继承为当前收益；尚无已验证收益时显示“—”，阶段发现保留在当前进展与详情中。Agent 文字中的提速不作为验证数据。
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

## 优化 PR

总览“优化 PR”显示已验证 best 对应的实现 PR 和 GitHub `updatedAt`，
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

命令工具操作远端工作区时，可配置 `"host_driven": true`，将 `command` 设为
`["kernel-envolve", "manage", "bridge", "--root", "/work/bridge"]`。
bridge 目录放在所有任务 workspace 之外。普通非交互 Agent 保持 `host_driven: false`。
所有命令使用同一 `manage --state /work/manager-state`。

`manage bridge-open TASK --owner HOST --claim-id UNIQUE` 申请一次轮次；同 claim 重放只返回
原收据。由 `manage status` 获取 round，读取 `/work/bridge/TASK/ROUND/prompt.txt` 后即可提交
原生局部检查、批量实验或 evaluator 命令，无需为每个参数点新开轮次。例如 request.json：

```json
{"kind":"gpu","argv":["python","experiment.py"],"expected_seconds":300}
```

```sh
kernel-envolve manage --state /work/manager-state bridge-request \
  --root /work/bridge --task TASK --round ROUND --owner HOST \
  --id REQUEST --file request.json --wait-seconds 30
```

同轮只允许一个未终态请求。前一请求完成后可直接提交下一请求，不要求先消费或撰写分析。
需要保留对已有结果的结论时，在下一请求内添加可选字段：

```json
"analysis": {
  "request": "PREVIOUS_REQUEST",
  "receipt_sha256": "已读取的完整收据SHA256",
  "outcome": "本次发现"
}
```

`analysis.round` 可省略，默认当前 round；显式指定可引用本任务同宿主的旧轮结果。
`next_action` 也可省略。分析必须对应真实完成收据，保存到原轮，不会改写执行状态或预算。
它不是运行许可；缺省分析不影响下一实验。结束轮次用 `{"kind":"finish"}`，有需要交接的
结论时附加 `report`，例如 `{"kind":"finish","report":{"outcome":"已排除当前融合方案"}}`。
外部手动宿主的新轮仍需新 claim；配置 `host_command` 后由管理器创建，并真正启动宿主命令。

查询原请求使用相同身份和 ID，省略 `--file`、`analysis` 及兼容参数；查询零写入，
不续租、不生成分析、不提交工作。CLI 默认最多等待 30 秒，`--wait-seconds` 范围 0–60；
有独立工作可做时设为 0，pending 后按需继续查询原 ID。完成与 pending 都可返回当前日志：
stdout/stderr 默认各末尾 4096 字节，`--log-bytes 0` 只取元数据，最大 65536 字节。
`log_files` 给出完整路径、字节数和截断状态；尾部不足以判断时再按路径读取完整内容。
日志投影不改变不可变收据及其 `receipt_sha256`。

`gpu` 由固定版本的 `manage gpu-run` 执行，受父轮次、设备锁和累计预算控制；预计执行时长
不等于排队超时。`cpu` 只运行无 GPU 命令，`expected_seconds` 同时作为超时上限；其登记
进程组在清理完成后才返回收据。每个 ID 只执行一次，不同内容不可复用 ID。响应不确定时
查询原 ID；`interrupted_without_receipt` 保留已领取但无完整收据的执行，先核对现场，不能换 ID 盲目重派。

没有在途命令，且距最近提交或执行完成超过 120 秒（可配置 1–300 秒）时，bridge 让出本轮（`yielded`），
不增加失败次数、不自动开启空轮。配置 `host_command` 时以真实宿主退出和任务预算为准，不用空闲租约打断源码分析。查询不会伪装为研究进展；真实执行恢复与下一研究决策
分别处理。用户停止、预算用尽和任务替换始终有效，补记旧结果不会恢复终态任务。

`bridge-consume` 与 `--consume-file` 保留旧客户端兼容；新流程使用可选的请求内 analysis，
不把独立消费作为继续实验的前置条件。


### Display source-bound research results

The existing recent-result column and result-detail table can show a stage result
without a full evaluator. Install a trusted **host** adapter once, outside the
candidate workspace, using the existing receipt watcher:

```sh
kernel-envolve manage --state /path/to/state watch-assessment task-id \
  --research --receipt /path/to/host-index.json --verifier /path/to/host-adapter.py
```

The host adapter reads existing experiment artifacts and emits `research-result-v1`.
This is an optional host integration, not a mandatory agent report/consume step;
the framework does not automatically understand arbitrary benchmark output.
It reuses the pinned verifier, CPU-only execution limit, receipt-change detection,
append-only history and dashboard archive. HTTP readers never execute adapters.

The output fields are `schema`, `status: "research"`, `promotion: false`, `task`,
`baseline_revision`, `source_sha256`, `evidence_sha256`, `recorded_at`, `scope`,
`summary`, `limitations`, and `binding`. SHA fields are lowercase SHA-256.
`recorded_at` is the original experiment time, not the import time. Numerical
claims in `summary` remain research claims: provenance checks do not establish
correctness, a holdout result, current-source validity or full-chain speedup.

Two bindings are supported:

- `manager_job`: `job`, `receipt_sha256`, `log_sha256`. The host verifies the
  artifacts; the manager additionally checks its DB-pinned source receipt,
  unchanged/complete source endpoints, stdout hash, terminal job and original end time.
- `transport_archive`: `request`, `request_sha256`, `archive_sha256`,
  `receipt_sha256`, `log_sha256`. A pinned host adapter validates legacy archives,
  exact source membership, request/receipt identity and original logs. It must use
  bounded, allowlisted local artifacts; the framework never follows paths or URLs
  from its output. This does not manufacture a new manager-job source receipt.

Task/job identities in bindings are 32 lowercase hexadecimal characters. Claims
and scopes are bounded text; arbitrary fields and verified-assessment schemas are
rejected in research mode. The adapter is trusted host code, not a sandbox for
candidate-authored Python. Preserve referenced source archives/logs with the
existing archive attachments when moving the review to another machine.

Late imports and amended audit descriptions are allowed after a task ends. They
append a distinct `indexed_at` audit time without changing task state, budgets,
experiment time, runs, promotions or best. A later cleanup job does not erase the
last semantic research/assessment result; execution failures remain in the job
history. Failed research updates retain the previous result and show the sync
error in details. Both the original input and host output are saved in archives.

## 共享进展旁白

旁白异步整理已有 trace、实验日志和评测记录，统一组织收益、优化 PR、当前进展和最近结果。按“改了什么 → 测到什么 → 保留还是放弃”解释研究结论，保留测量范围、真实数字和退化；结束任务展示最终结论。收益、PR、状态仍由系统提供，旁白只能引用，不能修改正式 best 或发布 PR，也不要求研究 Agent 增加报告。

四列绑定同一套系统事实：正式验收、源码结果、收益、PR、身份、错误、交付或任务停止变化时，旧模型内容整组失效，先显示当前系统事实。新的探索作业排队、开始运行或 phase 切换只更新实时状态，不隐藏已有发现，也不单独触发模型推理。新增 trace 仍按原有增量机制处理。

探索执行的成功、失败、取消、超时和中断会立即使旧的“最新结果”文案失效；最新结果读取当前系统记录。已引用的研究发现可以保留原完成来源与来源时间，不能重绑为新作业结论。即使每次模型调用期间都有新探索完成，真实发现仍可持续更新，不依赖重试或等待实验停顿。模型调用期间若正式事实变化，迟到输出只能存历史。最新尝试无收益不会抹掉已保留的真实最佳方案；局部加速也不能升级为完整调用链收益。

数字优先照录引用原文；非负小数可用十进制 `ROUND_HALF_UP` 显示为 1–6 位小数，至少保留 3 位有效数字，并标注“约”。这只允许确定性的显示舍入，不允许截断、百分比推导或单位换算，也不放行带符号、标识符或科学计数片段的舍入。正式收益仍由系统测量提供。

CPU 的最近终态独立保存在 `cpu_completed:<task>` 元数据中，新请求不会覆盖它；bridge 异常退出由原有 round 收尾记录中断。旧状态无需迁移，读取既有确认完成事件和当前终态记录；旧版本没有留存的异常退出历史无法事后补造。升级会刷新一次旁白策略与事实版本，旁白与 renderer 应使用同一框架版本。

宿主配置一次文本模型，随后正常启动 `dashboard` 或 `manage run`：

```bash
export NARRATOR_MODEL='<轻量文本模型>'
# 支持推理强度配置的模型可选 NARRATOR_REASONING_EFFORT=low。
export NARRATOR_RESPONSES_URL='https://<已授权模型服务>/v1/responses'
# NARRATOR_API_KEY 由宿主凭据环境注入。
kernel-envolve dashboard --research /work/operator-checkout --device GPU-... --seconds 3600 -- /opt/model-host-adapter
```

内置适配器按 [Responses API](https://developers.openai.com/api/reference/resources/responses/methods/create) 使用[结构化输出](https://developers.openai.com/api/docs/guides/structured-outputs?api-mode=responses)约束字段和证据引用，发起无工具、非流式文本请求，不创建远端会话，不保存凭据。也可用 `KERNEL_ENVOLVE_NARRATOR_COMMAND` 指定可信模型适配器的 JSON argv；输入为 stdin JSON（`instructions`、`tasks`），输出为任务 ID 到四项摘要的 JSON。每项为 `null` 或 `{"text":"简短结论","sources":["输入中的证据ID"]}`，字段为 `direction`、`finding`、`failure`、`next`。新整行输出仅在 `finding` 和最近结果写正文，其余三项为 `null`，避免重复生成同一结论。同时返回 `row`：绑定输入 `row_facts.revision`，用 `system:benefit`、`system:optimization_pr` 引用系统字段，`current_progress` 引用 `finding`，`latest_result` 提供引用 `system:latest_result` 的短结论与来源。模型不可填写收益值或 PR 地址。旧四项摘要仍可读取历史归档，新事实版本下须生成整行摘要。适配器只负责模型推理，不授予模型 shell 或研究工具。

一个管理状态共用旁白执行锁，每批最多八个任务，来源合计不超过 48 KB，单任务不超过 40 KB；优先保留验收和候选结论。后台尾随持久事件，合并短时间内的更新；每 30 秒检查有界日志增量，补上只写 trace、未产生事件的变化。没有新内容不调用模型；同一失败输入或不确定请求不自动重发，新输入可继续处理。模型调用不占数据库写锁，45 秒超时，退出清理子进程。旁白失败保留历史；仅当事实版本仍匹配时展示原摘要，否则使用系统事实，优化照常推进。

已有多任务管理器可单独运行同一个旁白服务；已结束任务可补录一次：

```bash
kernel-envolve manage --state /work/manager-state narrate
kernel-envolve manage --state /work/manager-state narrate --once
```

需要替换运行中管理器的旁白版本而不重启调度器时，使用独立框架路径启动
`narrate --exclusive-service`。服务先等待旧旁白当前批次结束，再全程持有原来的
`narrator.lock`；旧线程后续批次会立即跳过，不读取或消费游标。新服务复用原
SQLite、游标、模型适配器与事实版本事务，不复制状态、不启动研究或评测。
退出或异常时自动释放锁，旧线程可继续工作。不要同时启动两个普通 `narrate`
来替代版本。节点启动包装应绑定原管理器的 PID 与进程出生身份，并在管理器退出、
研究结束或原分配截止时间到达时终止这个独立旁白；本命令自身不延长研究预算。

旁白和引用摘录随现有 SQLite、看板归档保存，归档重开无需模型或 GPU。它是有来源的模型解释，不是数值验收；模型无依据时应留空，作业状态、卡时、指标、正式验收和 PR 继续读取系统记录。

### 外部验收结果回流

使用 `validate_candidate` 的本地作业自动归入原任务。验收在另一个 manager 运行时，
宿主只需绑定一次来源，后续已结束作业自动增量导入：

```sh
kernel-envolve manage --state /path/to/research-state watch-validation operator-a \
  --source-state /path/to/validation-state --source-task validation-task
```

来源可以是 manager 状态目录，也可以是带完整日志的官方归档目录；远程状态通过已有
持久化同步到宿主，不由此命令申请 GPU 或启动验收。绑定要求两边基线相同，来源安装
在候选工作区之外。每条记录检查终态作业、前后源码身份、数据库固定收据、完整日志
及结果哈希；无收益、失败、中断和通过均保留历史，但不写 best 或已验证收益。

重复刷新不重复发事件。旧结果不覆盖新结论，晚到结果不恢复终态任务或改变预算。
旁白失败时看板直接显示系统结论。不可用来源恢复后自动清除连接故障；不完整收据
单独归档，不能阻止其他任务复盘。已有自定义宿主 verifier 可用
`watch-assessment --validation` 输出 `validation-result-v1`；旧数据不伪造新的作业绑定。


### Linked native delivery in the dashboard

`policy-bind-delivery` records one explicit research owner for an observation-only native delivery task. Rebinding to a new attempt retains earlier delivery links and their original native run history. The dashboard shows one research row by default and opens each native attempt from that row's delivery details; names never imply this relationship. Research `best` and publication records remain empty. Only a verified native full, independent confirmation, matching baseline/source, and current source-bound PR can supply the row's separate delivery benefit and PR. No evaluator eligibility or scheduling state is transferred.

The existing background scanner projects the frozen native full/confirmation/PR relationship at most once per 30 seconds and yields to GPU work. It reuses the existing deep-audited native index and checks the small result/config receipts, without a second source-tree audit. Missing/revoked index entries and changed receipts cannot supply a displayed benefit. Previously verified results remain visible when research proceeds to another candidate, when a refresh is unavailable, and when a terminal archive is reopened. New stop-policy bindings retain a witness of all frozen delivery sources and their research owner, so later research edits do not invalidate delivery. Legacy bindings without that witness retain the live research-source check. Neither rule erases historical delivered results. Indexed corruption of the frozen evidence withdraws the affected result. A changed or stale PR head removes the current PR link from the research row while retaining the previously evaluated commit in delivery details. New attempts retain prior verified results until another published result is verified; unlike contracts are not ranked by speedup. All four narrated columns use the resulting row-facts revision, and hidden native child tasks do not generate duplicate narration.
