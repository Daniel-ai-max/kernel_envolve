# 持续优化管理

管理层将一个算子的长期任务与单轮 Agent 执行分开，复用现有 evaluator 和 best。
一个 POSIX 主机运行一个协调器、一到两个会话共用一张 GPU；所有 GPU 作业排队互斥。
Agent 自主选择优化路线，协调器只管理生命周期、资源和证据。没有消息推送或模型路由。

## 启动

在实际 GPU 主机的项目环境内安装本包。先准备每个算子的独立 worktree、task
目录和原始 baseline，再编写部署 JSON。管理状态目录必须位于所有任务工作区之外。
GPU 用 `nvidia-smi --query-gpu=uuid --format=csv,noheader` 得到的物理 UUID 标识，
不接受 `cuda:0` 或 MIG 切片作为调度身份。进入 GPU 作业后只暴露该卡，任务使用 `cuda:0`。

```sh
kernel-envolve manage contract --task /work/probs-scale/task --kind native
kernel-envolve manage --state /work/manager-state init deployment.json
kernel-envolve manage --state /work/manager-state run
# 另一个终端；远程主机可用 SSH -L 8787:127.0.0.1:8787 转发
kernel-envolve manage --state /work/manager-state serve --port 8787
```

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

额度单位为秒，null 表示明确不设上限；两种额度都是累计值，不是每轮重置。
session wall 时间包含该轮的等待；GPU 时间是作业占用设备的时间（包括导入、编译、
正确性和测量），不是 GPU 内核活跃时间或费用。主管失联期间采用保守的经过时间记账。

## Agent 适配接口

`command` 是 argv 数组，不经 shell 拼接。使用你现有 coding-agent 的非交互入口，或
一个转接它的可执行脚本。管理器不附带模型、API key 或新的推理循环，也不直接调用
桌面应用中的 chat。适配器必须在当前进程生命周期内运行，不得 daemonize。

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
kernel-envolve manage gpu-run -- kernel-envolve native-evaluate --task ./task --mode quick
kernel-envolve manage gpu-run -- kernel-envolve native-evaluate --task ./task --mode full
kernel-envolve native-promote ./task/.kernel_native/runs/<run>
kernel-envolve manage report --file progress.json
```

GPU wrapper 会转发命令输出的最后 64 KiB，完整日志在管理状态目录的 `jobs/<id>/`。
Agent 主进程默认看不到 CUDA 设备；GPU wrapper 给子进程设置目标 UUID。不能再嵌套
gpu-run。运行需要其他 GPU 的分布式验证不在单卡队列契约内，应单独安排并保存证据。

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
按退出码处理失败，不靠分析自由文本；三轮连续失败后进入 blocked，需人工检查后恢复。

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

协调器退出时，独立的轮次主管继续执行并检查预算；重启协调器会核对进程身份和锁，
接管观察，而不重新运行活着的轮次。新主机不能仅复制 SQLite 后自动重放作业。
部署时可用 systemd/现有作业系统守护 `manage run`，配置失败重启与启动退避。
面板服务独立运行，关闭页面不影响任务，页面会显示协调器是否在线。

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

更新分为采集、发布、读取和显示四步。协调器约每 5 秒尝试一次增量证据扫描；
同一部署只允许一个扫描执行。源码/收据的读取、指纹检查和判定计算在写事务之外
完成，每个算子的 runs、best、晋升历史、错误和观测时间再一次性提交。新实验先落盘、
校验，再进入面板；面板展示最近一次已发布的证据，不直接拼接正在变化的结果文件。
扫描耗时较长时不重叠启动，也不保证固定的 5 秒端到端延迟。

面板采用短轮询，适合当前单主机、少量观察者的工作方式：

| 数据/场景 | 更新规则 |
|---|---|
| 运行阶段、队列、用量 | 读 supervisor/协调器已经提交的状态；一次成功响应结束约 3 秒后再请求 |
| 总览与当前详情 | 同一次 HTTP 请求、同一个 SQLite WAL 只读快照；总览省略逐 case/source 大字段，只给选中算子传详情；历史只传曲线/实验摘要，逐 case 数据仅传当前 best |
| best、逐 case 延迟、验证范围 | 来自同一份已发布证据；后台索引独立于打开面板的数量运行 |
| 切换算子、手动刷新、控制完成 | 立即读取新快照，取消旧读取并拒绝它迟到的结果 |
| 暂停/恢复/保存 | 等待服务端确认后回读；不会预先把界面改成成功，也不会自动重试写操作 |
| 隐藏标签页 | 停止请求；再次可见时立即刷新 |
| 网络或服务故障 | 单次请求 8 秒超时；保留旧数据，以 6/12/24/30 秒退避重连；成功后恢复 3 秒间隔 |
| 编辑和阅读 | 复用 DOM 节点，仅更新变化内容；保留筛选、焦点、表格滚动和展开项；未保存的表单值即使失去焦点也保留 |

“同步时间”是面板读到状态快照的时间；“评测扫描时间”是该算子后台检查的观测时间，
放在验证范围和 best 标题的 tooltip 中。15 秒没有新证据扫描时出现短状态“评测更新延迟”；
协调器离线、浏览器断线分别显示，收到 HTTP 响应本身不代表评测或协调器仍在推进。
正常时不额外增加说明区。

只读请求不执行 schema 初始化或持久化写入；WAL 读快照允许调度器同时提交变化。
后台每轮仍需遍历历史文件的元数据以发现篡改，只有变化的内容重新校验；大量历史
文件时扫描可能变慢，延迟状态会如实显示。手动 `manage refresh --force` 在已有扫描
执行时明确拒绝并要求稍后重试，不排入无界重扫队列。

自动更新回归覆盖见 `tests/test_manager.py` 与 `tests/manager_sync.mjs`：包含并发读取
一致性、发布失败回滚、读查询不刷新证据时间、单扫描互斥、迟到响应、隐藏页和退避。
