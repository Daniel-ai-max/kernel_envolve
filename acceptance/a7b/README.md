# A7B P0–P2 算子验收设计

本目录覆盖 [A7B 训练算子拆解](https://s.apodex.site/WJuWiAQQtVaXMvsNOPePY6)
v8（2026-11-17 读取）中 E65 / E98 的全部 P0、P1、P2：**9 类算子、16 个 profile × 算子组合**。
这里保存可审查的验收设计和原生入口映射，尚未生成针对真实运行输入的 adapter，尚未运行 GPU。
`design_check=passed` 只表示目录完整且内部一致，不是数值、性能、运行就绪或晋升通过。

研究继续使用现有轻量流程；只在任务接入、候选交付时使用本目录，不新增逐实验确认、报告、
独立 consume、模型调用或后台守护进程。复用 [native evaluator](../../docs/native-harness.md)
和 [Manager](../../docs/optimization-manager.md)，不建立第二套评测引擎。

## 完整范围

| 页面行 | 算子设计 | E65 · 256K | E98 · 32K |
|---|---|---|---|
| 3 | [Dense Linear / GEMM](operators/dense-linear.md) | P1 | P1 |
| 4 | [Grouped Linear / GEMM](operators/grouped-linear.md) | P2 | P1 |
| 5 | [Router Linear](operators/router-linear.md) | P2 | P2 |
| 12 | [Q/K/V materialize · QK assemble](operators/mla-materialize.md) | P1 | P2 |
| 13 | [Materialized varlen Attention](operators/varlen-attention.md) | P0 | P0 |
| 26 | [Fused iHC pre](operators/ihc-pre.md) | P1 | P1 |
| 43 | [CP AllToAll](operators/cp-alltoall.md) | P0 | P2 |
| 50 | [CP ReduceScatter](operators/cp-reducescatter.md) | P0 | — |
| 44 | [EP dispatch / combine](operators/ep-dispatch-combine.md) | — | P0 |

“—”表示该 profile 未将该项列为 P0–P2，不表示没有实现或不用验证。
两处“基础 Tensor 原语 / 优先排查”保存在来源清单，但不是单一 P0–P2 目标，未偷偷扩入范围。
本页没有将 DSA 六叶算子单列 P0–P2；此前 DSA 实验不能替代这里的 MLA/FA4 验收。

## 来源与版本边界

- [source-inventory.json](source-inventory.json)：从分享页 JSON 抽取的优先级、原因、行号和来源提交。
  不复制巨大 trace、内部运行地址或凭据；保留原始命名（页面的 E98 内部 title 有旧标签，以切换页 scope 为准）。
- [catalog.json](catalog.json)：机器可读规范，是九份算子文档的唯一编辑来源。
  包含行为、输出、覆盖轴、oracle、已有入口、性能边界、负对照、独立确认和缺口。
- `operators/*.md`：由目录生成的逐算子阅读版，带固定提交的源码链接。

| 资料 | Core SHA | Megatron SHA | 可以说明什么 |
|---|---|---|---|
| E65 页面记录 | `d2d47877c96b859224d91a184280eb1fde7236ad` | `20052e11f2dbb566e98e821f66bb21a38edec36e` | 历史 profiling 归属/排序；不能证明无运行时补丁 |
| E98 页面记录 | `803a32def3e1e38c91438bfda0ee2bd4d665f288` | `25b0994c29f1b64f7871eea91c77398605dc94c1` | E98 配置与 trace 的归属依据 |
| 本次源码入口审计 | `415c21375d871c2fd475cba37891c6cb65c658dc` | `218adfeec230963181ef994056f37d0944a1aee1` | 25 个文件内复用入口/符号的静态存在性，非历史或当前 GPU 通过 |

源码审计版本来自本地可读快照，不声称是两个生产仓库最新 main。采用不同 baseline 时，
重新核查真实 `submodules/apodex-core` pin、导入路径、所有变更及适用测试。
旧 gate 通过记录不能跨源码/配置直接沿用。

## 两个 profile 分开建立任务

| 参数 | E65 | E98 |
|---|---|---|
| 全局序列长度 | 256K（262144） | 32K（32768） |
| 原 profiling 设备 | 256 × B300 | 256 × B300 |
| CP / EP / TP / PP | 32 / 32 / 1 / 1 | 4 / 32 / 1 / 1 |
| GBS / GA | 64 / 8 | 256 / 4（MBS1） |
| 重计算 | 无 | 无 |
| force balance | 开 | 开 |

E65 页面还介绍 2048 卡生产配置及前 5 层 block recompute；不能用该说明覆盖 E65 的运行配置。
E98 的 weight cache、deferred wgrad、RMS Graph 等开关须从实际生效配置绑定；RMS Graph
不表示所有算子都在 Graph 内。拟改变开关需独立对照，不把算法改变和 kernel 优化混成一个变量。

页面没有给出完整 tensor shape、stride、CPU sequence number 或 Python 栈。
不能从“256K/CP32”直接将每个算子的 M 都写成 8192：packing、head/latent、专家 padding、
GA 和具体调用层会改变实际形状。接入时一次捕获下列 manifest，之后复用：

- 真实模块/调用点、前反向角色、全局和 local shape、dtype、stride/offset、alias、设备；
- packed `cu_seqlens` / 文档长度、有效 token、mask、head/latent 维度；
- GEMM M/N/K、专家 counts/padding/top-k、每 rank 分配与实际同步/累加行为；
- 解析后的 flags、镜像 digest、CUDA/Torch/Triton/FA4/DeepGEMM/DeepEP/NCCL 版本及数学开关；
- shape 频次权重的来源；缺权重时按场景分别报告，不制造“训练加权收益”。

## 公共验收协议

### 1. 一次性接入与冻结

每个 profile × 算子独立建 native 任务，复用已有可信测试。`native.json` 固定来源、
`implementation_paths`、cases/seeds/weights、精度、性能/显存限制、`checks` 和 `confirmation`。
`adapter.py` 导入真实生产 API；声明全部输入、参数、缓冲、允许修改状态，返回前向与所有相关梯度。
源码差分只允许最小实现范围；测试、oracle、阈值和 adapter 不能随候选改变。

开发矩阵按每份设计的覆盖轴展开成**具体有限用例**，至少包括真实热尺寸、合法尾部/布局、
dispatch 边界两侧、容易退化的反例，以及所有已声明的精度/执行路径。非法输入单独检查拒绝行为，
不能把拒绝当作合法输入的计算通过。未参与调参的组合和种子在首次 full 前冻结到独立确认集。

`min_gain`、`max_regression`、`max_baseline_regression` 和 `max_peak_bytes` 由任务按现有契约
明确填写；此目录不从 P0/P1/P2 或 trace 占比推导阈值，不设一个跨算子通用数值容差。
若沿用 evaluator 默认值，也必须在冻结的任务中明确落值，并说明依据。

### 2. 基线资格与验证器可检出性

先复用源码、环境、契约匹配的基线资格证据；缺失时运行独立 baseline 任务的
`native-evaluate --mode full --check-only`。完整正确性专用检查不测性能，不消费确认集。
已有独立数值 FAIL 保留；只允许另外定义未受影响且有独立数学契约的研究子任务。

每类设计提供针对性的错误实现/错误状态作为负对照：应检出错梯度、错布局、错缩放、
漏 rank 或静默 fallback。纯 CPU 静态测试只能证明检查逻辑，不替代真实 GPU 负对照。

### 3. 正确性与集成

原生 full 默认要求 `operator` / `numerical` / `integration` 三类证据。
纯函数数值门放在 core 的 `verification/operations`；模块/collective 数值与训练接线放在
Megatron `tests/unit_tests/apodex` 等原生层。引用 module gate 时用受信 pytest wrapper 产生
明确报告；不能把 Megatron registry 名称直接交给只认识 core registry 的 gate 命令。

复用独立 naive reference 与 FP64 golden，并核对实际计算/输出精度；保留 U/z、bitwise、
确定性和 batch-invariance 的原约束。baseline parity 与独立数值通过分别报告。
每份报告核对本轮源码、预期 node、参数集合、非零数量和所有必需 rank；意外 skip、缺文件、
超时、旧报告、只通过 CPU 路由 mock、后端未执行均不能算通过。

### 4. 性能 full

复用 evaluator 的两批交错配对、当前 best/固定 baseline 对照及噪声/逐用例退化检查。
每个角色使用等值输入及相同初始状态；梯度累加在定义好的序列内部保留、序列之间复位。
编译和预热单列；真实调用中的转换、清零、分配、同步与异步完成计入对应计时边界。

目标 kernel 可以是 objective，完整 API 为 guard；完整阶段/前反向收益另给结论。
Graph 与 eager 分开。page 的 overlap、时间并集或通信驻留不能直接相加、解释为关键路径，
或当作优化收益上限。训练收益必须来自同配置、未开 profiler 的 step/tokens/s 配对，
不能仅凭叶子微基准或 CI 的防严重退化门宣称。

### 5. 独立确认与交付

固定候选后运行 `native-confirm`：重跑原生 checks、独立输入的正确性与性能；该阶段相对固定
baseline 验证，不虚称在留出集胜过旧 best。每任务只消费一次，失败/中断仍保留消费记录。
后续新版本将已见输入并入回归，预先建立新的确认集，不重复抽样直到通过。

通过后 `native-promote` 深核验并写 best，Agent 创建/更新实现 PR，再由 `manage sync-pr`
绑定同源码交付。普通验收不自动成为 [4h/8h 策略](../../docs/bounded-research-policy.md) 的成功：
该策略还要求其固定收益门槛、独立 native delivery 绑定及当前 PR 证明。

## 通信验收与框架边界

CP32、EP32 必须使用真实目标 process group 和拓扑；小规模测试只用于定位错误。
不同卡的时间戳不能直接求全局区间，使用各 rank 的局部 elapsed、同步定义和最慢 rank 汇总，
保留每 rank 样本。所有 rank 的退出、超时、NCCL 错误和清理都必须进入总结果。

本项目 Manager 只能锁定同一主机的 GPU 组，不是跨节点调度器。现有 native adapter 本身也
不会自动把单进程 Workload 变成 CP32/EP32。通信任务的最小接入设计如下：

1. 在 Megatron 的受信测试层实现/扩充目标模式的分布式 correctness + timing wrapper，
   由平台 launcher 启动独立 rank；测试 worker 不能放进可变候选目录。
2. wrapper 接收固定 run ID、角色源码、输入清单和种子；按同一角色/批次顺序执行，
   每 rank 写唯一报告与原始样本，聚合器检查**完整的预期集合**，不取“已有报告”的交集。
3. 可由单机 native `checks` 运行的子集继续用既有 native 门；跨节点完整链用受信 verifier
   通过 `watch-assessment` 产生明确 scope 的外部验收。外部 PASS 不修改 native best。
4. 若要求正式 native best/自动策略成功，必须再实现并验证完整分布式任务与 native evaluator
   的角色/样本/状态映射。当前目录不提供虚假的单卡 adapter 或跳过集成的 `native.json`。

这项接入边界对 CP AllToAll、CP ReduceScatter、EP dispatch/combine 都适用；Grouped GEMM
的单卡 kernel 可以先正式验收，EP32 整链仍须另有证据。现有 CP2/CP4 **reuse** 测试不能当
materialized CP32 证据；CP ReduceScatter 在归因之前保持“未就绪”。

## 接入顺序和复用关系

- Dense、Grouped、iHC：先复用现有数值门与真实模块测试，绑定实际 profile 的输入/flags。
- MLA materialize 与 FA4：共享 packed 输入与完整 MLA guard，各自测目标阶段，避免重复计入收益。
- Router Linear：补目标 FP32 Linear 独立三腿，已有 fused-router/triton 门只作邻近回归。
- CP A2A：补 materialized + 目标 CP32/4 接线；EP：补 EP32 全链参考与拓扑。
- CP ReduceScatter：先定位 E65 实际 producer/consumer/process group，再落地专用 adapter。

正式验收可以复用相同源码/契约/环境的已有结果引用；不同 profile、后端、精度或拓扑不能
互相冒用。基础设施准备和完整验收费用记入资源账本；不在每轮探索重复做接入审计。

## 维护与 CPU 检查

```bash
# 检查全部优先级覆盖、来源一致性和生成文档；不导入生产代码，不申请 GPU
python tools/check_a7b_acceptance.py

# 修改 catalog 后生成九份阅读版
python tools/check_a7b_acceptance.py --render

# 可选：对本地 Git 对象中的固定版本逐文件/符号检查，不执行外部测试
python tools/check_a7b_acceptance.py --core /path/to/apodex-core --megatron /path/to/Megatron-LM

KERNEL_ENVOLVE_AUTO_UPDATE=0 python -m pytest -q tests/test_a7b_acceptance_catalog.py
```

源页面以后更新时，先更新来源快照，再审查每个 profile 的增删与优先级变化；检查器会拒绝
漏项、重复项、错误优先级、把“优先排查”混入 P0–P2、陈旧生成文档和设计被误标为 PASS。
源码检查只证明固定 Git 对象中的入口存在，不证明所有参数已被 pytest 收集或 GPU 可运行。
