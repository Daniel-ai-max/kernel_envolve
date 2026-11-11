---
name: kernel-envolve
description: 持续优化 apodex-core 与 Megatron-LM 训练算子，复用原生验证，确认收益后自动创建或更新中文 PR，并保存可恢复的优化记录；也支持使用通用 evaluator 迭代独立 CPU/GPU 算子。
---

# Kernel Envolve

使用普通文件、现有 coding agent 和任务 evaluator 完成迭代。自主选择实验路线；目标是正确、快速、适配实际工作负载整个输入范围的算子，每次接受性能结论必须有可复现的验证和测量证据。

默认在目标 GPU 机器上使用 agent 的文件/命令工具，直接调用本地 evaluator；模型服务可在远端。无需独立 CPU 实例或 CPU 准备阶段。第一次实际评测同时检查环境和编译链，失败时根据 CLI 的 error（阶段、编译输出、退出码）修复，不为每轮增加固定环境自检。

内部项目先执行下方「Apodex 内部项目评测」的任务约定；独立算子使用本仓库 evaluator。两类任务共享假设驱动的迭代方式，验收入口分别记录。

## 使用前同步工具

每个新会话或管理轮次开始时运行 `kernel-envolve update`。读取返回的 `skill` 文件及其相对引用，本轮所有工具命令使用返回的 `executable` 绝对路径。已固定版本的会话会返回同一个提交；不要在 full 评测与 promote 之间切换版本。管理器的恢复提示也会提供该轮的 Skill 和入口。

复制安装的本文件作为发现入口，以命令返回的 Skill 为本轮流程来源；返回路径就是当前文件时继续阅读即可。首次使用需安装包含 `update` 命令的工具版本。若返回 `disabled`，按本地仓库流程执行；若首次离线且没有缓存入口，本轮通过 `KERNEL_ENVOLVE_AUTO_UPDATE=0 kernel-envolve ...` 使用已安装版本，下轮再检查。不要在任务工作区执行 `git pull` 或覆盖 Skill 文件来升级工具。

## 持续优化

优化任务默认持续推进。一次晋升、达到阶段目标或若干轮没有收益，都只结束当前实验，不结束优化任务。在用户限定的算子范围和资源额度内自主选择下一项实验；用户指定一次性任务时按其范围执行。

### 如何持续找到优化点

维护一个简短假设队列，按预期收益、现有证据和验证成本选择下一项，不机械轮询所有方向：

| 方向 | 关注点 |
|---|---|
| 算法与计算组织 | 数学等价的分解、归约树、分块、并行划分，消除重复计算 |
| 内存与硬件利用 | 访存合并、向量化、布局、缓存、寄存器/共享内存、流水线与目标硬件指令 |
| 融合与调用开销 | 中间张量、launch、Python dispatch、分配和同步；检查融合对占用率的影响 |
| 反向与训练状态 | dgrad/wgrad、saved tensors、重计算、梯度累加及更新，评估前反向总成本 |
| 不同输入区间 | 小尺寸、长上下文、非对齐布局、MoE 负载、dispatch 阈值及 fallback |
| 模块与执行方式 | 生产调用链、Graph、通信 overlap，以及与相邻算子的组合机会 |

从实测和源码找当前瓶颈，再提出能被实验否定的假设。已有方向连续缺乏收益时，重新分析瓶颈，转向另一种机制、输入区间或参考实现；不要只改几个 tile 参数重复碰运气。有晋升后重新判断剩余瓶颈，不沿用已经失效的判断。队列为空时，从逐 case 结果、必要的 profiling 和参考实现中生成新假设；某个尺寸接近硬件下限时，检查同一任务内其他尺寸、反向或调用开销，不据此宣布整个算子已最优。必要时读取相关硬件文档、新参考实现或有价值的失败原型，不每轮重读全部资料。

同一假设记录机制、代码起点、参数、输入范围、结果与失败原因。源码、参数、环境和评测契约相同的实验不重复提交；只有测量故障、预先约定的重复测量或新证据能说明必要性时才重跑。可以从尚未晋升的原型探索组合，但最终必须与当前 best 和固定原始基线比较。切换到其他算子或改变训练语义须属于用户已授权的范围。

### 状态保存与恢复

沿用任务的 `notes.md` 和已有 run 目录，不另建知识库。每次实验结束、交付阶段版本或即将交接时保存：

- 当前状态（进行中、等待资源/修复、用户暂停）、任务范围、评测契约版本、累计资源消耗及剩余额度。
- 原始基线、已验证 best、正在探索的 candidate 的源码位置/哈希和证据；失败原型保留可定位记录。
- 每个目标仓库的 PR URL/编号、目标分支、工作分支、最后发布的 commit 与评测 run；尚未发布的进展及原因。恢复后查询远端 PR 状态，不能只凭本地记录判断是否已合入。
- 已尝试机制及结论、下一项假设和可执行命令、仍在运行的 job/run ID。下一步应具体到改动或诊断，不只写「继续优化」。

恢复时先读这些记录、核对工作区及已有作业，收集未完成实验的结果，然后继续队列；不重建基线、不重复提交仍在运行的作业。环境或源码依赖改变时才重验受影响证据。中断实验标记未完成，不算失败或成功。

阶段交付后保留 best 并继续下一项实验。单轮执行时限或上下文交接只触发保存和续接；累计预算/资源授权用尽、外部故障或用户暂停则保存原因并进入等待，不宣称优化完成，也不超额消耗或反复提交失败作业。等待 GPU 时可继续有价值的源码分析；没有可推进工作时释放闲置资源，由执行器在条件满足后续接。用户暂停后仅在用户恢复时继续。

skill 规定决策与恢复流程；跨会话持续运行还需要宿主执行器唤起下一轮。可使用本仓库 `manage run` 管理层连接 coding-agent 命令，自动续接任务、检查累计额度并串行化 GPU 作业，参见 `docs/optimization-manager.md`。将「本轮返回」与「优化任务结束」区分；不能只靠提示词中的「不要停止」实现后台运行。

当环境包含 `KERNEL_ENVOLVE_ROUND` 时，当前会话属于管理层任务。所有 GPU 命令必须经过 `kernel-envolve manage gpu-run -- <命令>`，包括正确性测试、编译时会访问 GPU 的过程和 profiler；不得自行改可见设备、嵌套 gpu-run 或留后台 GPU 作业。普通 CPU 分析可并行。轮次结束前更新 notes.md，并将 hypothesis、outcome、next_action 等研究记录写到 `KERNEL_ENVOLVE_REPORT` 指定的 JSON 文件；过程更新可用 `manage report --file`。性能结论只来自 evaluator，报告中不自填 speedup。暂停或预算用尽不绕过执行器续跑。

### 持续搜索中的 evaluate

每次改动先获得最便宜的有效反馈，有希望的候选才做完整确认；无源码/契约/环境变化时不重复完整验证。best 只由满足正确性、收益和退化要求的证据更新，无收益轮次保留 best 并继续探索，不降低门槛来制造进展。

长期搜索保留全部尝试，按事先确定的实验批次和确认次数评测；完整确认失败后不能反复抽样直到偶然 PASS。交付使用未参与选择的新输入组合与新计时样本；已经用于修改候选的留出样例转入回归集合，不再声称其未见。新发现的失败永久保留并版本化扩展验证，原有覆盖不得缩水。持续报告相对固定原始基线及当前 best 的实际结果。

## 开始一项任务

- 明确调用语义、输入取值域（如正数、合法索引）、shape/dtype/layout/alias、允许的误差与原地更新、实际调用入口、目标硬件、测量范围和资源预算。换值验证仍须遵守取值域，通过任务输入工厂生成，不统一替换成 randn。缺少真实 workload 权重时，明确声明 benchmark 目标。
- 默认交付通用调用入口：明确 size/batch/上下文长度、dtype、布局等实际变化范围，不能为了取得加速缩成几个固定 shape。用代表点、分块/dispatch 阈值附近、非对齐尺寸和关键交叉组合构造覆盖；少量 case 用于 quick，广域 case 放入 full。读取 README 的覆盖示例，按实际算子选维度。
- 对每个输入独立变化布局：列 stride、行 stride/leading dimension、转置或 storage offset 按实际支持范围取样。至少检查实际会出现的行间 padding 和输入间不同 stride；不能仅用一个全局 stride 同时构造所有输入，再据此宣称布局通用。
- 可以按 shape/dtype/layout 选择不同 kernel，并为未命中优化分支的合法输入保留通用实现；测试完整入口、每个分支、切换邻域与 fallback 性能。尺寸变化应自动适配，不要求用户手调。总体收益之外检查各区间的最差退化，不能只报告赢的 shape。
- 对融合、归约次序、近似和精度变化，确认 oracle 的独立性及任务要求的误差/漂移/梯度检查；普通 allclose 不替代已有精度 gate。用已知正确和已知错误实现检查新增验证器。
- 读取当前任务、Git 状态、已有 best 和最近失败记录，避免覆盖未完成工作或重复实验。已有远程作业先查 job 状态，再决定是否提交。
- 区分固定 oracle、固定原始 baseline、已验证 best 和工作中的 candidate。优化候选不能修改 oracle/覆盖/精度来获取 PASS；契约确需改变时另建任务版本。
- 先确认实际环境和原始实现可运行。硬件型号、CUDA/PyTorch/Triton 版本以机器为准。按需读取 [硬件和参考实现入口](references/hardware.md)，不要把所有资料一次塞进上下文。

## 每轮实验

1. **假设。** 用一句话说明要解决的瓶颈和预期影响到的 case，例如减少中间张量、改善访存、消除同步或改变分块。记录代码起点；它可以是未晋升的原型。
2. **最便宜的有效反馈。** 尚未写完的结构先做编译/局部检查；正确性失败先重现对应 case/seed；参数调优可以成批探索。进入性能比较前先通过任务正确性验证。
3. **选择反馈范围。** 用代表用例筛选候选；若 quick 耗时接近 full，先按阶段定位并缩短重复准备和验证开销。只有候选已有足够依据且本轮预算允许时才直接 full。内部项目组合原生定向测试和 benchmark；通用 quick 自带正确性检查，无需先单独 check-only。只测候选的 signal 可以辅助选参数，但不能算 speedup。保留失败机制有价值的原型，不机械回滚所有未晋升的探索。
4. **确认。** 有希望且可交付的候选做完整评测（内部项目使用下文原生验证，独立任务使用 full），与当前 best、固定 baseline 同轮实测。检查各尺寸/布局区间的覆盖、逐 case 增量与累计回退、fallback 表现、噪声、目标权重和计时范围；有疑问时查看原始样本，不自行改 PASS/分数。仅为解释瓶颈时才跑 profiler。
5. **晋升、发布并继续。** 完整证据满足任务目标后更新已验证版本；独立任务使用 promote，内部任务使用 native-promote 保存已验证源码快照及原生证据。有明确收益时按下方流程自动创建或更新 PR，再继续探索。更快但未确认的代码仍是 candidate；正确但更慢的实现可以留下供组合。记录本轮假设、改动、run 路径、PR、结论与下一步，维护一个简短 `notes.md` 即可。

按 `phase_wall_seconds` 记录每轮成本。内部和通用 CLI 都支持 `--case/--seed`；已有正确性失败用 `--check-only` 定向复现，正常候选直接跑选定 case 的 quick，避免再重复一遍相同检查。内部 quick 默认 120 秒执行预算，full 默认 1800 秒；首次冷编译确需更久时显式指定 `--timeout` 并单独记录，后续超时先诊断瓶颈，不直接扩大每轮预算。

`snapshot` 或文件交换占主导时，将固定源码、候选工作树、任务和 run 目录一次性准备到 GPU 节点本地盘；后续只更新实际改动，不每轮从 FSx 重建整个项目。仍冻结并完整哈希每轮源码，不用跨角色硬链接或 mtime 缓存代替。每次评测结束将新证据及 manager 一致性备份保存到持久盘，确认保存成功再回收本地资源。导入/编译高时复用工具链按源码和配置区分的缓存；正确性高时缩小诊断集合；计时高时检查输出传输和 workload 范围。探索 trial 数在任务开始时约定，不因本轮结果缩减正式覆盖。失败 CUDA 进程不得复用，正确性 PASS 不跨候选沿用。

噪声导致的 inconclusive 表示收益尚未确认，不等于候选更慢。保留有希望的候选并检查测量环境；连续 inconclusive 不能单独作为达到性能极限的证据。

一次有效完整评测确认当前源码的固定开发覆盖；晋升后继续选择新假设。某个方向收益停止时切换方向，阶段目标达成后重新分析剩余瓶颈；资源等待和状态续接按「持续优化」执行。阶段交付仍需下面的未见尺寸验证。

正式确认前确定预算和门槛，保留全部尝试；不要反复抽样直到偶然赢。协议有缺陷时记录原因、版本化修正并重测受影响对照，不能观察到结果后降低门槛。相同源码不作为新的改进，注释或重命名也不算机制进展。

## 收益确认后自动维护 PR

PR 交付由 agent 使用 Git 和 GitHub 工具完成。用户启动此优化流程即包含创建及更新优化 PR，无需每轮再次确认或等用户手动汇总；遵守用户明确指定的其他交付方式。evaluator 负责验证与晋升，不持有 GitHub 凭据或负责发布。使用 `gh` 时显式指定仓库及 PR 编号，中文正文先保存为 UTF-8 文件，再通过 `--body-file` 创建或更新，保留 Markdown 格式。

### 何时提交与如何续更

1. **确认收益即发布。** `full` 满足既定正确性、收益、噪声和逐 case 退化要求，且完成本阶段适用的未见输入验证后，提交已验证的实现与必要测试，创建可评审 PR。不等整个算子“优化完”。quick 的优势、未确认的波动和相同源码不触发收益发布。
2. **同一算子复用 PR。** 按目标仓库、算子任务、目标分支和工作分支查找并核对远端 PR；同一仓库、算子任务和目标分支只维护一个未合入 PR；核对任务归属，不接管他人的同算子 PR。已有 open PR 就向原分支追加提交并更新标题、正文及收益表，不另开 PR，不只在评论里追加新数据。发布前重新核对状态，防止期间已合入；请求超时先查询是否已成功，避免重复创建。
3. **保持可评审状态。** PR 分支只保留已验证进展和必要修复，后续探索在独立工作区/分支进行。新分支默认使用 `codex/` 前缀。推送前核对 PR 中可执行代码及依赖与本轮评测源码一致，排除无关改动和实验残留；冲突解决或其他行为修改后重验受影响项。发布后核对远端 head、正文与检查状态，处理属于本改动的 CI/评审问题；等待评审期间继续探索。
4. **处理生命周期。** PR 已合入，从最新目标分支建立下一阶段工作分支，核对生产基线后为下一次明确收益创建新 PR，并关联上一阶段。PR 关闭未合入时先核对原因；允许继续且修复关闭原因后优先重开原 PR，不绕过拒绝意见或用户暂停。发布失败写入 `notes.md`，恢复发布时复用仍有效的证据，不因此重跑 GPU 实验。
5. **跨仓库分别交付。** kernel 与算子测试向 apodex-core 提 PR；确需 Megatron 调用接线或 pin 更新时，另提关联 PR，每仓库沿用同一个未合入 PR。依赖尚未就绪的 Megatron PR 可先用 draft 说明依赖，core 合入后更新为真实 main commit 并验证接入。合并沿用仓库评审与队列流程；不为提交 PR 额外重跑只由 merge queue 执行的整套 CI。宿主支持 PR 附件时同时关联到当前任务。

### 中文描述与收益口径

标题和正文使用中文，保留 API、算子名等必要技术标识；兼容目标仓库已有模板。标题说明算子和主要改动，例如 `perf(probs_scale): 融合反向计算，减少中间张量读写`；数字仅在条件清晰且已验证时写进标题。

正文始终描述当前 PR 的最终方案。更新时重写概述、收益和验证结果，删除已失效结论；保留一行本次新增进展与对应 commit/run，历史细节通过 Git 与证据链接追溯。目标仓库有 PR 模板时保留其必填章节，将下方收益和覆盖信息放入对应章节，不适用项说明原因。没有模板时采用以下结构，发布前替换全部占位内容，删去不适用项。向 kernel_envolve 提交时，使用其 `.github/PULL_REQUEST_TEMPLATE.md`，并运行 `python3 tools/check_pr_body.py --body-file <正文文件>`：

```markdown
## 改动说明

<哪个生产调用存在什么瓶颈；本次如何改善；涉及哪些实现和接入路径。>

## 性能收益

环境：<GPU、镜像 digest、软件版本>。口径：<前向+反向/状态更新、eager 或 Graph、输入覆盖与权重>。
对照：<固定原始基线 commit、上次已发布版本/本轮 best commit、本次候选 commit；对应 run 与源码哈希>。

| 场景 / 尺寸 / dtype / 布局 | 原始基线耗时 | 本次耗时 | 延迟下降 | 加速比 |
|---|---:|---:|---:|---:|
| <主要区间及最差变化场景，耗时注明 µs 或 ms> | <实测> | <实测> | <实测> | <实测> |
| <按任务权重汇总> | <实测> | <实测> | <实测> | <实测> |

本次进展：<相对本轮 best 的增量、它是否为上次 PR 版本；两批结果/波动、最差退化及显存变化>。

## 验证与复现

- 正确性：<前向/相关梯度、数值 gate、布局与状态检查；实际通过数量和结果>。
- 接入：<真实模块分发及适用的并行/Graph/重计算测试；CI 当前状态>。
- 复现：<命令、固定配置、原始测量与报告链接，绑定本次源码版本>。
- 关联：<确有需要时列依赖 PR、前一阶段 PR 和接入步骤>。
```

延迟下降 = `1 - 本次耗时 / 对照耗时`，加速比 = `对照耗时 / 本次耗时`；总体按任务权重汇总耗时后计算，不平均各 shape 的加速比。列出主要输入区间及最差变化，完整逐 case 结果放在可供评审者访问的证据链接中，不能只贴本机临时路径。证据沿用目标仓库已有存放方式，不把大量实验日志和中间产物塞进代码 diff。

累计收益相对固定原始基线；增量收益相对本轮同环境配对测量的 best。若 best 不是上次 PR 版本，明确注明，不能把两轮不同环境的百分比相减冒充新增收益。单算子收益只称算子收益，训练吞吐提升须有实际训练测量。缺失的显存或端到端指标写“未测”或省略，不推算填充；待运行的 CI 明确写待运行，不记为通过。

## Apodex 内部项目评测

### 原生评测是否足够

原生测试是验收组件，不能把「数值 gate PASS」或「CI 性能门通过」直接当成优化成功。每项任务在 `notes.md` 列出「生产要求 → 具体测试/benchmark → 缺失项 → 本轮结果」，补齐缺口后再冻结验收集合。

| 验收项 | 已有能力 | 任务必须核对或补充 |
|---|---|---|
| 数值与梯度 | core 的 candidate/reference/FP64 golden、偏差与误差幅度 gate | 所有受影响输出和梯度、实际 dtype、原有 bitwise 要求；统计 PASS 不等于逐位一致或训练轨迹保证 |
| 接口与执行语义 | 各算子的专项 pytest，部分已覆盖布局、设备、Graph、重算和累加 | shape/dtype/device、输入修改、alias、动态换值和所用执行模式；不是每个算子都有同等覆盖 |
| 输入泛化 | 注册的固定 fixture 与测试参数集合 | 生产尺寸/布局/数值分布、阈值邻域、fallback 和未参与调参的组合；增加 seed 不增加尺寸覆盖 |
| 性能收益 | core 微基准、Megatron 配对计时与性能回归门 | 原始实现及当前 best 的配对测量、噪声判断、逐场景退化、F+B 和显存；防止明显退化的 CI 门不能判定优化收益 |
| 训练接入 | 模块三路对照、专项并行/重算测试、训练指标与恢复检查 | 候选路径确实执行、受影响的训练配置及状态；DP 分摊采样不是 TP/EP 测试，loss/grad-norm 相等不是全部梯度相等 |

源码与复现依据：

- core 的 `_draw_on` 按 candidate → reference → golden 顺序传入同一输入，数值计算允许广播并转换到 FP64，不统一检查 API 契约。对约定保持输入/shape/dtype 的恒等算子，实际调用 `gate.run`（原配置 R_min=4096）时，错误 shape、错误 dtype、非法输入清零均 PASS；正确实现 PASS，明显数值错误和 NaN FAIL。这是**数值方法单独使用时的覆盖缺口**，不代表专项 pytest 或完整 CI 也放行。见[故障复现脚本](../../audit/probe_native_gate_contract.py)与[实际结果](../../audit/native_gate_contract_probe.json)。
- `shift_input` 会对浮点输入加偏移；本次行 padding 样例的 stride 从 `(11, 1)` 变成 `(4, 1)`。布局覆盖要在真正调用候选时确认，不能只看 `gen_input` 返回的布局。原生 gate 的尾部统计使用分布模型，拟合不足时会记录 empirical fallback；结果说明不得扩大为任意真实训练分布的误差保证。
- `test_fused_swiglu.py` 已有生产尺寸、BF16 位模式、FP16/FP32、特殊值、反向布局及 batch invariance；`test_embedding_wgrad.py` 已有累加、dtype、padding 和未出现 token 行检查；`test_router_gather.py` 已有真实 Graph 换值回放、checkpoint 和 stream 检查。先复用这些，不重新写一套相同测试。
- Megatron `test_deepgemm_perf_gate.py` 的 `MODULE_SPEEDUP_FLOOR=0.85` 用于防严重退化，不证明更快；其模块基准分别生成两份输入，`mod.zero_grad()` 也不清空闭包中输入 `x.grad`。可复用计时方法，正式优化对比需对齐输入/初始状态，并明确梯度累加或重置语义。

`native-evaluate` 已实现源码冻结、隔离执行、输入/输出与状态检查、原生测试报告核验及配对性能判定。任务定制集中在 `native.json` 的覆盖和测试入口，以及 `adapter.py` 的真实调用与状态声明；已有数值方法保持复用。

### 确定任务与环境

先阅读两仓当前的 `DEVELOPMENT_GUIDELINES.md`，从 Megatron 的调用点反查 core 公共入口。kernel 实现在 core，Megatron 只做接线；不在本工具仓库复制生产 kernel 或参考公式。

每项任务只维护一份简短 `notes.md`，写清：

- **版本与入口**：Megatron SHA、`git ls-tree HEAD submodules/apodex-core` 的 pin、实际 core 源码与改动哈希、导入文件路径、后端开关、镜像 digest 和 GPU 型号。开发期可挂载候选 core，但须显式记录；最终验证使用实际交付的组合。
- **工作负载**：从训练配置和调用处确定每 rank 的 local tokens、hidden/head/expert 维度、dtype、stride、packing、梯度累积等；全局上下文长度不等于本地算子输入长度。覆盖常用区间、非对齐尺寸、分支阈值及实际 fallback；MoE 包含空专家和不均衡负载，不能只测均匀分配。
- **验收约定**：相关测试/数值 gate 的名称、quick 代表集合、完整覆盖、性能对照与收益/退化/显存门槛、预算。没有实际频次就分尺寸报告，不虚构训练加权收益。
- **路径与精度**：将默认、显式后端和 fallback 分别对应到实际输入范围、working reference / golden 精度及具名测试。用路由断言确认执行路径；单个后端的长输入 gate 不代表完整默认组合，短输入也不能覆盖长输入才触发的 dispatch。

使用项目当前指定的开发镜像及依赖，不另行升级 torch/TE/FA4/DeepGEMM。首次运行确认导入确实来自候选 checkout，之后仅在版本或环境变化时重查。复用源码和编译配置匹配的扩展/Triton 缓存，只重建受影响扩展。默认直接在 GPU 环境工作，不设独立 CPU 准备阶段。

### 每轮流程与验证范围

| 时机 | 执行内容 | 用途 |
|---|---|---|
| 首次开始 | 原始版本运行选定测试和 benchmark，确认真实调用路径 | 建立可复现对照，排除既有失败 |
| 每轮修改 | 受影响前向/反向的定向测试；通过后测代表尺寸 | 快速淘汰错误或无收益方案 |
| 候选有收益 | 完整相关数值 gate、算子契约测试、固定性能集合及新尺寸组合 | 确认精度、泛化和收益 |
| 交付前 | 实际 Megatron 调用与模块测试；按影响补充执行模式和训练验证 | 确认优化能到达真实训练路径 |

不把整仓 CI、长跑或所有并行组合放入每轮循环。先沿调用链判断影响：纯局部 kernel 优化优先单卡；涉及 TP/CP/EP、通信或 overlap 时增加真实多卡路径。模块 gate 的多 rank 采样加速不等于 TP/EP 覆盖。Graph、recompute、梯度累积和 checkpoint 状态有变化时，分别验证捕获/换值回放、重算梯度、`main_grad` 累加及恢复行为。已有训练 smoke/bitwise/long-run 入口按仓库规范选用，不新建全量训练平台。

确认训练入口实际支持目标并行度、通信后端与每 rank 独立设备；mock 通信、分摊数值采样或单卡训练不作为真实 CP/TP/EP 训练证据。启动器拒绝目标配置时先修复入口，不把较小并行度的通过结果沿用到目标配置。

### 正确性 evaluate

- **core 数值验证**：复用 `apodex_core/verification/operations/local_ops.py` 的 `Case(candidate, reference, golden, gen_input)`。检查前向、输入梯度、权重/偏置梯度等实际输出；沿用原生偏差与误差幅度判据、精度配置和已批准的 WARN 规则，不用通用 allclose 替代。reference 与 golden 必须独立于候选；手写反向不能仅升为 FP64 就充当独立 golden。
- **不同精度契约**：历史严格 FP32 路径与已经声明的 BF16 默认路径分别验收。前者失败不能单独证明后者违约，也不能据此豁免默认路径；各自使用原契约及独立 reference/golden。测试绑定错误需独立评审、另建任务版本，保留原固定失败输入、阈值与报告，并补齐默认路径验证，不能仅切换后端让原测试变绿。
- **接口与覆盖**：结合 `tests/` 检查既有 bitwise、确定性、batch invariance、输入修改、alias、布局、设备/stream 等承诺。正确性 harness 在候选执行前保存原始输入，给对照提供独立但等值的输入/状态；保留每条执行内部应有的 alias/stride。数值转换前检查输出结构、shape/dtype/device，并断言只发生允许的原地更新；随机性任务还要对齐 RNG 状态。数值 gate 的固定 fixture 不代表全部生产尺寸/dtype；缺失覆盖补到算子测试或原生 case builder，不通过缩减支持范围取得加速。新增/调整验证器需用已知错误实现确认能检出目标错误。
- **Megatron 集成**：复用 `tests/unit_tests/apodex/operations/local_modules.py` 与相关模块测试，检查真实模块的 forward、dgrad、wgrad。用调用计数或定向断言确认候选后端确实执行，不能静默 fallback。涉及 microbatch 切分、padding、分片或归约时，核对相同有效 token 的归一化与梯度，防止引入系统性缩放或偏差。
- **新后端**：按 core 规范保留旧后端的执行路径和默认选择，验证旧路径不触达新 helper；性能对照使用改动前源码，不能只与当前代码中「关闭新开关」的路径比较。

例如，在 core 根目录对 SwiGLU 执行已有契约测试和生产数值 gate：

```bash
python -m pytest tests/test_fused_swiglu.py -q --junitxml=swiglu.xml
python -m apodex_core.verification.method.gate --source operations \
  --only local_fused_swiglu_fwd,local_fused_swiglu_dx
```

这两条命令分别覆盖已有单卡算子测试与数值 gate，是验收的一部分；是否足够由上面的覆盖对应关系决定。日常迭代按改动选择具体 pytest node。当前 gate CLI 以 BF16 和默认模型 fixture 构建 case，其他生产 dtype/shape 的覆盖需另外核对。Megatron 的 `test_local_modules.py` 默认 `APODEX_GATE_RMIN=512` 用于开发，正式验证使用当前 CI 配置（本次检查为 4096）。缩小抽样的结果仅作探索，不覆盖正式结果。

### 性能 evaluate

优先复用 core `benchmarks/` 和 Megatron 的模块 benchmark，检查其计时范围后再使用。训练目标以同一实际配置下的完整 forward+backward 为主，单独 fwd/bwd 用于定位；优化器还要覆盖状态更新。保留逐尺寸耗时、原始样本、峰值显存，以及启动/编译与稳态测量的耗时。

1. 对照固定原始实现和当前已验证版本；保持相同输入、权重、上游梯度、精度设置和执行模式。每组对照从等值的 `.grad`、`main_grad`、优化器状态及可变缓冲开始；按生产语义执行累加/更新，在相同边界重置，避免暖机状态泄漏。重置成本是否计入，按实际训练调用事先约定，两边一致。
2. GPU 独占，完成预热后交替 A/B 顺序，保存配对样本。Megatron `_paired_benchmark.py` 可复用其交替顺序逻辑，但还需检查调用者的输入和状态重置。core 某些 benchmark 按后端顺序报告均值，仅适合初筛；正式比较补配对测量，收益小于波动时记为未确认。
3. 计入实际调用中的 dispatch、布局转换、临时分配和必要同步。Graph 与 eager 分开评测，Graph 要覆盖重放时输入换值；不能把 CUDA event 一概解释为纯 kernel 时间。import 时读取的后端环境开关用新进程做对照，不能在导入后修改变量冒充 A/B。
4. 同轮确认候选与对照；需要不同源码进程时交替启动并保留各自版本证据，不在一个解释器里混载两份 `apodex_core`。通信/overlap 的收益在实际模块或 step 上测，局部同步微基准只用于诊断。

显存增加可能抵消融合收益，特别是 autograd saved tensors 与重计算。需要证明训练加速时，复用 Megatron `.ci/mfu_bench/` 的对应 recipe，比较相同配置的稳态 step time、tokens/s 和峰值显存；排除编译、预热与 profiler 窗口。mock/均衡路由结果单独标明，不作为实际数据分布下的吞吐。

### 结果与交付

原生命令直接运行即可，记录命令、cwd、必要环境变量、退出码、耗时和原始日志；pytest 保存 JUnit，gate 保存本轮实际算子记录。分别记录算子正确性、性能、集成验证的通过/失败/未执行，不合成一个含糊的 PASS。

验证结果须属于本轮源码：核对预期测试名称、非零通过数量和应有 rank 报告；意外 skip、缺报告、fallback 或旧记录不算通过。core `gate_history.json` 会保留以前的算子条目，只读取本轮预期名称且核对时间和版本；`--emit-observed` 是内部子进程接口，会跳过最终判定，不能作为验收命令。复用 Megatron `.ci/assert_no_pytest_skips.py` / `.ci/check_junit_results.py` 时仍保留 pytest 的退出码。

内部任务使用 `native-evaluate/native-confirm/native-promote`，best 指向 `.kernel_native/` 内冻结的源码与完整证据；后续比较从快照加载，修改工作区不会改变旧 best。通用 `evaluate/promote` 保持独立，不承担内部训练验收。

交付按 core 改动 → Megatron 接线和 pin 更新的依赖顺序进行；按两仓当前规范接入相关 CI。最终 pin 必须已进入 core main，核对实际导入路径与接线结果；新特性的 sweep 配置按 Megatron 规范更新。常规 GPU merge-queue 门禁仍由队列执行，review 不要求提前重跑整套 CI。

源码核对基准（2026-11-10）：Megatron `eb2c08444c46ad7bfca0f51c7554970cab6edf48` 的 core pin 为 `0b8b9b8da1624f8f994e4e54c9126f22c3b00361`。后续任务读取实际 checkout 的规范和入口：

- [core 开发规范](https://github.com/ApodexAI/apodex-core/blob/0b8b9b8da1624f8f994e4e54c9126f22c3b00361/DEVELOPMENT_GUIDELINES.md)、[数值 gate](https://github.com/ApodexAI/apodex-core/blob/0b8b9b8da1624f8f994e4e54c9126f22c3b00361/apodex_core/verification/method/gate.py)、[前反向 benchmark 示例](https://github.com/ApodexAI/apodex-core/blob/0b8b9b8da1624f8f994e4e54c9126f22c3b00361/benchmarks/bench_fused_hc_post.py)
- [Megatron 开发规范](https://github.com/ApodexAI/Megatron-LM/blob/eb2c08444c46ad7bfca0f51c7554970cab6edf48/DEVELOPMENT_GUIDELINES.md)、[模块 gate](https://github.com/ApodexAI/Megatron-LM/blob/eb2c08444c46ad7bfca0f51c7554970cab6edf48/tests/unit_tests/apodex/test_local_modules.py)、[配对计时](https://github.com/ApodexAI/Megatron-LM/blob/eb2c08444c46ad7bfca0f51c7554970cab6edf48/tests/unit_tests/apodex/_paired_benchmark.py)

## 使用内部 evaluator

从 [probs_scale 示例](../../examples/apodex_probs_scale) 建立任务，在项目指定镜像内运行。先固定原始 core/Megatron checkout，配置候选 checkout，调整生产输入与测试集合；不要把 eager reference 当作原始性能基线。

镜像从目标 Megatron 版本的 [`.ci/dlc/targets.yaml`](https://github.com/ApodexAI/Megatron-LM/blob/eb2c08444c46ad7bfca0f51c7554970cab6edf48/.ci/dlc/targets.yaml) 解析：HyperPod B300 使用 `aps2-b300` 的完整 image digest，并沿用该仓 `.ci/dlc/_image_env.sh` 的 Python/依赖初始化。镜像提供工具链，生产源码按固定 commit 单独准备；不要假设镜像已安装 apodex-core，也不要随意升级 Torch/Triton 来修补导入错误。

```bash
kernel-envolve native-evaluate --task <任务目录> --case <失败case> --seed <种子> --check-only --timeout 60
kernel-envolve native-evaluate --task <任务目录> --case <代表case> --timeout 120
kernel-envolve native-evaluate --task <任务目录> --mode quick
kernel-envolve native-evaluate --task <任务目录> --mode full --timeout 1800
kernel-envolve native-confirm <完整评测run目录> --timeout 1800  # 已声明 confirmation 的任务
kernel-envolve native-promote <完整评测run目录>
```

- `native.json` 声明源码根目录、生产包与根目录对应关系、输入 case/seed/weight、精度、时间和显存门槛、原生测试。`full` 默认要求 operator、numerical、integration 三类证据；pytest 指定命令、每个 rank 的预期 JUnit 文件、必需测试名称模式及可选 `min_tests`，gate 指定完整算子名称。`devices` 可指定测试使用的可见 GPU 索引，默认单卡使用任务设备。
- `adapter.py` 提供 `create(case, seed, device, oracle) -> Workload`。导入实际生产 API；oracle 使用固定基线源码中的独立参考。`Workload.tensors` 声明所有输入、参数、缓冲和更新状态，`run()` 返回前向、相关梯度等非空 tensor 结构；`mutable` 指定允许更新的状态，`validate(result)` 可断言后端实际执行。不能把依赖的张量状态藏在未声明的闭包中。
- `forward_backward(fn, inputs, grad_outputs)` 直接执行真实 autograd，返回输出及所有 `requires_grad` 输入的梯度。涉及 `.grad`、`main_grad` 或优化器状态时，在适配器中显式执行完整更新并声明/返回相应状态。单次 run 可包含完整梯度累积序列；评测在序列之间恢复初始状态，不截断序列内部的训练语义。
- 输入为目标设备上的 strided leaf tensor，可带 `requires_grad`；保持输入原有 stride/offset 和只读共享存储，复用同一对象换值检测旧值缓存。可写共享存储应声明单一 backing tensor，并在适配器中检查其视图契约；不能把互相覆盖的多个可写 view 当作独立状态重置。
- 各角色在独立常驻进程加载自己的源码快照，同一评测内复用导入和编译缓存；先完成便宜的契约检查，再执行 full 的原生门禁和配对计时。计时采用设备同步后的 wall time，覆盖实际调用中的 host 开销。Graph/多卡语义由显式适配器或原生测试覆盖，不将 eager 测量当作其结果。
- `--case`、`--seed` 可重复指定，仅用于 quick；`--check-only` 保留输入、换值、重复执行及前反向输出检查，跳过性能对照和原生 full 门禁。诊断使用完整冻结配置并记录实际覆盖，不能晋升。计时的每次迭代仍执行状态、运行条件、dispatch 和输出结构检查，仅最后一次将输出与可写状态复制到 CPU 做原有跨进程数值比较。
- 快照包含 Git 跟踪及未忽略文件。所需 Git 子模块另列为源码根目录；被忽略的已编译扩展用 `artifacts` 按根目录声明相对 glob，文件缺失直接报错。准备扩展时只编译受影响部分；不在冻结后改写源码或扩展。依赖镜像、环境与实际导入路径随记录核对，缺 CUDA 不自动退回 CPU。
- 输出结构、shape/dtype/device、只读输入和可写状态分别核验；与固定独立参考比较前向及梯度，必要时启用 bitwise。`required_outputs` 指定必需路径，例如 `output`、`gradients.0` 和 `gradients.1`，防止遗漏训练输出。完整收益判定沿用 `scoring.py`，同时检查原始基线与 best；`max_peak_bytes` 限制 CUDA 单进程峰值分配。跨进程常驻 CUDA context 的总占用不等于这一指标。
- 退出码 0 表示评测执行正确，能否晋升看 `eligible_for_best`；1 为正确性拒绝，2 为执行/配置错误。`native-promote` 重验源码、配置、报告和当前 best，禁止 quick、旧报告、缺测试或过期对照晋升。阶段交付前仍需执行任务之外的新尺寸组合与适用的生产验收；full 覆盖的是所声明的任务集合。

## 使用本仓库 evaluator

读取[仓库 README](https://github.com/Daniel-ai-max/kernel_envolve#readme) 中的任务接口和测量规则；任务由 `task.json`、`task.py`、`candidate.py` 组成。保持 `reference_files` 和 `candidate_files` 依赖清单完整，本地 helper 使用相对导入。

```bash
kernel-envolve evaluate --task <task目录> --case <失败case> --seed <种子> --check-only
kernel-envolve evaluate --task <task目录> --mode quick
kernel-envolve evaluate --task <task目录> --mode full
kernel-envolve promote <full输出的run目录>
```

- 先确认部署是 eager 还是 CUDA Graph。内置 CUDA event 测量可能包含 host 发射间隙，不能视为纯 kernel 时长；Graph 目标使用覆盖捕获、回放和换值语义的专用 evaluator，不混用 eager 分数。
- 在 task.json 明确 timer 和是否要求 deterministic；初始化精度/线程等公共设置放在 task.py，不让 candidate 持久改动其他实现的运行条件。误差敏感任务可参考仓库 reduction 示例增加独立 golden gate。
- 扩大正确性覆盖时可将补充 case 设为 `measure: false, quick: false`，省去反复计时；它仍是 full 的硬门槛。性能集合必须覆盖关键区间、dispatch 边界和 fallback，不能借此移除慢 case。分别查看 CLI 的 `coverage.correctness_cases` / `coverage.timed_cases`，未计时区域不能宣称已验证加速。
- quick 永不晋升；full 合格由 evaluator 计算。失败应修复或诊断，不能编辑结果绕过检查。
- 对照对象 best 来自保存的快照，不从可变工作目录读取。full 自动核对原始 baseline；`--baseline` 可单独诊断原始实现，不能用它绕过当前 best。
- 不在候选里按固定随机种子、输入地址、评测次数或评测状态投机。允许按公开 shape/dtype/layout 做合法 dispatch。
- 输出 alias/layout 等语义写入 task.validate。涉及 Graph、多流、通信、复杂 backward 或显存约束时，采用覆盖这些语义的专用 evaluator；明确记录实际验证和测量范围。
- 性能测量串行占用目标 GPU，避免与其他 profiling/benchmark 并发。非法访问、编译/运行异常后使用新进程。

## 阶段交付

内部任务在注册前用 `native.json` 的 `confirmation: {cases, seeds, trials}` 冻结独立确认集。完整评测合格后，经 manager GPU 队列执行 `native-confirm`，再执行 `native-promote`；确认复用全部原生门禁与阈值，独立比较固定基线。确认集每任务消费一次，失败、中断或换候选不能重用；下一阶段把已见输入转入开发回归，建立含新确认集的新任务契约。收据与配置见 [独立确认](../../docs/native-harness.md#独立确认)。未声明该字段的旧任务不具有独立确认收据。

冻结本阶段候选，在实际支持范围内用未参与调参的尺寸/维度组合和数值种子验证正确性与性能，覆盖各分支和阈值邻域，保存 case、采样规则、最终源码哈希和结果。仅换 seed 不算尺寸泛化。常规 full 只覆盖开发 case；内部任务使用上述独立确认，其他任务使用独立验证任务或项目已有 evaluator。发现失败就修复并验收最终版本，不删除失败 case。此项在阶段交付时执行，无需加到每轮循环；交付完成后按检查点继续下一项实验。

确认最终导入/调用入口实际使用已验证代码及 helper，独立任务通过 `load_best(task目录)` 加载 `best.json` 指向的 source；内部项目按已验证的 core 修订与 Megatron pin 交付，并确认 dispatcher 实际调用了候选。交付对应源码、运行命令、证据路径、逐 case 结果和简短结论；若只有失败或未确认候选，保留原 best 并明确说明本轮发现。
