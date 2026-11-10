---
name: kernel-envolve
description: 优化 apodex-core 与 Megatron-LM 中的训练算子，复用原生数值验证、前反向性能基准和模块测试；也支持使用通用 evaluator 迭代独立 CPU/GPU 算子。
---

# Kernel Envolve

使用普通文件、现有 coding agent 和任务 evaluator 完成迭代。自主选择实验路线；目标是正确、快速、适配实际工作负载整个输入范围的算子，每次接受性能结论必须有可复现的验证和测量证据。

默认在目标 GPU 机器上使用 agent 的文件/命令工具，直接调用本地 evaluator；模型服务可在远端。无需独立 CPU 实例或 CPU 准备阶段。第一次实际评测同时检查环境和编译链，失败时根据 CLI 的 error（阶段、编译输出、退出码）修复，不为每轮增加固定环境自检。

内部项目先执行下方「Apodex 内部项目评测」的任务约定；独立算子使用本仓库 evaluator。两类任务共享假设驱动的迭代方式，验收入口分别记录。

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
3. **选择反馈范围。** 用代表用例筛选候选；若快速与完整评测耗时接近，直接完整评测。内部项目组合原生定向测试和 benchmark；通用 quick 自带正确性检查，无需先单独 check-only。只测候选的 signal 可以辅助选参数，但不能算 speedup。保留失败机制有价值的原型，不机械回滚所有未晋升的探索。
4. **确认。** 有希望且可交付的候选做完整评测（内部项目使用下文原生验证，独立任务使用 full），与当前 best、固定 baseline 同轮实测。检查各尺寸/布局区间的覆盖、逐 case 增量与累计回退、fallback 表现、噪声、目标权重和计时范围；有疑问时查看原始样本，不自行改 PASS/分数。仅为解释瓶颈时才跑 profiler。
5. **晋升或继续。** 完整证据满足任务目标后更新已验证版本；独立任务使用 promote，内部项目保留对应 Git 修订及验证记录。更快但未确认的代码仍是 candidate；正确但更慢的实现可以留下供组合。记录本轮假设、改动、run 路径、结论与下一步，维护一个简短 `notes.md` 即可。

按阶段记录实际成本，通用 CLI 可读 phase_wall_seconds 和 worker_process_wall_seconds：setup 高时先区分导入与编译成本：减少无用启动，重复编译则优先使用工具链已有的、按源码和编译配置区分的缓存；correctness/measurement 高时用原生测试选择器（通用 CLI 用 --case/--seed）缩小探索范围。探索 trial 数在任务开始时约定；不因本轮结果修改契约或缩减正式 full 的固定覆盖。不要为省启动成本复用失败的 CUDA 进程，也不要跨候选复用正确性 PASS。

噪声导致的 inconclusive 表示收益尚未确认，不等于候选更慢。保留有希望的候选并检查测量环境；连续 inconclusive 不能单独作为达到性能极限的证据。

预算用完、收益停止或目标达成时收尾，不为凑流程启动新实验。一次有效完整评测已确认固定开发覆盖；无需无条件重复相同验证。通用算子的交付仍需下面的未见尺寸验证。

正式确认前确定预算和门槛，保留全部尝试；不要反复抽样直到偶然赢。协议有缺陷时记录原因、版本化修正并重测受影响对照，不能观察到结果后降低门槛。相同源码不作为新的改进，注释或重命名也不算机制进展。

## Apodex 内部项目评测

### 确定任务与环境

先阅读两仓当前的 `DEVELOPMENT_GUIDELINES.md`，从 Megatron 的调用点反查 core 公共入口。kernel 实现在 core，Megatron 只做接线；不在本工具仓库复制生产 kernel 或参考公式。

每项任务只维护一份简短 `notes.md`，写清：

- **版本与入口**：Megatron SHA、`git ls-tree HEAD submodules/apodex-core` 的 pin、实际 core 源码与改动哈希、导入文件路径、后端开关、镜像 digest 和 GPU 型号。开发期可挂载候选 core，但须显式记录；最终验证使用实际交付的组合。
- **工作负载**：从训练配置和调用处确定每 rank 的 local tokens、hidden/head/expert 维度、dtype、stride、packing、梯度累积等；全局上下文长度不等于本地算子输入长度。覆盖常用区间、非对齐尺寸、分支阈值及实际 fallback；MoE 包含空专家和不均衡负载，不能只测均匀分配。
- **验收约定**：相关测试/数值 gate 的名称、quick 代表集合、完整覆盖、性能对照与收益/退化/显存门槛、预算。没有实际频次就分尺寸报告，不虚构训练加权收益。

使用项目当前指定的开发镜像及依赖，不另行升级 torch/TE/FA4/DeepGEMM。首次运行确认导入确实来自候选 checkout，之后仅在版本或环境变化时重查。复用源码和编译配置匹配的扩展/Triton 缓存，只重建受影响扩展。默认直接在 GPU 环境工作，不设独立 CPU 准备阶段。

### 每轮流程与验证范围

| 时机 | 执行内容 | 用途 |
|---|---|---|
| 首次开始 | 原始版本运行选定测试和 benchmark，确认真实调用路径 | 建立可复现对照，排除既有失败 |
| 每轮修改 | 受影响前向/反向的定向测试；通过后测代表尺寸 | 快速淘汰错误或无收益方案 |
| 候选有收益 | 完整相关数值 gate、算子契约测试、固定性能集合及新尺寸组合 | 确认精度、泛化和收益 |
| 交付前 | 实际 Megatron 调用与模块测试；按影响补充执行模式和训练验证 | 确认优化能到达真实训练路径 |

不把整仓 CI、长跑或所有并行组合放入每轮循环。先沿调用链判断影响：纯局部 kernel 优化优先单卡；涉及 TP/CP/EP、通信或 overlap 时增加真实多卡路径。模块 gate 的多 rank 采样加速不等于 TP/EP 覆盖。Graph、recompute、梯度累积和 checkpoint 状态有变化时，分别验证捕获/换值回放、重算梯度、`main_grad` 累加及恢复行为。已有训练 smoke/bitwise/long-run 入口按仓库规范选用，不新建全量训练平台。

### 正确性 evaluate

- **core 数值验证**：复用 `apodex_core/verification/operations/local_ops.py` 的 `Case(candidate, reference, golden, gen_input)`。检查前向、输入梯度、权重/偏置梯度等实际输出；沿用原生偏差与误差幅度判据、精度配置和已批准的 WARN 规则，不用通用 allclose 替代。reference 与 golden 必须独立于候选；手写反向不能仅升为 FP64 就充当独立 golden。
- **接口与覆盖**：结合 `tests/` 检查既有 bitwise、确定性、batch invariance、输入修改、alias、布局、设备/stream 等承诺。数值 gate 的固定 fixture 不代表全部生产尺寸/dtype；缺失覆盖补到算子测试或原生 case builder，不通过缩减支持范围取得加速。新增/调整验证器需用已知错误实现确认能检出目标错误。
- **Megatron 集成**：复用 `tests/unit_tests/apodex/operations/local_modules.py` 与相关模块测试，检查真实模块的 forward、dgrad、wgrad。用调用计数或定向断言确认候选后端确实执行，不能静默 fallback。涉及 microbatch 切分、padding、分片或归约时，核对相同有效 token 的归一化与梯度，防止引入系统性缩放或偏差。
- **新后端**：按 core 规范保留旧后端的执行路径和默认选择，验证旧路径不触达新 helper；性能对照使用改动前源码，不能只与当前代码中「关闭新开关」的路径比较。

例如，在 core 根目录对 SwiGLU 执行已有契约测试和生产数值 gate：

```bash
python -m pytest tests/test_fused_swiglu.py -q --junitxml=swiglu.xml
python -m apodex_core.verification.method.gate --source operations \
  --only local_fused_swiglu_fwd,local_fused_swiglu_dx
```

这是完整确认的入口示例；日常迭代按改动选择具体 pytest node。当前 gate CLI 以 BF16 和默认模型 fixture 构建 case，其他生产 dtype/shape 的覆盖需另外核对。Megatron 的 `test_local_modules.py` 默认 `APODEX_GATE_RMIN=512` 用于开发，正式验证使用当前 CI 配置（本次检查为 4096）。缩小抽样的结果仅作探索，不覆盖正式结果。

### 性能 evaluate

优先复用 core `benchmarks/` 和 Megatron 的模块 benchmark，检查其计时范围后再使用。训练目标以同一实际配置下的完整 forward+backward 为主，单独 fwd/bwd 用于定位；优化器还要覆盖状态更新。保留逐尺寸耗时、原始样本、峰值显存，以及启动/编译与稳态测量的耗时。

1. 对照固定原始实现和当前已验证版本；保持相同输入、权重、上游梯度、精度设置和执行模式。每轮恢复 `.grad`、`main_grad`、优化器状态及可变缓冲；重置成本是否计入，按实际训练调用事先约定，两边一致。
2. GPU 独占，完成预热后交替 A/B 顺序，保存配对样本。Megatron `_paired_benchmark.py` 可复用其交替顺序逻辑，但还需检查调用者的输入和状态重置。core 某些 benchmark 按后端顺序报告均值，仅适合初筛；正式比较补配对测量，收益小于波动时记为未确认。
3. 计入实际调用中的 dispatch、布局转换、临时分配和必要同步。Graph 与 eager 分开评测，Graph 要覆盖重放时输入换值；不能把 CUDA event 一概解释为纯 kernel 时间。import 时读取的后端环境开关用新进程做对照，不能在导入后修改变量冒充 A/B。
4. 同轮确认候选与对照；需要不同源码进程时交替启动并保留各自版本证据，不在一个解释器里混载两份 `apodex_core`。通信/overlap 的收益在实际模块或 step 上测，局部同步微基准只用于诊断。

显存增加可能抵消融合收益，特别是 autograd saved tensors 与重计算。需要证明训练加速时，复用 Megatron `.ci/mfu_bench/` 的对应 recipe，比较相同配置的稳态 step time、tokens/s 和峰值显存；排除编译、预热与 profiler 窗口。mock/均衡路由结果单独标明，不作为实际数据分布下的吞吐。

### 结果与交付

原生命令直接运行即可，记录命令、cwd、必要环境变量、退出码、耗时和原始日志；pytest 保存 JUnit，gate 保存本轮实际算子记录。分别记录算子正确性、性能、集成验证的通过/失败/未执行，不合成一个含糊的 PASS。

验证结果须属于本轮源码：核对预期测试名称、非零通过数量和应有 rank 报告；意外 skip、缺报告、fallback 或旧记录不算通过。core `gate_history.json` 会保留以前的算子条目，只读取本轮预期名称且核对时间和版本；`--emit-observed` 是内部子进程接口，会跳过最终判定，不能作为验收命令。复用 Megatron `.ci/assert_no_pytest_skips.py` / `.ci/check_junit_results.py` 时仍保留 pytest 的退出码。

通用 `evaluate/full/promote` 不会调用以上内部检查，且不直接接受带 `requires_grad` 的输入。内部 best 用 Git 修订与完整证据定位，不写成通用 `best.json`；只有局部实验时才将明确封装的计算交给通用评测器。

交付按 core 改动 → Megatron 接线和 pin 更新的依赖顺序进行；按两仓当前规范接入相关 CI。最终 pin 必须已进入 core main，核对实际导入路径与接线结果；新特性的 sweep 配置按 Megatron 规范更新。常规 GPU merge-queue 门禁仍由队列执行，review 不要求提前重跑整套 CI。

源码核对基准（2026-11-10）：Megatron `eb2c08444c46ad7bfca0f51c7554970cab6edf48` 的 core pin 为 `0b8b9b8da1624f8f994e4e54c9126f22c3b00361`。后续任务读取实际 checkout 的规范和入口：

- [core 开发规范](https://github.com/ApodexAI/apodex-core/blob/0b8b9b8da1624f8f994e4e54c9126f22c3b00361/DEVELOPMENT_GUIDELINES.md)、[数值 gate](https://github.com/ApodexAI/apodex-core/blob/0b8b9b8da1624f8f994e4e54c9126f22c3b00361/apodex_core/verification/method/gate.py)、[前反向 benchmark 示例](https://github.com/ApodexAI/apodex-core/blob/0b8b9b8da1624f8f994e4e54c9126f22c3b00361/benchmarks/bench_fused_hc_post.py)
- [Megatron 开发规范](https://github.com/ApodexAI/Megatron-LM/blob/eb2c08444c46ad7bfca0f51c7554970cab6edf48/DEVELOPMENT_GUIDELINES.md)、[模块 gate](https://github.com/ApodexAI/Megatron-LM/blob/eb2c08444c46ad7bfca0f51c7554970cab6edf48/tests/unit_tests/apodex/test_local_modules.py)、[配对计时](https://github.com/ApodexAI/Megatron-LM/blob/eb2c08444c46ad7bfca0f51c7554970cab6edf48/tests/unit_tests/apodex/_paired_benchmark.py)

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

## 交付

冻结最终候选，在实际支持范围内用未参与调参的尺寸/维度组合和数值种子验证正确性与性能，覆盖各分支和阈值邻域，保存 case、采样规则、最终源码哈希和结果。仅换 seed 不算尺寸泛化。内置 full 只覆盖 task.json 中的 case，不自动完成这项验证；使用独立验证任务或项目已有 evaluator。发现失败就修复并验收最终版本，不删除失败 case。此项在交付时执行，无需加到每轮循环。

确认最终导入/调用入口实际使用已验证代码及 helper，独立任务通过 `load_best(task目录)` 加载 `best.json` 指向的 source；内部项目按已验证的 core 修订与 Megatron pin 交付，并确认 dispatcher 实际调用了候选。交付对应源码、运行命令、证据路径、逐 case 结果和简短结论；若只有失败或未确认候选，保留原 best 并明确说明本轮发现。
