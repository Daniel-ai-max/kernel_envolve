---
name: kernel-envolve
description: 通过假设、代码实验、独立正确性验证和配对计时迭代优化 CPU/GPU 算子，维护可复现的 verified best。适用于在明确硬件、语义与预算下开发 Triton、CUDA、CuTe DSL、TileLang 或其他算子实现。
---

# Kernel Envolve

使用普通文件、现有 coding agent 和任务 evaluator 完成迭代。自主选择实验路线；每次接受性能结论必须有可复现的验证和测量证据。

## 开始一项任务

- 明确调用语义、shape/dtype/layout/alias、允许的误差与原地更新、实际调用入口、目标硬件、测量范围和资源预算。缺少真实 workload 权重时，明确声明 benchmark 目标。
- 对融合、归约次序、近似和精度变化，确认 oracle 的独立性及任务要求的误差/漂移/梯度检查；普通 allclose 不替代已有精度 gate。用已知正确和已知错误实现检查新增验证器。
- 读取当前任务、Git 状态、已有 best 和最近失败记录，避免覆盖未完成工作或重复实验。已有远程作业先查 job 状态，再决定是否提交。
- 区分固定 oracle、固定原始 baseline、已验证 best 和工作中的 candidate。优化候选不能修改 oracle/覆盖/精度来获取 PASS；契约确需改变时另建任务版本。
- 先确认实际环境和原始实现可运行。硬件型号、CUDA/PyTorch/Triton 版本以机器为准。按需读取 [硬件和参考实现入口](references/hardware.md)，不要把所有资料一次塞进上下文。

## 每轮实验

1. **假设。** 用一句话说明要解决的瓶颈和预期影响到的 case，例如减少中间张量、改善访存、消除同步或改变分块。记录代码起点；它可以是未晋升的原型。
2. **最便宜的有效反馈。** 尚未写完的结构先做编译/局部检查；正确性失败先重现对应 case/seed；参数调优可以成批探索。进入性能比较前先通过任务正确性验证。
3. **选择反馈范围。** quick 自带正确性检查，正常迭代无需先单独 check-only。用 quick 筛掉大量不值得完整评测的候选；若近期记录显示 quick 与 full 耗时接近、候选又很可能进入 full，直接 full。只测候选的 signal 可以辅助选参数，但不能算 speedup。保留失败机制有价值的候选，不要因为未晋升就机械回滚所有探索。
4. **确认。** 有希望且可交付的候选做 full，与当前 best、固定 baseline 同轮实测。检查覆盖、逐 case 增量与累计回退、噪声、目标权重和计时范围；有疑问时查看原始样本，不自行改 PASS/分数。仅为解释瓶颈时才跑 profiler。
5. **晋升或继续。** full 证据满足任务目标后 promote。更快但未确认的代码仍是 candidate；正确但更慢的实现可以留下供组合。记录本轮假设、改动、run 路径、结论与下一步，维护一个简短 `notes.md` 即可。

根据 CLI 的 phase_wall_seconds 和 worker_process_wall_seconds 判断成本：setup 高时减少重复启动，correctness/measurement 高时用 --case/--seed 缩小探索范围。探索 trial 数在任务开始时约定；不因本轮结果修改契约或缩减正式 full 的固定覆盖。不要为省启动成本复用失败的 CUDA 进程，也不要跨候选复用正确性 PASS。

预算用完、收益停止或目标达成时收尾，不为凑流程启动新实验。一次有效 full 已包含正式确认；无需无条件再跑第三次相同 full。

正式确认前确定预算和门槛，保留全部尝试；不要反复抽样直到偶然赢。协议有缺陷时记录原因、版本化修正并重测受影响对照，不能观察到结果后降低门槛。相同源码不作为新的改进，注释或重命名也不算机制进展。

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
- quick 永不晋升；full 合格由 evaluator 计算。失败应修复或诊断，不能编辑结果绕过检查。
- 对照对象 best 来自保存的快照，不从可变工作目录读取。full 自动核对原始 baseline；`--baseline` 可单独诊断原始实现，不能用它绕过当前 best。
- 不在候选里按固定随机种子、输入地址、评测次数或评测状态投机。允许按公开 shape/dtype/layout 做合法 dispatch。
- 输出 alias/layout 等语义写入 task.validate。涉及 Graph、多流、通信、复杂 backward 或显存约束时，采用覆盖这些语义的专用 evaluator；明确记录实际验证和测量范围。
- 性能测量串行占用目标 GPU，避免与其他 profiling/benchmark 并发。非法访问、编译/运行异常后使用新进程。

## 交付

确认最终导入/调用入口实际使用已验证代码及 helper，通过 `load_best(task目录)` 加载 `best.json` 指向的 source；已有模型的 dispatcher 需额外确认实际调用了它。交付对应源码、运行命令、证据路径、逐 case 结果和简短结论；若只有失败或未确认候选，保留原 best 并明确说明本轮发现。
