# 受信原生检查

原生测试不能由待优化的候选实现自行修改。`native-evaluate` 将固定 baseline 的完整项目树复制为检查环境，仅覆盖任务明确允许的实现路径。pytest、conftest、独立 reference、数值 gate 和阈值均来自 baseline。

## 配置

```json
"implementation_paths": {
  "core": [
    "apodex_core/ops/candidate/triton_probs_scale.py"
  ]
}
```

路径相对于对应源码根目录，文件和目录均可。没有列出的根目录使用固定 baseline。应选择最小实现范围；例如优化 DSA 时可以列出具体 kernel、调用器及 source guard 文件，不应覆盖整个 `apodex_core/ops`，其中包含独立 reference。

路径不能是绝对路径、含 `..` 或 symlink，彼此不能重叠，不能包含测试、reference、verification、pytest 配置或声明的检查入口。目录覆盖仅接受实现源码类型；配置文件需要保留在受信 baseline。候选在允许范围之外修改代码或配置会在启动 worker 前被拒绝；Markdown/RST 文档可以随候选更新，但不进入检查环境。

白名单属于任务契约。增减路径、修改原生测试或参考阈值时，应先单独审查，再建立新任务契约。旧任务缺少这个字段时明确拒绝运行，历史证据保持原样；不能将旧版接受的结果自动转换为新版验收。

## 执行和证据

- correctness 和性能 worker 分别使用完整的冻结 candidate、baseline 或 best 源码。
- 原生检查的工作目录、`PYTHONPATH` 及 `{core}`、`{megatron}`、`{hf}` 等替换路径统一指向 `sources/validation`。因此 conftest 将项目根目录插入 `sys.path` 后，仍能加载候选实现和受信检查代码。
- 检查环境记录 baseline、candidate 的 Git 身份、实现覆盖路径及每个文件的 SHA256。执行结束和晋升时重新验证，文件被修改则拒绝晋升。
- `harness_seconds` 单独记录检查环境的复制与哈希耗时，未计入算子性能。按冻结清单直接合成基线与候选文件，每个目标文件只复制一次并核对原 SHA256；候选删除的实现不会残留。复制与哈希之间检查剩余执行预算，没有额外缓存。

这解决候选修改检查代码而获得 PASS 的问题；同权限 Python 代码不构成安全沙箱。

## 快速迭代

```sh
# 复现已知错误，不运行计时或完整原生门禁
kernel-envolve native-evaluate --task ./task --case tail --seed 17 --check-only --timeout 60
# 新候选直接做代表输入的正确性及 ABBA 筛选
kernel-envolve native-evaluate --task ./task --case small --case tail --timeout 120
```

case/seed 选择和 check-only 仅用于 quick，配置与完整源码仍冻结，实际覆盖保存在结果中，不能晋升。check-only 只启动候选与独立 oracle；其余 quick 保留同轮性能对照。full 禁止缩小覆盖，独立确认的规则不变。

quick 默认执行预算 120 秒、full 1800 秒，可显式覆盖；源码快照与最终完整性遍历也检查预算。系统 I/O 返回和进程清理仍需时间，超时保留失败阶段与收据。首次冷编译单独记录，不把重复超时当作自动延长预算的理由。

计时循环每次仍检查输入/状态、运行设置、adapter 验证和输出结构；仅最后一次生成 CPU 输出/可写状态快照，供原有数值比较使用。中间快照原先被立即丢弃，省去其传输不减少正式数值比较或计时样本。

大型项目应一次性将工作树及任务准备到 GPU 节点本地盘，避免每个候选重复承担共享文件系统元数据和临时张量交换成本。每轮仍完整哈希、隔离源码；及时保存新增 run、源码改动与 manager 一致性备份到持久盘，再回收节点。不使用跨角色硬链接、仅按 mtime 命中的快照缓存或跨候选 PASS 缓存。

## 原生拒绝与执行错误

完整 JUnit 覆盖所有配置的报告和必需用例、没有跳过或运行错误，且失败均为明确的断言并与退出码 `1` 一致时，结果为 `execution=completed`、`correctness=fail`。原生数值 gate 则要求本次完整算子记录、CLI 的最终 `BLOCKED` 判定和退出码 `1` 同时成立。WARN 是否阻塞仍由项目原生策略决定。

缺报告、覆盖不足、计数矛盾、导入或运行异常、退出码与证据不一致，继续视为执行错误。两类结果都不会进入性能测量或晋升；每项已尝试的检查在 `native_checks` 保留状态、退出码、完整耗时及失败用例，原始报告和日志纳入收据。历史结果不会被重新分类。

命令启动失败、超时或清理异常均保留结构化错误记录，未知退出码为 `null`，耗时包含准备、执行、报告解析及清理。清理异常单独保存为 `cleanup_error`；即使命令已通过或产生断言拒绝，清理失败仍归为执行错误，同时保留原始报告、失败用例和原始异常。JUnit 用例必须直接属于 `testsuite`，不接受缺少 suite 的不完整报告。

## 独立确认

内部任务可在首次评测前声明 `confirmation`，将阶段交付的预留输入纳入晋升门禁：

```json
{
  "confirmation": {
    "cases": [
      {"id": "heldout-tail", "args": {"size": 32769}, "weight": 1}
    ],
    "seeds": [1009, 2027],
    "trials": 6
  }
}
```

case 的参数组合和名称须与开发集不同，seed 不得与开发集重合，trials 至少 3。案例字段与普通 `cases` 相同；adapter 必须实际使用声明的尺寸、布局等参数。整个确认配置属于冻结任务契约，不能在看完 full 后更换。

```sh
kernel-envolve native-evaluate --task ./task --mode full
kernel-envolve native-confirm ./task/.kernel_native/runs/<full-run> --timeout 1800
kernel-envolve native-promote ./task/.kernel_native/runs/<full-run>
```

管理器内通过 `manage gpu-run --expected-seconds ... -- kernel-envolve native-confirm ...` 分配资源。确认复用冻结 adapter、全部原生 checks、正确性和性能阈值，执行独立 full；独立收据绑定父 full、子 raw evidence、源码、配置及 evaluator，并要求各角色的 GPU、PyTorch/CUDA 版本和数学开关与父 full 一致。声明确认的任务缺失或未通过确认时不能晋升，已晋升证据被篡改也不能继续显示为有效 best。

开发 full 仍比较当前 best 和固定基线；确认阶段独立比较固定基线，不宣称已在预留输入上胜过旧 best。确认集每个任务只能消费一次，失败、中断、换候选或重跑 full 都不重用。继续迭代时建立新版本任务，把已见确认输入并入开发回归，预先声明新的确认集；不能删除消费记录来重抽。工具核验参数组合与收据，agent 仍负责选择真实生产维度并保留跨任务的已见输入记录。

确认收据同时记录总墙钟耗时 `seconds` 和 `phase_wall_seconds`：锁等待、父证据审计、前后 live source 哈希、独立 full、后验审计均计入；源码复制及文件系统读取成本不只归为子 full 时间。确认仅在阶段交付执行，不加入每轮 quick。

未声明 `confirmation` 的旧任务保持原行为，其 full 仅代表开发覆盖，不因此获得独立确认声明。

后续迭代加载已晋升 best 时，仅核验绑定的结果、配置、收据、消费记录及实际加载的父候选源码，不重复扫描独立确认阶段未使用的子源码与日志。晋升、管理器正式证据复核和 PR 发布继续完整核验这些归档文件；快速加载不替代交付验收。

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

- core 的 `_draw_on` 按 candidate → reference → golden 顺序传入同一输入，数值计算允许广播并转换到 FP64，不统一检查 API 契约。对约定保持输入/shape/dtype 的恒等算子，实际调用 `gate.run`（原配置 R_min=4096）时，错误 shape、错误 dtype、非法输入清零均 PASS；正确实现 PASS，明显数值错误和 NaN FAIL。这是**数值方法单独使用时的覆盖缺口**，不代表专项 pytest 或完整 CI 也放行。见[故障复现脚本](../audit/probe_native_gate_contract.py)与[实际结果](../audit/native_gate_contract_probe.json)。
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

从 [probs_scale 示例](../examples/apodex_probs_scale) 建立任务，在项目指定镜像内运行。先固定原始 core/Megatron checkout，配置候选 checkout，调整生产输入与测试集合；不要把 eager reference 当作原始性能基线。

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
