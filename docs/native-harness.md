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
- `harness_seconds` 单独记录检查环境的复制与哈希耗时，未计入算子性能。此版本使用普通文件复制，没有额外缓存。

这解决候选修改检查代码而获得 PASS 的问题；同权限 Python 代码不构成安全沙箱。

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
