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
