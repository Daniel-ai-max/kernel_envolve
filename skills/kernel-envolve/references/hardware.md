# 按需查阅的硬件与参考实现

先确认机器实际型号、compute capability、显存、驱动和软件版本，再查对应资料。B200/B300 等 Blackwell 型号不应只凭“B 卡”推断能力；参考代码的适用架构、dtype、shape 和依赖版本要逐项核对。

## 设备与语言

| 入口 | 适合解决的问题 |
|---|---|
| [Blackwell tuning guide](https://docs.nvidia.com/cuda/blackwell-tuning-guide/index.html) | Blackwell 架构特性、资源限制与调优方向 |
| [CUDA programming guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/) | 执行模型、内存、同步和流语义 |
| [PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/) | 指令语义、支持架构、对齐与约束 |
| [CuTe DSL](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/overview.html) · [示例](https://github.com/NVIDIA/cutlass/tree/main/examples/python/CuTeDSL) | 新架构 MMA、layout、pipeline 与低层实现 |
| [Triton tutorials](https://triton-lang.org/main/getting-started/tutorials/index.html) | 分块、归约、attention、autotune 的可执行起点 |
| [TileLang examples](https://github.com/tile-ai/tilelang/tree/main/examples) | tile/pipeline 模式与不同 kernel 家族 |
| [Nsight Compute profiling guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html) | 访存、占用率、指令和同步瓶颈；计数器解释 |
| [Compute Sanitizer](https://docs.nvidia.com/compute-sanitizer/ComputeSanitizer/index.html) | 越界、竞态和同步问题定位 |
| [AMD ROCm documentation](https://rocm.docs.amd.com/en/latest/) | AMD 平台对应的工具链与调优文档 |

## 算子参考库

| 仓库 | 优先参考的部分 |
|---|---|
| [NVIDIA/CUTLASS](https://github.com/NVIDIA/cutlass) | GEMM、架构专用 MMA、layout 与 pipeline |
| [Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention) | attention 的数值稳定性、分块和融合 |
| [flashinfer-ai/flashinfer](https://github.com/flashinfer-ai/flashinfer) | 推理 attention、KV cache、布局与 dispatch |
| [deepseek-ai/DeepGEMM](https://github.com/deepseek-ai/DeepGEMM) | 低精度 GEMM、grouped GEMM 和架构约束 |
| [FlagOpen/FlagGems](https://github.com/FlagOpen/FlagGems) | 通用算子覆盖、测试 case 与 Triton 实现 |
| [HazyResearch/ThunderKittens](https://github.com/HazyResearch/ThunderKittens) | tile 抽象、流水和融合实现 |
| [pytorch/pytorch](https://github.com/pytorch/pytorch) | ATen 语义、边界 case、测试与原生实现 |

先搜索目标算子名和目标架构，读取最相关的源码与测试。记录实际借鉴的 commit/路径和思路；发现可复用知识时，在任务 notes 中保留“观察 + 证据 run + 适用硬件”的短条目。无需另建 wiki 或复制整份文档。
