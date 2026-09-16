# Triton Decode Lab

面向实习求职的单 GPU LLM Decode 优化项目：用 Triton 实现基础融合算子与 GQA Decode Attention，再扩展到分页 KV Cache，并在真实模型中验证。

> 当前状态：项目骨架与设计文档。尚未实现 kernel、模型接入或基准脚本，没有实测加速数据。下文的接口、路径和实验均为实现计划，不代表已经支持。

## 项目目标

- 基础算子：RMSNorm、Residual Add + RMSNorm、SiLU × Mul。
- 核心算子：连续 KV GQA Decode、Online Softmax、Split-KV、Paged Decode Attention。
- 系统集成：页分配与回收、KV 追加、变长 batch、多步 Decode、一个固定模型的接入。
- 实验闭环：正确性 → kernel 性能 → 模块性能 → 模型性能，记录优化的收益与退化范围。

第一版仅做前向推理：单 GPU、普通单 token Decode、FP16/BF16 输入、关键归约使用 FP32。Q/K/V 和 MLP 线性层先沿用成熟实现。完整服务调度、前缀共享、量化、多 GPU、反向传播不属于首版验收范围。

## 项目结构

```text
triton-decode-lab/
├── README.md
├── pyproject.toml              # 仅定义可编辑安装骨架
├── .gitignore
├── configs/
│   ├── smoke.json             # 少量正确性用例
│   └── benchmark.json         # 实验候选参数，须按显存筛选
├── src/triton_decode_lab/
│   ├── __init__.py
│   ├── ops/                   # Python 参数检查、kernel 选择、公开 API
│   ├── kernels/               # Triton kernel
│   ├── reference/             # 独立 PyTorch 正确性实现
│   ├── cache/                 # 页池、页表、追加和回收
│   ├── runtime/               # 最小 Decode 循环
│   └── integration/           # 固定模型适配
├── tests/
│   ├── unit/                  # 数值和边界
│   └── integration/           # 缓存生命周期、多步 Decode
├── benchmarks/                # kernel、模块、模型测量脚本
├── examples/                  # 最小复现和模型运行入口
├── scripts/                   # 环境采集、结果汇总
├── results/                   # 实验数据与图表，禁止虚构数据
└── docs/
    ├── architecture.md        # 模块职责、数据流、文件规划
    ├── interfaces.md          # 张量布局、数学语义、API 契约
    ├── roadmap.md             # 实现顺序与验收门槛
    ├── testing.md             # 正确性与边界覆盖
    ├── benchmarking.md        # 公平计时和消融实验
    ├── environment.md         # GPU 环境与版本固定
    └── report-template.md     # 实验报告和简历素材模板
```

空目录使用 `.gitkeep` 保留；计划中的实现文件详见架构文档。

## 开始使用

1. 阅读 [实现路线](docs/roadmap.md) 和 [接口约定](docs/interfaces.md)。
2. 按 [环境说明](docs/environment.md) 准备受支持的 NVIDIA GPU 环境。本项目首版以 NVIDIA 为目标；本地无 GPU 时只能编辑文档和开发 CPU 参考逻辑。
3. 在仓库根目录执行 `python -m pip install -e .`，仅安装本项目骨架。它不会安装 PyTorch、Triton，也不会运行任何算子。
4. 从 M0 的 PyTorch 参考实现和测试开始，然后逐项实现 Triton 版本。当前没有可运行的 benchmark 或模型命令。

## 验收标准

- 正确性：随机输入、尾块、页边界、变长、随机物理页映射、跨页追加均通过。
- 性能：同一硬件和固定软件版本下，公开原始延迟、精度误差、实验配置和适用范围。
- 工程：通过页表直接读取 KV；无需先拼接成连续缓存；页池支持安全回收和复用。
- 模型：至少一个固定模型完成多步 Decode 对齐；分别报告基础算子和 Attention 的贡献。

不预设必须达到的加速倍数。没有优化收益也必须保留结果并解释瓶颈。

## 文档导航

| 文档 | 解决的问题 |
|---|---|
| [架构](docs/architecture.md) | 每个模块负责什么，文件怎么拆 |
| [接口](docs/interfaces.md) | Q/K/V、页表和输出的精确语义 |
| [路线](docs/roadmap.md) | 先实现什么，做到哪里算完成 |
| [测试](docs/testing.md) | 怎样避免看似正确的错误 kernel |
| [基准](docs/benchmarking.md) | 怎样测出可信而可复现的性能 |
| [环境](docs/environment.md) | 怎样记录和复现硬件软件环境 |
| [报告模板](docs/report-template.md) | 怎样沉淀优化证据和求职素材 |

## 参考资料

- [Triton 官方教程](https://triton-lang.org/main/getting-started/tutorials/index.html)：Softmax、矩阵乘和 Attention。
- [GQA 论文](https://arxiv.org/abs/2305.13245)：Query 头与 KV 头分组共享。
- [PagedAttention 论文](https://arxiv.org/abs/2309.06180)：分页 KV 缓存与服务系统。
- [vLLM Paged Attention 设计，固定版本文档](https://docs.vllm.ai/en/v0.10.1/design/paged_attention.html)：实现参考，不是本项目的布局要求。
- [Liger-Kernel](https://github.com/linkedin/Liger-Kernel)：融合算子与模型集成参考。
- [FlashInfer](https://github.com/flashinfer-ai/flashinfer)：成熟推理算子对比候选。

引用或改编代码时保留来源和原有许可证要求。当前骨架未选择发布许可证，公开发布前由维护者确定。
