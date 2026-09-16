# 正确性测试规范

状态：待实现测试清单。测试不能只比较两个共享同一索引逻辑的实现。

## 独立真值

Attention 用小规模 PyTorch 显式 FP32 运算计算 `softmax(QK^T * scale)V`。分页参考先按页表 gather 为逻辑顺序，再用独立 Attention 公式；gather 只用于真值，不放入被测 kernel。

使用手工可核算案例验证参考：单 token 时输出等于对应 V；全部分数相同时输出为有效 V 的均值；每个 KV 头填不同模式以验证 GQA 映射。归约公式可用 CPU FP64 小用例复核。

## 必测矩阵

| 维度 | 用例 |
|---|---|
| B | 1、2、8，混合请求长度 |
| Hq/Hkv | 1/1、8/8、8/2、8/1、32/8 |
| D | 64、128；不支持值明确报错 |
| T | 1、S-1、S、S+1、2S-1、非分块整数倍 |
| 精度 | FP16、BF16，硬件不支持则显式 skip |
| 数值 | 零、正负混合、大但有限值、固定随机种子 |
| 分页 | 随机物理页、无效尾部填异常模式、-1 未分配页 |
| Split | 1 段、多段、段数大于有效 token 数 |
| 基础算子 | 非二次幂 H、M=1、多个 M、不同 epsilon |

非连续输入、非法头数比例、错误 dtype/设备、超出容量、有效位置指向 -1 等在边界检查测试中覆盖。NaN/Inf 输入不属于首版有限输入保证，应明确文档化，而非声称支持。

## 数值门槛

可作为初始调查门槛：FP16 `atol=1e-2, rtol=1e-2`；BF16 `atol=3e-2, rtol=3e-2`。这些不是已验证阈值；必须结合输入分布、长度和模型确认，在最终报告中固定并解释，禁止为掩盖缺陷随意放宽。

同时报告 max_abs_error、mean_abs_error、带分母下限的 relative error、NaN/Inf 数量和失败用例。用 `abs(actual-ref) <= atol + rtol*abs(ref)` 判断通过，避免近零相对误差误导。

## 生命周期与模型

- 追加 token 恰好跨页；写入后 Attention 能看到当前 token。
- 不同请求写入互不污染；释放后再分配不读取旧有效数据。
- 容量不足无部分提交；多层长度一致；batch 压缩后页表仍匹配请求。
- 首先使用固定的 token 序列逐步输入，比较每步 logits，避免一次采样分歧导致后续不可比。
- 之后再做固定种子的生成演示；生成文本看起来正常不能代替数值测试。

测试文件建议：`test_norm.py`、`test_activation.py`、`test_decode.py`、`test_split_kv.py`、`test_paged_decode.py`、`test_cache_lifecycle.py`、`test_model_decode.py`。
