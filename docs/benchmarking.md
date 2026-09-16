# 性能评估规范

所有数值必须来自实测。当前没有结果，不能把示例参数写成性能结论。

## 分层计时

1. Kernel：提前分配输入输出和临时缓冲；数据已在 GPU；仅测计算。Split-KV 必须包含分段与合并两部分。
2. Operator API：可另外测 Python 调度与必要分配，明确是否包含开销。
3. Decode step：包含各层计算、KV 追加和必要管理；说明是否包括采样与主机调度。
4. 模型运行：Prefill、缓存转换、Decode 分开测；如报告总耗时，要包含全部路径。

固定 batch 的 tokens/s 定义为总生成 token 数 / Decode 总秒数。每步延迟与单请求体验不同；没有服务负载发生器时不报告线上 TTFT、排队延迟或服务 p99。

## 公平基线

- 基础算子：PyTorch eager、适用的 torch.compile、可选 Liger。
- 连续 Attention：显式 PyTorch 为正确性和朴素性能基线；成熟 SDPA/FlashInfer 为可选性能基线。
- 分页 Attention：选择支持相同 GQA、dtype、页布局语义的成熟库。不能兼容时注明不可直接比较。
- 记录实际 backend。避免隐式 KV repeat、数据转换、不同精度或不同计算语义造成不公平对比。
- 成熟库需要 plan/workspace 时，分别说明预处理与重复运行成本。

## 计时流程

先做正确性；编译和 autotune 完成后充分预热；用 CUDA events 或经过验证的 GPU benchmark 工具测量；在读取时间前同步。普通主机计时若没有同步不能用于 GPU 延迟。

建议起点：25 次预热、100 次测量、至少 3 轮独立重复，必要时增加时长。保存原始样本，报告中位数、p10/p90 和轮间波动。不要平均加速比来替代原始延迟。

双方使用一致的 CUDA Graph 条件。冷启动编译独立报告；不混入稳态。记录 GPU 是否被其他进程占用、功耗/时钟设置、输入是否反复命中缓存。profiler 另开运行，不能把其插桩时间当正式延迟。

## 扫描参数

候选 B={1,4,16,32}，T={128,512,2048,8192}，D={64,128}，头组合={(32,8),(16,4),(8,8),(8,1)}，page_size={16,32}，dtype={fp16,bf16}。

不要盲跑笛卡尔积：单层有效 KV 字节数为 `2 * sum(seq_lens) * Hkv * D * element_size`，实际页池分配为 `2 * P * S * Hkv * D * element_size`，多层再乘层数，并为模型权重、工作区和参考张量预留空间。OOM 用例记录为未运行。

## 消融

A 原模型；B 仅基础算子；C 在 B 上加连续 GQA；D 在 C 上加 Split-KV；E 在匹配条件下改用分页 Attention。所有版本对齐模型语义。

分页可能增加单步寻址开销，其价值通过页池分配量、有效 token 占用、页尾浪费和可容纳请求数评估。若模拟动态请求，公开到达与长度轨迹；不要把预分配整个页池的显存称为“仅按需占用”。

## 结果格式

每条记录至少包括：run_id、git_commit、timestamp、GPU/driver、torch/triton/CUDA 版本、kernel、baseline、B/Hq/Hkv/D、实际长度列表或生成种子、dtype、page_size、num_splits、num_warps、num_stages、预热/重复数、计时范围、CUDA Graph 状态、原始样本路径、median_us、p10_us、p90_us、精度误差、峰值显存统计口径。

每个配置独立计算 `speedup = baseline_median / candidate_median`。报告所有目标用例，包括退化；若聚合，写明几何平均及纳入用例。GB/s 需注明是估算的算法有效字节量，不是硬件实测 DRAM 流量。

建议图：延迟随 T 变化、加速比随 B/T 变化、不同 split 数对比、页大小与占用、模型消融。每张图附原始数据。
