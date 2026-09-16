# 架构与文件规划

状态：设计方案，以下 `.py` 文件除包初始化外均待实现。

## 分层职责

`reference` 提供独立正确性真值；`kernels` 只负责计算；`ops` 负责参数验证与 dispatch；`cache` 管理存储；`runtime` 串联步骤；`integration` 处理具体模型差异。kernel 不执行 Python 侧页分配。

| 目录 | 计划文件 | 职责 |
|---|---|---|
| reference | norm.py、activation.py、attention.py | 显式 FP32 参考公式；测试规模保持可控 |
| kernels | rmsnorm.py、add_rmsnorm.py、silu_mul.py | 三个基础算子 |
| kernels | decode.py、split_kv.py、merge.py | 连续 GQA、分段统计量及稳定合并 |
| kernels | paged_decode.py、cache_append.py | 页表读取、批量 KV 写入 |
| ops | norm.py、activation.py、attention.py | 检查布局、dtype、形状，选择 kernel |
| cache | page_pool.py、kv_cache.py | 空闲页、页表、长度、追加和回收 |
| runtime | decode_runner.py | 活跃请求组织、KV 追加、逐步推理 |
| integration | model_adapter.py | 首个固定模型，记录版本与模型配置 |
| benchmarks | bench_ops.py、bench_attention.py、bench_model.py | 分层测量与 JSON 结果 |
| scripts | collect_env.py、summarize_results.py | 环境采集、汇总曲线 |

## 单层 Decode 数据流

输入 → RMSNorm → Q/K/V 投影 → RoPE → 当前 K/V 写入缓存 → GQA Attention → O 投影 → 残差相加 → RMSNorm → Gate/Up 投影 → SiLU×Mul → Down 投影 → 残差相加。

投影使用已有 PyTorch 实现。RoPE 第一阶段使用模型原实现。残差加法与其后 RMSNorm 在数据依赖允许时融合；必须同时保留后续需要的残差分支，不可只输出归一化结果。

## 缓存生命周期

1. 注册请求并分配请求槽位。
2. 原模型完成 Prefill；将各层已应用 RoPE 的 K 和对应 V 导入页池。此转换单独计时。
3. Decode 前为即将写入的位置预留足够页面；内存不足应明确失败，不得部分更新请求状态。
4. 各层写入当前 K/V，Attention 可见长度为旧长度加一。所有层完成后再统一提交长度。
5. 请求结束后释放其独占页面，清理页表和槽位。

第一版每个请求独占页面，不实现共享引用计数。每个物理页编号对应跨层的同一组 token 位置；各层保存独立 K/V 张量。多层分配必须保持一致。

## 首版集成边界

只接入一个架构和一个固定 checkpoint/revision，记录 head 数、head_dim、RoPE 参数、归一化 epsilon、精度和软件版本。选择普通因果全注意力模型；滑动窗口、特殊位置编码等不匹配语义必须显式拒绝。

静态 batch + 不同请求长度即可满足首版目标。动态到达、continuous batching 和真实服务指标作为扩展；没有调度器时不得声称提升线上服务吞吐。
