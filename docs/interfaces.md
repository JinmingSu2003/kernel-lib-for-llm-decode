# 张量与接口契约

状态：计划接口。布局为本项目选择，不承诺兼容其他库。

## 通用约定

- B：活跃请求数；Hq：Query 头数；Hkv：KV 头数；D：头维度；Tmax：连续缓存容量。
- Hq 必须能被 Hkv 整除；Query 头 h 使用 KV 头 `h // (Hq // Hkv)`。
- 输入 FP16/BF16，FP32 归约和累加，输出与输入 dtype 一致；参数必须在同一设备。
- 首版要求下列布局连续，非连续输入明确报错；不要在计时路径中隐式复制。
- 普通单 token 因果 Decode；可见缓存只包含过去及当前 token，因此不需要额外三角 mask，但必须按每个请求的实际长度屏蔽。
- Attention 的 Q/K 已完成必要的位置编码；本接口不执行 RoPE。

## 基础算子

```python
rms_norm(x, weight, eps) -> y
add_rms_norm(x, residual, weight, eps) -> (y, residual_out)
silu_mul(gate, up) -> y
```

`x/residual/gate/up` 均为 `[M, H]`；weight 为 `[H]`，首版与输入 dtype 相同。epsilon 必须为有限正数。禁止输入输出别名与原地改写。

- RMSNorm：`y = x * rsqrt(mean(x², dim=-1) + eps) * weight`。
- Add RMSNorm：先用 FP32 相加，再将和舍入为输入 dtype 得到 `residual_out`，然后按上述公式归一化 residual_out，输出 y。这一定义使融合前后的舍入位置明确；接入模型时必须与模型基线一致。
- SiLU×Mul：`y = (gate * sigmoid(gate)) * up`，计算使用 FP32，最终转回输入 dtype。这里只融合逐元素激活和乘法，不包含 Gate/Up/Down GEMM。

## 连续 GQA Decode

```python
gqa_decode(q, k, v, seq_lens, scale=None) -> out
```

| 参数 | 形状 | 语义 |
|---|---|---|
| q | `[B, Hq, D]` | 当前 token Query |
| k、v | `[B, Hkv, Tmax, D]` | 连续 KV 缓存，当前 token 已写入 |
| seq_lens | `[B]`，int32 | 有效长度，含当前 token，范围 1..Tmax |
| out | `[B, Hq, D]` | 各 Query 头的 Attention 输出 |

默认 `scale = 1 / sqrt(D)`；首版要求有限正数。无效 token 的缓存内容可以是任意值，不能影响结果。长度为零的请求不进入活跃 batch。

## Split-KV

首版由显式 `num_splits` 参数控制内部实现，验证后再做按形状 dispatch。第 j 段输出 FP32 的 `m_j`、`l_j`、`a_j[D]`：

```text
m_j = max(scores_j)
l_j = sum(exp(scores_j - m_j))
a_j = sum(exp(scores_j - m_j) * V_j)
m = max_j(m_j)
out = sum_j(exp(m_j-m)*a_j) / sum_j(exp(m_j-m)*l_j)
```

空分段写 `m_j=-inf, l_j=0, a_j=0`，分段 kernel 要单独处理全空分段，避免 `-inf - -inf`。总有效长度至少 1；合并不可对分段输出直接求平均。

## 分页 GQA Decode

```python
paged_gqa_decode(q, k_pages, v_pages, block_table, seq_lens,
                 scale=None) -> out
```

| 参数 | 形状 | 语义 |
|---|---|---|
| k_pages、v_pages | `[P, S, Hkv, D]` | 单层物理页池；S 为每页 token 数 |
| block_table | `[B, max_pages]`，int32 | 活跃 batch 顺序对应的逻辑页→物理页映射 |
| seq_lens | `[B]`，int32 | 已写入、含当前 token 的有效长度 |

逻辑 token t：`logical=t//S`，`offset=t%S`，`physical=block_table[b,logical]`，读取 `pages[physical,offset,kv_head,:]`。

有效页号必须处于 `[0,P)`；未分配页用 -1。先屏蔽逻辑无效 token/页表项，再进行安全的物理地址访问，禁止用 -1 做实际读取。调试模式在计时之外验证页表合法性。

## KV 追加

```python
append_kv(k_new, v_new, k_pages, v_pages, block_table, old_seq_lens)
```

新 K/V 为 `[B,Hkv,D]`；写入逻辑位置 `old_seq_lens[b]`；调用前完成页面预分配。返回前完成约定流上的写入，Attention 在同一流或明确同步后读取。此 API 不修改长度，由 runtime 统一提交。

本项目首版不处理重复槽位、跨请求共享可写页或并发修改同一缓存；这些输入必须被管理层拒绝。
