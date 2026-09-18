import torch
import triton
import triton.language as tl


@triton.jit
def split_kv_kernel(
    Q, K, V,
    PART_M, PART_L, PART_U,
    head_num: tl.constexpr,
    d: tl.constexpr,
    L: tl.constexpr,
    kvsize: tl.constexpr,
    num_splits: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    batch_id = tl.program_id(0)
    head_id = tl.program_id(1)
    kv_id = tl.program_id(2)

    bh = batch_id * head_num + head_id

    offs_d = tl.arange(0, BLOCK_D)
    offs_l = kv_id * kvsize + tl.arange(0, kvsize)

    # q: [BLOCK_D]
    q = tl.load(
        Q + bh * d + offs_d,
        mask=offs_d < d,
        other=0.0,
    ).to(tl.float32)

    # k, v: [kvsize, BLOCK_D]
    kv_offsets = (
        bh * L * d
        + offs_l[:, None] * d
        + offs_d[None, :]
    )
    kv_mask = (offs_l[:, None] < L) & (offs_d[None, :] < d)

    k = tl.load(K + kv_offsets, mask=kv_mask, other=0.0).to(tl.float32)
    v = tl.load(V + kv_offsets, mask=kv_mask, other=0.0).to(tl.float32)

    # 沿 D 求点积，每个 KV token 得到一个 score
    # [kvsize, BLOCK_D] -> [kvsize]
    scores = tl.sum(q[None, :] * k, axis=1) * (d ** -0.5)

    # 无效 token 必须设为 -inf，避免进入 softmax 分母
    scores = tl.where(offs_l < L, scores, -float("inf"))

    # 当前分段的统计量：两个标量
    m = tl.max(scores, axis=0)
    p = tl.exp(scores - m)
    l = tl.sum(p, axis=0)

    # 未归一化的加权 V
    # [kvsize, BLOCK_D] -> [BLOCK_D]
    u = tl.sum(p[:, None] * v, axis=0)

    # 每个 split 写独立位置
    part_id = bh * num_splits + kv_id
    tl.store(PART_M + part_id, m)
    tl.store(PART_L + part_id, l)
    tl.store(
        PART_U + part_id * d + offs_d,
        u,
        mask=offs_d < d,
    )
