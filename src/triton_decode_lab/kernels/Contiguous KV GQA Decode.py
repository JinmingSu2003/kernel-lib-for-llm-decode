import torch
import triton
import triton.language as tl


@triton.jit
def gqa_decode_kernel(
    Q, K, V, O,
    H_Q: tl.constexpr,
    H_KV: tl.constexpr,
    L: tl.constexpr,
    D: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    batch_id = tl.program_id(0)
    head_id = tl.program_id(1)

    groupsize = H_Q // H_KV#多少个q头共享1个kv头
    head_kv = head_id // groupsize#当前q头使用哪个kv
    

    q_base = Q + (batch_id * H_Q + head_id) * D
    k_base = K + (batch_id * H_KV + head_kv) * L * D
    v_base = V + (batch_id * H_KV + head_kv) * L * D
    o_base = O + (batch_id * H_Q + head_id) * D

    col = tl.arange(0, D)
    q = tl.load(q_base + col).to(tl.float32)

    m = -float("inf")
    l = 0.0
    acc = tl.zeros((D,), dtype=tl.float32)

    for start in range(0, L, BLOCK_N):
        row = start + tl.arange(0, BLOCK_N)
        offsets = row[:, None] * D + col[None, :]
        mask = row[:, None] < L

        k = tl.load(
            k_base + offsets, mask=mask, other=0.0
        ).to(tl.float32)

        v = tl.load(
            v_base + offsets, mask=mask, other=0.0
        ).to(tl.float32)

        scores = tl.sum(q * k, axis=1) * (D ** -0.5)
        scores = tl.where(row < L, scores, -float("inf"))
        m_new = tl.maximum(m, tl.max(scores, axis=0))
        alpha = tl.exp(m - m_new)
        p = tl.exp(scores - m_new)
        l = alpha * l + tl.sum(p, axis=0)
        acc = alpha * acc + tl.sum(p[:, None] * v, axis=0)
        m = m_new

    out = acc / l
    tl.store(o_base + col, out)