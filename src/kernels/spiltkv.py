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


def split_kv_with_torch_merge(q, k, v, kvsize=256):
    # 仅支持连续存储、MHA、单 token decode
    B, H, one, D = q.shape
    L = k.shape[2]

    assert one == 1 and L > 0
    assert k.shape == v.shape == (B, H, L, D)
    assert q.is_contiguous() and k.is_contiguous() and v.is_contiguous()
    assert q.is_cuda and q.device == k.device == v.device
    assert kvsize > 0 and (kvsize & (kvsize - 1)) == 0

    # ceil 除法保证所有 split 至少包含一个有效 token
    S = triton.cdiv(L, kvsize)

    part_m = torch.empty((B, H, S), device=q.device, dtype=torch.float32)
    part_l = torch.empty_like(part_m)
    part_u = torch.empty((B, H, S, D), device=q.device, dtype=torch.float32)

    split_kv_kernel[(B, H, S)](
        q, k, v,
        part_m, part_l, part_u,
        head_num=H,
        d=D,
        L=L,
        kvsize=kvsize,
        num_splits=S,
        BLOCK_D=triton.next_power_of_2(D),
        num_warps=4,
    )

    # 暂时用 PyTorch 合并，后续可替换成第二个 Triton kernel
    m = part_m.amax(dim=2, keepdim=True)      # [B, H, 1]
    alpha = torch.exp(part_m - m)            # [B, H, S]

    numerator = (alpha[..., None] * part_u).sum(dim=2)  # [B, H, D]
    denominator = (alpha * part_l).sum(dim=2)          # [B, H]

    out = numerator / denominator[..., None]
    return out.unsqueeze(2)  # [B, H, 1, D]，保留 FP32 便于检查误差


def reference_attention(q, k, v):
    # 使用 FP32 的完整 Attention 作为参考
    scores = torch.matmul(q.float(), k.float().transpose(-1, -2))
    scores = scores * (q.shape[-1] ** -0.5)
    return torch.matmul(torch.softmax(scores, dim=-1), v.float())


def test_correctness():
    torch.manual_seed(0)
    # 避免参考 matmul 使用 TF32，干扰误差比较
    torch.backends.cuda.matmul.allow_tf32 = False

    # B, H, L, D, kvsize
    cases = [
        (1, 4, 128,   64, 256),  # 单段，且有 padding
        (2, 3, 1024,  64, 256),  # 检查 batch/head 偏移
        (2, 3, 1003,  80, 256),  # L、D 均需要 mask
        (1, 16, 16384, 64, 256), # 长 KV、多段
    ]

    for B, H, L, D, kvsize in cases:
        q = torch.randn((B, H, 1, D), device="cuda", dtype=torch.float16)
        k = torch.randn((B, H, L, D), device="cuda", dtype=torch.float16)
        v = torch.randn_like(k)

        actual = split_kv_with_torch_merge(q, k, v, kvsize)
        expected = reference_attention(q, k, v)

        error = (actual - expected).abs()
        print(
            f"B={B}, H={H}, L={L}, D={D}, "
            f"splits={triton.cdiv(L, kvsize)} | "
            f"max_abs={error.max().item():.3e}, "
            f"mean_abs={error.mean().item():.3e}"
        )

        assert torch.isfinite(actual).all().item()
        torch.testing.assert_close(
            actual, expected,
            atol=1e-4,
            rtol=1e-3,
        )

    print("All tests passed!")


if __name__ == "__main__":
    test_correctness()