import torch
import triton
import triton.language as tl

@triton.jit
def rmsnorm_kernel(
    x,
    w,
    y,
    N: tl.constexpr,
    BLOCK_N: tl.constexpr,
    eps: tl.constexpr = 1e-6,
):
    pid=tl.program_id(0)
    col=tl.arange(0,BLOCK_N)
    mask=col<N
    off_x=pid*N+col
    x_d=tl.load(
        x+off_x,
        mask=mask,
        other=0.0
    )
    x_d=x_d.to(tl.float32)
    w=tl.load(
        w+col,
        mask=mask,
        other=0.0
    )
    w=w.to(tl.float32)
    mean_square=tl.sum(x_d*x_d,axis=-1)/N
    r=tl.rsqrt(mean_square+eps)
    y_d=x_d*r*w
    tl.store(
        y+off_x,
        y_d,
        mask=mask
    )

@triton.jit
def rmsnorm_kernel_large_program(
        X,W,Y,
        N:tl.constexpr,
        BLCOK_M:tl.constexpr,
        BLOCK_N:tl.constexpr,
        eps:tl.constexpr=1e-6
):
    pid_m=tl.program_id(0)

    rows=pid_m*BLCOK_M+tl.arange(0,BLCOK_M)
    cols=tl.arange(0,BLOCK_N)
    mask=cols<N
    off_x=rows[:,None]*N+cols
    x_d=tl.load(
        X+off_x,
        mask=mask,
        other=0.0
    ).to(tl.float32)
    w_d=tl.load(
        W+cols,
        mask=mask,
        other=0.0
    ).to(tl.float32)

    mean_square=tl.sum(x_d*x_d,axis=-1)/N
    r=tl.rsqrt(mean_square+eps)
    y_d=x_d*r[:,None]*w_d
    tl.store(
        Y+off_x,
        y_d,
        mask=mask
    )




M,N=4096,128
BLCOK_M=16
x=torch.randn((M,N),device="cuda",dtype=torch.float16)
weight = torch.randn(N, device="cuda", dtype=torch.float16)
y = torch.empty_like(x)
grid=triton.cdiv(M,BLCOK_M)


# def run():
#     rmsnorm_kernel_large_program[(grid,)](
#         x, weight, y,
#         N=N,
#         BLCOK_M=BLCOK_M,
#         BLOCK_N=triton.next_power_of_2(N),
#         eps=1e-6,
#         num_warps=4,
#     )


def run():
    rmsnorm_kernel[(M,)](
        x, weight, y,
        N=N,
        BLOCK_N=triton.next_power_of_2(N),
        eps=1e-6,
        num_warps=1,
    )


for _ in range(10):
    run()

torch.cuda.synchronize()
torch.cuda.cudart().cudaProfilerStart()
run()
torch.cuda.synchronize()
torch.cuda.cudart().cudaProfilerStop()
