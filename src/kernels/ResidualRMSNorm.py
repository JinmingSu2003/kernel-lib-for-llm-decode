import torch
import triton
import triton.language as tl

@triton.jit
def residual_rmsnorm_kernel(
        X,W,Y,R,Z,
        N:tl.constexpr,
        BLOCK_M:tl.constexpr,
        BLOCK_N:tl.constexpr,
        eps:tl.constexpr=1e-6
):
    pid_m=tl.program_id(0)

    rows=pid_m*BLOCK_M+tl.arange(0,BLOCK_M)
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
    r_d=tl.load(
        R+off_x,
        mask=mask,
        other=0.0
    ).to(tl.float32)

    z=x_d+r_d

    mean_square=tl.sum(z*z,axis=-1)/N
    r=tl.rsqrt(mean_square+eps)
    y_d=z*r[:,None]*w_d
    tl.store(
        Y+off_x,
        y_d,
        mask=mask
    )
    tl.store(
        Z+off_x,
        z,
        mask=mask
    )


# torch.cuda.synchronize()
# torch.cuda.cudart().cudaProfilerStart()
# run()
# torch.cuda.synchronize()
# torch.cuda.cudart().cudaProfilerStop()
