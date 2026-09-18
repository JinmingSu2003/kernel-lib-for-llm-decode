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

M,N=4096,128
BLOCK_M=16
x=torch.randn((M,N),device="cuda",dtype=torch.float16)
weight=torch.randn((N),device="cuda",dtype=torch.float16)
y=torch.empty_like(x)
residual=torch.rand_like(x)
z=torch.empty_like(x)
grid=triton.cdiv(M,BLOCK_M)
def run():
    residual_rmsnorm_kernel[(grid,)](
        x,weight,y,residual,z,N,BLOCK_M,
        BLOCK_N=triton.next_power_of_2(N),
        eps=1e-6
    )

z_ref=(x.float() + residual.float()).to(torch.float16)
y_ref=torch.rms_norm(z_ref,(N,),weight=weight,eps=1e-6).to(torch.float16)

run()

ok1=torch.allclose(
    z,
    z_ref,
    1e-3,
    1e-3
)
print(ok1)

ok2=torch.allclose(
    y,
    y_ref,
    1e-3,
    1e-3
)
print(ok2)


# torch.cuda.synchronize()
# torch.cuda.cudart().cudaProfilerStart()
# run()
# torch.cuda.synchronize()
# torch.cuda.cudart().cudaProfilerStop()
