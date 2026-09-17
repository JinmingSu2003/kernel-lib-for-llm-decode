import torch
import torch.nn.functional as F
import triton
import triton.language as tl

def siluxmul_ref(gate,up):
    return F.silu(gate)*up

@triton.jit
def siluxmul_kernel(gate,up,out,
                    BLOCK_SIZE:tl.constexpr,
                    size:tl.constexpr
                    ):
    pid=tl.program_id(0)
    col=tl.arange(0,BLOCK_SIZE)
    mask=(pid*BLOCK_SIZE+col)<size
    gate_base=gate+pid*BLOCK_SIZE+col
    up_base=up+pid*BLOCK_SIZE+col
    g=tl.load(
        gate_base,
        mask=mask,
        other=0.0
    ).to(tl.float32)
    u=tl.load(
        up_base,
        mask=mask,
        other=0.0
    ).to(tl.float32)
    y=g/(1+tl.exp(-g))*u
    tl.store(
        out+pid*BLOCK_SIZE+col,
        y,
        mask=mask
    )


size=409600
x=torch.randn(size,device="cuda",dtype=torch.float16)
y=torch.empty_like(x)
u=torch.rand_like(x)
blocksize=1024
grid=triton.cdiv(size,blocksize)

def run():
    siluxmul_kernel[(grid,)](
        gate=x,
        up=u,
        out=y,
        BLOCK_SIZE=blocksize,
        size=size,
        num_warps=4
    )


out1=siluxmul_ref(x,u)
run()
ok=torch.allclose(
    y,
    out1,
    1e-3,
    1e-3
)
print(ok)





