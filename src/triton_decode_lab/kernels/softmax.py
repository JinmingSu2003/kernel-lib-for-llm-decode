import torch
import triton
import triton.language as tl
import triton.testing

@triton.jit
def softmax_kernel(X,OUT,
                   BLOCK_N:tl.constexpr,
                   M:tl.constexpr,
                   BLOCK_M:tl.constexpr
                   ):
    pid=tl.program_id(0)
    x_base=X+pid*BLOCK_N*M
    out_base=OUT+pid*BLOCK_N*M
    row=tl.arange(0,BLOCK_N)
    l=tl.zeros((BLOCK_N,),dtype=tl.float32)
    m=tl.full(
        (BLOCK_N,),
        -float("inf"),
        dtype=tl.float32
    )
    for start in range (0,M,BLOCK_M):
        col=start+tl.arange(0,BLOCK_M)
        x=tl.load(
            x_base+row[:,None]*M+col
        ).to(tl.float32)
        m_new=tl.maximum(m,tl.max(x,axis=-1))
        l=l*tl.exp(m-m_new)+tl.sum(tl.exp(x-m_new[:,None]),axis=1)
        m=m_new
    for start in range (0,M,BLOCK_M):
        col=start+tl.arange(0,BLOCK_M)
        x=tl.load(
            x_base+row[:,None]*M+col
        ).to(tl.float32)
        y=tl.exp(x-m[:,None])/l[:, None]
        tl.store(
            out_base+row[:,None]*M+col,
            y
        )

M,N=512,409600
x=torch.randn((M,N),device="cuda",dtype=torch.float16)
y=torch.zeros_like(x)
y_ref=y
block_n=4
block_m=1024
grid=triton.cdiv(M,block_n)
def run():
    softmax_kernel[(grid,)](
        x,y,block_n,N,block_m,
        num_warps=4
    )

def run_ref():
    torch.softmax(x, dim=-1, out=y_ref)

for _ in range (10):
    run()
    run_ref()

torch.cuda.synchronize()

torch_us = triton.testing.do_bench(run_ref) * 1000
triton_us = triton.testing.do_bench(run) * 1000

print(f"PyTorch Softmax: {torch_us:.3f} us")
print(f"Triton  Softmax: {triton_us:.3f} us")
print(f"加速比: {torch_us / triton_us:.2f}x")



