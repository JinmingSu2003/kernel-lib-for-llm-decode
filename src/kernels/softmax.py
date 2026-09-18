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
