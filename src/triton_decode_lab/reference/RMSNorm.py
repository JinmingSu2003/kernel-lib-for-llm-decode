import torch
import triton
import triton.language as tl

def rmsnorm_ref(x,weight,eps=1e-6):
    x_fp32=x.float()
    w_fp32=weight.float()

    acc=(x_fp32*x_fp32).mean(dim=-1,keepdim=True)

    r=torch.rsqrt(acc+eps)
    y=x_fp32*r*w_fp32
    return y.to(x.dtype)

x = torch.tensor([[3.0, 4.0]], device="cuda")
w = torch.ones(2, device="cuda")

print(rmsnorm_ref(x, w))
# 约为 [[0.8485, 1.1314]]


