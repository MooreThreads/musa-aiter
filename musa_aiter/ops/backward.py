from __future__ import annotations

import torch

from musa_aiter.dispatch import DispatchRecord, emit, tensor_device_dtype
from musa_aiter.policy import ensure_reference_allowed


def rmsnorm_backward_ref(
    grad_out: torch.Tensor,
    x: torch.Tensor,
    weight: torch.Tensor,
    *,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reference RMSNorm backward using PyTorch autograd."""
    if grad_out.shape != x.shape:
        raise ValueError("grad_out and x shapes must match")
    if x.shape[-1] != weight.shape[-1]:
        raise ValueError("weight last dim must match x")
    device, dtype = tensor_device_dtype(x)
    ensure_reference_allowed("rmsnorm_backward_ref")
    emit(DispatchRecord("rmsnorm_backward_ref", "torch-autograd-reference", device, dtype, "phase3 native backward pending"))

    x_req = x.detach().clone().requires_grad_(True)
    w_req = weight.detach().clone().requires_grad_(True)
    variance = x_req.float().pow(2).mean(dim=-1, keepdim=True)
    y = x_req.float() * torch.rsqrt(variance + eps)
    y = y.to(dtype=x_req.dtype) * w_req
    y.backward(grad_out)
    return x_req.grad, w_req.grad


def gemm_backward_ref(
    grad_out: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reference GEMM backward for y = a @ b."""
    if grad_out.shape[-2] != a.shape[-2] or grad_out.shape[-1] != b.shape[-1]:
        raise ValueError("grad_out shape incompatible with a @ b")
    device, dtype = tensor_device_dtype(a)
    ensure_reference_allowed("gemm_backward_ref")
    emit(DispatchRecord("gemm_backward_ref", "torch-reference", device, dtype, "phase3 native backward pending"))
    grad_a = torch.matmul(grad_out, b.transpose(-2, -1))
    grad_b = torch.matmul(a.transpose(-2, -1), grad_out)
    return grad_a, grad_b


def attention_backward_ref(
    grad_out: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    causal: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Reference dense attention backward using PyTorch autograd."""
    if grad_out.shape != q.shape:
        raise ValueError("grad_out shape must match attention output/q shape")
    device, dtype = tensor_device_dtype(q)
    ensure_reference_allowed("attention_backward_ref")
    emit(DispatchRecord("attention_backward_ref", "torch-autograd-reference", device, dtype, "phase3 native backward pending"))

    from musa_aiter.ops.attention import flash_attn_func

    q_req = q.detach().clone().requires_grad_(True)
    k_req = k.detach().clone().requires_grad_(True)
    v_req = v.detach().clone().requires_grad_(True)
    out = flash_attn_func(q_req, k_req, v_req, causal=causal)
    out.backward(grad_out)
    return q_req.grad, k_req.grad, v_req.grad
