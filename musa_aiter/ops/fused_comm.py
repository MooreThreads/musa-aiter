from __future__ import annotations

import torch

from musa_aiter.dispatch import DispatchRecord, emit, tensor_device_dtype
from musa_aiter.policy import ensure_reference_allowed
from musa_aiter.ops.communication import DistributedContext, all_reduce, reduce_scatter, all_gather


def gemm_all_reduce(
    a: torch.Tensor,
    b: torch.Tensor,
    *,
    ctx: DistributedContext | None = None,
    op: str = "sum",
) -> torch.Tensor:
    """Reference fused GEMM + all-reduce."""
    if a.shape[-1] != b.shape[-2]:
        raise ValueError("K mismatch")
    device, dtype = tensor_device_dtype(a)
    ensure_reference_allowed("gemm_all_reduce")
    emit(DispatchRecord("gemm_all_reduce", "torch-reference", device, dtype, "phase3 native fusion pending"))
    out = torch.matmul(a, b)
    return all_reduce(out, ctx=ctx, op=op)


def gemm_reduce_scatter(
    a: torch.Tensor,
    b: torch.Tensor,
    *,
    ctx: DistributedContext | None = None,
    dim: int = 0,
) -> torch.Tensor:
    """Reference fused GEMM + reduce-scatter."""
    if a.shape[-1] != b.shape[-2]:
        raise ValueError("K mismatch")
    device, dtype = tensor_device_dtype(a)
    ensure_reference_allowed("gemm_reduce_scatter")
    emit(DispatchRecord("gemm_reduce_scatter", "torch-reference", device, dtype, "phase3 native fusion pending"))
    out = torch.matmul(a, b)
    return reduce_scatter(out, ctx=ctx, dim=dim)


def all_gather_gemm(
    a: torch.Tensor,
    b: torch.Tensor,
    *,
    ctx: DistributedContext | None = None,
    dim: int = 0,
) -> torch.Tensor:
    """Reference fused all-gather + GEMM."""
    device, dtype = tensor_device_dtype(a)
    ensure_reference_allowed("all_gather_gemm")
    emit(DispatchRecord("all_gather_gemm", "torch-reference", device, dtype, "phase3 native fusion pending"))
    gathered = all_gather(a, ctx=ctx, dim=dim)
    if gathered.shape[-1] != b.shape[-2]:
        raise ValueError("K mismatch after all_gather")
    return torch.matmul(gathered, b)
