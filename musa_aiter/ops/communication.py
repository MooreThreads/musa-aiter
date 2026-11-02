from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from musa_aiter.dispatch import DispatchRecord, emit, tensor_device_dtype
from musa_aiter.policy import ensure_reference_allowed


@dataclass
class DistributedContext:
    world_size: int = 1
    rank: int = 0
    backend: str = "reference"
    initialized: bool = True


def init_dist_env(
    tensor_model_parallel_size: int = 1,
    rank: int = 0,
    *,
    backend: str = "reference",
    **_: Any,
) -> DistributedContext:
    """Phase-3 reference distributed context initializer.

    This does not initialize torch.distributed. It provides an API-compatible
    placeholder for single-process reference tests while native/MUSA distributed
    support is unavailable in this environment.
    """
    if tensor_model_parallel_size < 1:
        raise ValueError("tensor_model_parallel_size must be >= 1")
    if rank < 0 or rank >= tensor_model_parallel_size:
        raise ValueError("rank must be in [0, tensor_model_parallel_size)")
    return DistributedContext(world_size=tensor_model_parallel_size, rank=rank, backend=backend)


def destroy_dist_env(ctx: DistributedContext | None = None) -> None:
    if ctx is not None:
        ctx.initialized = False


def all_reduce(x: torch.Tensor, *, ctx: DistributedContext | None = None, op: str = "sum") -> torch.Tensor:
    """Reference all-reduce.

    Single-rank reference returns input clone for sum/mean/max/min. Multi-rank is
    intentionally unsupported until a real communication backend is selected.
    """
    device, dtype = tensor_device_dtype(x)
    ensure_reference_allowed("all_reduce")
    emit(DispatchRecord("all_reduce", "torch-reference", device, dtype, "phase3 communication backend pending"))
    world_size = ctx.world_size if ctx else 1
    if world_size != 1:
        raise NotImplementedError("reference all_reduce only supports world_size=1")
    if op not in {"sum", "mean", "max", "min"}:
        raise ValueError("op must be one of sum/mean/max/min")
    return x.clone()


def reduce_scatter(x: torch.Tensor, *, ctx: DistributedContext | None = None, dim: int = 0) -> torch.Tensor:
    """Reference reduce-scatter.

    world_size=1 returns a clone. Multi-rank requires native/distributed backend.
    """
    device, dtype = tensor_device_dtype(x)
    ensure_reference_allowed("reduce_scatter")
    emit(DispatchRecord("reduce_scatter", "torch-reference", device, dtype, "phase3 communication backend pending"))
    world_size = ctx.world_size if ctx else 1
    if world_size != 1:
        raise NotImplementedError("reference reduce_scatter only supports world_size=1")
    return x.clone()


def all_gather(x: torch.Tensor, *, ctx: DistributedContext | None = None, dim: int = 0) -> torch.Tensor:
    """Reference all-gather.

    world_size=1 returns a clone. Multi-rank requires native/distributed backend.
    """
    device, dtype = tensor_device_dtype(x)
    ensure_reference_allowed("all_gather")
    emit(DispatchRecord("all_gather", "torch-reference", device, dtype, "phase3 communication backend pending"))
    world_size = ctx.world_size if ctx else 1
    if world_size != 1:
        raise NotImplementedError("reference all_gather only supports world_size=1")
    return x.clone()


def fused_all_reduce_rmsnorm(
    x: torch.Tensor,
    weight: torch.Tensor,
    *,
    ctx: DistributedContext | None = None,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Reference fused all-reduce + RMSNorm."""
    reduced = all_reduce(x, ctx=ctx, op="sum")
    variance = reduced.float().pow(2).mean(dim=-1, keepdim=True)
    out = reduced.float() * torch.rsqrt(variance + eps)
    return (out.to(dtype=x.dtype) * weight).to(dtype=x.dtype)
