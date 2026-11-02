from __future__ import annotations

import torch

from musa_aiter.dispatch import DispatchRecord, emit, tensor_device_dtype
from musa_aiter.policy import ensure_reference_allowed
from musa_aiter.ops.smoke_rmsnorm import build_native_rmsnorm, try_native_rmsnorm
from musa_aiter.vendors.planning import emit_vendor_planning_note


def rmsnorm(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
    *,
    prefer_native: bool = False,
) -> torch.Tensor:
    """Reference RMSNorm API for Phase 1.

    Args:
        x: input tensor with hidden dimension at the last axis.
        weight: scale tensor broadcastable to the last dimension of x.
        eps: numerical epsilon.
        prefer_native: when ``True``, attempt MUSA float32 ctypes JIT path (requires
          ``MUSA_AITER_NATIVE_SMOKE=1``, 1-D ``weight``, last dim ``≤ RMSNORM_NATIVE_MAX_DIM``);
          falls back to the PyTorch recipe otherwise.

    Raises:
        ValueError: inconsistent shapes/devices.
    """
    if x.shape[-1] != weight.shape[-1]:
        raise ValueError(f"last dim mismatch: x.shape[-1]={x.shape[-1]} weight.shape[-1]={weight.shape[-1]}")
    if x.device != weight.device:
        raise ValueError(f"device mismatch: {x.device} != {weight.device}")
    device, dtype = tensor_device_dtype(x)
    ensure_reference_allowed("rmsnorm")

    if prefer_native:
        native = try_native_rmsnorm(x, weight, eps)
        if native is not None:
            emit(DispatchRecord("rmsnorm", "ctypes-musa-native", device, dtype, "native smoke rmsnorm opt-in"))
            emit_vendor_planning_note("rmsnorm")
            return native
        build_native_rmsnorm()

    msg = (
        "native unavailable or constraints not met (float32 musa 1-D weight smoke)"
        if prefer_native
        else "phase1 native backend pending"
    )
    emit(DispatchRecord("rmsnorm", "torch-reference", device, dtype, msg))
    emit_vendor_planning_note("rmsnorm")
    variance = x.float().pow(2).mean(dim=-1, keepdim=True)
    y = x.float() * torch.rsqrt(variance + eps)
    y = y.to(dtype=x.dtype) * weight
    return y
