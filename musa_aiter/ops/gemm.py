from __future__ import annotations

import torch

from musa_aiter.dispatch import DispatchRecord, emit, tensor_device_dtype
from musa_aiter.policy import ensure_reference_allowed
from musa_aiter.tune_runtime import gemm_dispatch_reason, lookup_gemm_config
from musa_aiter.vendors.planning import emit_vendor_planning_note
from musa_aiter.vendors.routing import resolve_dispatch_backend


def gemm(a: torch.Tensor, b: torch.Tensor, *, out: torch.Tensor | None = None) -> torch.Tensor:
    """Phase-1 GEMM wrapper with torch reference backend.

    Computes `a @ b`. Intended API placeholder for future MUSA native GEMM.
    """
    if a.dim() < 2 or b.dim() < 2:
        raise ValueError("a and b must be at least 2D")
    if a.shape[-1] != b.shape[-2]:
        raise ValueError(f"K mismatch: {a.shape[-1]} != {b.shape[-2]}")
    if a.device != b.device:
        raise ValueError("a and b must be on same device")
    device, dtype = tensor_device_dtype(a)
    ensure_reference_allowed("gemm")
    tune_row = lookup_gemm_config(a, b)
    backend, v_suffix = resolve_dispatch_backend("gemm")
    reason = gemm_dispatch_reason("phase1 native backend pending", tune_row) + v_suffix
    emit(DispatchRecord("gemm", backend, device, dtype, reason))
    emit_vendor_planning_note("gemm")
    result = torch.matmul(a, b)
    if out is not None:
        if out.shape != result.shape:
            raise ValueError(f"out shape mismatch: {tuple(out.shape)} != {tuple(result.shape)}")
        out.copy_(result)
        return out
    return result
