from __future__ import annotations

import torch

from musa_aiter.dispatch import DispatchRecord, emit, tensor_device_dtype
from musa_aiter.policy import ensure_reference_allowed
from musa_aiter.ops.quant import (
    quantize_per_tensor,
    dequantize_per_tensor,
    quantize_per_token,
    dequantize_per_token,
)


def gemm_a8w8(
    a: torch.Tensor,
    b: torch.Tensor,
    *,
    a_scale: torch.Tensor | None = None,
    b_scale: torch.Tensor | None = None,
    out_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Phase-2 INT8/FP8-style GEMM reference API.

    If scales are not provided, this function quantizes/dequantizes per tensor
    and computes a reference matmul. Native low-precision MUSA GEMM is pending.
    """
    if a.dim() < 2 or b.dim() < 2:
        raise ValueError("a and b must be at least 2D")
    if a.shape[-1] != b.shape[-2]:
        raise ValueError(f"K mismatch: {a.shape[-1]} != {b.shape[-2]}")
    if a.device != b.device:
        raise ValueError("a and b must be on same device")

    device, dtype = tensor_device_dtype(a)
    ensure_reference_allowed("gemm_a8w8")
    emit(DispatchRecord("gemm_a8w8", "torch-reference", device, dtype, "phase2 native backend pending"))

    if a_scale is None:
        a_q, a_scale = quantize_per_tensor(a.float(), dtype=torch.float8_e4m3fn)
        a_ref = dequantize_per_tensor(a_q, a_scale, dtype=torch.float32)
    else:
        a_ref = a.float() * a_scale.float()
    if b_scale is None:
        b_q, b_scale = quantize_per_tensor(b.float(), dtype=torch.float8_e4m3fn)
        b_ref = dequantize_per_tensor(b_q, b_scale, dtype=torch.float32)
    else:
        b_ref = b.float() * b_scale.float()
    return torch.matmul(a_ref, b_ref).to(out_dtype)


def gemm_a8w8_per_token(
    a: torch.Tensor,
    b: torch.Tensor,
    *,
    out_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Phase-2 per-token activation quantized GEMM reference API."""
    if a.dim() != 2 or b.dim() != 2:
        raise ValueError("a and b must be 2D")
    if a.shape[-1] != b.shape[-2]:
        raise ValueError("K mismatch")
    device, dtype = tensor_device_dtype(a)
    ensure_reference_allowed("gemm_a8w8_per_token")
    emit(DispatchRecord("gemm_a8w8_per_token", "torch-reference", device, dtype, "phase2 native backend pending"))

    a_q, a_scale = quantize_per_token(a.float(), dtype=torch.float8_e4m3fn)
    a_ref = dequantize_per_token(a_q, a_scale, dtype=torch.float32)
    b_q, b_scale = quantize_per_tensor(b.float(), dtype=torch.float8_e4m3fn)
    b_ref = dequantize_per_tensor(b_q, b_scale, dtype=torch.float32)
    return torch.matmul(a_ref, b_ref).to(out_dtype)


def gemm_a4w4(
    a: torch.Tensor,
    b: torch.Tensor,
    *,
    out_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Phase-2 INT4-style GEMM reference API.

    Simulates signed 4-bit quantization with range [-7, 7]. Native packed INT4
    MUSA GEMM is pending.
    """
    if a.shape[-1] != b.shape[-2]:
        raise ValueError("K mismatch")
    device, dtype = tensor_device_dtype(a)
    ensure_reference_allowed("gemm_a4w4")
    emit(DispatchRecord("gemm_a4w4", "torch-reference", device, dtype, "phase2 native backend pending"))

    def q4(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        scale = (x.float().abs().amax() / 7.0).clamp(min=1e-12)
        return (x.float() / scale).round().clamp(-7, 7), scale

    a_q, a_s = q4(a)
    b_q, b_s = q4(b)
    return torch.matmul(a_q * a_s, b_q * b_s).to(out_dtype)


def grouped_gemm_a8w8(
    a: torch.Tensor,
    b: torch.Tensor,
    m_indices: torch.Tensor,
    *,
    out_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Phase-2 grouped low-precision GEMM reference API."""
    if a.dim() != 2 or b.dim() != 3:
        raise ValueError("a must be 2D and b must be 3D")
    if m_indices.numel() != a.shape[0]:
        raise ValueError("m_indices length mismatch")
    device, dtype = tensor_device_dtype(a)
    ensure_reference_allowed("grouped_gemm_a8w8")
    emit(DispatchRecord("grouped_gemm_a8w8", "torch-reference", device, dtype, "phase2 native backend pending"))

    out = torch.empty(a.shape[0], b.shape[-1], device=a.device, dtype=out_dtype)
    for expert in range(b.shape[0]):
        mask = m_indices == expert
        if mask.sum() == 0:
            continue
        out[mask] = gemm_a8w8_per_token(a[mask], b[expert], out_dtype=out_dtype)
    return out
