from __future__ import annotations

import torch

from musa_aiter.dispatch import DispatchRecord, emit, tensor_device_dtype
from musa_aiter.policy import ensure_reference_allowed


def quantize_per_token(
    x: torch.Tensor,
    *,
    dtype: torch.dtype = torch.float8_e4m3fn,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Phase-2 per-token quantization reference API.

    Returns (quantized_tensor, scale_per_token).
    """
    if x.dim() < 2:
        raise ValueError("x must be at least 2D")

    device, orig_dtype = tensor_device_dtype(x)
    ensure_reference_allowed("quantize_per_token")
    emit(DispatchRecord("quantize_per_token", "torch-reference", device, orig_dtype, "phase2 native backend pending"))

    num_tokens = x.shape[0]
    flat = x.view(num_tokens, -1).float()
    max_val = flat.abs().amax(dim=-1, keepdim=True)
    scale = (max_val / 127.0).clamp(min=1e-12)
    quantized = (flat / scale).clamp(-127, 127).to(dtype=torch.int8)
    return quantized.view_as(x).to(dtype), scale.squeeze(-1)


def dequantize_per_token(
    x_q: torch.Tensor,
    scale: torch.Tensor,
    *,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Phase-2 per-token dequantization reference API."""
    if x_q.dim() < 2:
        raise ValueError("x_q must be at least 2D")
    if x_q.shape[0] != scale.shape[0]:
        raise ValueError("scale must match token count")

    device, orig_dtype = tensor_device_dtype(x_q)
    ensure_reference_allowed("dequantize_per_token")
    emit(DispatchRecord("dequantize_per_token", "torch-reference", device, orig_dtype, "phase2 native backend pending"))

    num_tokens = x_q.shape[0]
    flat = x_q.view(num_tokens, -1).float()
    scale_expanded = scale.view(num_tokens, 1)
    return (flat * scale_expanded).view_as(x_q).to(dtype)


def quantize_per_tensor(
    x: torch.Tensor,
    *,
    dtype: torch.dtype = torch.float8_e4m3fn,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Phase-2 per-tensor quantization reference API."""
    device, orig_dtype = tensor_device_dtype(x)
    ensure_reference_allowed("quantize_per_tensor")
    emit(DispatchRecord("quantize_per_tensor", "torch-reference", device, orig_dtype, "phase2 native backend pending"))

    max_val = x.abs().amax().float()
    scale = (max_val / 127.0).clamp(min=1e-12)
    quantized = (x.float() / scale).clamp(-127, 127).to(dtype=torch.int8)
    return quantized.to(dtype), scale


def dequantize_per_tensor(
    x_q: torch.Tensor,
    scale: torch.Tensor,
    *,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Phase-2 per-tensor dequantization reference API."""
    device, orig_dtype = tensor_device_dtype(x_q)
    ensure_reference_allowed("dequantize_per_tensor")
    emit(DispatchRecord("dequantize_per_tensor", "torch-reference", device, orig_dtype, "phase2 native backend pending"))

    return (x_q.float() * scale.float()).to(dtype)


def rmsnorm_quant(
    x: torch.Tensor,
    weight: torch.Tensor,
    *,
    eps: float = 1e-6,
    quant_dtype: torch.dtype = torch.float8_e4m3fn,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused RMSNorm + per-token quantization reference API."""
    device, orig_dtype = tensor_device_dtype(x)
    ensure_reference_allowed("rmsnorm_quant")
    emit(DispatchRecord("rmsnorm_quant", "torch-reference", device, orig_dtype, "phase2 native backend pending"))

    variance = x.float().pow(2).mean(dim=-1, keepdim=True)
    y = x.float() * torch.rsqrt(variance + eps)
    y = y.to(dtype=x.dtype) * weight
    return quantize_per_token(y, dtype=quant_dtype)


def gated_rmsnorm_quant(
    x: torch.Tensor,
    weight: torch.Tensor,
    *,
    eps: float = 1e-6,
    quant_dtype: torch.dtype = torch.float8_e4m3fn,
    group_size: int = 128,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Gated RMSNorm + group quantization reference API."""
    device, orig_dtype = tensor_device_dtype(x)
    ensure_reference_allowed("gated_rmsnorm_quant")
    emit(DispatchRecord("gated_rmsnorm_quant", "torch-reference", device, orig_dtype, "phase2 native backend pending"))

    variance = x.float().pow(2).mean(dim=-1, keepdim=True)
    y = x.float() * torch.rsqrt(variance + eps)
    y = y.to(dtype=x.dtype) * weight

    num_tokens = y.shape[0]
    hidden = y.shape[-1]
    num_groups = (hidden + group_size - 1) // group_size

    y_reshaped = y.view(num_tokens, num_groups, group_size)
    max_val = y_reshaped.abs().amax(dim=-1, keepdim=True)
    scale = (max_val / 127.0).clamp(min=1e-12).view(num_tokens, num_groups)
    quantized = (y_reshaped / scale.unsqueeze(-1)).clamp(-127, 127).view_as(y).to(dtype=torch.int8)
    return quantized.to(quant_dtype), scale
