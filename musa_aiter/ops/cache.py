from __future__ import annotations

import torch

from musa_aiter.dispatch import DispatchRecord, emit, tensor_device_dtype
from musa_aiter.policy import ensure_reference_allowed


def apply_rope(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    *,
    interleaved: bool = False,
) -> torch.Tensor:
    """Apply rotary positional embedding to a tensor.

    Shape contract:
      x:   [..., seq, heads, head_dim] or any shape whose last dim is head_dim
      cos: broadcastable to x[..., :head_dim/2]
      sin: broadcastable to x[..., :head_dim/2]

    Phase-1 reference implementation.
    """
    if x.shape[-1] % 2 != 0:
        raise ValueError("head_dim must be even for RoPE")
    device, dtype = tensor_device_dtype(x)
    ensure_reference_allowed("apply_rope")
    emit(DispatchRecord("apply_rope", "torch-reference", device, dtype, "phase1 native backend pending"))

    half = x.shape[-1] // 2
    if interleaved:
        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]
        out_even = x_even * cos - x_odd * sin
        out_odd = x_even * sin + x_odd * cos
        return torch.stack((out_even, out_odd), dim=-1).flatten(-2).to(dtype=x.dtype)
    x1 = x[..., :half]
    x2 = x[..., half:]
    out1 = x1 * cos - x2 * sin
    out2 = x1 * sin + x2 * cos
    return torch.cat((out1, out2), dim=-1).to(dtype=x.dtype)


def apply_rope_to_key_cache(
    key_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    positions: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    *,
    interleaved: bool = False,
) -> torch.Tensor:
    if key_cache.dim() != 4:
        raise ValueError("apply_rope_to_key_cache: key_cache must be [blocks, heads, head_dim, block_size]")
    if slot_mapping.dim() != 1 or positions.dim() != 1 or slot_mapping.numel() != positions.numel():
        raise ValueError("apply_rope_to_key_cache: slot_mapping and positions must be same-length vectors")
    if slot_mapping.dtype not in (torch.int16, torch.int32, torch.int64):
        raise ValueError("apply_rope_to_key_cache: slot_mapping must use an integer dtype")
    if positions.dtype not in (torch.int16, torch.int32, torch.int64):
        raise ValueError("apply_rope_to_key_cache: positions must use an integer dtype")
    if key_cache.shape[2] % 2 != 0:
        raise ValueError("apply_rope_to_key_cache: head_dim must be even for RoPE")
    if slot_mapping.device != key_cache.device or positions.device != key_cache.device:
        raise ValueError("apply_rope_to_key_cache: slot_mapping and positions must be on key_cache device")
    if cos.device != key_cache.device or sin.device != key_cache.device:
        raise ValueError("apply_rope_to_key_cache: cos/sin must be on key_cache device")
    if cos.shape != sin.shape or cos.dim() not in (1, 2):
        raise ValueError("apply_rope_to_key_cache: cos/sin must be [half_dim] or [max_positions, half_dim]")
    half = key_cache.shape[2] // 2
    if cos.shape[-1] != half:
        raise ValueError("apply_rope_to_key_cache: cos/sin last dim must equal head_dim / 2")

    device, dtype = tensor_device_dtype(key_cache)
    ensure_reference_allowed("apply_rope_to_key_cache")
    emit(DispatchRecord("apply_rope_to_key_cache", "torch-reference", device, dtype, "Phase PKV key-cache RoPE reposition reference"))

    block_size = int(key_cache.shape[-1])
    num_blocks = int(key_cache.shape[0])
    slots_cpu = slot_mapping.to(device="cpu", dtype=torch.long)
    positions_cpu = positions.to(device="cpu", dtype=torch.long)
    for token_idx, slot in enumerate(slots_cpu.tolist()):
        if slot < 0:
            continue
        block_id = slot // block_size
        offset = slot - block_id * block_size
        if block_id < 0 or block_id >= num_blocks:
            raise ValueError(f"apply_rope_to_key_cache: slot maps to invalid block_id={block_id}")
        position = int(positions_cpu[token_idx].item())
        if position < 0:
            raise ValueError("apply_rope_to_key_cache: positions must be non-negative")
        if cos.dim() == 1:
            cos_row = cos
            sin_row = sin
        else:
            if position >= cos.shape[0]:
                raise ValueError("apply_rope_to_key_cache: position exceeds cos/sin table")
            cos_row = cos[position]
            sin_row = sin[position]
        rotated = apply_rope(key_cache[block_id, :, :, offset], cos_row, sin_row, interleaved=interleaved)
        key_cache[block_id, :, :, offset].copy_(rotated)
    return key_cache


def copy_to_kv_cache(
    k: torch.Tensor,
    v: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
) -> None:
    """Reference KV-cache update.

    Shape contract:
      k/v:         [num_tokens, num_kv_heads, head_dim]
      key_cache:   [num_blocks, num_kv_heads, head_dim, block_size]
      value_cache: [num_blocks, num_kv_heads, head_dim, block_size]
      slot_mapping:[num_tokens] flattened slot id = block_id * block_size + offset
    """
    if k.shape != v.shape:
        raise ValueError("k and v shapes must match")
    if key_cache.shape != value_cache.shape:
        raise ValueError("key_cache and value_cache shapes must match")
    if k.dim() != 3 or key_cache.dim() != 4:
        raise ValueError("k/v must be 3D and caches must be 4D")
    if slot_mapping.numel() != k.shape[0]:
        raise ValueError("slot_mapping length must equal number of tokens")
    if k.device != key_cache.device or v.device != value_cache.device or slot_mapping.device != k.device:
        raise ValueError("all tensors must be on same device")
    device, dtype = tensor_device_dtype(k)
    ensure_reference_allowed("copy_to_kv_cache")
    emit(DispatchRecord("copy_to_kv_cache", "torch-reference", device, dtype, "phase1 native backend pending"))

    block_size = key_cache.shape[-1]
    for token_idx in range(k.shape[0]):
        slot = int(slot_mapping[token_idx].item())
        if slot < 0:
            continue
        block_id = slot // block_size
        offset = slot % block_size
        if block_id >= key_cache.shape[0]:
            raise ValueError(f"slot maps to invalid block_id={block_id}")
        key_cache[block_id, :, :, offset].copy_(k[token_idx])
        value_cache[block_id, :, :, offset].copy_(v[token_idx])
