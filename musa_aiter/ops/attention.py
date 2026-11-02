from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from musa_aiter.attention_type import (
    AttentionType,
    coerce_attention_type,
    default_causal_for_attention_type,
)
from musa_aiter.dispatch import DispatchRecord, emit, tensor_device_dtype
from musa_aiter.policy import ensure_reference_allowed
from musa_aiter.vendors.planning import emit_vendor_planning_note


def _manual_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, causal: bool, scale: float) -> torch.Tensor:
    # q/k/v expected [B, S, H, D]. Compute as [B, H, S, D].
    qh = q.transpose(1, 2).float()
    kh = k.transpose(1, 2).float()
    vh = v.transpose(1, 2).float()
    scores = torch.matmul(qh, kh.transpose(-2, -1)) * scale
    if causal:
        sq = q.shape[1]
        sk = k.shape[1]
        mask = torch.ones((sq, sk), dtype=torch.bool, device=q.device).tril(diagonal=sk - sq)
        scores = scores.masked_fill(~mask, float("-inf"))
    probs = torch.softmax(scores, dim=-1)
    out = torch.matmul(probs, vh).transpose(1, 2)
    return out.to(dtype=q.dtype)


def flash_attn_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    causal: bool | None = None,
    softmax_scale: float | None = None,
    attention_type: AttentionType | str | None = None,
) -> torch.Tensor:
    """Dense FlashAttention-style Phase-1 reference API.

    Shape contract: q/k/v are [batch, seq, heads, head_dim].
    Native MUSA backend is pending; this uses PyTorch SDPA/reference fallback.

    ``attention_type`` mirrors **vLLM** / **SGLang** attention labels (string ``Enum``).
    If ``causal`` is ``None`` and ``attention_type`` is set, a default mask policy
    applies (:data:`AttentionType.DECODER` → causal; other roles → non-causal).
    If ``causal`` is explicitly ``True``/``False``, it always wins.
    """
    if q.dim() != 4 or k.dim() != 4 or v.dim() != 4:
        raise ValueError("q, k, v must be 4D tensors [B, S, H, D]")
    if q.device != k.device or q.device != v.device:
        raise ValueError("q, k, v must be on the same device")
    if q.dtype != k.dtype or q.dtype != v.dtype:
        raise ValueError("q, k, v must have the same dtype")
    if q.shape[0] != k.shape[0] or q.shape[0] != v.shape[0]:
        raise ValueError("batch dimensions must match")
    if k.shape[1] != v.shape[1]:
        raise ValueError("k and v sequence lengths must match")
    if k.shape[2] != v.shape[2]:
        raise ValueError("k and v head counts must match")
    if q.shape[2] % k.shape[2] != 0:
        raise ValueError("q heads must be divisible by kv heads")
    if q.shape[-1] != k.shape[-1]:
        raise ValueError("q and k head_dim must match")

    scale = softmax_scale if softmax_scale is not None else 1.0 / math.sqrt(q.shape[-1])
    device, dtype = tensor_device_dtype(q)
    at = coerce_attention_type(attention_type)
    if causal is not None:
        effective_causal = bool(causal)
    elif at is not None:
        effective_causal = default_causal_for_attention_type(at)
    else:
        effective_causal = False

    ensure_reference_allowed("flash_attn_func")
    reason = "phase1 native backend pending"
    if at is not None:
        reason += f"; attention_type={at.value}"
    emit(DispatchRecord("flash_attn_func", "torch-sdpa-reference", device, dtype, reason))
    emit_vendor_planning_note("flash_attn_func")

    # Expand GQA/MQA KV heads for reference implementation.
    if q.shape[2] != k.shape[2]:
        repeat = q.shape[2] // k.shape[2]
        k = k.repeat_interleave(repeat, dim=2)
        v = v.repeat_interleave(repeat, dim=2)

    try:
        qh = q.transpose(1, 2)
        kh = k.transpose(1, 2)
        vh = v.transpose(1, 2)
        out = F.scaled_dot_product_attention(qh, kh, vh, attn_mask=None, dropout_p=0.0, is_causal=effective_causal, scale=scale)
        return out.transpose(1, 2).contiguous()
    except Exception:
        return _manual_attention(q, k, v, causal=effective_causal, scale=scale)
