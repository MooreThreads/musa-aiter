from __future__ import annotations

import math

import torch

from musa_aiter.attention_type import (
    AttentionType,
    coerce_attention_type,
    default_causal_for_attention_type,
)
from musa_aiter.dispatch import DispatchRecord, emit, tensor_device_dtype
from musa_aiter.policy import ensure_reference_allowed
from musa_aiter.vendors.planning import emit_vendor_planning_note
from musa_aiter.vendors.routing import resolve_dispatch_backend


def _query_batched_4d(query: torch.Tensor) -> tuple[torch.Tensor, int]:
    """``[num_seqs, H, dim] → [num_seqs, 1, H, dim]``; keeps 4D as-is."""

    if query.dim() == 3:
        return query.unsqueeze(1), 1
    if query.dim() == 4:
        return query, int(query.shape[1])
    raise ValueError(
        "query must be [num_seqs, num_q_heads, head_dim] (decode) or "
        "[num_seqs, Q, num_q_heads, head_dim] (prefill / multi-query)"
    )


def paged_attention_v1(
    out: torch.Tensor,
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    *,
    num_kv_heads: int,
    scale: float,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    block_size: int,
    max_seq_len: int | None = None,
    causal: bool | None = None,
    attention_type: AttentionType | str | None = None,
) -> torch.Tensor:
    """Paged KV attention (reference).

    **Query shapes**

    - Decode-style: ``[num_seqs, num_q_heads, head_dim]`` (implicit ``Q == 1``).
    - Prefill / chunked: ``[num_seqs, Q, num_q_heads, head_dim]`` — each row uses the
      same paged KV prefix of length ``seq_lens[i]``. When ``causal`` is resolved
      ``True``, query token ``qi`` attends to cache positions ``0 .. seq_lens - Q + qi``
      (inclusive); requires ``seq_lens[i] >= Q``.

    ``attention_type`` matches :class:`AttentionType` (vLLM / SGLang string labels).
    If ``causal`` is omitted, decoder roles default to a causal mask rule when ``Q > 1``;
    omitted ``attention_type`` + ``causal`` keeps the legacy behavior (full attention to
    all cached tokens).

    KV layout:
      ``key_cache/value_cache``: ``[num_blocks, num_kv_heads, head_dim, block_size]``
    """

    q_4d, Q = _query_batched_4d(query)
    if out.shape != query.shape:
        raise ValueError(f"out shape must match query: {tuple(out.shape)} != {tuple(query.shape)}")
    if key_cache.dim() != 4 or value_cache.dim() != 4:
        raise ValueError(
            "key_cache/value_cache must be [num_blocks, num_kv_heads, head_dim, block_size]"
        )
    num_seqs, _, num_q_heads, head_dim = q_4d.shape
    if query.device != key_cache.device or query.device != value_cache.device or query.device != out.device:
        raise ValueError("query, key_cache, value_cache, out must be on same device")
    if block_tables.device != query.device or seq_lens.device != query.device:
        raise ValueError("block_tables and seq_lens must be on same device as query")
    if key_cache.shape != value_cache.shape:
        raise ValueError("key_cache and value_cache shape must match")
    if key_cache.shape[1] != num_kv_heads:
        raise ValueError("num_kv_heads does not match key_cache.shape[1]")
    if key_cache.shape[2] != head_dim:
        raise ValueError("cache head_dim must match query head_dim")
    if key_cache.shape[3] != block_size:
        raise ValueError("block_size does not match cache shape")
    if num_q_heads % num_kv_heads != 0:
        raise ValueError("query heads must be divisible by num_kv_heads")

    at = coerce_attention_type(attention_type)
    if causal is not None:
        effective_causal = bool(causal)
    elif at is not None:
        effective_causal = default_causal_for_attention_type(at)
    else:
        effective_causal = False

    device, dtype = tensor_device_dtype(query)
    ensure_reference_allowed("paged_attention_v1")
    reason = "phase1 native backend pending"
    if at is not None:
        reason += f"; attention_type={at.value}"
    backend, v_suffix = resolve_dispatch_backend("paged_attention_v1")
    reason += v_suffix
    emit(DispatchRecord("paged_attention_v1", backend, device, dtype, reason))
    emit_vendor_planning_note("paged_attention_v1")

    repeat = num_q_heads // num_kv_heads

    for seq_idx in range(num_seqs):
        seq_len = int(seq_lens[seq_idx].item())
        if seq_len < 0:
            raise ValueError("seq_lens must be non-negative")
        if effective_causal and seq_len < Q:
            raise ValueError(
                "causal paged_attention requires seq_lens >= Q for each row"
            )
        if max_seq_len is not None and seq_len > max_seq_len:
            raise ValueError("seq_len exceeds max_seq_len")
        needed_blocks = (seq_len + block_size - 1) // block_size
        if needed_blocks > block_tables.shape[1]:
            raise ValueError("block_tables does not contain enough blocks for seq_len")
        if seq_len == 0:
            out[seq_idx].zero_()
            continue

        k_chunks = []
        v_chunks = []
        for logical_block in range(needed_blocks):
            block_id = int(block_tables[seq_idx, logical_block].item())
            if block_id < 0 or block_id >= key_cache.shape[0]:
                raise ValueError(f"invalid block id {block_id}")
            k_block = key_cache[block_id].permute(2, 0, 1)  # [block, kv_heads, D]
            v_block = value_cache[block_id].permute(2, 0, 1)
            k_chunks.append(k_block)
            v_chunks.append(v_block)
        k = torch.cat(k_chunks, dim=0)[:seq_len]
        v = torch.cat(v_chunks, dim=0)[:seq_len]
        if repeat != 1:
            k = k.repeat_interleave(repeat, dim=1)
            v = v.repeat_interleave(repeat, dim=1)

        for qi in range(Q):
            if effective_causal:
                kv_lim = seq_len - Q + qi + 1
                k_sub = k[:kv_lim]
                v_sub = v[:kv_lim]
            else:
                k_sub = k
                v_sub = v

            qh = q_4d[seq_idx, qi].float()  # [H, D]
            kh = k_sub.transpose(0, 1).float()  # [H, S', D]
            vh = v_sub.transpose(0, 1).float()
            scores = torch.einsum("hd,hsd->hs", qh, kh) * scale
            probs = torch.softmax(scores, dim=-1)
            result = torch.einsum("hs,hsd->hd", probs, vh).to(dtype=query.dtype)
            if query.dim() == 3:
                out[seq_idx].copy_(result)
            else:
                out[seq_idx, qi].copy_(result)

    return out


def paged_attention(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    *,
    num_kv_heads: int,
    block_size: int,
    scale: float | None = None,
    max_seq_len: int | None = None,
    causal: bool | None = None,
    attention_type: AttentionType | str | None = None,
) -> torch.Tensor:
    """Convenience wrapper allocating ``out``; see :func:`paged_attention_v1`."""

    out = torch.empty_like(query)
    if scale is None:
        scale = 1.0 / math.sqrt(query.shape[-1])
    return paged_attention_v1(
        out,
        query,
        key_cache,
        value_cache,
        num_kv_heads=num_kv_heads,
        scale=scale,
        block_tables=block_tables,
        seq_lens=seq_lens,
        block_size=block_size,
        max_seq_len=max_seq_len,
        causal=causal,
        attention_type=attention_type,
    )
