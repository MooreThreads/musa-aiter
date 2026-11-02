from __future__ import annotations

import math

import torch

from musa_aiter.dispatch import DispatchRecord, emit, tensor_device_dtype
from musa_aiter.policy import ensure_reference_allowed


def mla_decode_fwd(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    k_pe_cache: torch.Tensor,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    *,
    num_q_heads: int,
    block_size: int,
    softmax_scale: float | None = None,
) -> torch.Tensor:
    """Phase-2 MLA decode reference API.

    Multi-head Latent Attention (MLA) decode with KV cache.

    Shape contract (simplified reference):
      q:            [num_seqs, num_q_heads, head_dim]
      kv_cache:     [num_blocks, block_size, kv_hidden_dim]
      k_pe_cache:   [num_blocks, block_size, pe_hidden_dim]
      block_tables: [num_seqs, max_blocks_per_seq]
      seq_lens:     [num_seqs]

    Returns:
      output: [num_seqs, num_q_heads, head_dim]
    """
    if q.dim() != 3:
        raise ValueError("q must be [num_seqs, num_q_heads, head_dim]")
    if kv_cache.dim() != 3 or k_pe_cache.dim() != 3:
        raise ValueError("caches must be 3D")
    if block_tables.dim() != 2 or seq_lens.dim() != 1:
        raise ValueError("block_tables must be 2D, seq_lens must be 1D")

    num_seqs, num_q_heads, head_dim = q.shape
    device = q.device
    ensure_reference_allowed("mla_decode_fwd")
    emit(DispatchRecord("mla_decode_fwd", "torch-reference", str(device), str(q.dtype), "phase2 native backend pending"))

    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(head_dim)

    out = torch.empty_like(q)
    for seq_idx in range(num_seqs):
        seq_len = int(seq_lens[seq_idx].item())
        if seq_len == 0:
            out[seq_idx].zero_()
            continue
        needed_blocks = (seq_len + block_size - 1) // block_size
        if needed_blocks > block_tables.shape[1]:
            raise ValueError("insufficient blocks")

        kv_chunks = []
        pe_chunks = []
        for logical_block in range(needed_blocks):
            block_id = int(block_tables[seq_idx, logical_block].item())
            kv_chunks.append(kv_cache[block_id])
            pe_chunks.append(k_pe_cache[block_id])

        kv = torch.cat(kv_chunks, dim=0)[:seq_len]
        pe = torch.cat(pe_chunks, dim=0)[:seq_len]

        # For each head, compute attention over KV cache
        for head_idx in range(num_q_heads):
            q_head = q[seq_idx, head_idx].float()  # [head_dim]
            scores = torch.einsum(
                "d,sd->s",
                q_head,
                kv.float(),
            ) * softmax_scale
            probs = torch.softmax(scores, dim=-1)
            result = torch.einsum("s,sd->d", probs, kv.float())
            out[seq_idx, head_idx].copy_(result.to(dtype=q.dtype))

    return out


def mla_prefill_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    cu_seqlens: torch.Tensor | None = None,
    softmax_scale: float | None = None,
    causal: bool = True,
) -> torch.Tensor:
    """Phase-2 MLA prefill reference API.

    Multi-head Latent Attention prefill.

    Shape contract (dense):
      q/k/v: [batch, seq, heads, head_dim]

    Returns:
      output: [batch, seq, heads, head_dim]
    """
    if q.dim() != 4 or k.dim() != 4 or v.dim() != 4:
        raise ValueError("q/k/v must be 4D")
    if q.device != k.device or q.device != v.device:
        raise ValueError("q/k/v must be on same device")

    device = q.device
    ensure_reference_allowed("mla_prefill_fwd")
    emit(DispatchRecord("mla_prefill_fwd", "torch-reference", str(device), str(q.dtype), "phase2 native backend pending"))

    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(q.shape[-1])

    batch, seq_q, heads_q, head_dim = q.shape
    _, seq_kv, heads_kv, _ = k.shape
    repeat = heads_q // heads_kv

    # Expand KV heads if GQA
    if repeat != 1:
        k_expanded = k.repeat_interleave(repeat, dim=2)
        v_expanded = v.repeat_interleave(repeat, dim=2)
    else:
        k_expanded = k
        v_expanded = v

    qh = q.transpose(1, 2).float()   # [batch, heads, seq_q, head_dim]
    kh = k_expanded.transpose(1, 2).float()
    vh = v_expanded.transpose(1, 2).float()

    scores = torch.matmul(qh, kh.transpose(-2, -1)) * softmax_scale
    if causal:
        mask = torch.ones((seq_q, seq_kv), dtype=torch.bool, device=q.device).tril(diagonal=seq_kv - seq_q)
        scores = scores.masked_fill(~mask, float("-inf"))

    probs = torch.softmax(scores, dim=-1)
    out = torch.matmul(probs, vh).transpose(1, 2)
    return out.to(dtype=q.dtype)
