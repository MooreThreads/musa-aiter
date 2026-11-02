from __future__ import annotations

import torch

from musa_aiter.dispatch import DispatchRecord, emit, tensor_device_dtype
from musa_aiter.policy import ensure_reference_allowed
from musa_aiter.tune_runtime import lookup_fused_moe_config, tuning_dispatch_reason
from musa_aiter.vendors.planning import emit_vendor_planning_note
from musa_aiter.vendors.routing import resolve_dispatch_backend


def topk_softmax(
    gating_output: torch.Tensor,
    top_k: int,
    *,
    out_weights: torch.Tensor | None = None,
    out_indices: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Phase-2 MoE top-k softmax reference API.

    Shape contract:
      gating_output: [num_tokens, num_experts]
      out_weights:   [num_tokens, top_k]
      out_indices:   [num_tokens, top_k]

    Returns (weights, indices) tensors for MoE routing.
    """
    if gating_output.dim() != 2:
        raise ValueError("gating_output must be 2D [num_tokens, num_experts]")
    if top_k < 1 or top_k > gating_output.shape[-1]:
        raise ValueError(f"top_k={top_k} out of range [1, {gating_output.shape[-1]}]")

    device, dtype = tensor_device_dtype(gating_output)
    ensure_reference_allowed("topk_softmax")
    emit(DispatchRecord("topk_softmax", "torch-reference", device, dtype, "phase2 native backend pending"))

    weights, indices = torch.topk(gating_output, top_k, dim=-1)
    weights = torch.softmax(weights, dim=-1)

    if out_weights is not None:
        if out_weights.shape != weights.shape:
            raise ValueError(f"out_weights shape mismatch: {tuple(out_weights.shape)} != {tuple(weights.shape)}")
        out_weights.copy_(weights)
    if out_indices is not None:
        if out_indices.shape != indices.shape:
            raise ValueError(f"out_indices shape mismatch: {tuple(out_indices.shape)} != {tuple(indices.shape)}")
        out_indices.copy_(indices)

    return weights if out_weights is None else out_weights, indices if out_indices is None else out_indices


def topk_sigmoid(
    gating_output: torch.Tensor,
    top_k: int,
    *,
    out_weights: torch.Tensor | None = None,
    out_indices: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Phase-2 MoE top-k sigmoid reference API.

    Same contract as topk_softmax but uses sigmoid normalization.
    """
    if gating_output.dim() != 2:
        raise ValueError("gating_output must be 2D")
    if top_k < 1 or top_k > gating_output.shape[-1]:
        raise ValueError(f"top_k={top_k} out of range")

    device, dtype = tensor_device_dtype(gating_output)
    ensure_reference_allowed("topk_sigmoid")
    emit(DispatchRecord("topk_sigmoid", "torch-reference", device, dtype, "phase2 native backend pending"))

    weights, indices = torch.topk(gating_output, top_k, dim=-1)
    weights = torch.sigmoid(weights)

    if out_weights is not None:
        if out_weights.shape != weights.shape:
            raise ValueError(f"out_weights shape mismatch")
        out_weights.copy_(weights)
    if out_indices is not None:
        if out_indices.shape != indices.shape:
            raise ValueError(f"out_indices shape mismatch")
        out_indices.copy_(indices)

    return weights if out_weights is None else out_weights, indices if out_indices is None else out_indices


def moe_sorting(
    topk_indices: torch.Tensor,
    num_experts: int,
    block_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Phase-2 MoE token sorting reference API.

    Groups tokens by expert and computes sorting metadata for grouped GEMM.

    Args:
      topk_indices: [num_tokens, top_k] expert indices
      num_experts: total number of experts
      block_size: block size for padding alignment

    Returns:
      sorted_token_ids, expert_ids, tokens_per_expert, num_tokens_post_pad
    """
    if topk_indices.dim() != 2:
        raise ValueError("topk_indices must be 2D")

    device, dtype = tensor_device_dtype(topk_indices)
    ensure_reference_allowed("moe_sorting")
    emit(DispatchRecord("moe_sorting", "torch-reference", device, dtype, "phase2 native backend pending"))

    num_tokens, top_k = topk_indices.shape
    tokens_per_expert = torch.zeros(num_experts, dtype=torch.int64, device=device)
    for expert_id in range(num_experts):
        tokens_per_expert[expert_id] = (topk_indices == expert_id).sum()

    sorted_token_ids = []
    expert_ids = []
    for token_idx in range(num_tokens):
        for k_idx in range(top_k):
            expert_id = int(topk_indices[token_idx, k_idx].item())
            sorted_token_ids.append(token_idx * top_k + k_idx)
            expert_ids.append(expert_id)

    sorted_token_ids = torch.tensor(sorted_token_ids, dtype=torch.int64, device=device)
    expert_ids = torch.tensor(expert_ids, dtype=torch.int64, device=device)
    num_tokens_post_pad = torch.tensor(0, dtype=torch.int64, device=device)

    return sorted_token_ids, expert_ids, tokens_per_expert, num_tokens_post_pad


def fused_moe(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weight: torch.Tensor,
    topk_ids: torch.Tensor,
    *,
    activation: str = "silu",
    quant_type: str = "none",
    w1_scale: torch.Tensor | None = None,
    w2_scale: torch.Tensor | None = None,
) -> torch.Tensor:
    """Phase-2 Fused MoE reference API.

    Shape contract:
      hidden_states: [num_tokens, hidden_dim]
      w1:            [num_experts, inter_dim * 2, hidden_dim]
      w2:            [num_experts, hidden_dim, inter_dim]
      topk_weight:   [num_tokens, top_k]
      topk_ids:      [num_tokens, top_k]

    This is a simple reference implementation that processes each token/expert
    pair sequentially. Native MUSA kernels will use grouped GEMM and fusion.
    """
    if hidden_states.dim() != 2:
        raise ValueError("hidden_states must be 2D")
    if w1.dim() != 3 or w2.dim() != 3:
        raise ValueError("w1 and w2 must be 3D")
    if topk_weight.shape != topk_ids.shape:
        raise ValueError("topk_weight and topk_ids must have same shape")
    if hidden_states.device != w1.device or hidden_states.device != w2.device:
        raise ValueError("all tensors must be on same device")

    num_tokens, hidden_dim = hidden_states.shape
    num_experts, inter_dim_x2, _ = w1.shape
    top_k = topk_ids.shape[-1]

    device, dtype = tensor_device_dtype(hidden_states)
    ensure_reference_allowed("fused_moe")
    tune_row = lookup_fused_moe_config(hidden_states, w1, w2, topk_weight)
    backend, v_suffix = resolve_dispatch_backend("fused_moe")
    reason = tuning_dispatch_reason("phase2 native backend pending", tune_row) + v_suffix
    emit(DispatchRecord("fused_moe", backend, device, dtype, reason))
    emit_vendor_planning_note("fused_moe")

    out = torch.zeros(num_tokens, hidden_dim, dtype=hidden_states.dtype, device=hidden_states.device)

    for token_idx in range(num_tokens):
        for k_idx in range(top_k):
            expert_id = int(topk_ids[token_idx, k_idx].item())
            if expert_id < 0 or expert_id >= num_experts:
                raise ValueError(f"invalid expert_id={expert_id}")
            weight = topk_weight[token_idx, k_idx]

            x = hidden_states[token_idx]
            gate = w1[expert_id]
            down = w2[expert_id]

            intermediate = torch.matmul(gate, x)
            if activation == "silu":
                gate_val, up_val = intermediate.chunk(2, dim=0)
                intermediate = up_val * torch.sigmoid(gate_val)
            elif activation == "gelu":
                intermediate = torch.nn.functional.gelu(intermediate)

            token_out = torch.matmul(down, intermediate)
            out[token_idx] += weight * token_out

    return out
