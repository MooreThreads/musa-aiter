from __future__ import annotations

import torch

from musa_aiter.dispatch import DispatchRecord, emit, tensor_device_dtype
from musa_aiter.policy import ensure_reference_allowed
from musa_aiter.tune_runtime import grouped_gemm_dispatch_reason, lookup_grouped_gemm_config
from musa_aiter.vendors.planning import emit_vendor_planning_note
from musa_aiter.vendors.routing import resolve_dispatch_backend


def grouped_gemm(
    a: torch.Tensor,
    b: torch.Tensor,
    m_indices: torch.Tensor,
    *,
    out: torch.Tensor | None = None,
    num_groups: int | None = None,
) -> torch.Tensor:
    """Phase-2 grouped GEMM reference API.

    Shape contract:
      a: [total_tokens, hidden_dim]
      b: [num_groups, hidden_dim, out_dim]
      m_indices: [total_tokens] group assignment for each token
      out: [total_tokens, out_dim] (optional)

    Returns:
      out: [total_tokens, out_dim]

    Each token in 'a' is multiplied by the corresponding expert matrix in 'b'
    based on its group assignment in m_indices.
    """
    if a.dim() != 2 or b.dim() != 3:
        raise ValueError("a must be 2D, b must be 3D [num_groups, K, N]")
    if a.device != b.device:
        raise ValueError("a and b must be on same device")
    if m_indices.dim() != 1 or m_indices.shape[0] != a.shape[0]:
        raise ValueError("m_indices must be 1D with same length as a.shape[0]")

    num_tokens, hidden = a.shape
    num_experts, k_dim, out_dim = b.shape

    if hidden != k_dim:
        raise ValueError(f"hidden mismatch: a.shape[1]={hidden} != b.shape[1]={k_dim}")

    ensure_reference_allowed("grouped_gemm")
    tune_row = lookup_grouped_gemm_config(a, b)
    backend, v_suffix = resolve_dispatch_backend("grouped_gemm")
    reason = grouped_gemm_dispatch_reason("phase2 native backend pending", tune_row) + v_suffix
    device, dtype = tensor_device_dtype(a)
    emit(DispatchRecord("grouped_gemm", backend, device, dtype, reason))
    emit_vendor_planning_note("grouped_gemm")

    if num_groups is None:
        num_groups = num_experts

    out_tensor = torch.empty(num_tokens, out_dim, dtype=a.dtype, device=device) if out is None else out
    if out_tensor.shape != (num_tokens, out_dim):
        raise ValueError(f"out shape mismatch: {tuple(out_tensor.shape)} != ({num_tokens}, {out_dim})")

    for expert_id in range(num_groups):
        mask = m_indices == expert_id
        if mask.sum() == 0:
            continue
        out_tensor[mask] = torch.matmul(a[mask], b[expert_id])

    return out_tensor


def ragged_gemm(
    a_list: list[torch.Tensor],
    b: torch.Tensor,
    *,
    out_list: list[torch.Tensor] | None = None,
) -> list[torch.Tensor]:
    """Phase-2 ragged GEMM reference API.

    Multiplies each tensor in a_list by the corresponding slice of b.

    Shape contract:
      a_list: list of [m_i, k] tensors
      b: [num_groups, k, n]
      out_list: optional list of output tensors

    Returns:
      out_list: list of [m_i, n] tensors
    """
    if b.dim() != 3:
        raise ValueError("b must be 3D [num_groups, k, n]")

    ensure_reference_allowed("ragged_gemm")
    device = a_list[0].device if a_list else "cpu"
    emit(DispatchRecord("ragged_gemm", "torch-reference", str(device), str(a_list[0].dtype if a_list else "unknown"), "phase2 native backend pending"))

    if out_list is None:
        out_list = []
        for i, a in enumerate(a_list):
            out_list.append(torch.matmul(a, b[i]))
    else:
        for i, (a, out) in enumerate(zip(a_list, out_list)):
            out.copy_(torch.matmul(a, b[i]))

    return out_list
