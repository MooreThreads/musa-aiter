from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Sequence

import torch

from .kv_layout import KVLayoutKind, KVLayoutSpec, KVOrganization


class LMCacheFormat(str, Enum):
    KV_2LTD = "kv_2ltd"
    KV_T2D = "kv_t2d"
    KV_2TD = "kv_2td"
    SGLANG_SPLIT_LIST = "sglang_split_list"


@dataclass(frozen=True)
class LMCacheFormatFacts:
    format: LMCacheFormat
    spec: KVLayoutSpec


def _dtype_itemsize(tensor: torch.Tensor) -> int:
    if tensor.dtype not in (torch.float32, torch.float16, torch.bfloat16, torch.uint8):
        raise ValueError(f"unsupported LMCache dtype={tensor.dtype}")
    return tensor.element_size()


def _linear_spec(
    tensor: torch.Tensor,
    *,
    kind: KVLayoutKind,
    organization: KVOrganization,
    num_layers: int,
    num_tokens: int,
    hidden_size: int,
) -> KVLayoutSpec:
    return KVLayoutSpec.linear(
        num_layers=num_layers,
        num_tokens=num_tokens,
        num_heads=1,
        head_size=hidden_size,
        dtype_name=str(tensor.dtype).removeprefix("torch."),
        organization=organization,
        kind=kind,
        device_kind=tensor.device.type,
    )


def describe_lmcache_format(memory_tensor: torch.Tensor) -> LMCacheFormatFacts:
    if not isinstance(memory_tensor, torch.Tensor) or not memory_tensor.is_contiguous():
        raise ValueError("LMCache memory tensor must be contiguous")
    if memory_tensor.dim() == 4:
        if memory_tensor.shape[0] != 2:
            raise ValueError("KV_2LTD memory tensor must have leading K/V dimension 2")
        _dtype_itemsize(memory_tensor)
        _, num_layers, num_tokens, hidden_size = map(int, memory_tensor.shape)
        spec = _linear_spec(
            memory_tensor,
            kind=KVLayoutKind.KV_2LTD,
            organization=KVOrganization.SEPARATE,
            num_layers=num_layers,
            num_tokens=num_tokens,
            hidden_size=hidden_size,
        )
        return LMCacheFormatFacts(LMCacheFormat.KV_2LTD, spec)
    if memory_tensor.dim() != 3:
        raise ValueError("LMCache memory tensor rank must be 3 or 4")
    first, second, num_tokens, hidden_size = (
        int(memory_tensor.shape[0]),
        int(memory_tensor.shape[1]),
        int(memory_tensor.shape[0]),
        int(memory_tensor.shape[-1]),
    )
    _dtype_itemsize(memory_tensor)
    if first == 2 and second == 2:
        raise ValueError("3-D LMCache tensor is ambiguous when both leading dimensions are 2")
    if second == 2:
        spec = _linear_spec(
            memory_tensor,
            kind=KVLayoutKind.KV_T2D,
            organization=KVOrganization.MERGED,
            num_layers=1,
            num_tokens=num_tokens,
            hidden_size=hidden_size,
        )
        return LMCacheFormatFacts(LMCacheFormat.KV_T2D, spec)
    if first == 2:
        spec = _linear_spec(
            memory_tensor,
            kind=KVLayoutKind.KV_2TD,
            organization=KVOrganization.SEPARATE,
            num_layers=1,
            num_tokens=second,
            hidden_size=hidden_size,
        )
        return LMCacheFormatFacts(LMCacheFormat.KV_2TD, spec)
    raise ValueError("3-D LMCache memory tensor must be [T, 2, H] or [2, T, H]")


def describe_sglang_split_list(
    kvcaches: Sequence[Sequence[torch.Tensor]],
) -> LMCacheFormatFacts:
    if len(kvcaches) != 2:
        raise ValueError("SGLang split-list must contain K and V layer lists")
    if not kvcaches[0] or not kvcaches[1]:
        raise ValueError("SGLang split-list must be non-empty")
    if len(kvcaches[0]) != len(kvcaches[1]):
        raise ValueError("SGLang K/V layer lists must have the same length")
    first = kvcaches[0][0]
    if not isinstance(first, torch.Tensor) or first.dim() != 4:
        raise ValueError("SGLang split-list entries must be [NB, BS, NH, HS] tensors")
    _dtype_itemsize(first)
    nb, block_size, num_heads, head_size = map(int, first.shape)
    for axis in kvcaches:
        for tensor in axis:
            if not isinstance(tensor, torch.Tensor) or tensor.shape != first.shape:
                raise ValueError("all SGLang split-list entries must share one shape")
            if tensor.dtype != first.dtype:
                raise ValueError("all SGLang split-list entries must share one dtype")
    row_bytes = num_heads * head_size * first.element_size()
    spec = KVLayoutSpec(
        kind=KVLayoutKind.TWO_X_NL,
        num_layers=len(kvcaches[0]),
        num_blocks=nb,
        block_size=block_size,
        num_heads=num_heads,
        head_size=head_size,
        row_bytes=row_bytes,
        block_stride_bytes=block_size * row_bytes,
        dtype_name=str(first.dtype).removeprefix("torch."),
        organization=KVOrganization.SPLIT_LIST,
        is_contiguous=False,
        pointer_table_length=2 * len(kvcaches[0]),
        device_kind=first.device.type,
    )
    return LMCacheFormatFacts(LMCacheFormat.SGLANG_SPLIT_LIST, spec)
