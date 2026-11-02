"""LMCache native KV transfer ABI facade.

This module implements the Python-level ABI expected by LMCache MUSA support.
The functions validate LMCache's documented layout contract and delegate the
actual slot semantics to simple reference-compatible tensor operations.
"""

from __future__ import annotations

import ctypes
import os
import sys
import time
from typing import Any

import torch

from musa_aiter.jit.ctypes_loader import has_symbol
from musa_aiter.ops.kv_transfer import _current_musa_stream_ptr, load_native_kv_transfer
from musa_aiter.ops.lmcache_kv_transfer_ext import load_lmcache_kv_transfer_ext
from musa_aiter.ops.native_gate import native_requested

_LMCACHE_KV_TRANSFER_ABI_VERSION = 1
_NATIVE_FN_CACHE: dict[str, Any] = {}
_NATIVE_AVAILABLE_CACHE: bool | None = None


def clear_native_lmcache_kv_transfer_cache() -> None:
    """Clear symbol/capability caches after rebuilding the native shared object."""

    global _NATIVE_AVAILABLE_CACHE
    _NATIVE_AVAILABLE_CACHE = None
    _NATIVE_FN_CACHE.clear()


def _timing_enabled() -> bool:
    return os.environ.get("MUSA_AITER_LMCACHE_PY_TIMING", os.environ.get("MUSA_AITER_LMCACHE_TIMING", "0")) == "1"


def _log_timing(op: str, elapsed_us: float, **fields: Any) -> None:
    if not _timing_enabled():
        return
    suffix = " ".join(f"{key}={value}" for key, value in fields.items())
    print(f"[musa_aiter_lmcache_py_timing] op={op} elapsed_us={elapsed_us:.3f} {suffix}", file=sys.stderr)


def native_lmcache_kv_transfer_abi_version() -> int:
    return _LMCACHE_KV_TRANSFER_ABI_VERSION


def lmcache_kv_buffer_to_paged(
    kv_buffer: torch.Tensor,
    paged_kv_cache: torch.Tensor | list[torch.Tensor] | list[list[torch.Tensor]],
    slot_mapping: torch.Tensor,
    *args: Any,
) -> bool:
    """Scatter a LMCache non-MLA chunk into paged KV cache.

    Expected layouts:
    - legacy ``paged_kv_cache`` tensor: ``[num_layers, 2, num_blocks, block_size, num_heads, head_size]``
    - LMCache layer list: list of ``[2, num_blocks, block_size, num_heads, head_size]`` tensors
    - SGLang MP cache: ``[K_layers, V_layers]`` where every tensor is
      ``[num_blocks, block_size, num_heads, head_size]``
    - ``kv_buffer``: ``[2, num_layers, num_tokens, num_heads * head_size]``
    - ``slot_mapping``: flattened ``block_id * block_size + offset`` slots.
    """

    if isinstance(paged_kv_cache, list):
        if len(args) != 4:
            raise TypeError("LMCache list-cache buffer_to_paged expects skip_prefix_n_tokens, block_size, num_heads, head_size")
        skip_prefix_n_tokens, block_size, num_heads, head_size = map(int, args)
        if _contains_nested_cache_list(paged_kv_cache):
            return lmcache_kv_buffer_to_paged_two_x_nl(
                kv_buffer,
                paged_kv_cache,
                slot_mapping,
                skip_prefix_n_tokens,
                block_size,
                num_heads,
                head_size,
            )
        return _lmcache_kv_buffer_to_paged_list(kv_buffer, paged_kv_cache, slot_mapping, skip_prefix_n_tokens, block_size, num_heads, head_size)

    _validate_non_mla_contract(kv_buffer, paged_kv_cache, slot_mapping)
    if _try_native_non_mla_to_paged(kv_buffer, paged_kv_cache, slot_mapping):
        return True
    _, num_layers, num_tokens, hidden = map(int, kv_buffer.shape)
    _, _, _, block_size, num_heads, head_size = map(int, paged_kv_cache.shape)
    reshaped = kv_buffer.reshape(2, num_layers, num_tokens, num_heads, head_size)
    for token_index, slot in enumerate(_slot_values(slot_mapping)):
        if slot < 0:
            continue
        block_id, block_offset = divmod(slot, block_size)
        paged_kv_cache[:, 0, block_id, block_offset, :, :] = reshaped[0, :, token_index]
        paged_kv_cache[:, 1, block_id, block_offset, :, :] = reshaped[1, :, token_index]
    return True


def lmcache_paged_to_kv_buffer(paged_kv_cache: torch.Tensor, kv_buffer: torch.Tensor, slot_mapping: torch.Tensor) -> bool:
    """Gather paged non-MLA KV cache into a LMCache chunk buffer."""

    _validate_non_mla_contract(kv_buffer, paged_kv_cache, slot_mapping)
    if _try_native_non_mla_to_buffer(paged_kv_cache, kv_buffer, slot_mapping):
        return True
    _, num_layers, num_tokens, _ = map(int, kv_buffer.shape)
    _, _, _, block_size, num_heads, head_size = map(int, paged_kv_cache.shape)
    gathered = torch.zeros((2, num_layers, num_tokens, num_heads, head_size), dtype=kv_buffer.dtype, device=kv_buffer.device)
    for token_index, slot in enumerate(_slot_values(slot_mapping)):
        if slot < 0:
            continue
        block_id, block_offset = divmod(slot, block_size)
        gathered[0, :, token_index] = paged_kv_cache[:, 0, block_id, block_offset, :, :]
        gathered[1, :, token_index] = paged_kv_cache[:, 1, block_id, block_offset, :, :]
    kv_buffer.copy_(gathered.reshape_as(kv_buffer))
    return True


def lmcache_kv_buffer_to_paged_mla(kv_buffer: torch.Tensor, paged_kv_cache: torch.Tensor, slot_mapping: torch.Tensor) -> bool:
    """Scatter a LMCache MLA chunk into paged MLA cache.

    Expected layouts:
    - ``kv_buffer``: ``[num_layers, num_tokens, hidden_size]``
    - ``paged_kv_cache``: ``[num_layers, num_blocks, block_size, hidden_size]``
    """

    _validate_mla_contract(kv_buffer, paged_kv_cache, slot_mapping)
    if _try_native_mla_to_paged(kv_buffer, paged_kv_cache, slot_mapping):
        return True
    _, _, block_size, _ = map(int, paged_kv_cache.shape)
    for token_index, slot in enumerate(_slot_values(slot_mapping)):
        if slot < 0:
            continue
        block_id, block_offset = divmod(slot, block_size)
        paged_kv_cache[:, block_id, block_offset, :] = kv_buffer[:, token_index, :]
    return True


def lmcache_paged_to_kv_buffer_mla(paged_kv_cache: torch.Tensor, kv_buffer: torch.Tensor, slot_mapping: torch.Tensor) -> bool:
    """Gather paged MLA cache into a LMCache MLA chunk buffer."""

    _validate_mla_contract(kv_buffer, paged_kv_cache, slot_mapping)
    if _try_native_mla_to_buffer(paged_kv_cache, kv_buffer, slot_mapping):
        return True
    _, _, block_size, _ = map(int, paged_kv_cache.shape)
    kv_buffer.zero_()
    for token_index, slot in enumerate(_slot_values(slot_mapping)):
        if slot < 0:
            continue
        block_id, block_offset = divmod(slot, block_size)
        kv_buffer[:, token_index, :] = paged_kv_cache[:, block_id, block_offset, :]
    return True


def lmcache_kv_paged_to_buffer(
    kvcaches: list[torch.Tensor] | list[list[torch.Tensor]],
    memory_tensor: torch.Tensor,
    slot_mapping: torch.Tensor,
    block_size: int,
    num_heads: int,
    head_size: int,
) -> bool:
    """Gather LMCache non-MLA list-cache tensors into a contiguous buffer."""

    if _contains_nested_cache_list(kvcaches):
        return lmcache_kv_paged_to_buffer_two_x_nl(
            kvcaches,
            memory_tensor,
            slot_mapping,
            block_size,
            num_heads,
            head_size,
        )
    return _lmcache_kv_paged_to_buffer_list(kvcaches, memory_tensor, slot_mapping, block_size, num_heads, head_size)


def lmcache_kv_buffer_to_paged_two_x_nl(
    memory_tensor: torch.Tensor,
    kvcaches: list[list[torch.Tensor]],
    slot_mapping: torch.Tensor,
    skip_prefix_n_tokens: int,
    block_size: int,
    num_heads: int,
    head_size: int,
) -> bool:
    """Scatter into SGLang MP ``2 x NL x [NB, BS, NH, HS]`` caches.

    ``kvcaches[0]`` contains all key-cache layers and ``kvcaches[1]`` contains
    all value-cache layers. The tensors remain independent; this operator does
    not pack or transpose them into an intermediate cache.
    """

    _validate_two_x_nl_contract(
        memory_tensor,
        kvcaches,
        slot_mapping,
        skip_prefix_n_tokens,
        block_size,
        num_heads,
        head_size,
    )
    if _try_native_two_x_nl_to_paged(
        memory_tensor,
        kvcaches,
        slot_mapping,
        skip_prefix_n_tokens,
        block_size,
        num_heads,
        head_size,
    ):
        return True

    num_layers = len(kvcaches[0])
    num_tokens = int(slot_mapping.numel())
    values = memory_tensor[
        :, :, skip_prefix_n_tokens : skip_prefix_n_tokens + num_tokens
    ].reshape(2, num_layers, num_tokens, num_heads, head_size)
    for token_index, slot in enumerate(_slot_values(slot_mapping)):
        if slot < 0:
            continue
        block_id, block_offset = divmod(slot, block_size)
        for kv_axis in range(2):
            for layer in range(num_layers):
                kvcaches[kv_axis][layer][block_id, block_offset] = values[
                    kv_axis, layer, token_index
                ]
    return True


def lmcache_kv_paged_to_buffer_two_x_nl(
    kvcaches: list[list[torch.Tensor]],
    memory_tensor: torch.Tensor,
    slot_mapping: torch.Tensor,
    block_size: int,
    num_heads: int,
    head_size: int,
) -> bool:
    """Gather SGLang MP ``2 x NL x [NB, BS, NH, HS]`` caches."""

    _validate_two_x_nl_contract(
        memory_tensor,
        kvcaches,
        slot_mapping,
        0,
        block_size,
        num_heads,
        head_size,
    )
    if _try_native_two_x_nl_to_buffer(
        kvcaches,
        memory_tensor,
        slot_mapping,
        block_size,
        num_heads,
        head_size,
    ):
        return True

    memory_tensor.zero_()
    num_layers = len(kvcaches[0])
    for token_index, slot in enumerate(_slot_values(slot_mapping)):
        if slot < 0:
            continue
        block_id, block_offset = divmod(slot, block_size)
        for kv_axis in range(2):
            for layer in range(num_layers):
                memory_tensor[kv_axis, layer, token_index].copy_(
                    kvcaches[kv_axis][layer][block_id, block_offset].reshape(-1)
                )
    return True


def lmcache_mla_buffer_to_paged(
    memory_tensor: torch.Tensor,
    kvcaches: list[torch.Tensor],
    slot_mapping: torch.Tensor,
    skip_prefix_n_tokens: int,
    block_size: int,
    head_size: int,
) -> bool:
    """Scatter a LMCache MLA buffer into list-cache tensors."""

    return _lmcache_mla_buffer_to_paged_list(memory_tensor, kvcaches, slot_mapping, skip_prefix_n_tokens, block_size, head_size)


def lmcache_mla_paged_to_buffer(
    kvcaches: list[torch.Tensor],
    memory_tensor: torch.Tensor,
    slot_mapping: torch.Tensor,
    block_size: int,
    head_size: int,
) -> bool:
    """Gather LMCache MLA list-cache tensors into a contiguous buffer."""

    return _lmcache_mla_paged_to_buffer_list(kvcaches, memory_tensor, slot_mapping, block_size, head_size)


def _validate_non_mla_contract(kv_buffer: torch.Tensor, paged_kv_cache: torch.Tensor, slot_mapping: torch.Tensor) -> None:
    if kv_buffer.dim() != 4:
        raise ValueError("LMCache non-MLA kv_buffer rank must be 4: [2, NL, T, NH*HS]")
    if paged_kv_cache.dim() != 6:
        raise ValueError("LMCache non-MLA paged_kv_cache rank must be 6: [NL, 2, NB, BS, NH, HS]")
    if int(kv_buffer.shape[0]) != 2 or int(paged_kv_cache.shape[1]) != 2:
        raise ValueError("LMCache non-MLA layout requires K/V axis size 2")
    num_layers = int(kv_buffer.shape[1])
    num_tokens = int(kv_buffer.shape[2])
    hidden = int(kv_buffer.shape[3])
    cache_layers = int(paged_kv_cache.shape[0])
    num_blocks = int(paged_kv_cache.shape[2])
    block_size = int(paged_kv_cache.shape[3])
    num_heads = int(paged_kv_cache.shape[4])
    head_size = int(paged_kv_cache.shape[5])
    if num_layers != cache_layers:
        raise ValueError("LMCache non-MLA layer count mismatch")
    if hidden != num_heads * head_size:
        raise ValueError("LMCache non-MLA hidden size must equal num_heads * head_size")
    _validate_slots(slot_mapping, num_tokens=num_tokens, num_blocks=num_blocks, block_size=block_size)


def _contains_nested_cache_list(kvcaches: list[Any]) -> bool:
    return any(isinstance(axis_or_tensor, list) for axis_or_tensor in kvcaches)


def _validate_two_x_nl_contract(
    memory_tensor: torch.Tensor,
    kvcaches: list[list[torch.Tensor]],
    slot_mapping: torch.Tensor,
    skip_prefix_n_tokens: int,
    block_size: int,
    num_heads: int,
    head_size: int,
) -> None:
    name = "LMCache TWO_X_NL_X_NB_BS_NH_HS"
    if memory_tensor.dim() != 4 or int(memory_tensor.shape[0]) != 2:
        raise ValueError(f"{name} memory_tensor must be [2, NL, T, NH*HS]")
    if len(kvcaches) != 2 or not all(isinstance(axis, list) for axis in kvcaches):
        raise ValueError(f"{name} cache must be [K_layers, V_layers]")
    if not kvcaches[0] or len(kvcaches[0]) != len(kvcaches[1]):
        raise ValueError(f"{name} K/V layer lists must be non-empty and have equal length")

    num_layers = len(kvcaches[0])
    if int(memory_tensor.shape[1]) != num_layers:
        raise ValueError(f"{name} layer count mismatch")
    if min(block_size, num_heads, head_size) <= 0:
        raise ValueError(f"{name} block_size, num_heads, and head_size must be positive")
    if int(memory_tensor.shape[3]) != num_heads * head_size:
        raise ValueError(f"{name} hidden size must equal num_heads * head_size")
    if memory_tensor.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError(f"{name} supports float16, bfloat16, and float32")
    if slot_mapping.dtype not in (torch.int32, torch.int64):
        raise ValueError(f"{name} slot_mapping must be int32 or int64")
    if skip_prefix_n_tokens < 0:
        raise ValueError(f"{name} skip_prefix_n_tokens must be non-negative")
    if skip_prefix_n_tokens + int(slot_mapping.numel()) > int(memory_tensor.shape[2]):
        raise ValueError(f"{name} transfer range exceeds memory_tensor token extent")

    first = kvcaches[0][0]
    if not isinstance(first, torch.Tensor) or first.dim() != 4:
        raise ValueError(f"{name} each cache tensor must be [NB, BS, NH, HS]")
    num_blocks = int(first.shape[0])
    expected_shape = (num_blocks, block_size, num_heads, head_size)
    if num_blocks <= 0:
        raise ValueError(f"{name} num_blocks must be positive")
    for kv_axis, layers in enumerate(kvcaches):
        for layer, cache in enumerate(layers):
            if not isinstance(cache, torch.Tensor) or tuple(cache.shape) != expected_shape:
                raise ValueError(
                    f"{name} cache[{kv_axis}][{layer}] must have shape "
                    f"{expected_shape}"
                )
            if cache.dtype != memory_tensor.dtype:
                raise ValueError(f"{name} cache dtype must match memory_tensor")
            if cache.device != memory_tensor.device:
                raise ValueError(f"{name} caches and memory_tensor must share a device")
    if slot_mapping.device != memory_tensor.device:
        raise ValueError(f"{name} slot_mapping and data tensors must share a device")
    _validate_slots(
        slot_mapping,
        num_tokens=int(slot_mapping.numel()),
        num_blocks=num_blocks,
        block_size=block_size,
    )


def _validate_mla_contract(kv_buffer: torch.Tensor, paged_kv_cache: torch.Tensor, slot_mapping: torch.Tensor) -> None:
    if kv_buffer.dim() != 3:
        raise ValueError("LMCache MLA kv_buffer rank must be 3: [NL, T, HS]")
    if paged_kv_cache.dim() != 4:
        raise ValueError("LMCache MLA paged_kv_cache rank must be 4: [NL, NB, BS, HS]")
    num_layers = int(kv_buffer.shape[0])
    num_tokens = int(kv_buffer.shape[1])
    hidden = int(kv_buffer.shape[2])
    cache_layers = int(paged_kv_cache.shape[0])
    num_blocks = int(paged_kv_cache.shape[1])
    block_size = int(paged_kv_cache.shape[2])
    cache_hidden = int(paged_kv_cache.shape[3])
    if num_layers != cache_layers:
        raise ValueError("LMCache MLA layer count mismatch")
    if hidden != cache_hidden:
        raise ValueError("LMCache MLA hidden size mismatch")
    _validate_slots(slot_mapping, num_tokens=num_tokens, num_blocks=num_blocks, block_size=block_size)


def _validate_slots(slot_mapping: torch.Tensor, *, num_tokens: int, num_blocks: int, block_size: int) -> None:
    if slot_mapping.dim() != 1:
        raise ValueError("LMCache slot_mapping rank must be 1")
    if int(slot_mapping.numel()) != num_tokens:
        raise ValueError("LMCache slot_mapping length must equal num_tokens")
    if slot_mapping.device.type != "cpu":
        return
    max_slot = num_blocks * block_size
    for slot in _slot_values(slot_mapping):
        if slot >= max_slot:
            raise ValueError(f"LMCache slot_mapping value out of range: slot={slot} max={max_slot - 1}")


def _lmcache_kv_buffer_to_paged_list(
    memory_tensor: torch.Tensor,
    kvcaches: list[torch.Tensor],
    slot_mapping: torch.Tensor,
    skip_prefix_n_tokens: int,
    block_size: int,
    num_heads: int,
    head_size: int,
) -> bool:
    _validate_non_mla_list_contract(memory_tensor, kvcaches, slot_mapping, block_size, num_heads, head_size)
    if _try_native_non_mla_list_to_paged(memory_tensor, kvcaches, slot_mapping, skip_prefix_n_tokens, block_size, num_heads, head_size):
        return True
    hidden = num_heads * head_size
    valid_memory = memory_tensor[:, :, skip_prefix_n_tokens:].reshape(2, len(kvcaches), -1, num_heads, head_size)
    for token_index, slot in enumerate(_slot_values(slot_mapping)):
        if slot < 0:
            continue
        block_id, block_offset = divmod(slot, block_size)
        for layer, kvcache in enumerate(kvcaches):
            kvcache[0, block_id, block_offset] = valid_memory[0, layer, token_index]
            kvcache[1, block_id, block_offset] = valid_memory[1, layer, token_index]
    return True


def _lmcache_kv_paged_to_buffer_list(
    kvcaches: list[torch.Tensor],
    memory_tensor: torch.Tensor,
    slot_mapping: torch.Tensor,
    block_size: int,
    num_heads: int,
    head_size: int,
) -> bool:
    _validate_non_mla_list_contract(memory_tensor, kvcaches, slot_mapping, block_size, num_heads, head_size)
    if _try_native_non_mla_list_to_buffer(kvcaches, memory_tensor, slot_mapping, block_size, num_heads, head_size):
        return True
    memory_tensor.zero_()
    out = memory_tensor.reshape(2, len(kvcaches), -1, num_heads, head_size)
    for token_index, slot in enumerate(_slot_values(slot_mapping)):
        if slot < 0:
            continue
        block_id, block_offset = divmod(slot, block_size)
        for layer, kvcache in enumerate(kvcaches):
            out[0, layer, token_index] = kvcache[0, block_id, block_offset]
            out[1, layer, token_index] = kvcache[1, block_id, block_offset]
    return True


def _lmcache_mla_buffer_to_paged_list(
    memory_tensor: torch.Tensor,
    kvcaches: list[torch.Tensor],
    slot_mapping: torch.Tensor,
    skip_prefix_n_tokens: int,
    block_size: int,
    head_size: int,
) -> bool:
    _validate_mla_list_contract(memory_tensor, kvcaches, slot_mapping, block_size, head_size)
    if _try_native_mla_list_to_paged(memory_tensor, kvcaches, slot_mapping, skip_prefix_n_tokens, block_size, head_size):
        return True
    values = memory_tensor[0, :, skip_prefix_n_tokens:]
    for token_index, slot in enumerate(_slot_values(slot_mapping)):
        if slot < 0:
            continue
        block_id, block_offset = divmod(slot, block_size)
        for layer, kvcache in enumerate(kvcaches):
            kvcache[block_id, block_offset] = values[layer, token_index]
    return True


def _lmcache_mla_paged_to_buffer_list(
    kvcaches: list[torch.Tensor],
    memory_tensor: torch.Tensor,
    slot_mapping: torch.Tensor,
    block_size: int,
    head_size: int,
) -> bool:
    _validate_mla_list_contract(memory_tensor, kvcaches, slot_mapping, block_size, head_size)
    if _try_native_mla_list_to_buffer(kvcaches, memory_tensor, slot_mapping, block_size, head_size):
        return True
    memory_tensor.zero_()
    out = memory_tensor[0]
    for token_index, slot in enumerate(_slot_values(slot_mapping)):
        if slot < 0:
            continue
        block_id, block_offset = divmod(slot, block_size)
        for layer, kvcache in enumerate(kvcaches):
            out[layer, token_index] = kvcache[block_id, block_offset]
    return True


def _validate_non_mla_list_contract(
    memory_tensor: torch.Tensor,
    kvcaches: list[torch.Tensor],
    slot_mapping: torch.Tensor,
    block_size: int,
    num_heads: int,
    head_size: int,
) -> None:
    if memory_tensor.dim() != 4 or int(memory_tensor.shape[0]) != 2:
        raise ValueError("LMCache list non-MLA memory_tensor rank must be 4: [2, NL, T, NH*HS]")
    if len(kvcaches) != int(memory_tensor.shape[1]):
        raise ValueError("LMCache list non-MLA layer count mismatch")
    hidden = num_heads * head_size
    if int(memory_tensor.shape[3]) != hidden:
        raise ValueError("LMCache list non-MLA hidden size must equal num_heads * head_size")
    if not kvcaches:
        raise ValueError("LMCache list non-MLA kvcaches must not be empty")
    num_blocks = int(kvcaches[0].shape[1])
    for kvcache in kvcaches:
        if kvcache.dim() != 5 or int(kvcache.shape[0]) != 2 or int(kvcache.shape[2]) != block_size or int(kvcache.shape[3]) != num_heads or int(kvcache.shape[4]) != head_size:
            raise ValueError("LMCache list non-MLA kvcache rank must be 5: [2, NB, BS, NH, HS]")
        if int(kvcache.shape[1]) != num_blocks:
            raise ValueError("LMCache list non-MLA num_blocks mismatch")
    _validate_slots(slot_mapping, num_tokens=int(slot_mapping.numel()), num_blocks=num_blocks, block_size=block_size)


def _validate_mla_list_contract(memory_tensor: torch.Tensor, kvcaches: list[torch.Tensor], slot_mapping: torch.Tensor, block_size: int, head_size: int) -> None:
    if memory_tensor.dim() != 4 or int(memory_tensor.shape[0]) != 1:
        raise ValueError("LMCache list MLA memory_tensor rank must be 4: [1, NL, T, HS]")
    if len(kvcaches) != int(memory_tensor.shape[1]):
        raise ValueError("LMCache list MLA layer count mismatch")
    if not kvcaches:
        raise ValueError("LMCache list MLA kvcaches must not be empty")
    num_blocks = int(kvcaches[0].shape[0])
    for kvcache in kvcaches:
        if kvcache.dim() != 3 or int(kvcache.shape[1]) != block_size or int(kvcache.shape[2]) != head_size:
            raise ValueError("LMCache list MLA kvcache rank must be 3: [NB, BS, HS]")
        if int(kvcache.shape[0]) != num_blocks:
            raise ValueError("LMCache list MLA num_blocks mismatch")
    _validate_slots(slot_mapping, num_tokens=int(slot_mapping.numel()), num_blocks=num_blocks, block_size=block_size)


def _native_requested() -> bool:
    return native_requested(
        "MUSA_AITER_NATIVE_LMCACHE_KV_TRANSFER",
        "LMCACHE_MUSA_NATIVE_KV_TRANSFER",
        "MUSA_AITER_NATIVE_KV_TRANSFER",
    )


def is_native_lmcache_kv_transfer_available() -> bool:
    global _NATIVE_AVAILABLE_CACHE
    if _NATIVE_AVAILABLE_CACHE is not None:
        return _NATIVE_AVAILABLE_CACHE
    try:
        lib = load_native_kv_transfer()
        _NATIVE_AVAILABLE_CACHE = (
            has_symbol(lib, "native_lmcache_kv_transfer_abi_version")
            and has_symbol(lib, "lmcache_kv_buffer_to_paged")
            and has_symbol(lib, "lmcache_paged_to_kv_buffer")
            and has_symbol(lib, "lmcache_kv_buffer_to_paged_mla")
            and has_symbol(lib, "lmcache_paged_to_kv_buffer_mla")
            and has_symbol(lib, "lmcache_kv_buffer_to_paged_list_u16")
            and has_symbol(lib, "lmcache_paged_to_kv_buffer_list_u16")
            and has_symbol(lib, "lmcache_kv_buffer_to_paged_list_float")
            and has_symbol(lib, "lmcache_paged_to_kv_buffer_list_float")
            and has_symbol(lib, "lmcache_kv_buffer_to_paged_two_x_nl_float")
            and has_symbol(lib, "lmcache_paged_to_kv_buffer_two_x_nl_float")
            and has_symbol(lib, "lmcache_mla_buffer_to_paged_list_float")
            and has_symbol(lib, "lmcache_mla_paged_to_buffer_list_float")
            and has_symbol(lib, "lmcache_mla_buffer_to_paged_list_u16")
            and has_symbol(lib, "lmcache_mla_paged_to_buffer_list_u16")
            and has_symbol(lib, "musa_aiter_kv_paged_to_merged_linear_half")
            and has_symbol(lib, "musa_aiter_kv_merged_linear_to_paged_half")
            and has_symbol(lib, "musa_aiter_kv_paged_to_merged_linear_multi_layer_half")
            and has_symbol(lib, "musa_aiter_kv_merged_linear_to_paged_multi_layer_half")
            and has_symbol(lib, "musa_aiter_kv_paged_to_mla_linear_half")
            and has_symbol(lib, "musa_aiter_kv_mla_linear_to_paged_half")
            and has_symbol(lib, "musa_aiter_kv_paged_to_merged_linear_bfloat16")
            and has_symbol(lib, "musa_aiter_kv_merged_linear_to_paged_bfloat16")
            and has_symbol(lib, "musa_aiter_kv_paged_to_merged_linear_multi_layer_bfloat16")
            and has_symbol(lib, "musa_aiter_kv_merged_linear_to_paged_multi_layer_bfloat16")
            and has_symbol(lib, "musa_aiter_kv_paged_to_mla_linear_bfloat16")
            and has_symbol(lib, "musa_aiter_kv_mla_linear_to_paged_bfloat16")
        )
    except Exception:
        _NATIVE_AVAILABLE_CACHE = False
    return _NATIVE_AVAILABLE_CACHE


def is_native_lmcache_two_x_nl_kv_transfer_available() -> bool:
    """Return whether the SGLang MP split K/V native symbols are built."""

    if not is_native_lmcache_kv_transfer_available():
        return False
    try:
        lib = load_native_kv_transfer()
        return all(
            has_symbol(lib, symbol)
            for symbol in (
                "lmcache_kv_buffer_to_paged_two_x_nl_float",
                "lmcache_paged_to_kv_buffer_two_x_nl_float",
                "lmcache_kv_buffer_to_paged_two_x_nl_u16",
                "lmcache_paged_to_kv_buffer_two_x_nl_u16",
            )
        )
    except Exception:
        return False


def _can_native_data(*tensors: torch.Tensor) -> bool:
    if not _native_requested():
        return False
    for tensor in tensors:
        if str(tensor.device).split(":")[0] != "musa" or tensor.dtype not in (torch.float32, torch.float16, torch.bfloat16) or not tensor.is_contiguous():
            return False
    return is_native_lmcache_kv_transfer_available()


def _can_native_slot_mapping(slot_mapping: torch.Tensor) -> bool:
    return str(slot_mapping.device).split(":")[0] == "musa" and slot_mapping.dtype is torch.int64 and slot_mapping.is_contiguous()


def _call_non_mla_symbol(symbol: str, first: torch.Tensor, second: torch.Tensor, slot_mapping: torch.Tensor) -> bool:
    if first.dim() == 4:
        _, num_layers, num_tokens, _ = map(int, first.shape)
        _, _, num_blocks, block_size, num_heads, head_size = map(int, second.shape)
    else:
        _, _, num_blocks, block_size, num_heads, head_size = map(int, first.shape)
        _, num_layers, num_tokens, _ = map(int, second.shape)
    lib = load_native_kv_transfer()
    fn = getattr(lib.handle, symbol)
    fn.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_void_p,
    ]
    fn.restype = None
    fn(
        ctypes.c_void_p(first.data_ptr()),
        ctypes.c_void_p(second.data_ptr()),
        ctypes.c_void_p(slot_mapping.data_ptr()),
        num_layers,
        num_tokens,
        num_blocks,
        block_size,
        num_heads,
        head_size,
        _current_musa_stream_ptr(),
    )
    return True


def _call_mla_symbol(symbol: str, first: torch.Tensor, second: torch.Tensor, slot_mapping: torch.Tensor) -> bool:
    _, num_tokens, hidden_size = map(int, first.shape) if first.dim() == 3 else (int(second.shape[0]), int(second.shape[1]), int(second.shape[2]))
    if second.dim() == 4:
        num_layers, num_blocks, block_size, hidden_size = map(int, second.shape)
    else:
        num_layers, num_blocks, block_size, hidden_size = map(int, first.shape)
    lib = load_native_kv_transfer()
    fn = getattr(lib.handle, symbol)
    fn.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_void_p,
    ]
    fn.restype = None
    fn(
        ctypes.c_void_p(first.data_ptr()),
        ctypes.c_void_p(second.data_ptr()),
        ctypes.c_void_p(slot_mapping.data_ptr()),
        num_layers,
        num_tokens,
        num_blocks,
        block_size,
        hidden_size,
        _current_musa_stream_ptr(),
    )
    return True


def _lmcache_symbol(base_symbol: str, dtype: torch.dtype) -> str:
    if dtype in (torch.float16, torch.bfloat16):
        return f"{base_symbol}_u16"
    return base_symbol


def _lmcache_list_symbol(base_symbol: str, dtype: torch.dtype) -> str:
    return f"{base_symbol}_{'float' if dtype is torch.float32 else 'u16'}"


def _can_native_list(memory_tensor: torch.Tensor, kvcaches: list[torch.Tensor], slot_mapping: torch.Tensor) -> bool:
    return _can_native_data(memory_tensor, *kvcaches) and _can_native_slot_mapping(slot_mapping)


def _call_two_x_nl_symbol(
    symbol: str,
    cache: torch.Tensor,
    memory_tensor: torch.Tensor,
    slot_mapping: torch.Tensor,
    kv_axis: int,
    layer_index: int,
    skip_prefix_n_tokens: int,
    block_size: int,
    num_heads: int,
    head_size: int,
) -> bool:
    fn = _cached_native_fn(
        symbol,
        [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_int64,
            ctypes.c_int64,
            ctypes.c_int64,
            ctypes.c_int64,
            ctypes.c_int64,
            ctypes.c_int64,
            ctypes.c_int64,
            ctypes.c_int64,
            ctypes.c_int64,
            ctypes.c_int64,
            ctypes.c_void_p,
        ],
    )
    if fn is None:
        return False
    fn(
        ctypes.c_void_p(memory_tensor.data_ptr()),
        ctypes.c_void_p(cache.data_ptr()),
        ctypes.c_void_p(slot_mapping.data_ptr()),
        kv_axis,
        layer_index,
        int(memory_tensor.shape[1]),
        int(memory_tensor.shape[2]),
        int(slot_mapping.numel()),
        skip_prefix_n_tokens,
        int(cache.shape[0]),
        block_size,
        num_heads,
        head_size,
        _current_musa_stream_ptr(),
    )
    return True


def _try_native_two_x_nl_to_paged(
    memory_tensor: torch.Tensor,
    kvcaches: list[list[torch.Tensor]],
    slot_mapping: torch.Tensor,
    skip_prefix_n_tokens: int,
    block_size: int,
    num_heads: int,
    head_size: int,
) -> bool:
    flat_caches = kvcaches[0] + kvcaches[1]
    if (
        memory_tensor.dtype not in (torch.float32, torch.float16, torch.bfloat16)
        or (
            memory_tensor.dtype is not torch.float32
            and (num_heads * head_size) % 4 != 0
        )
        or not _can_native_list(memory_tensor, flat_caches, slot_mapping)
        or not is_native_lmcache_two_x_nl_kv_transfer_available()
    ):
        return False
    if _try_cpp_extension_two_x_nl_to_paged(
        memory_tensor,
        kvcaches,
        slot_mapping,
        skip_prefix_n_tokens,
        block_size,
        num_heads,
        head_size,
    ):
        return True
    for kv_axis, layers in enumerate(kvcaches):
        for layer_index, cache in enumerate(layers):
            if not _call_two_x_nl_symbol(
                _lmcache_list_symbol("lmcache_kv_buffer_to_paged_two_x_nl", memory_tensor.dtype),
                cache,
                memory_tensor,
                slot_mapping,
                kv_axis,
                layer_index,
                skip_prefix_n_tokens,
                block_size,
                num_heads,
                head_size,
            ):
                return False
    return True


def _try_native_two_x_nl_to_buffer(
    kvcaches: list[list[torch.Tensor]],
    memory_tensor: torch.Tensor,
    slot_mapping: torch.Tensor,
    block_size: int,
    num_heads: int,
    head_size: int,
) -> bool:
    flat_caches = kvcaches[0] + kvcaches[1]
    if (
        memory_tensor.dtype not in (torch.float32, torch.float16, torch.bfloat16)
        or (
            memory_tensor.dtype is not torch.float32
            and (num_heads * head_size) % 4 != 0
        )
        or not _can_native_list(memory_tensor, flat_caches, slot_mapping)
        or not is_native_lmcache_two_x_nl_kv_transfer_available()
    ):
        return False
    if _try_cpp_extension_two_x_nl_to_buffer(
        kvcaches,
        memory_tensor,
        slot_mapping,
        block_size,
        num_heads,
        head_size,
    ):
        return True
    memory_tensor.zero_()
    for kv_axis, layers in enumerate(kvcaches):
        for layer_index, cache in enumerate(layers):
            if not _call_two_x_nl_symbol(
                _lmcache_list_symbol("lmcache_paged_to_kv_buffer_two_x_nl", memory_tensor.dtype),
                cache,
                memory_tensor,
                slot_mapping,
                kv_axis,
                layer_index,
                0,
                block_size,
                num_heads,
                head_size,
            ):
                return False
    return True


def _cached_native_fn(symbol: str, argtypes: list[Any]) -> Any | None:
    fn = _NATIVE_FN_CACHE.get(symbol)
    if fn is not None:
        return fn
    lib = load_native_kv_transfer()
    if not has_symbol(lib, symbol):
        return None
    fn = getattr(lib.handle, symbol)
    fn.argtypes = argtypes
    fn.restype = None
    _NATIVE_FN_CACHE[symbol] = fn
    return fn


def _call_non_mla_list_symbol(
    symbol: str,
    kvcache: torch.Tensor,
    memory_tensor: torch.Tensor,
    slot_mapping: torch.Tensor,
    layer_index: int,
    skip_prefix_n_tokens: int,
    block_size: int,
    num_heads: int,
    head_size: int,
) -> bool:
    fn = _cached_native_fn(symbol, [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_void_p,
    ])
    if fn is None:
        return False
    fn(
        ctypes.c_void_p(memory_tensor.data_ptr()),
        ctypes.c_void_p(kvcache.data_ptr()),
        ctypes.c_void_p(slot_mapping.data_ptr()),
        layer_index,
        int(memory_tensor.shape[1]),
        int(memory_tensor.shape[2]),
        int(slot_mapping.numel()),
        skip_prefix_n_tokens,
        int(kvcache.shape[1]),
        block_size,
        num_heads,
        head_size,
        _current_musa_stream_ptr(),
    )
    return True


def _call_mla_list_symbol(
    symbol: str,
    kvcache: torch.Tensor,
    memory_tensor: torch.Tensor,
    slot_mapping: torch.Tensor,
    layer_index: int,
    skip_prefix_n_tokens: int,
    block_size: int,
    head_size: int,
    with_skip_prefix: bool,
) -> bool:
    # The MLA list ABI carries ``num_blocks`` in both directions; only the
    # memory-to-paged direction also carries ``skip_prefix``.
    scalars = [layer_index, int(memory_tensor.shape[2]), int(slot_mapping.numel())]
    if with_skip_prefix:
        scalars.append(skip_prefix_n_tokens)
    scalars.extend([int(kvcache.shape[0]), block_size, head_size])
    fn = _cached_native_fn(
        symbol,
        [ctypes.c_void_p] * 3 + [ctypes.c_int64] * len(scalars) + [ctypes.c_void_p],
    )
    if fn is None:
        return False
    fn(
        ctypes.c_void_p(memory_tensor.data_ptr()),
        ctypes.c_void_p(kvcache.data_ptr()),
        ctypes.c_void_p(slot_mapping.data_ptr()),
        *scalars,
        _current_musa_stream_ptr(),
    )
    return True


def _call_non_mla_list2_u16_symbol(
    symbol: str,
    kvcaches: list[torch.Tensor],
    memory_tensor: torch.Tensor,
    slot_mapping: torch.Tensor,
    skip_prefix_n_tokens: int,
    block_size: int,
    num_heads: int,
    head_size: int,
) -> bool:
    fn = _cached_native_fn(symbol, [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_void_p,
    ])
    if fn is None:
        return False
    fn(
        ctypes.c_void_p(memory_tensor.data_ptr()),
        ctypes.c_void_p(kvcaches[0].data_ptr()),
        ctypes.c_void_p(kvcaches[1].data_ptr()),
        ctypes.c_void_p(slot_mapping.data_ptr()),
        int(memory_tensor.shape[2]),
        int(slot_mapping.numel()),
        skip_prefix_n_tokens,
        int(kvcaches[0].shape[1]),
        block_size,
        num_heads,
        head_size,
        _current_musa_stream_ptr(),
    )
    return True


def _try_cpp_extension_two_x_nl_to_paged(
    memory_tensor: torch.Tensor,
    kvcaches: list[list[torch.Tensor]],
    slot_mapping: torch.Tensor,
    skip_prefix_n_tokens: int,
    block_size: int,
    num_heads: int,
    head_size: int,
) -> bool:
    if memory_tensor.dtype not in (torch.float16, torch.bfloat16):
        return False
    ext = load_lmcache_kv_transfer_ext()
    if ext is None or not callable(getattr(ext, "kv_buffer_to_paged_two_x_nl", None)):
        return False
    return bool(ext.kv_buffer_to_paged_two_x_nl(
        memory_tensor, kvcaches, slot_mapping, skip_prefix_n_tokens,
        block_size, num_heads, head_size
    ))


def _try_cpp_extension_two_x_nl_to_buffer(
    kvcaches: list[list[torch.Tensor]],
    memory_tensor: torch.Tensor,
    slot_mapping: torch.Tensor,
    block_size: int,
    num_heads: int,
    head_size: int,
) -> bool:
    if memory_tensor.dtype not in (torch.float16, torch.bfloat16):
        return False
    ext = load_lmcache_kv_transfer_ext()
    if ext is None or not callable(getattr(ext, "paged_to_kv_buffer_two_x_nl", None)):
        return False
    return bool(ext.paged_to_kv_buffer_two_x_nl(
        kvcaches, memory_tensor, slot_mapping, block_size, num_heads, head_size
    ))


def _try_cpp_extension_non_mla_list_to_paged(
    memory_tensor: torch.Tensor,
    kvcaches: list[torch.Tensor],
    slot_mapping: torch.Tensor,
    skip_prefix_n_tokens: int,
    block_size: int,
    num_heads: int,
    head_size: int,
) -> bool:
    # The ext list2 kernels vectorize 16 u16 lanes per row and have no scalar tail,
    # so hidden must be a multiple of 16 or the trailing lanes are silently dropped.
    if (
        len(kvcaches) != 2
        or memory_tensor.dtype not in (torch.float16, torch.bfloat16)
        or (num_heads * head_size) % 16 != 0
        or not _can_native_list(memory_tensor, kvcaches, slot_mapping)
    ):
        return False
    start = time.perf_counter()
    ext = load_lmcache_kv_transfer_ext()
    load_elapsed_us = (time.perf_counter() - start) * 1_000_000.0
    if ext is None:
        _log_timing("facade_load_ext_to_gpu_failed", load_elapsed_us, tokens=int(slot_mapping.numel()), layers=len(kvcaches))
        return False
    start = time.perf_counter()
    result = bool(ext.kv_buffer_to_paged_list2(memory_tensor, kvcaches, slot_mapping, skip_prefix_n_tokens, block_size, num_heads, head_size))
    call_elapsed_us = (time.perf_counter() - start) * 1_000_000.0
    _log_timing("facade_ext_to_gpu", call_elapsed_us, load_ext_us=f"{load_elapsed_us:.3f}", tokens=int(slot_mapping.numel()), layers=len(kvcaches))
    return result


def _try_cpp_extension_non_mla_list_to_buffer(
    kvcaches: list[torch.Tensor],
    memory_tensor: torch.Tensor,
    slot_mapping: torch.Tensor,
    block_size: int,
    num_heads: int,
    head_size: int,
) -> bool:
    # Same vec-16-without-scalar-tail constraint as the to-paged variant.
    if (
        len(kvcaches) != 2
        or memory_tensor.dtype not in (torch.float16, torch.bfloat16)
        or (num_heads * head_size) % 16 != 0
        or not _can_native_list(memory_tensor, kvcaches, slot_mapping)
    ):
        return False
    start = time.perf_counter()
    ext = load_lmcache_kv_transfer_ext()
    load_elapsed_us = (time.perf_counter() - start) * 1_000_000.0
    if ext is None:
        _log_timing("facade_load_ext_from_gpu_failed", load_elapsed_us, tokens=int(slot_mapping.numel()), layers=len(kvcaches))
        return False
    start = time.perf_counter()
    result = bool(ext.paged_to_kv_buffer_list2(kvcaches, memory_tensor, slot_mapping, block_size, num_heads, head_size))
    call_elapsed_us = (time.perf_counter() - start) * 1_000_000.0
    _log_timing("facade_ext_from_gpu", call_elapsed_us, load_ext_us=f"{load_elapsed_us:.3f}", tokens=int(slot_mapping.numel()), layers=len(kvcaches))
    return result


def _try_native_non_mla_list_to_paged(
    memory_tensor: torch.Tensor,
    kvcaches: list[torch.Tensor],
    slot_mapping: torch.Tensor,
    skip_prefix_n_tokens: int,
    block_size: int,
    num_heads: int,
    head_size: int,
) -> bool:
    if (
        memory_tensor.dtype not in (torch.float32, torch.float16, torch.bfloat16)
        or (
            memory_tensor.dtype is not torch.float32
            and (num_heads * head_size) % 4 != 0
        )
        or not _can_native_list(memory_tensor, kvcaches, slot_mapping)
    ):
        return False
    if (
        memory_tensor.dtype is not torch.float32
        and _try_cpp_extension_non_mla_list_to_paged(
            memory_tensor,
            kvcaches,
            slot_mapping,
            skip_prefix_n_tokens,
            block_size,
            num_heads,
            head_size,
        )
    ):
        return True
    if (
        memory_tensor.dtype is not torch.float32
        and len(kvcaches) == 2
        and (num_heads * head_size) % 16 == 0
        and _call_non_mla_list2_u16_symbol(
            "lmcache_kv_buffer_to_paged_list2_u16",
            kvcaches,
            memory_tensor,
            slot_mapping,
            skip_prefix_n_tokens,
            block_size,
            num_heads,
            head_size,
        )
    ):
        return True
    for layer, kvcache in enumerate(kvcaches):
        if not _call_non_mla_list_symbol(
            _lmcache_list_symbol("lmcache_kv_buffer_to_paged_list", memory_tensor.dtype),
            kvcache,
            memory_tensor,
            slot_mapping,
            layer,
            skip_prefix_n_tokens,
            block_size,
            num_heads,
            head_size,
        ):
            return False
    return True


def _try_native_non_mla_list_to_buffer(
    kvcaches: list[torch.Tensor],
    memory_tensor: torch.Tensor,
    slot_mapping: torch.Tensor,
    block_size: int,
    num_heads: int,
    head_size: int,
) -> bool:
    if (
        memory_tensor.dtype not in (torch.float32, torch.float16, torch.bfloat16)
        or (
            memory_tensor.dtype is not torch.float32
            and (num_heads * head_size) % 4 != 0
        )
        or not _can_native_list(memory_tensor, kvcaches, slot_mapping)
    ):
        return False
    if (
        memory_tensor.dtype is not torch.float32
        and _try_cpp_extension_non_mla_list_to_buffer(
            kvcaches, memory_tensor, slot_mapping, block_size, num_heads, head_size
        )
    ):
        return True
    if (
        memory_tensor.dtype is not torch.float32
        and len(kvcaches) == 2
        and (num_heads * head_size) % 16 == 0
        and _call_non_mla_list2_u16_symbol(
            "lmcache_paged_to_kv_buffer_list2_u16",
            kvcaches,
            memory_tensor,
            slot_mapping,
            0,
            block_size,
            num_heads,
            head_size,
        )
    ):
        return True
    for layer, kvcache in enumerate(kvcaches):
        if not _call_non_mla_list_symbol(
            _lmcache_list_symbol("lmcache_paged_to_kv_buffer_list", memory_tensor.dtype),
            kvcache,
            memory_tensor,
            slot_mapping,
            layer,
            0,
            block_size,
            num_heads,
            head_size,
        ):
            return False
    return True


def _try_native_mla_list_to_paged(
    memory_tensor: torch.Tensor,
    kvcaches: list[torch.Tensor],
    slot_mapping: torch.Tensor,
    skip_prefix_n_tokens: int,
    block_size: int,
    head_size: int,
) -> bool:
    if (
        memory_tensor.dtype not in (torch.float32, torch.float16, torch.bfloat16)
        or not _can_native_list(memory_tensor, kvcaches, slot_mapping)
    ):
        return False
    for layer, kvcache in enumerate(kvcaches):
        if not _call_mla_list_symbol(
            _lmcache_list_symbol("lmcache_mla_buffer_to_paged_list", memory_tensor.dtype),
            kvcache,
            memory_tensor,
            slot_mapping,
            layer,
            skip_prefix_n_tokens,
            block_size,
            head_size,
            True,
        ):
            return False
    return True


def _try_native_mla_list_to_buffer(
    kvcaches: list[torch.Tensor],
    memory_tensor: torch.Tensor,
    slot_mapping: torch.Tensor,
    block_size: int,
    head_size: int,
) -> bool:
    if (
        memory_tensor.dtype not in (torch.float32, torch.float16, torch.bfloat16)
        or not _can_native_list(memory_tensor, kvcaches, slot_mapping)
    ):
        return False
    for layer, kvcache in enumerate(kvcaches):
        if not _call_mla_list_symbol(
            _lmcache_list_symbol("lmcache_mla_paged_to_buffer_list", memory_tensor.dtype),
            kvcache,
            memory_tensor,
            slot_mapping,
            layer,
            0,
            block_size,
            head_size,
            False,
        ):
            return False
    return True


def _try_native_non_mla_to_paged(kv_buffer: torch.Tensor, paged_kv_cache: torch.Tensor, slot_mapping: torch.Tensor) -> bool:
    if not (_can_native_data(kv_buffer, paged_kv_cache) and _can_native_slot_mapping(slot_mapping)):
        return False
    return _call_non_mla_symbol(_lmcache_symbol("lmcache_kv_buffer_to_paged", kv_buffer.dtype), kv_buffer, paged_kv_cache, slot_mapping)


def _try_native_non_mla_to_buffer(paged_kv_cache: torch.Tensor, kv_buffer: torch.Tensor, slot_mapping: torch.Tensor) -> bool:
    if not (_can_native_data(paged_kv_cache, kv_buffer) and _can_native_slot_mapping(slot_mapping)):
        return False
    return _call_non_mla_symbol(_lmcache_symbol("lmcache_paged_to_kv_buffer", kv_buffer.dtype), paged_kv_cache, kv_buffer, slot_mapping)


def _try_native_mla_to_paged(kv_buffer: torch.Tensor, paged_kv_cache: torch.Tensor, slot_mapping: torch.Tensor) -> bool:
    if not (_can_native_data(kv_buffer, paged_kv_cache) and _can_native_slot_mapping(slot_mapping)):
        return False
    return _call_mla_symbol(_lmcache_symbol("lmcache_kv_buffer_to_paged_mla", kv_buffer.dtype), kv_buffer, paged_kv_cache, slot_mapping)


def _try_native_mla_to_buffer(paged_kv_cache: torch.Tensor, kv_buffer: torch.Tensor, slot_mapping: torch.Tensor) -> bool:
    if not (_can_native_data(paged_kv_cache, kv_buffer) and _can_native_slot_mapping(slot_mapping)):
        return False
    return _call_mla_symbol(_lmcache_symbol("lmcache_paged_to_kv_buffer_mla", kv_buffer.dtype), paged_kv_cache, kv_buffer, slot_mapping)


def lmcache_kv_strided_block_transfer(
    staging: "torch.Tensor | int",
    paged_ptrs: list[int],
    block_ids: torch.Tensor,
    direction: int,
    num_layers: int,
    idx_start: int,
    n_valid: int,
    offset_in_object: int,
    staging_tokens: int,
    block_size: int,
    row_bytes: int,
    paged_block_stride: int,
    mode: int,
    value_bytes: int = 0,
    scale_bytes: int = 0,
) -> bool:
    """Stride-aware byte-block transfer for DSV4 MLA/blocked-scale groups.

    One kernel launch moves a whole kernel-group batch (all layers, all
    blocks, value+scale planes) between a contiguous LMCache staging object
    ``[NL, staging_tokens, row_bytes]`` and padded per-layer paged pages.
    ``mode 0`` addresses plain ``[NB, BS, HS]`` pages; ``mode 1`` addresses
    blocked-scale pages (per block: all tokens' values, then all tokens'
    scales). ``block_ids`` must be a contiguous MUSA int64 tensor;
    ``paged_ptrs`` is a host list of per-layer base pointers.
    """

    if not _native_requested():
        return False
    if num_layers <= 0 or num_layers > 128:
        return False
    if n_valid <= 0 or block_size <= 0 or staging_tokens <= 0:
        return False
    if row_bytes <= 0 or row_bytes % 4 != 0:
        return False
    if mode not in (0, 1) or int(direction) not in (0, 1):
        return False
    if mode == 1 and (
        value_bytes <= 0 or scale_bytes <= 0
        or value_bytes % 4 != 0 or scale_bytes % 4 != 0
    ):
        return False
    if not isinstance(paged_ptrs, (list, tuple)) or len(paged_ptrs) != num_layers:
        return False
    if not isinstance(block_ids, torch.Tensor):
        return False
    if (
        block_ids.dtype is not torch.int64
        or str(block_ids.device).split(":")[0] != "musa"
        or not block_ids.is_contiguous()
    ):
        return False
    if idx_start < 0 or idx_start + n_valid > int(block_ids.numel()):
        return False
    staging_ptr = (
        int(staging.data_ptr()) if isinstance(staging, torch.Tensor) else int(staging)
    )
    if staging_ptr <= 0:
        return False

    fn = _cached_native_fn(
        "lmcache_kv_strided_block_transfer",
        [
            ctypes.c_void_p,                      # staging
            ctypes.POINTER(ctypes.c_uint64),      # paged_ptrs (host)
            ctypes.c_void_p,                      # block_ids (device)
            ctypes.c_int64,                       # direction
            ctypes.c_int64,                       # num_layers
            ctypes.c_int64,                       # idx_start
            ctypes.c_int64,                       # n_valid
            ctypes.c_int64,                       # offset_in_object
            ctypes.c_int64,                       # staging_tokens
            ctypes.c_int64,                       # block_size
            ctypes.c_int64,                       # row_bytes
            ctypes.c_int64,                       # paged_block_stride
            ctypes.c_int64,                       # mode
            ctypes.c_int64,                       # value_bytes
            ctypes.c_int64,                       # scale_bytes
            ctypes.c_void_p,                      # stream
        ],
    )
    if fn is None:
        return False
    ptr_array = (ctypes.c_uint64 * num_layers)(*map(int, paged_ptrs))
    device = block_ids.device
    try:
        with torch.musa.device(device):
            stream_ptr = _current_musa_stream_ptr()
            fn(
                ctypes.c_void_p(staging_ptr),
                ptr_array,
                ctypes.c_void_p(block_ids.data_ptr()),
                int(direction),
                int(num_layers),
                int(idx_start),
                int(n_valid),
                int(offset_in_object),
                int(staging_tokens),
                int(block_size),
                int(row_bytes),
                int(paged_block_stride),
                int(mode),
                int(value_bytes),
                int(scale_bytes),
                stream_ptr,
            )
    except Exception:
        return False
    return True


def _slot_values(slot_mapping: torch.Tensor) -> list[int]:
    return [int(value) for value in slot_mapping.detach().cpu().reshape(-1).tolist()]
