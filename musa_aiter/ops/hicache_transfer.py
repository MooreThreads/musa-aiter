"""SGLang HiCache layout transfer façade.

The public functions use the same two-index-table and byte-stride model as
SGLang's CUDA KVCacheIO kernels. Tensors are three-dimensional:

* ``layer_first``: ``[layers, pages, item_elements]``
* ``page_first``: ``[pages, layers, item_elements]``

The reference path is device agnostic. The opt-in native path copies 64-bit
words on the caller's current MUSA stream and keeps the same offset contract.
"""

from __future__ import annotations

import ctypes
from pathlib import Path
from typing import Literal

import torch

from musa_aiter._paths import project_data_path
from musa_aiter.dispatch import DispatchRecord, emit, tensor_device_dtype
from musa_aiter.jit.core import build_shared_object
from musa_aiter.jit.ctypes_loader import has_symbol, load_library
from musa_aiter.ops.kv_transfer import _current_musa_stream_ptr
from musa_aiter.ops.native_gate import native_requested
from musa_aiter.policy import ensure_reference_allowed

_SOURCE = project_data_path("csrc", "kvcache", "hicache_transfer.mu")
_ABI_VERSION = 1
_BUILD_ATTEMPTED = False
_BUILD_RESULT = None
_LIBRARY = None

LayoutName = Literal["layer_first", "page_first"]


def build_native_hicache_transfer():
    global _BUILD_ATTEMPTED, _BUILD_RESULT
    if not _BUILD_ATTEMPTED:
        _BUILD_RESULT = build_shared_object(
            name="hicache_transfer",
            sources=[Path(_SOURCE)],
        )
        _BUILD_ATTEMPTED = True
    return _BUILD_RESULT


def load_native_hicache_transfer():
    global _LIBRARY
    result = build_native_hicache_transfer()
    if not result.ok or result.library_path is None:
        raise RuntimeError(result.message)
    if _LIBRARY is None:
        _LIBRARY = load_library(result.library_path)
    return _LIBRARY


def native_hicache_transfer_abi_version() -> int | None:
    try:
        library = load_native_hicache_transfer()
        if not has_symbol(library, "musa_aiter_hicache_transfer_abi_version"):
            return None
        fn = library.handle.musa_aiter_hicache_transfer_abi_version
        fn.argtypes = []
        fn.restype = ctypes.c_int
        return int(fn())
    except Exception:
        return None


def is_native_hicache_transfer_available() -> bool:
    result = build_native_hicache_transfer()
    if not result.ok or result.library_path is None:
        return False
    try:
        library = load_native_hicache_transfer()
        return (
            native_hicache_transfer_abi_version() == _ABI_VERSION
            and has_symbol(library, "musa_aiter_hicache_transfer_bytes")
        )
    except Exception:
        return False


def _native_requested() -> bool:
    return native_requested(
        "MUSA_AITER_NATIVE_HICACHE_TRANSFER",
        "MUSA_AITER_NATIVE_KV_TRANSFER",
    )


def _is_musa_tensor(tensor: torch.Tensor) -> bool:
    return str(tensor.device).split(":")[0] in {"musa", "privateuseone"}


def _layout_geometry(
    tensor: torch.Tensor, layout: LayoutName
) -> tuple[int, int, int, int, int]:
    if not isinstance(tensor, torch.Tensor) or tensor.dim() != 3:
        raise ValueError("HiCache tensors must be rank-3 [layers/pages/item] or [pages/layers/item]")
    if not tensor.is_contiguous():
        raise ValueError("HiCache tensors must be contiguous")
    if layout == "layer_first":
        layers, pages, item_elements = map(int, tensor.shape)
        page_stride = item_elements * tensor.element_size()
        layer_stride = pages * page_stride
    elif layout == "page_first":
        pages, layers, item_elements = map(int, tensor.shape)
        layer_stride = item_elements * tensor.element_size()
        page_stride = layers * layer_stride
    else:
        raise ValueError(f"unsupported HiCache layout={layout!r}")
    if layers <= 0 or pages <= 0 or item_elements <= 0:
        raise ValueError("HiCache layer/page/item dimensions must be positive")
    return layers, pages, item_elements, page_stride, layer_stride


def _validate_inputs(
    src_k: torch.Tensor,
    dst_k: torch.Tensor,
    src_v: torch.Tensor | None,
    dst_v: torch.Tensor | None,
    src_indices: torch.Tensor,
    dst_indices: torch.Tensor,
    src_layout: LayoutName,
    dst_layout: LayoutName,
    start_layer: int,
    num_layers: int,
    *,
    mla: bool,
) -> tuple[tuple[int, int, int, int, int], tuple[int, int, int, int, int], int]:
    if not isinstance(src_indices, torch.Tensor) or not isinstance(dst_indices, torch.Tensor):
        raise TypeError("HiCache index tables must be torch.Tensor values")
    if src_indices.dim() != 1 or dst_indices.dim() != 1:
        raise ValueError("HiCache index tables must be 1-D")
    if src_indices.numel() != dst_indices.numel():
        raise ValueError("source and destination index tables must have the same length")
    if src_indices.dtype not in (torch.int32, torch.int64) or dst_indices.dtype not in (torch.int32, torch.int64):
        raise ValueError("HiCache index tables must use int32 or int64")
    if src_k.dtype != dst_k.dtype:
        raise ValueError("source and destination K tensors must have the same dtype")
    if src_k.device != dst_k.device:
        raise ValueError("source and destination K tensors must be on the same device")
    if src_v is not None or dst_v is not None:
        if mla:
            raise ValueError("MLA HiCache transfer does not accept a value plane")
        if src_v is None or dst_v is None:
            raise ValueError("source and destination value tensors must be provided together")
        if src_v.dtype != src_k.dtype or dst_v.dtype != dst_k.dtype:
            raise ValueError("K and V tensors must use the same dtype")
        if src_v.device != src_k.device or dst_v.device != dst_k.device:
            raise ValueError("K and V tensors must be on the same device")
    elif not mla:
        raise ValueError("non-MLA HiCache transfer requires K and V tensors")

    src_geometry = _layout_geometry(src_k, src_layout)
    dst_geometry = _layout_geometry(dst_k, dst_layout)
    if src_geometry[2] * src_k.element_size() != dst_geometry[2] * dst_k.element_size():
        raise ValueError("source and destination item byte sizes must match")
    if src_v is not None and _layout_geometry(src_v, src_layout) != src_geometry:
        raise ValueError("source K and V layouts must match")
    if dst_v is not None and _layout_geometry(dst_v, dst_layout) != dst_geometry:
        raise ValueError("destination K and V layouts must match")
    if start_layer < 0 or num_layers <= 0:
        raise ValueError("start_layer and num_layers must be non-negative/positive")
    if start_layer + num_layers > min(src_geometry[0], dst_geometry[0]):
        raise ValueError("requested layer range exceeds source or destination")
    item_size_bytes = src_geometry[2] * src_k.element_size()
    if item_size_bytes % 8 != 0:
        raise ValueError("HiCache item byte size must be divisible by 8")
    return src_geometry, dst_geometry, item_size_bytes


def _validate_page_indices(
    src_indices: torch.Tensor,
    dst_indices: torch.Tensor,
    src_pages: int,
    dst_pages: int,
) -> None:
    src_values = src_indices.to(device="cpu", dtype=torch.long).tolist()
    dst_values = dst_indices.to(device="cpu", dtype=torch.long).tolist()
    for src_page, dst_page in zip(src_values, dst_values):
        if src_page < -1 or dst_page < -1:
            raise ValueError("HiCache page indices must be -1 or non-negative")
        if src_page >= src_pages or dst_page >= dst_pages:
            raise ValueError("HiCache page index is out of range")


def _native_transfer(
    src_k: torch.Tensor,
    dst_k: torch.Tensor,
    src_v: torch.Tensor | None,
    dst_v: torch.Tensor | None,
    src_indices: torch.Tensor,
    dst_indices: torch.Tensor,
    src_geometry: tuple[int, int, int, int, int],
    dst_geometry: tuple[int, int, int, int, int],
    item_size_bytes: int,
    start_layer: int,
    num_layers: int,
) -> bool:
    if not (_is_musa_tensor(src_k) and _is_musa_tensor(dst_k)):
        return False
    if src_indices.device != src_k.device or dst_indices.device != dst_k.device:
        return False
    if src_indices.dtype is not torch.int64 or dst_indices.dtype is not torch.int64:
        return False
    if not src_indices.is_contiguous() or not dst_indices.is_contiguous():
        return False
    try:
        library = load_native_hicache_transfer()
        fn = library.handle.musa_aiter_hicache_transfer_bytes
        fn.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
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
            ctypes.c_int64,
            ctypes.c_void_p,
        ]
        fn.restype = None
        fn(
            ctypes.c_void_p(src_k.data_ptr()),
            ctypes.c_void_p(dst_k.data_ptr()),
            ctypes.c_void_p(src_v.data_ptr()) if src_v is not None else ctypes.c_void_p(0),
            ctypes.c_void_p(dst_v.data_ptr()) if dst_v is not None else ctypes.c_void_p(0),
            ctypes.c_void_p(src_indices.data_ptr()),
            ctypes.c_void_p(dst_indices.data_ptr()),
            ctypes.c_int64(src_indices.numel()),
            ctypes.c_int64(start_layer),
            ctypes.c_int64(num_layers),
            ctypes.c_int64(item_size_bytes),
            ctypes.c_int64(src_geometry[3]),
            ctypes.c_int64(src_geometry[4]),
            ctypes.c_int64(dst_geometry[3]),
            ctypes.c_int64(dst_geometry[4]),
            _current_musa_stream_ptr(),
        )
        return True
    except Exception:
        return False


def _transfer(
    src_k: torch.Tensor,
    dst_k: torch.Tensor,
    src_v: torch.Tensor | None,
    dst_v: torch.Tensor | None,
    src_indices: torch.Tensor,
    dst_indices: torch.Tensor,
    *,
    src_layout: LayoutName,
    dst_layout: LayoutName,
    start_layer: int,
    num_layers: int,
    mla: bool,
) -> bool:
    src_geometry, dst_geometry, item_size_bytes = _validate_inputs(
        src_k,
        dst_k,
        src_v,
        dst_v,
        src_indices,
        dst_indices,
        src_layout,
        dst_layout,
        start_layer,
        num_layers,
        mla=mla,
    )
    _validate_page_indices(
        src_indices,
        dst_indices,
        src_geometry[1],
        dst_geometry[1],
    )
    if _native_requested() and _native_transfer(
        src_k,
        dst_k,
        src_v,
        dst_v,
        src_indices,
        dst_indices,
        src_geometry,
        dst_geometry,
        item_size_bytes,
        start_layer,
        num_layers,
    ):
        emit(
            DispatchRecord(
                "hicache_transfer",
                "hicache-musa-native",
                *tensor_device_dtype(src_k),
                "SGLang CUDA-compatible byte-stride transfer",
            )
        )
        return True

    ensure_reference_allowed("hicache_transfer", native_available=False)
    emit(
        DispatchRecord(
            "hicache_transfer",
            "torch-reference",
            *tensor_device_dtype(src_k),
            "HiCache layout transfer reference fallback",
        )
    )
    src_values = src_indices.to(device="cpu", dtype=torch.long).tolist()
    dst_values = dst_indices.to(device="cpu", dtype=torch.long).tolist()
    for src_page, dst_page in zip(src_values, dst_values):
        if src_page < 0 or dst_page < 0:
            continue
        for layer in range(start_layer, start_layer + num_layers):
            src_k_row = (
                src_k[layer, src_page] if src_layout == "layer_first" else src_k[src_page, layer]
            )
            dst_k_row = (
                dst_k[layer, dst_page] if dst_layout == "layer_first" else dst_k[dst_page, layer]
            )
            dst_k_row.copy_(src_k_row)
            if src_v is not None and dst_v is not None:
                src_v_row = (
                    src_v[layer, src_page] if src_layout == "layer_first" else src_v[src_page, layer]
                )
                dst_v_row = (
                    dst_v[layer, dst_page] if dst_layout == "layer_first" else dst_v[dst_page, layer]
                )
                dst_v_row.copy_(src_v_row)
    return True


def hicache_transfer_kv_per_layer(
    src_k: torch.Tensor,
    dst_k: torch.Tensor,
    src_v: torch.Tensor,
    dst_v: torch.Tensor,
    src_indices: torch.Tensor,
    dst_indices: torch.Tensor,
    *,
    layer_id: int = 0,
) -> bool:
    return _transfer(
        src_k, dst_k, src_v, dst_v, src_indices, dst_indices,
        src_layout="layer_first", dst_layout="layer_first",
        start_layer=layer_id, num_layers=1, mla=False,
    )


def hicache_transfer_kv_per_layer_pf_lf(
    src_k: torch.Tensor,
    dst_k: torch.Tensor,
    src_v: torch.Tensor,
    dst_v: torch.Tensor,
    src_indices: torch.Tensor,
    dst_indices: torch.Tensor,
    *,
    layer_id: int = 0,
) -> bool:
    return _transfer(
        src_k, dst_k, src_v, dst_v, src_indices, dst_indices,
        src_layout="page_first", dst_layout="layer_first",
        start_layer=layer_id, num_layers=1, mla=False,
    )


def hicache_transfer_kv_all_layer(
    src_k: torch.Tensor,
    dst_k: torch.Tensor,
    src_v: torch.Tensor,
    dst_v: torch.Tensor,
    src_indices: torch.Tensor,
    dst_indices: torch.Tensor,
) -> bool:
    layers = int(src_k.shape[0])
    return _transfer(
        src_k, dst_k, src_v, dst_v, src_indices, dst_indices,
        src_layout="layer_first", dst_layout="layer_first",
        start_layer=0, num_layers=layers, mla=False,
    )


def hicache_transfer_kv_all_layer_lf_pf(
    src_k: torch.Tensor,
    dst_k: torch.Tensor,
    src_v: torch.Tensor,
    dst_v: torch.Tensor,
    src_indices: torch.Tensor,
    dst_indices: torch.Tensor,
) -> bool:
    layers = int(src_k.shape[0])
    return _transfer(
        src_k, dst_k, src_v, dst_v, src_indices, dst_indices,
        src_layout="layer_first", dst_layout="page_first",
        start_layer=0, num_layers=layers, mla=False,
    )


def hicache_transfer_kv_per_layer_mla(
    src: torch.Tensor,
    dst: torch.Tensor,
    src_indices: torch.Tensor,
    dst_indices: torch.Tensor,
    *,
    layer_id: int = 0,
    value: torch.Tensor | None = None,
) -> bool:
    if value is not None:
        raise ValueError("MLA HiCache transfer does not accept a value plane")
    return _transfer(
        src, dst, None, None, src_indices, dst_indices,
        src_layout="layer_first", dst_layout="layer_first",
        start_layer=layer_id, num_layers=1, mla=True,
    )


def hicache_transfer_kv_per_layer_mla_pf_lf(
    src: torch.Tensor,
    dst: torch.Tensor,
    src_indices: torch.Tensor,
    dst_indices: torch.Tensor,
    *,
    layer_id: int = 0,
) -> bool:
    return _transfer(
        src, dst, None, None, src_indices, dst_indices,
        src_layout="page_first", dst_layout="layer_first",
        start_layer=layer_id, num_layers=1, mla=True,
    )


def hicache_transfer_kv_all_layer_mla(
    src: torch.Tensor,
    dst: torch.Tensor,
    src_indices: torch.Tensor,
    dst_indices: torch.Tensor,
) -> bool:
    layers = int(src.shape[0])
    return _transfer(
        src, dst, None, None, src_indices, dst_indices,
        src_layout="layer_first", dst_layout="layer_first",
        start_layer=0, num_layers=layers, mla=True,
    )


def hicache_transfer_kv_all_layer_mla_lf_pf(
    src: torch.Tensor,
    dst: torch.Tensor,
    src_indices: torch.Tensor,
    dst_indices: torch.Tensor,
) -> bool:
    """Transfer all MLA layers from layer-first to page-first storage."""

    layers = int(src.shape[0])
    return _transfer(
        src,
        dst,
        None,
        None,
        src_indices,
        dst_indices,
        src_layout="layer_first",
        dst_layout="page_first",
        start_layer=0,
        num_layers=layers,
        mla=True,
    )


def _validate_direct_indices(
    src_indices: torch.Tensor,
    dst_indices: torch.Tensor,
    page_size: int,
) -> tuple[list[int], list[int]]:
    if not isinstance(src_indices, torch.Tensor) or not isinstance(dst_indices, torch.Tensor):
        raise TypeError("HiCache direct index tables must be torch.Tensor values")
    if src_indices.dim() != 1 or dst_indices.dim() != 1:
        raise ValueError("HiCache direct index tables must be 1-D")
    if src_indices.numel() != dst_indices.numel():
        raise ValueError("source and destination index tables must have the same length")
    if src_indices.dtype not in (torch.int32, torch.int64) or dst_indices.dtype not in (torch.int32, torch.int64):
        raise ValueError("HiCache direct index tables must use int32 or int64")
    if page_size <= 0:
        raise ValueError("HiCache direct page_size must be positive")
    if src_indices.numel() % page_size != 0:
        raise ValueError("HiCache direct index count must be divisible by page_size")
    src_values = src_indices.detach().to(device="cpu", dtype=torch.long).tolist()
    dst_values = dst_indices.detach().to(device="cpu", dtype=torch.long).tolist()
    if any(value < 0 for value in src_values + dst_values):
        raise ValueError("HiCache direct indices must be non-negative")
    return src_values, dst_values


def _validate_direct_tensor_lists(
    src_layers: list[torch.Tensor],
    dst_layers: list[torch.Tensor],
) -> None:
    if not isinstance(src_layers, (list, tuple)) or not isinstance(dst_layers, (list, tuple)):
        raise TypeError("HiCache direct layers must be tensor lists")
    if not src_layers or len(src_layers) != len(dst_layers):
        raise ValueError("HiCache direct source and destination lists must have equal non-zero length")
    for src, dst in zip(src_layers, dst_layers):
        if not isinstance(src, torch.Tensor) or not isinstance(dst, torch.Tensor):
            raise TypeError("HiCache direct layer entries must be tensors")
        if src.dim() < 1 or dst.dim() < 1:
            raise ValueError("HiCache direct tensors must have at least one dimension")
        if tuple(src.shape[1:]) != tuple(dst.shape[1:]):
            raise ValueError("HiCache direct source and destination tails must match")
        if src.dtype != dst.dtype:
            raise ValueError("HiCache direct source and destination dtypes must match")
        if not src.is_contiguous() or not dst.is_contiguous():
            raise ValueError("HiCache direct tensors must be contiguous")


def _emit_direct_dispatch(operation: str, tensor: torch.Tensor) -> None:
    backend = (
        "hicache-musa-direct"
        if _is_musa_tensor(tensor) and _native_requested()
        else "torch-reference-direct"
    )
    emit(
        DispatchRecord(
            operation,
            backend,
            *tensor_device_dtype(tensor),
            "SGLang direct range copy",
        )
    )


def _copy_direct_runs(
    src: torch.Tensor,
    dst: torch.Tensor,
    src_values: list[int],
    dst_values: list[int],
    page_size: int,
) -> None:
    start = 0
    while start < len(src_values):
        end = start + page_size
        run_src = src_values[start]
        run_dst = dst_values[start]
        while end < len(src_values):
            if src_values[end] != src_values[end - 1] + 1 or dst_values[end] != dst_values[end - 1] + 1:
                break
            end += 1
        count = end - start
        if run_src + count > int(src.shape[0]) or run_dst + count > int(dst.shape[0]):
            raise ValueError("HiCache direct token range exceeds source or destination")
        dst.narrow(0, run_dst, count).copy_(
            src.narrow(0, run_src, count),
            non_blocking=True,
        )
        start = end


def hicache_transfer_kv_direct(
    src_layers: list[torch.Tensor],
    dst_layers: list[torch.Tensor],
    src_indices: torch.Tensor,
    dst_indices: torch.Tensor,
    page_size: int,
) -> bool:
    """Copy contiguous token ranges for SGLang's direct HiCache path."""

    _validate_direct_tensor_lists(src_layers, dst_layers)
    src_values, dst_values = _validate_direct_indices(src_indices, dst_indices, int(page_size))
    for src, dst in zip(src_layers, dst_layers):
        _copy_direct_runs(src, dst, src_values, dst_values, int(page_size))
    _emit_direct_dispatch("hicache_transfer_direct", dst_layers[0])
    return True


def hicache_transfer_kv_per_layer_direct_pf_lf(
    src_ptrs: list[torch.Tensor],
    dst_ptrs: list[torch.Tensor],
    src_indices: torch.Tensor,
    dst_indices: torch.Tensor,
    layer_id: int,
    page_size: int,
) -> bool:
    """Copy page-first source pages into one layer-first destination layer."""

    if layer_id < 0:
        raise ValueError("HiCache direct layer_id must be non-negative")
    if len(src_ptrs) not in (1, 2) or len(dst_ptrs) != len(src_ptrs):
        raise ValueError("HiCache per-layer direct expects one MLA plane or K/V plane pairs")
    src_values, dst_values = _validate_direct_indices(src_indices, dst_indices, int(page_size))
    for src, dst in zip(src_ptrs, dst_ptrs):
        if src.dim() != 4 or dst.dim() != 2:
            raise ValueError("HiCache per-layer direct expects source [pages,layers,page,item] and destination [tokens,item]")
        if layer_id >= int(src.shape[1]):
            raise ValueError("HiCache direct layer_id exceeds source layers")
        if not src.is_contiguous() or not dst.is_contiguous():
            raise ValueError("HiCache direct tensors must be contiguous")
        if src.dtype != dst.dtype or tuple(src.shape[3:]) != tuple(dst.shape[1:]):
            raise ValueError("HiCache direct source and destination item shapes/dtypes must match")
        for start in range(0, len(src_values), int(page_size)):
            src_token = src_values[start]
            dst_token = dst_values[start]
            src_page = src_token // int(page_size)
            if src_page >= int(src.shape[0]) or dst_token + int(page_size) > int(dst.shape[0]):
                raise ValueError("HiCache direct page range exceeds source or destination")
            dst.narrow(0, dst_token, int(page_size)).copy_(
                src[src_page, layer_id],
                non_blocking=True,
            )
    _emit_direct_dispatch("hicache_transfer_per_layer_direct_pf_lf", dst_ptrs[0])
    return True


def hicache_transfer_kv_all_layer_direct_lf_pf(
    src_ptrs: list[torch.Tensor],
    dst_ptrs: list[torch.Tensor],
    src_indices: torch.Tensor,
    dst_indices: torch.Tensor,
    page_size: int,
) -> bool:
    """Copy layer-first source pages into an all-layer page-first destination."""

    if len(dst_ptrs) not in (1, 2):
        raise ValueError("HiCache all-layer direct expects one MLA plane or K/V plane pairs")
    is_mla = len(dst_ptrs) == 1
    if len(src_ptrs) == 0 or (not is_mla and len(src_ptrs) % 2 != 0):
        raise ValueError("HiCache all-layer direct source planes do not match destination planes")
    num_layers = len(src_ptrs) if is_mla else len(src_ptrs) // 2
    _validate_direct_indices(src_indices, dst_indices, int(page_size))
    if not all(isinstance(tensor, torch.Tensor) for tensor in src_ptrs + dst_ptrs):
        raise TypeError("HiCache direct layer entries must be tensors")
    for tensor in src_ptrs:
        if tensor.dim() != 2 or not tensor.is_contiguous():
            raise ValueError("HiCache all-layer direct sources must be contiguous [tokens,item] tensors")
    for tensor in dst_ptrs:
        if tensor.dim() != 4 or not tensor.is_contiguous():
            raise ValueError("HiCache all-layer direct destinations must be contiguous [pages,layers,page,item] tensors")
    for index in range(num_layers):
        if (
            src_ptrs[index].dtype != dst_ptrs[0].dtype
            or tuple(src_ptrs[index].shape[1:]) != tuple(dst_ptrs[0].shape[3:])
        ):
            raise ValueError("HiCache direct source and destination item shapes/dtypes must match")
    if not is_mla:
        if (
            src_ptrs[num_layers].dtype != dst_ptrs[1].dtype
            or tuple(src_ptrs[num_layers].shape[1:]) != tuple(dst_ptrs[1].shape[3:])
        ):
            raise ValueError("HiCache direct K/V item shapes/dtypes must match")

    src_values, dst_values = _validate_direct_indices(src_indices, dst_indices, int(page_size))
    for start in range(0, len(src_values), int(page_size)):
        src_token = src_values[start]
        dst_token = dst_values[start]
        dst_page = dst_token // int(page_size)
        if dst_page >= int(dst_ptrs[0].shape[0]):
            raise ValueError("HiCache direct destination page exceeds destination")
        for layer in range(num_layers):
            if src_token + int(page_size) > int(src_ptrs[layer].shape[0]):
                raise ValueError("HiCache direct source range exceeds source")
            dst_ptrs[0][dst_page, layer].copy_(
                src_ptrs[layer].narrow(0, src_token, int(page_size)),
                non_blocking=True,
            )
            if not is_mla:
                dst_ptrs[1][dst_page, layer].copy_(
                    src_ptrs[layer + num_layers].narrow(0, src_token, int(page_size)),
                    non_blocking=True,
                )
    _emit_direct_dispatch("hicache_transfer_all_layer_direct_lf_pf", dst_ptrs[0])
    return True
