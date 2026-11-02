"""Phase PKV — paged KV ⇄ contiguous linear offload buffers (reference).

Tensor layout matches :func:`~musa_aiter.ops.cache.copy_to_kv_cache` and ``paged_attention_v1``:

- **Single-layer paged caches:** ``[num_blocks, num_kv_heads, head_dim, block_size]``.
- **Multi-layer paged caches:** leading dimension **layers** →
  ``[num_layers, num_blocks, num_kv_heads, head_dim, block_size]`` (same ``slot_mapping`` per layer).
- **Linear buffers:** ``[num_tokens, num_kv_heads, head_dim]`` or, for multi-layer,
  ``[num_layers, num_tokens, num_kv_heads, head_dim]`` (**layer-major**).
- **Merged linear buffers:** ``[num_tokens, 2, num_kv_heads, head_dim]`` or, for multi-layer,
  ``[num_layers, num_tokens, 2, num_kv_heads, head_dim]`` where axis 2 stores key then value.
- ``slot_mapping`` length ``num_tokens``; flattened slot ``block_id * block_size + offset``.
  Indices ``slot < 0`` are **skipped** (prefix-cache / inactive positions); gather writes zeros there;
  scatter does not overwrite paged slots for skipped tokens.

Native MUSA backends are available for separate-K/V float32, float16, and bfloat16
transfer paths. Merged-layout, MLA/DSA flat-cache, and CacheGen native paths currently use
float32 kernels. Native execution remains opt-in via ``MUSA_AITER_NATIVE_KV_TRANSFER=1``.
"""

from __future__ import annotations

import ctypes
import os
from typing import Literal

import torch

from musa_aiter.dispatch import DispatchRecord, emit, tensor_device_dtype
from musa_aiter._paths import project_data_path
from musa_aiter.jit.core import build_shared_object
from musa_aiter.jit.ctypes_loader import has_symbol, load_library
from musa_aiter.policy import ensure_reference_allowed
from musa_aiter.ops.native_gate import native_requested
from musa_aiter.vendors.planning import emit_vendor_planning_note
from musa_aiter.ops.kv_layout import KVLayoutSpec, KVScaleKind

_KVCACHE_SOURCE = project_data_path("csrc", "kvcache", "kv_transfer_float.mu")
_KVCACHE_EXTENDED_SOURCE = project_data_path(
    "csrc", "kvcache", "lmcache_kv_transfer_extended.mu"
)
_FP8_KVCACHE_SOURCE = project_data_path("csrc", "kvcache", "fp8_kv_cache.mu")

_KV_NATIVE_ATTEMPTED = False
_KV_NATIVE_RESULT = None
_KV_NATIVE_LIB = None
_FP8_KV_NATIVE_ATTEMPTED = False
_FP8_KV_NATIVE_RESULT = None
_FP8_KV_NATIVE_LIB = None


def build_native_kv_transfer():
    global _KV_NATIVE_ATTEMPTED, _KV_NATIVE_RESULT
    if not _KV_NATIVE_ATTEMPTED:
        _KV_NATIVE_RESULT = build_shared_object(
            name="kv_transfer_float",
            sources=[_KVCACHE_SOURCE, _KVCACHE_EXTENDED_SOURCE],
            extra_flags=["-DMUSA_AITER_ENABLE_CACHEGEN_NATIVE"],
        )
        _KV_NATIVE_ATTEMPTED = True
    return _KV_NATIVE_RESULT


def load_native_kv_transfer():
    global _KV_NATIVE_LIB
    result = build_native_kv_transfer()
    if not result.ok or result.library_path is None:
        raise RuntimeError(result.message)
    if _KV_NATIVE_LIB is None:
        _KV_NATIVE_LIB = load_library(result.library_path)
    return _KV_NATIVE_LIB


def build_native_fp8_kv_cache():
    global _FP8_KV_NATIVE_ATTEMPTED, _FP8_KV_NATIVE_RESULT
    if not _FP8_KV_NATIVE_ATTEMPTED:
        _FP8_KV_NATIVE_RESULT = build_shared_object(name="fp8_kv_cache", sources=[_FP8_KVCACHE_SOURCE])
        _FP8_KV_NATIVE_ATTEMPTED = True
    return _FP8_KV_NATIVE_RESULT


def load_native_fp8_kv_cache():
    global _FP8_KV_NATIVE_LIB
    result = build_native_fp8_kv_cache()
    if not result.ok or result.library_path is None:
        raise RuntimeError(result.message)
    if _FP8_KV_NATIVE_LIB is None:
        _FP8_KV_NATIVE_LIB = load_library(result.library_path)
    return _FP8_KV_NATIVE_LIB


def native_kv_transfer_abi_version() -> int | None:
    try:
        lib = load_native_kv_transfer()
        if not has_symbol(lib, "musa_aiter_kv_transfer_abi_version"):
            return None
        fn = lib.handle.musa_aiter_kv_transfer_abi_version
        fn.argtypes = []
        fn.restype = ctypes.c_int
        return int(fn())
    except Exception:
        return None


def is_native_kv_transfer_available() -> bool:
    result = build_native_kv_transfer()
    if not bool(result and result.ok and result.library_path):
        return False
    try:
        lib = load_native_kv_transfer()
        return (
            has_symbol(lib, "musa_aiter_kv_paged_to_linear_float")
            and has_symbol(lib, "musa_aiter_kv_linear_to_paged_float")
            and has_symbol(lib, "musa_aiter_kv_paged_to_linear_multi_layer_float")
            and has_symbol(lib, "musa_aiter_kv_linear_to_paged_multi_layer_float")
            and has_symbol(lib, "musa_aiter_kv_paged_to_merged_linear_float")
            and has_symbol(lib, "musa_aiter_kv_merged_linear_to_paged_float")
            and has_symbol(lib, "musa_aiter_kv_paged_to_merged_linear_multi_layer_float")
            and has_symbol(lib, "musa_aiter_kv_merged_linear_to_paged_multi_layer_float")
            and has_symbol(lib, "musa_aiter_kv_paged_to_mla_linear_float")
            and has_symbol(lib, "musa_aiter_kv_mla_linear_to_paged_float")
            and has_symbol(lib, "musa_aiter_kv_load_and_reshape_flash_float")
            and has_symbol(lib, "musa_aiter_kv_reshape_and_cache_back_flash_float")
            and has_symbol(lib, "musa_aiter_kv_triton_indexed_to_dense_float")
            and has_symbol(lib, "musa_aiter_kv_copy_paged_blocks_float")
            and has_symbol(lib, "musa_aiter_kv_cachegen_encode_float")
            and has_symbol(lib, "musa_aiter_kv_cachegen_decode_float")
            and has_symbol(lib, "musa_aiter_kv_paged_to_linear_half")
            and has_symbol(lib, "musa_aiter_kv_linear_to_paged_half")
            and has_symbol(lib, "musa_aiter_kv_paged_to_linear_multi_layer_half")
            and has_symbol(lib, "musa_aiter_kv_linear_to_paged_multi_layer_half")
            and has_symbol(lib, "musa_aiter_kv_load_and_reshape_flash_half")
            and has_symbol(lib, "musa_aiter_kv_reshape_and_cache_back_flash_half")
            and has_symbol(lib, "musa_aiter_kv_triton_indexed_to_dense_half")
            and has_symbol(lib, "musa_aiter_kv_copy_paged_blocks_half")
            and has_symbol(lib, "musa_aiter_kv_paged_to_linear_bfloat16")
            and has_symbol(lib, "musa_aiter_kv_linear_to_paged_bfloat16")
            and has_symbol(lib, "musa_aiter_kv_paged_to_linear_multi_layer_bfloat16")
            and has_symbol(lib, "musa_aiter_kv_linear_to_paged_multi_layer_bfloat16")
            and has_symbol(lib, "musa_aiter_kv_load_and_reshape_flash_bfloat16")
            and has_symbol(lib, "musa_aiter_kv_reshape_and_cache_back_flash_bfloat16")
            and has_symbol(lib, "musa_aiter_kv_triton_indexed_to_dense_bfloat16")
            and has_symbol(lib, "musa_aiter_kv_copy_paged_blocks_bfloat16")
        )
    except Exception:
        return False


def native_fp8_kv_cache_abi_version() -> int | None:
    try:
        lib = load_native_fp8_kv_cache()
        if not has_symbol(lib, "musa_aiter_fp8_kv_cache_abi_version"):
            return None
        fn = lib.handle.musa_aiter_fp8_kv_cache_abi_version
        fn.argtypes = []
        fn.restype = ctypes.c_int
        return int(fn())
    except Exception:
        return None


def is_native_fp8_kv_cache_available(fp8_format: str = "e4m3") -> bool:
    result = build_native_fp8_kv_cache()
    if not bool(result and result.ok and result.library_path):
        return False
    try:
        lib = load_native_fp8_kv_cache()
        fmt = _normalize_fp8_format(fp8_format)
        return (
            has_symbol(lib, f"musa_aiter_kv_linear_to_paged_fp8_{fmt}")
            and has_symbol(lib, f"musa_aiter_kv_paged_fp8_to_linear_{fmt}")
        )
    except Exception:
        return False


def is_native_fp8_kv_vectorized_available() -> bool:
    result = build_native_fp8_kv_cache()
    if not bool(result and result.ok and result.library_path):
        return False
    try:
        lib = load_native_fp8_kv_cache()
        return (
            has_symbol(lib, "musa_aiter_kv_copy_paged_blocks_fp8_vec8")
            and has_symbol(lib, "musa_aiter_kv_linear_to_paged_fp8_vec4_e4m3")
            and has_symbol(lib, "musa_aiter_kv_linear_to_paged_fp8_vec4_e5m2")
            and has_symbol(lib, "musa_aiter_kv_paged_fp8_to_linear_hd64_bs16_e4m3")
            and has_symbol(lib, "musa_aiter_kv_paged_fp8_to_linear_hd64_bs16_e5m2")
            and has_symbol(lib, "musa_aiter_kv_paged_fp8_to_linear_hd128_bs16_e4m3")
            and has_symbol(lib, "musa_aiter_kv_paged_fp8_to_linear_hd128_bs16_e5m2")
            and has_symbol(lib, "musa_aiter_kv_paged_fp8_to_linear_hd128_bs32_e4m3")
            and has_symbol(lib, "musa_aiter_kv_paged_fp8_to_linear_hd128_bs32_e5m2")
            and has_symbol(lib, "musa_aiter_kv_paged_fp8_to_linear_vec4_e4m3")
            and has_symbol(lib, "musa_aiter_kv_paged_fp8_to_linear_vec4_e5m2")
        )
    except Exception:
        return False


def _native_fp8_kv_cache_requested() -> bool:
    return os.environ.get("MUSA_AITER_NATIVE_FP8_KV_CACHE", "0") == "1"


def _native_fp8_kv_vectorize_requested() -> bool:
    return os.environ.get("MUSA_AITER_NATIVE_FP8_KV_VECTORIZE", "0") == "1"


def _native_kv_transfer_requested() -> bool:
    return native_requested("MUSA_AITER_NATIVE_KV_TRANSFER")


def _current_musa_stream_ptr() -> ctypes.c_void_p:
    try:
        stream = torch.musa.current_stream()
        param = getattr(stream, "_as_parameter_", None)
        if isinstance(param, ctypes.c_void_p):
            return param
        stream_value = int(getattr(stream, "musa_stream", 0) or 0)
        return ctypes.c_void_p(stream_value)
    except Exception:
        return ctypes.c_void_p(0)


def _is_musa_float32_tensor(tensor: torch.Tensor) -> bool:
    return str(tensor.device).split(":")[0] == "musa" and tensor.dtype is torch.float32


def _is_musa_native_kv_dtype_tensor(tensor: torch.Tensor) -> bool:
    return str(tensor.device).split(":")[0] == "musa" and tensor.dtype in (torch.float32, torch.float16, torch.bfloat16)


def _is_musa_native_fp8_cache_tensor(tensor: torch.Tensor) -> bool:
    if str(tensor.device).split(":")[0] != "musa":
        return False
    return tensor.dtype is torch.uint8 or str(tensor.dtype) in ("torch.float8_e4m3fn", "torch.float8_e5m2")


def _native_fp8_linear_dtype_code(dtype: torch.dtype) -> int | None:
    if dtype is torch.float16:
        return 1
    if dtype is torch.bfloat16:
        return 2
    return None


def _as_musa_float32_scale_ptr(scale: torch.Tensor | float, *, device: torch.device) -> torch.Tensor | None:
    if isinstance(scale, torch.Tensor):
        scale_tensor = scale
    else:
        scale_tensor = torch.tensor(scale, dtype=torch.float32, device=device)
    if scale_tensor.device != device:
        return None
    if scale_tensor.dtype is not torch.float32 or scale_tensor.numel() != 1 or not scale_tensor.is_contiguous():
        return None
    return scale_tensor.reshape(())


def _as_musa_float32_scale_vec_ptr(scale: torch.Tensor | float, *, device: torch.device, num_heads: int) -> tuple[torch.Tensor, int] | None:
    if isinstance(scale, torch.Tensor):
        scale_tensor = scale
    else:
        scale_tensor = torch.tensor(scale, dtype=torch.float32, device=device)
    if scale_tensor.device != device or scale_tensor.dtype is not torch.float32:
        return None
    if scale_tensor.numel() == 1:
        return scale_tensor.reshape(()), 0
    if scale_tensor.dim() == 1 and int(scale_tensor.shape[0]) == num_heads and scale_tensor.is_contiguous():
        return scale_tensor, 1
    return None


def _as_musa_float32_scale_layer_head_ptr(scale: torch.Tensor | float, *, device: torch.device, num_layers: int, num_heads: int) -> tuple[torch.Tensor, int] | None:
    if isinstance(scale, torch.Tensor):
        scale_tensor = scale
    else:
        scale_tensor = torch.tensor(scale, dtype=torch.float32, device=device)
    if scale_tensor.device != device or scale_tensor.dtype is not torch.float32:
        return None
    if scale_tensor.numel() == 1:
        return scale_tensor.reshape(()), 0
    if scale_tensor.dim() == 1 and int(scale_tensor.shape[0]) == num_heads and scale_tensor.is_contiguous():
        return scale_tensor, 1
    if scale_tensor.dim() == 2 and tuple(scale_tensor.shape) == (num_layers, num_heads) and scale_tensor.is_contiguous():
        return scale_tensor, 2
    return None


def _native_kv_dtype_suffix(dtype: torch.dtype) -> str | None:
    if dtype is torch.float32:
        return "float"
    if dtype is torch.float16:
        return "half"
    if dtype is torch.bfloat16:
        return "bfloat16"
    return None


def _native_kv_symbol(base: str, dtype: torch.dtype) -> str | None:
    suffix = _native_kv_dtype_suffix(dtype)
    return None if suffix is None else f"{base}_{suffix}"


def _get_native_kv_function(lib, base: str, dtype: torch.dtype):
    symbol = _native_kv_symbol(base, dtype)
    if symbol is None or not has_symbol(lib, symbol):
        return None
    return getattr(lib.handle, symbol)


def _try_native_kv_paged_to_linear(
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    slots_long: torch.Tensor,
    valid: torch.Tensor,
    tcount: int,
    nh: int,
    hd: int,
    bs: int,
    *,
    fill_skipped: Literal["zeros", "preserve"],
    out_k: torch.Tensor | None,
    out_v: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    if not _native_kv_transfer_requested():
        return None
    if fill_skipped != "zeros":
        return None
    if not (_is_musa_native_kv_dtype_tensor(key_cache) and _is_musa_native_kv_dtype_tensor(value_cache)):
        return None
    if key_cache.dtype != value_cache.dtype:
        return None
    slots64 = slots_long.contiguous()
    key_work = key_cache if key_cache.is_contiguous() else key_cache.contiguous()
    value_work = value_cache if value_cache.is_contiguous() else value_cache.contiguous()
    out_k_work = out_k if out_k is not None else torch.empty((tcount, nh, hd), dtype=key_cache.dtype, device=key_cache.device)
    out_v_work = out_v if out_v is not None else torch.empty((tcount, nh, hd), dtype=value_cache.dtype, device=value_cache.device)
    if not (_is_musa_native_kv_dtype_tensor(out_k_work) and _is_musa_native_kv_dtype_tensor(out_v_work)):
        return None
    if out_k_work.dtype != key_cache.dtype or out_v_work.dtype != value_cache.dtype:
        return None
    if not out_k_work.is_contiguous() or not out_v_work.is_contiguous():
        return None
    try:
        lib = load_native_kv_transfer()
        fn = _get_native_kv_function(lib, "musa_aiter_kv_paged_to_linear", key_cache.dtype)
        if fn is None:
            return None
    except Exception:
        return None
    fn.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_longlong,
        ctypes.c_longlong,
        ctypes.c_longlong,
        ctypes.c_longlong,
        ctypes.c_longlong,
        ctypes.c_void_p,
    ]
    fn.restype = None
    fn(
        ctypes.c_void_p(key_work.data_ptr()),
        ctypes.c_void_p(value_work.data_ptr()),
        ctypes.c_void_p(slots64.data_ptr()),
        ctypes.c_void_p(out_k_work.data_ptr()),
        ctypes.c_void_p(out_v_work.data_ptr()),
        ctypes.c_longlong(tcount),
        ctypes.c_longlong(key_cache.shape[0]),
        ctypes.c_longlong(nh),
        ctypes.c_longlong(hd),
        ctypes.c_longlong(bs),
        ctypes.c_void_p(0),
    )
    return out_k_work, out_v_work


def _try_native_kv_linear_to_paged(
    key_linear: torch.Tensor,
    value_linear: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    slots_long: torch.Tensor,
) -> bool:
    if not _native_kv_transfer_requested():
        return False
    if not (
        _is_musa_native_kv_dtype_tensor(key_linear)
        and _is_musa_native_kv_dtype_tensor(value_linear)
        and _is_musa_native_kv_dtype_tensor(key_cache)
        and _is_musa_native_kv_dtype_tensor(value_cache)
    ):
        return False
    if not (key_linear.dtype == value_linear.dtype == key_cache.dtype == value_cache.dtype):
        return False
    key_linear_work = key_linear if key_linear.is_contiguous() else key_linear.contiguous()
    value_linear_work = value_linear if value_linear.is_contiguous() else value_linear.contiguous()
    if not key_cache.is_contiguous() or not value_cache.is_contiguous():
        return False
    slots64 = slots_long.contiguous()
    try:
        lib = load_native_kv_transfer()
        fn = _get_native_kv_function(lib, "musa_aiter_kv_linear_to_paged", key_linear.dtype)
        if fn is None:
            return False
    except Exception:
        return False
    fn.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_longlong,
        ctypes.c_longlong,
        ctypes.c_longlong,
        ctypes.c_longlong,
        ctypes.c_longlong,
        ctypes.c_void_p,
    ]
    fn.restype = None
    fn(
        ctypes.c_void_p(key_linear_work.data_ptr()),
        ctypes.c_void_p(value_linear_work.data_ptr()),
        ctypes.c_void_p(slots64.data_ptr()),
        ctypes.c_void_p(key_cache.data_ptr()),
        ctypes.c_void_p(value_cache.data_ptr()),
        ctypes.c_longlong(key_linear.shape[0]),
        ctypes.c_longlong(key_cache.shape[0]),
        ctypes.c_longlong(key_cache.shape[1]),
        ctypes.c_longlong(key_cache.shape[2]),
        ctypes.c_longlong(key_cache.shape[3]),
        ctypes.c_void_p(0),
    )
    return True


def _try_native_kv_paged_to_linear_multi_layer(
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    slots_long: torch.Tensor,
    tcount: int,
    nl: int,
    nh: int,
    hd: int,
    bs: int,
    *,
    fill_skipped: Literal["zeros", "preserve"],
    out_k: torch.Tensor | None,
    out_v: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    if not _native_kv_transfer_requested():
        return None
    if fill_skipped != "zeros":
        return None
    if not (_is_musa_native_kv_dtype_tensor(key_cache) and _is_musa_native_kv_dtype_tensor(value_cache)):
        return None
    if key_cache.dtype != value_cache.dtype:
        return None
    slots64 = slots_long.contiguous()
    key_work = key_cache if key_cache.is_contiguous() else key_cache.contiguous()
    value_work = value_cache if value_cache.is_contiguous() else value_cache.contiguous()
    out_k_work = out_k if out_k is not None else torch.empty((nl, tcount, nh, hd), dtype=key_cache.dtype, device=key_cache.device)
    out_v_work = out_v if out_v is not None else torch.empty((nl, tcount, nh, hd), dtype=value_cache.dtype, device=value_cache.device)
    if not (_is_musa_native_kv_dtype_tensor(out_k_work) and _is_musa_native_kv_dtype_tensor(out_v_work)):
        return None
    if out_k_work.dtype != key_cache.dtype or out_v_work.dtype != value_cache.dtype:
        return None
    if not out_k_work.is_contiguous() or not out_v_work.is_contiguous():
        return None
    try:
        lib = load_native_kv_transfer()
        fn = _get_native_kv_function(lib, "musa_aiter_kv_paged_to_linear_multi_layer", key_cache.dtype)
        if fn is None:
            return None
    except Exception:
        return None
    fn.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_longlong,
        ctypes.c_longlong,
        ctypes.c_longlong,
        ctypes.c_longlong,
        ctypes.c_longlong,
        ctypes.c_longlong,
        ctypes.c_void_p,
    ]
    fn.restype = None
    fn(
        ctypes.c_void_p(key_work.data_ptr()),
        ctypes.c_void_p(value_work.data_ptr()),
        ctypes.c_void_p(slots64.data_ptr()),
        ctypes.c_void_p(out_k_work.data_ptr()),
        ctypes.c_void_p(out_v_work.data_ptr()),
        ctypes.c_longlong(nl),
        ctypes.c_longlong(tcount),
        ctypes.c_longlong(key_cache.shape[1]),
        ctypes.c_longlong(nh),
        ctypes.c_longlong(hd),
        ctypes.c_longlong(bs),
        ctypes.c_void_p(0),
    )
    return out_k_work, out_v_work


def _try_native_kv_linear_to_paged_multi_layer(
    key_linear: torch.Tensor,
    value_linear: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    slots_long: torch.Tensor,
) -> bool:
    if not _native_kv_transfer_requested():
        return False
    if not (
        _is_musa_native_kv_dtype_tensor(key_linear)
        and _is_musa_native_kv_dtype_tensor(value_linear)
        and _is_musa_native_kv_dtype_tensor(key_cache)
        and _is_musa_native_kv_dtype_tensor(value_cache)
    ):
        return False
    if not (key_linear.dtype == value_linear.dtype == key_cache.dtype == value_cache.dtype):
        return False
    key_linear_work = key_linear if key_linear.is_contiguous() else key_linear.contiguous()
    value_linear_work = value_linear if value_linear.is_contiguous() else value_linear.contiguous()
    if not key_cache.is_contiguous() or not value_cache.is_contiguous():
        return False
    slots64 = slots_long.contiguous()
    try:
        lib = load_native_kv_transfer()
        fn = _get_native_kv_function(lib, "musa_aiter_kv_linear_to_paged_multi_layer", key_linear.dtype)
        if fn is None:
            return False
    except Exception:
        return False
    fn.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_longlong,
        ctypes.c_longlong,
        ctypes.c_longlong,
        ctypes.c_longlong,
        ctypes.c_longlong,
        ctypes.c_longlong,
        ctypes.c_void_p,
    ]
    fn.restype = None
    fn(
        ctypes.c_void_p(key_linear_work.data_ptr()),
        ctypes.c_void_p(value_linear_work.data_ptr()),
        ctypes.c_void_p(slots64.data_ptr()),
        ctypes.c_void_p(key_cache.data_ptr()),
        ctypes.c_void_p(value_cache.data_ptr()),
        ctypes.c_longlong(key_linear.shape[0]),
        ctypes.c_longlong(key_linear.shape[1]),
        ctypes.c_longlong(key_cache.shape[1]),
        ctypes.c_longlong(key_cache.shape[2]),
        ctypes.c_longlong(key_cache.shape[3]),
        ctypes.c_longlong(key_cache.shape[4]),
        ctypes.c_void_p(0),
    )
    return True


def _try_native_kv_paged_to_merged_linear(
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    slots_long: torch.Tensor,
    tcount: int,
    nh: int,
    hd: int,
    bs: int,
    *,
    fill_skipped: Literal["zeros", "preserve"],
    out: torch.Tensor | None,
) -> torch.Tensor | None:
    if not _native_kv_transfer_requested():
        return None
    if fill_skipped != "zeros":
        return None
    if not (
        _is_musa_native_kv_dtype_tensor(key_cache)
        and _is_musa_native_kv_dtype_tensor(value_cache)
        and key_cache.dtype == value_cache.dtype
    ):
        return None
    slots64 = slots_long.contiguous()
    key_work = key_cache if key_cache.is_contiguous() else key_cache.contiguous()
    value_work = value_cache if value_cache.is_contiguous() else value_cache.contiguous()
    out_work = out if out is not None else torch.empty((tcount, 2, nh, hd), dtype=key_cache.dtype, device=key_cache.device)
    if (
        not _is_musa_native_kv_dtype_tensor(out_work)
        or out_work.dtype != key_cache.dtype
        or not out_work.is_contiguous()
    ):
        return None
    try:
        lib = load_native_kv_transfer()
    except Exception:
        return None
    fn = _get_native_kv_function(
        lib, "musa_aiter_kv_paged_to_merged_linear", key_cache.dtype
    )
    if fn is None:
        return None
    fn.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_longlong,
        ctypes.c_longlong,
        ctypes.c_longlong,
        ctypes.c_longlong,
        ctypes.c_longlong,
        ctypes.c_void_p,
    ]
    fn.restype = None
    fn(
        ctypes.c_void_p(key_work.data_ptr()),
        ctypes.c_void_p(value_work.data_ptr()),
        ctypes.c_void_p(slots64.data_ptr()),
        ctypes.c_void_p(out_work.data_ptr()),
        ctypes.c_longlong(tcount),
        ctypes.c_longlong(key_cache.shape[0]),
        ctypes.c_longlong(nh),
        ctypes.c_longlong(hd),
        ctypes.c_longlong(bs),
        _current_musa_stream_ptr(),
    )
    return out_work


def _try_native_kv_merged_linear_to_paged(
    merged_linear: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    slots_long: torch.Tensor,
) -> bool:
    if not _native_kv_transfer_requested():
        return False
    if not (
        _is_musa_native_kv_dtype_tensor(merged_linear)
        and _is_musa_native_kv_dtype_tensor(key_cache)
        and _is_musa_native_kv_dtype_tensor(value_cache)
        and merged_linear.dtype == key_cache.dtype == value_cache.dtype
    ):
        return False
    if not key_cache.is_contiguous() or not value_cache.is_contiguous():
        return False
    merged_work = merged_linear if merged_linear.is_contiguous() else merged_linear.contiguous()
    slots64 = slots_long.contiguous()
    try:
        lib = load_native_kv_transfer()
    except Exception:
        return False
    fn = _get_native_kv_function(
        lib, "musa_aiter_kv_merged_linear_to_paged", merged_linear.dtype
    )
    if fn is None:
        return False
    fn.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_longlong,
        ctypes.c_longlong,
        ctypes.c_longlong,
        ctypes.c_longlong,
        ctypes.c_longlong,
        ctypes.c_void_p,
    ]
    fn.restype = None
    fn(
        ctypes.c_void_p(merged_work.data_ptr()),
        ctypes.c_void_p(slots64.data_ptr()),
        ctypes.c_void_p(key_cache.data_ptr()),
        ctypes.c_void_p(value_cache.data_ptr()),
        ctypes.c_longlong(merged_linear.shape[0]),
        ctypes.c_longlong(key_cache.shape[0]),
        ctypes.c_longlong(key_cache.shape[1]),
        ctypes.c_longlong(key_cache.shape[2]),
        ctypes.c_longlong(key_cache.shape[3]),
        _current_musa_stream_ptr(),
    )
    return True


def _try_native_kv_paged_to_merged_linear_multi_layer(
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    slots_long: torch.Tensor,
    tcount: int,
    nl: int,
    nh: int,
    hd: int,
    bs: int,
    *,
    fill_skipped: Literal["zeros", "preserve"],
    out: torch.Tensor | None,
) -> torch.Tensor | None:
    if not _native_kv_transfer_requested():
        return None
    if fill_skipped != "zeros":
        return None
    if not (
        _is_musa_native_kv_dtype_tensor(key_cache)
        and _is_musa_native_kv_dtype_tensor(value_cache)
        and key_cache.dtype == value_cache.dtype
    ):
        return None
    slots64 = slots_long.contiguous()
    key_work = key_cache if key_cache.is_contiguous() else key_cache.contiguous()
    value_work = value_cache if value_cache.is_contiguous() else value_cache.contiguous()
    out_work = out if out is not None else torch.empty((nl, tcount, 2, nh, hd), dtype=key_cache.dtype, device=key_cache.device)
    if (
        not _is_musa_native_kv_dtype_tensor(out_work)
        or out_work.dtype != key_cache.dtype
        or not out_work.is_contiguous()
    ):
        return None
    try:
        lib = load_native_kv_transfer()
    except Exception:
        return None
    fn = _get_native_kv_function(
        lib, "musa_aiter_kv_paged_to_merged_linear_multi_layer", key_cache.dtype
    )
    if fn is None:
        return None
    fn.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_longlong,
        ctypes.c_longlong,
        ctypes.c_longlong,
        ctypes.c_longlong,
        ctypes.c_longlong,
        ctypes.c_longlong,
        ctypes.c_void_p,
    ]
    fn.restype = None
    fn(
        ctypes.c_void_p(key_work.data_ptr()),
        ctypes.c_void_p(value_work.data_ptr()),
        ctypes.c_void_p(slots64.data_ptr()),
        ctypes.c_void_p(out_work.data_ptr()),
        ctypes.c_longlong(nl),
        ctypes.c_longlong(tcount),
        ctypes.c_longlong(key_cache.shape[1]),
        ctypes.c_longlong(nh),
        ctypes.c_longlong(hd),
        ctypes.c_longlong(bs),
        _current_musa_stream_ptr(),
    )
    return out_work


def _try_native_kv_merged_linear_to_paged_multi_layer(
    merged_linear: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    slots_long: torch.Tensor,
) -> bool:
    if not _native_kv_transfer_requested():
        return False
    if not (
        _is_musa_native_kv_dtype_tensor(merged_linear)
        and _is_musa_native_kv_dtype_tensor(key_cache)
        and _is_musa_native_kv_dtype_tensor(value_cache)
        and merged_linear.dtype == key_cache.dtype == value_cache.dtype
    ):
        return False
    if not key_cache.is_contiguous() or not value_cache.is_contiguous():
        return False
    merged_work = merged_linear if merged_linear.is_contiguous() else merged_linear.contiguous()
    slots64 = slots_long.contiguous()
    try:
        lib = load_native_kv_transfer()
    except Exception:
        return False
    fn = _get_native_kv_function(
        lib, "musa_aiter_kv_merged_linear_to_paged_multi_layer", merged_linear.dtype
    )
    if fn is None:
        return False
    fn.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_longlong,
        ctypes.c_longlong,
        ctypes.c_longlong,
        ctypes.c_longlong,
        ctypes.c_longlong,
        ctypes.c_longlong,
        ctypes.c_void_p,
    ]
    fn.restype = None
    fn(
        ctypes.c_void_p(merged_work.data_ptr()),
        ctypes.c_void_p(slots64.data_ptr()),
        ctypes.c_void_p(key_cache.data_ptr()),
        ctypes.c_void_p(value_cache.data_ptr()),
        ctypes.c_longlong(merged_linear.shape[0]),
        ctypes.c_longlong(merged_linear.shape[1]),
        ctypes.c_longlong(key_cache.shape[1]),
        ctypes.c_longlong(key_cache.shape[2]),
        ctypes.c_longlong(key_cache.shape[3]),
        ctypes.c_longlong(key_cache.shape[4]),
        _current_musa_stream_ptr(),
    )
    return True


def _try_native_kv_paged_to_mla_linear(
    kv_cache: torch.Tensor,
    k_pe_cache: torch.Tensor,
    slots_long: torch.Tensor,
    tcount: int,
    nb: int,
    bs: int,
    kv_dim: int,
    pe_dim: int,
    *,
    out_kv: torch.Tensor | None,
    out_k_pe: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    if not _native_kv_transfer_requested():
        return None
    if not (
        _is_musa_native_kv_dtype_tensor(kv_cache)
        and _is_musa_native_kv_dtype_tensor(k_pe_cache)
        and kv_cache.dtype == k_pe_cache.dtype
    ):
        return None
    kv_work = kv_cache if kv_cache.is_contiguous() else kv_cache.contiguous()
    k_pe_work = k_pe_cache if k_pe_cache.is_contiguous() else k_pe_cache.contiguous()
    slots64 = slots_long.contiguous()
    out_kv_work = out_kv if out_kv is not None else torch.empty((tcount, kv_dim), dtype=kv_cache.dtype, device=kv_cache.device)
    out_k_pe_work = out_k_pe if out_k_pe is not None else torch.empty((tcount, pe_dim), dtype=k_pe_cache.dtype, device=k_pe_cache.device)
    if not (
        _is_musa_native_kv_dtype_tensor(out_kv_work)
        and _is_musa_native_kv_dtype_tensor(out_k_pe_work)
        and out_kv_work.dtype == out_k_pe_work.dtype == kv_cache.dtype
    ):
        return None
    if not out_kv_work.is_contiguous() or not out_k_pe_work.is_contiguous():
        return None
    try:
        lib = load_native_kv_transfer()
    except Exception:
        return None
    fn = _get_native_kv_function(
        lib, "musa_aiter_kv_paged_to_mla_linear", kv_cache.dtype
    )
    if fn is None:
        return None
    fn.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_longlong,
        ctypes.c_longlong,
        ctypes.c_longlong,
        ctypes.c_longlong,
        ctypes.c_longlong,
        ctypes.c_void_p,
    ]
    fn.restype = None
    fn(
        ctypes.c_void_p(kv_work.data_ptr()),
        ctypes.c_void_p(k_pe_work.data_ptr()),
        ctypes.c_void_p(slots64.data_ptr()),
        ctypes.c_void_p(out_kv_work.data_ptr()),
        ctypes.c_void_p(out_k_pe_work.data_ptr()),
        ctypes.c_longlong(tcount),
        ctypes.c_longlong(nb),
        ctypes.c_longlong(bs),
        ctypes.c_longlong(kv_dim),
        ctypes.c_longlong(pe_dim),
        _current_musa_stream_ptr(),
    )
    return out_kv_work, out_k_pe_work


def _try_native_kv_mla_linear_to_paged(
    kv_linear: torch.Tensor,
    k_pe_linear: torch.Tensor,
    kv_cache: torch.Tensor,
    k_pe_cache: torch.Tensor,
    slots_long: torch.Tensor,
) -> bool:
    if not _native_kv_transfer_requested():
        return False
    if not (
        _is_musa_native_kv_dtype_tensor(kv_linear)
        and _is_musa_native_kv_dtype_tensor(k_pe_linear)
        and _is_musa_native_kv_dtype_tensor(kv_cache)
        and _is_musa_native_kv_dtype_tensor(k_pe_cache)
        and kv_linear.dtype == k_pe_linear.dtype == kv_cache.dtype == k_pe_cache.dtype
    ):
        return False
    if not kv_cache.is_contiguous() or not k_pe_cache.is_contiguous():
        return False
    kv_linear_work = kv_linear if kv_linear.is_contiguous() else kv_linear.contiguous()
    k_pe_linear_work = k_pe_linear if k_pe_linear.is_contiguous() else k_pe_linear.contiguous()
    slots64 = slots_long.contiguous()
    try:
        lib = load_native_kv_transfer()
    except Exception:
        return False
    fn = _get_native_kv_function(
        lib, "musa_aiter_kv_mla_linear_to_paged", kv_linear.dtype
    )
    if fn is None:
        return False
    fn.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_longlong,
        ctypes.c_longlong,
        ctypes.c_longlong,
        ctypes.c_longlong,
        ctypes.c_longlong,
        ctypes.c_void_p,
    ]
    fn.restype = None
    fn(
        ctypes.c_void_p(kv_linear_work.data_ptr()),
        ctypes.c_void_p(k_pe_linear_work.data_ptr()),
        ctypes.c_void_p(slots64.data_ptr()),
        ctypes.c_void_p(kv_cache.data_ptr()),
        ctypes.c_void_p(k_pe_cache.data_ptr()),
        ctypes.c_longlong(kv_linear.shape[0]),
        ctypes.c_longlong(kv_cache.shape[0]),
        ctypes.c_longlong(kv_cache.shape[1]),
        ctypes.c_longlong(kv_cache.shape[2]),
        ctypes.c_longlong(k_pe_cache.shape[2]),
        _current_musa_stream_ptr(),
    )
    return True


def _try_native_kv_paged_to_mla_linear_multi_layer(
    kv_cache: torch.Tensor,
    k_pe_cache: torch.Tensor,
    slots_long: torch.Tensor,
    nl: int,
    kv_dim: int,
    pe_dim: int,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    if not _native_kv_transfer_requested():
        return None
    if not (
        _is_musa_native_kv_dtype_tensor(kv_cache)
        and _is_musa_native_kv_dtype_tensor(k_pe_cache)
        and kv_cache.dtype == k_pe_cache.dtype
        and kv_cache.is_contiguous()
        and k_pe_cache.is_contiguous()
    ):
        return None
    slots64 = slots_long.contiguous()
    kv_out = torch.empty(
        (nl, slots64.numel(), kv_dim),
        dtype=kv_cache.dtype,
        device=kv_cache.device,
    )
    pe_out = torch.empty(
        (nl, slots64.numel(), pe_dim),
        dtype=k_pe_cache.dtype,
        device=k_pe_cache.device,
    )
    try:
        from musa_aiter.ops.lmcache_kv_transfer import (
            _try_native_mla_to_buffer,
            is_native_lmcache_kv_transfer_available,
        )

        if not is_native_lmcache_kv_transfer_available():
            return None
        if not _try_native_mla_to_buffer(kv_cache, kv_out, slots64):
            return None
        if not _try_native_mla_to_buffer(k_pe_cache, pe_out, slots64):
            return None
    except Exception:
        return None
    return kv_out, pe_out


def _try_native_kv_mla_linear_to_paged_multi_layer(
    kv_linear: torch.Tensor,
    k_pe_linear: torch.Tensor,
    kv_cache: torch.Tensor,
    k_pe_cache: torch.Tensor,
    slots_long: torch.Tensor,
) -> bool:
    if not _native_kv_transfer_requested():
        return False
    if not (
        _is_musa_native_kv_dtype_tensor(kv_linear)
        and _is_musa_native_kv_dtype_tensor(k_pe_linear)
        and _is_musa_native_kv_dtype_tensor(kv_cache)
        and _is_musa_native_kv_dtype_tensor(k_pe_cache)
        and kv_linear.dtype == k_pe_linear.dtype == kv_cache.dtype == k_pe_cache.dtype
        and kv_linear.is_contiguous()
        and k_pe_linear.is_contiguous()
        and kv_cache.is_contiguous()
        and k_pe_cache.is_contiguous()
    ):
        return False
    slots64 = slots_long.contiguous()
    try:
        from musa_aiter.ops.lmcache_kv_transfer import (
            _try_native_mla_to_paged,
            is_native_lmcache_kv_transfer_available,
        )

        if not is_native_lmcache_kv_transfer_available():
            return False
        if not _try_native_mla_to_paged(kv_linear, kv_cache, slots64):
            return False
        if not _try_native_mla_to_paged(k_pe_linear, k_pe_cache, slots64):
            return False
    except Exception:
        return False
    return True


def _try_native_kv_cachegen_encode(
    tensor: torch.Tensor,
    min_val: torch.Tensor,
    max_val: torch.Tensor,
    num_bins: int,
) -> torch.Tensor | None:
    if not _native_kv_transfer_requested() or not _is_musa_float32_tensor(tensor):
        return None
    work = tensor if tensor.is_contiguous() else tensor.contiguous()
    encoded = torch.empty_like(work, dtype=torch.uint8)
    min_item = float(min_val.item())
    max_item = float(max_val.item())
    scale = float((num_bins - 1) / (max_item - min_item))
    try:
        lib = load_native_kv_transfer()
        if not has_symbol(lib, "musa_aiter_kv_cachegen_encode_float"):
            return None
    except Exception:
        return None
    fn = lib.handle.musa_aiter_kv_cachegen_encode_float
    fn.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_float, ctypes.c_float, ctypes.c_longlong, ctypes.c_longlong, ctypes.c_void_p]
    fn.restype = None
    fn(
        ctypes.c_void_p(work.data_ptr()),
        ctypes.c_void_p(encoded.data_ptr()),
        ctypes.c_float(min_item),
        ctypes.c_float(scale),
        ctypes.c_longlong(work.numel()),
        ctypes.c_longlong(num_bins),
        ctypes.c_void_p(0),
    )
    return encoded.reshape(tensor.shape)


def _try_native_kv_cachegen_decode(encoded: torch.Tensor, codebook: torch.Tensor, out_dtype: torch.dtype | None) -> torch.Tensor | None:
    if not _native_kv_transfer_requested():
        return None
    if encoded.dtype != torch.uint8 or not _is_musa_float32_tensor(codebook):
        return None
    if out_dtype not in (None, torch.float32):
        return None
    encoded_work = encoded if encoded.is_contiguous() else encoded.contiguous()
    codebook_work = codebook if codebook.is_contiguous() else codebook.contiguous()
    out = torch.empty(encoded_work.shape, dtype=torch.float32, device=encoded.device)
    try:
        lib = load_native_kv_transfer()
        if not has_symbol(lib, "musa_aiter_kv_cachegen_decode_float"):
            return None
    except Exception:
        return None
    fn = lib.handle.musa_aiter_kv_cachegen_decode_float
    fn.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_longlong, ctypes.c_void_p]
    fn.restype = None
    fn(
        ctypes.c_void_p(encoded_work.data_ptr()),
        ctypes.c_void_p(codebook_work.data_ptr()),
        ctypes.c_void_p(out.data_ptr()),
        ctypes.c_longlong(encoded_work.numel()),
        ctypes.c_void_p(0),
    )
    return out.reshape(encoded.shape)


def kv_cachegen_encode(
    tensor: torch.Tensor,
    *,
    num_bins: int = 256,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Encode a KV tensor into uint8 bins plus dequant codebook and empirical CDF."""
    device, dtype = tensor_device_dtype(tensor)
    ensure_reference_allowed("kv_cachegen_encode")
    if num_bins < 2 or num_bins > 256:
        raise ValueError("kv_cachegen_encode: num_bins must be in [2, 256]")
    if not tensor.is_floating_point():
        raise ValueError("kv_cachegen_encode: tensor must use a floating dtype")
    work = tensor.float()
    if work.numel() == 0:
        codebook = torch.zeros(num_bins, dtype=torch.float32, device=tensor.device)
        cdf = torch.zeros(num_bins + 1, dtype=torch.float32, device=tensor.device)
        return torch.empty_like(tensor, dtype=torch.uint8), codebook, cdf
    min_val = work.min()
    max_val = work.max()
    if bool((max_val == min_val).item()):
        encoded = torch.zeros_like(tensor, dtype=torch.uint8)
        codebook = torch.full((num_bins,), min_val.item(), dtype=torch.float32, device=tensor.device)
        cdf = torch.zeros(num_bins + 1, dtype=torch.float32, device=tensor.device)
        cdf[1:] = 1.0
        return encoded, codebook, cdf
    native_encoded = _try_native_kv_cachegen_encode(tensor, min_val, max_val, num_bins)
    if native_encoded is not None:
        emit(
            DispatchRecord(
                "kv_cachegen_encode",
                "ctypes-musa-native",
                device,
                str(dtype),
                "Phase PKV native CacheGen-style uniform-bin encode with empirical CDF",
            )
        )
        codebook = torch.linspace(min_val.item(), max_val.item(), num_bins, dtype=torch.float32, device=tensor.device)
        counts = torch.bincount(native_encoded.reshape(-1).to(device="cpu", dtype=torch.long), minlength=num_bins).to(device=tensor.device, dtype=torch.float32)
        cdf = torch.empty(num_bins + 1, dtype=torch.float32, device=tensor.device)
        cdf[0] = 0.0
        cdf[1:] = torch.cumsum(counts, dim=0) / float(native_encoded.numel())
        return native_encoded, codebook, cdf

    emit(
        DispatchRecord(
            "kv_cachegen_encode",
            "torch-reference",
            device,
            str(dtype),
            "Phase PKV CacheGen-style uniform-bin encode with empirical CDF",
        )
    )
    emit_vendor_planning_note("kv_cachegen_encode")
    scale = (num_bins - 1) / (max_val - min_val)
    encoded_long = torch.round((work - min_val) * scale).clamp_(0, num_bins - 1).to(torch.long)
    encoded = encoded_long.to(torch.uint8)
    codebook = torch.linspace(min_val.item(), max_val.item(), num_bins, dtype=torch.float32, device=tensor.device)
    counts = torch.bincount(encoded_long.reshape(-1).to(device="cpu"), minlength=num_bins).to(device=tensor.device, dtype=torch.float32)
    cdf = torch.empty(num_bins + 1, dtype=torch.float32, device=tensor.device)
    cdf[0] = 0.0
    cdf[1:] = torch.cumsum(counts, dim=0) / float(encoded_long.numel())
    return encoded, codebook, cdf


def kv_cachegen_decode(
    encoded: torch.Tensor,
    codebook: torch.Tensor,
    *,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Decode uint8 CacheGen-style bins using a dequant codebook."""
    device, encoded_dtype = tensor_device_dtype(encoded)
    ensure_reference_allowed("kv_cachegen_decode")
    if encoded.dtype != torch.uint8:
        raise ValueError("kv_cachegen_decode: encoded must be uint8")
    if codebook.dim() != 1 or codebook.numel() < 2 or codebook.numel() > 256:
        raise ValueError("kv_cachegen_decode: codebook must be 1-D with 2..256 bins")
    if codebook.device != encoded.device:
        raise ValueError("kv_cachegen_decode: encoded and codebook must be on same device")
    if not codebook.is_floating_point():
        raise ValueError("kv_cachegen_decode: codebook must use a floating dtype")
    if encoded.numel() > 0 and int(encoded.max().item()) >= int(codebook.numel()):
        raise ValueError("kv_cachegen_decode: encoded bin exceeds codebook size")
    native_out = _try_native_kv_cachegen_decode(encoded, codebook, dtype)
    if native_out is not None:
        emit(
            DispatchRecord(
                "kv_cachegen_decode",
                "ctypes-musa-native",
                device,
                str(encoded_dtype),
                "Phase PKV native CacheGen-style codebook decode",
            )
        )
        return native_out
    emit(
        DispatchRecord(
            "kv_cachegen_decode",
            "torch-reference",
            device,
            str(encoded_dtype),
            "Phase PKV CacheGen-style codebook decode",
        )
    )
    emit_vendor_planning_note("kv_cachegen_decode")
    out = codebook[encoded.long()]
    return out if dtype is None else out.to(dtype=dtype)


def _try_native_kv_triton_indexed_to_dense(
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    kv_indptr: torch.Tensor,
    kv_indices: torch.Tensor,
    out_k: torch.Tensor,
    out_v: torch.Tensor,
    batch: int,
    dense_len: int,
    nh: int,
    hd: int,
) -> bool:
    if not _native_kv_transfer_requested():
        return False
    if not (_is_musa_native_kv_dtype_tensor(key_cache) and _is_musa_native_kv_dtype_tensor(value_cache)):
        return False
    if key_cache.dtype != value_cache.dtype:
        return False
    if not (_is_musa_native_kv_dtype_tensor(out_k) and _is_musa_native_kv_dtype_tensor(out_v)):
        return False
    if out_k.dtype != key_cache.dtype or out_v.dtype != value_cache.dtype:
        return False
    if not out_k.is_contiguous() or not out_v.is_contiguous():
        return False
    if str(kv_indptr.device).split(":")[0] != "musa" or str(kv_indices.device).split(":")[0] != "musa":
        return False
    key_work = key_cache if key_cache.is_contiguous() else key_cache.contiguous()
    value_work = value_cache if value_cache.is_contiguous() else value_cache.contiguous()
    indptr64 = kv_indptr.to(dtype=torch.long).contiguous()
    indices64 = kv_indices.to(dtype=torch.long).contiguous()
    try:
        lib = load_native_kv_transfer()
        fn = _get_native_kv_function(lib, "musa_aiter_kv_triton_indexed_to_dense", key_cache.dtype)
        if fn is None:
            return False
    except Exception:
        return False
    fn.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_longlong,
        ctypes.c_longlong,
        ctypes.c_longlong,
        ctypes.c_longlong,
        ctypes.c_longlong,
        ctypes.c_longlong,
        ctypes.c_void_p,
    ]
    fn.restype = None
    fn(
        ctypes.c_void_p(key_work.data_ptr()),
        ctypes.c_void_p(value_work.data_ptr()),
        ctypes.c_void_p(indptr64.data_ptr()),
        ctypes.c_void_p(indices64.data_ptr()),
        ctypes.c_void_p(out_k.data_ptr()),
        ctypes.c_void_p(out_v.data_ptr()),
        ctypes.c_longlong(batch),
        ctypes.c_longlong(dense_len),
        ctypes.c_longlong(indices64.numel()),
        ctypes.c_longlong(key_cache.shape[0]),
        ctypes.c_longlong(nh),
        ctypes.c_longlong(hd),
        _current_musa_stream_ptr(),
    )
    return True


def kv_triton_indexed_to_dense(
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    kv_indptr: torch.Tensor,
    kv_indices: torch.Tensor,
    *,
    max_seq_len: int,
    out_k: torch.Tensor | None = None,
    out_v: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Load SGLang Triton decode flat KV indices into dense ``[B, S, kv_heads, head_dim]`` tensors."""
    device, dtype = tensor_device_dtype(key_cache)
    ensure_reference_allowed("kv_triton_indexed_to_dense")
    if key_cache.dim() != 3 or value_cache.dim() != 3:
        raise ValueError("kv_triton_indexed_to_dense: key/value cache must be [tokens, kv_heads, head_dim]")
    if key_cache.shape != value_cache.shape or key_cache.dtype != value_cache.dtype or key_cache.device != value_cache.device:
        raise ValueError("kv_triton_indexed_to_dense: key/value cache shape, dtype, and device must match")
    if kv_indptr.dim() != 1 or kv_indptr.numel() < 2:
        raise ValueError("kv_triton_indexed_to_dense: kv_indptr must be [batch + 1]")
    if kv_indices.dim() != 1:
        raise ValueError("kv_triton_indexed_to_dense: kv_indices must be 1-D")
    if kv_indptr.device != key_cache.device or kv_indices.device != key_cache.device:
        raise ValueError("kv_triton_indexed_to_dense: all tensors must be on the same device")
    if kv_indptr.dtype not in (torch.int16, torch.int32, torch.int64) or kv_indices.dtype not in (torch.int16, torch.int32, torch.int64):
        raise ValueError("kv_triton_indexed_to_dense: indices must use integer dtype")
    batch = int(kv_indptr.shape[0] - 1)
    dense_len = int(max_seq_len)
    if dense_len < 0:
        raise ValueError("kv_triton_indexed_to_dense: max_seq_len must be non-negative")
    _, nh, hd = map(int, key_cache.shape)
    shape = (batch, dense_len, nh, hd)
    if out_k is None:
        out_k = torch.empty(shape, dtype=key_cache.dtype, device=key_cache.device)
    elif out_k.shape != shape or out_k.dtype != key_cache.dtype or out_k.device != key_cache.device or not out_k.is_contiguous():
        raise ValueError("kv_triton_indexed_to_dense: out_k must match [B, S, kv_heads, head_dim] dtype/device and be contiguous")
    if out_v is None:
        out_v = torch.empty(shape, dtype=value_cache.dtype, device=value_cache.device)
    elif out_v.shape != shape or out_v.dtype != value_cache.dtype or out_v.device != value_cache.device or not out_v.is_contiguous():
        raise ValueError("kv_triton_indexed_to_dense: out_v must match [B, S, kv_heads, head_dim] dtype/device and be contiguous")

    if _try_native_kv_triton_indexed_to_dense(key_cache, value_cache, kv_indptr, kv_indices, out_k, out_v, batch, dense_len, nh, hd):
        emit(
            DispatchRecord(
                "kv_triton_indexed_to_dense",
                "ctypes-musa-native",
                device,
                str(dtype),
                "SGLang Triton-indexed graph-capture-safe dense KV preload",
            )
        )
        return out_k, out_v

    emit(
        DispatchRecord(
            "kv_triton_indexed_to_dense",
            "torch-reference",
            device,
            str(dtype),
            "SGLang Triton-indexed dense KV preload reference",
        )
    )
    emit_vendor_planning_note("kv_triton_indexed_to_dense")
    out_k.zero_()
    out_v.zero_()
    indptr_cpu = kv_indptr.to(device="cpu", dtype=torch.long)
    indices_long = kv_indices.to(dtype=torch.long)
    for batch_idx in range(batch):
        start = int(indptr_cpu[batch_idx].item())
        end = int(indptr_cpu[batch_idx + 1].item())
        copy_len = min(max(end - start, 0), dense_len)
        if copy_len <= 0:
            continue
        slots = indices_long[start : start + copy_len]
        out_k[batch_idx, :copy_len].copy_(key_cache.index_select(0, slots))
        out_v[batch_idx, :copy_len].copy_(value_cache.index_select(0, slots))
    return out_k, out_v


def _try_native_kv_load_and_reshape_flash(
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    out_k: torch.Tensor,
    out_v: torch.Tensor,
    batch: int,
    dense_len: int,
    nh: int,
    hd: int,
    bs: int,
) -> bool:
    if not _native_kv_transfer_requested():
        return False
    if not (_is_musa_native_kv_dtype_tensor(key_cache) and _is_musa_native_kv_dtype_tensor(value_cache)):
        return False
    if key_cache.dtype != value_cache.dtype:
        return False
    if not (_is_musa_native_kv_dtype_tensor(out_k) and _is_musa_native_kv_dtype_tensor(out_v)):
        return False
    if out_k.dtype != key_cache.dtype or out_v.dtype != value_cache.dtype:
        return False
    if not out_k.is_contiguous() or not out_v.is_contiguous():
        return False
    if str(block_tables.device).split(":")[0] != "musa" or str(seq_lens.device).split(":")[0] != "musa":
        return False
    key_work = key_cache if key_cache.is_contiguous() else key_cache.contiguous()
    value_work = value_cache if value_cache.is_contiguous() else value_cache.contiguous()
    block_tables64 = block_tables.to(dtype=torch.long).contiguous()
    seq_lens64 = seq_lens.to(dtype=torch.long).contiguous()
    try:
        lib = load_native_kv_transfer()
        fn = _get_native_kv_function(lib, "musa_aiter_kv_load_and_reshape_flash", key_cache.dtype)
        if fn is None:
            return False
    except Exception:
        return False
    fn.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_longlong,
        ctypes.c_longlong,
        ctypes.c_longlong,
        ctypes.c_longlong,
        ctypes.c_longlong,
        ctypes.c_longlong,
        ctypes.c_longlong,
        ctypes.c_void_p,
    ]
    fn.restype = None
    fn(
        ctypes.c_void_p(key_work.data_ptr()),
        ctypes.c_void_p(value_work.data_ptr()),
        ctypes.c_void_p(block_tables64.data_ptr()),
        ctypes.c_void_p(seq_lens64.data_ptr()),
        ctypes.c_void_p(out_k.data_ptr()),
        ctypes.c_void_p(out_v.data_ptr()),
        ctypes.c_longlong(batch),
        ctypes.c_longlong(dense_len),
        ctypes.c_longlong(block_tables.shape[1]),
        ctypes.c_longlong(key_cache.shape[0]),
        ctypes.c_longlong(nh),
        ctypes.c_longlong(hd),
        ctypes.c_longlong(bs),
        ctypes.c_void_p(0),
    )
    return True


def _try_native_kv_reshape_and_cache_back_flash(
    dense_k: torch.Tensor,
    dense_v: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    batch: int,
    dense_len: int,
    max_blocks: int,
    num_blocks: int,
    nh: int,
    hd: int,
    bs: int,
) -> bool:
    if not _native_kv_transfer_requested():
        return False
    if not (
        _is_musa_native_kv_dtype_tensor(dense_k)
        and _is_musa_native_kv_dtype_tensor(dense_v)
        and _is_musa_native_kv_dtype_tensor(key_cache)
        and _is_musa_native_kv_dtype_tensor(value_cache)
    ):
        return False
    if dense_k.dtype != dense_v.dtype or dense_k.dtype != key_cache.dtype or dense_k.dtype != value_cache.dtype:
        return False
    if not (
        str(block_tables.device).split(":")[0] == "musa"
        and str(seq_lens.device).split(":")[0] == "musa"
    ):
        return False
    if not key_cache.is_contiguous() or not value_cache.is_contiguous():
        return False
    dense_k_work = dense_k if dense_k.is_contiguous() else dense_k.contiguous()
    dense_v_work = dense_v if dense_v.is_contiguous() else dense_v.contiguous()
    block_tables64 = block_tables.to(dtype=torch.long).contiguous()
    seq_lens64 = seq_lens.to(dtype=torch.long).contiguous()
    try:
        lib = load_native_kv_transfer()
        fn = _get_native_kv_function(
            lib,
            "musa_aiter_kv_reshape_and_cache_back_flash",
            dense_k.dtype,
        )
        if fn is None:
            return False
    except Exception:
        return False
    fn.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_longlong,
        ctypes.c_longlong,
        ctypes.c_longlong,
        ctypes.c_longlong,
        ctypes.c_longlong,
        ctypes.c_longlong,
        ctypes.c_longlong,
        ctypes.c_void_p,
    ]
    fn.restype = None
    fn(
        ctypes.c_void_p(dense_k_work.data_ptr()),
        ctypes.c_void_p(dense_v_work.data_ptr()),
        ctypes.c_void_p(block_tables64.data_ptr()),
        ctypes.c_void_p(seq_lens64.data_ptr()),
        ctypes.c_void_p(key_cache.data_ptr()),
        ctypes.c_void_p(value_cache.data_ptr()),
        ctypes.c_longlong(batch),
        ctypes.c_longlong(dense_len),
        ctypes.c_longlong(max_blocks),
        ctypes.c_longlong(num_blocks),
        ctypes.c_longlong(nh),
        ctypes.c_longlong(hd),
        ctypes.c_longlong(bs),
        _current_musa_stream_ptr(),
    )
    return True


def kv_load_and_reshape_flash(
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    *,
    max_seq_len: int | None = None,
    out_k: torch.Tensor | None = None,
    out_v: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Load paged KV blocks into dense Flash-style ``[B, S, kv_heads, head_dim]`` tensors."""
    device, dtype = tensor_device_dtype(key_cache)
    ensure_reference_allowed("kv_load_and_reshape_flash")
    nb, nh, hd, bs = _validate_paged_shapes(key_cache, value_cache, name="kv_load_and_reshape_flash")
    if block_tables.dim() != 2:
        raise ValueError("kv_load_and_reshape_flash: block_tables must be [batch, max_blocks]")
    if seq_lens.dim() != 1 or seq_lens.shape[0] != block_tables.shape[0]:
        raise ValueError("kv_load_and_reshape_flash: seq_lens must be [batch]")
    if key_cache.device != value_cache.device or block_tables.device != key_cache.device or seq_lens.device != key_cache.device:
        raise ValueError("kv_load_and_reshape_flash: all tensors must be on the same device")
    if block_tables.dtype not in (torch.int16, torch.int32, torch.int64):
        raise ValueError("kv_load_and_reshape_flash: block_tables must use an integer dtype")
    if seq_lens.dtype not in (torch.int16, torch.int32, torch.int64):
        raise ValueError("kv_load_and_reshape_flash: seq_lens must use an integer dtype")

    batch = int(block_tables.shape[0])
    seq_lens_cpu = seq_lens.to(device="cpu", dtype=torch.long)
    if seq_lens_cpu.numel() > 0 and int(seq_lens_cpu.min().item()) < 0:
        raise ValueError("kv_load_and_reshape_flash: seq_lens must be non-negative")
    inferred_max = int(seq_lens_cpu.max().item()) if seq_lens_cpu.numel() > 0 else 0
    dense_len = inferred_max if max_seq_len is None else int(max_seq_len)
    if dense_len < inferred_max:
        raise ValueError("kv_load_and_reshape_flash: max_seq_len is smaller than seq_lens max")
    needed_blocks_max = (dense_len + bs - 1) // bs
    if needed_blocks_max > int(block_tables.shape[1]):
        raise ValueError("kv_load_and_reshape_flash: block_tables does not contain enough columns")

    shape = (batch, dense_len, nh, hd)
    if out_k is None:
        out_k = torch.zeros(shape, dtype=key_cache.dtype, device=key_cache.device)
    elif out_k.shape != shape or out_k.dtype != key_cache.dtype or out_k.device != key_cache.device:
        raise ValueError("kv_load_and_reshape_flash: out_k must match [B, S, kv_heads, head_dim] dtype/device")
    else:
        out_k.zero_()
    if out_v is None:
        out_v = torch.zeros(shape, dtype=value_cache.dtype, device=value_cache.device)
    elif out_v.shape != shape or out_v.dtype != value_cache.dtype or out_v.device != value_cache.device:
        raise ValueError("kv_load_and_reshape_flash: out_v must match [B, S, kv_heads, head_dim] dtype/device")
    else:
        out_v.zero_()

    block_tables_cpu = block_tables.to(device="cpu", dtype=torch.long)
    for seq_idx in range(batch):
        seq_len = int(seq_lens_cpu[seq_idx].item())
        needed_blocks = (seq_len + bs - 1) // bs
        for logical_block in range(needed_blocks):
            block_id = int(block_tables_cpu[seq_idx, logical_block].item())
            if block_id < 0 or block_id >= nb:
                raise ValueError(f"kv_load_and_reshape_flash: invalid block id {block_id}")

    if _try_native_kv_load_and_reshape_flash(key_cache, value_cache, block_tables, seq_lens, out_k, out_v, batch, dense_len, nh, hd, bs):
        emit(
            DispatchRecord(
                "kv_load_and_reshape_flash",
                "ctypes-musa-native",
                device,
                str(dtype),
                "Phase PKV native Flash-style paged KV load/reshape",
            )
        )
        return out_k, out_v

    emit(
        DispatchRecord(
            "kv_load_and_reshape_flash",
            "torch-reference",
            device,
            str(dtype),
            "Phase PKV Flash-style paged KV load/reshape façade",
        )
    )
    emit_vendor_planning_note("kv_load_and_reshape_flash")

    for seq_idx in range(batch):
        seq_len = int(seq_lens_cpu[seq_idx].item())
        needed_blocks = (seq_len + bs - 1) // bs
        for logical_block in range(needed_blocks):
            block_id = int(block_tables_cpu[seq_idx, logical_block].item())
            if block_id < 0 or block_id >= nb:
                raise ValueError(f"kv_load_and_reshape_flash: invalid block id {block_id}")
            start = logical_block * bs
            end = min(start + bs, seq_len)
            width = end - start
            out_k[seq_idx, start:end].copy_(key_cache[block_id, :, :, :width].permute(2, 0, 1))
            out_v[seq_idx, start:end].copy_(value_cache[block_id, :, :, :width].permute(2, 0, 1))
    return out_k, out_v


def kv_reshape_and_cache_back_flash(
    dense_k: torch.Tensor,
    dense_v: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
) -> bool:
    """Scatter dense Flash-style KV tensors back into paged cache blocks."""
    device, dtype = tensor_device_dtype(dense_k)
    ensure_reference_allowed("kv_reshape_and_cache_back_flash")
    if dense_k.dim() != 4 or dense_v.dim() != 4:
        raise ValueError("dense Flash KV tensors must be [batch, seq, heads, head_dim]")
    if dense_k.shape != dense_v.shape or dense_k.dtype != dense_v.dtype:
        raise ValueError("dense K and V tensors must have matching shape and dtype")
    if dense_k.device != dense_v.device:
        raise ValueError("dense K and V tensors must be on the same device")
    nb, nh, hd, bs = _validate_paged_shapes(
        key_cache, value_cache, name="kv_reshape_and_cache_back_flash"
    )
    if key_cache.dtype != dense_k.dtype or value_cache.dtype != dense_v.dtype:
        raise ValueError("dense and paged KV tensors must have matching dtypes")
    if key_cache.device != dense_k.device or value_cache.device != dense_k.device:
        raise ValueError("dense and paged KV tensors must be on the same device")
    if dense_k.shape[2:] != (nh, hd):
        raise ValueError("dense KV head shape does not match paged cache")
    if block_tables.dim() != 2:
        raise ValueError("kv_reshape_and_cache_back_flash: block_tables must be [batch, max_blocks]")
    if seq_lens.dim() != 1 or seq_lens.shape[0] != block_tables.shape[0]:
        raise ValueError("kv_reshape_and_cache_back_flash: seq_lens must be [batch]")
    if dense_k.shape[0] != block_tables.shape[0]:
        raise ValueError("kv_reshape_and_cache_back_flash: dense batch does not match block_tables")
    if block_tables.device != dense_k.device or seq_lens.device != dense_k.device:
        raise ValueError("kv_reshape_and_cache_back_flash: all tensors must be on the same device")
    if block_tables.dtype not in (torch.int16, torch.int32, torch.int64):
        raise ValueError("kv_reshape_and_cache_back_flash: block_tables must use an integer dtype")
    if seq_lens.dtype not in (torch.int16, torch.int32, torch.int64):
        raise ValueError("kv_reshape_and_cache_back_flash: seq_lens must use an integer dtype")

    seq_lens_cpu = seq_lens.to(device="cpu", dtype=torch.long)
    block_tables_cpu = block_tables.to(device="cpu", dtype=torch.long)
    batch, dense_len = int(dense_k.shape[0]), int(dense_k.shape[1])
    for batch_idx in range(batch):
        seq_len = int(seq_lens_cpu[batch_idx].item())
        if seq_len < 0 or seq_len > dense_len:
            raise ValueError("kv_reshape_and_cache_back_flash: seq_lens exceeds dense sequence length")
        needed_blocks = (seq_len + bs - 1) // bs
        if needed_blocks > int(block_tables.shape[1]):
            raise ValueError("kv_reshape_and_cache_back_flash: block_tables does not contain enough columns")
        for logical_block in range(needed_blocks):
            block_id = int(block_tables_cpu[batch_idx, logical_block].item())
            if block_id < 0 or block_id >= nb:
                raise ValueError(f"kv_reshape_and_cache_back_flash: invalid block id {block_id}")
    if _try_native_kv_reshape_and_cache_back_flash(
        dense_k,
        dense_v,
        key_cache,
        value_cache,
        block_tables,
        seq_lens,
        batch,
        dense_len,
        int(block_tables.shape[1]),
        nb,
        nh,
        hd,
        bs,
    ):
        emit(
            DispatchRecord(
                "kv_reshape_and_cache_back_flash",
                "ctypes-musa-native",
                device,
                str(dtype),
                "native LMCache-compatible reverse Flash reshape",
            )
        )
        return True
    for batch_idx in range(batch):
        seq_len = int(seq_lens_cpu[batch_idx].item())
        needed_blocks = (seq_len + bs - 1) // bs
        for logical_block in range(needed_blocks):
            block_id = int(block_tables_cpu[batch_idx, logical_block].item())
            start = logical_block * bs
            end = min(start + bs, seq_len)
            width = end - start
            key_cache[block_id, :, :, :width].copy_(
                dense_k[batch_idx, start:end].permute(1, 2, 0)
            )
            value_cache[block_id, :, :, :width].copy_(
                dense_v[batch_idx, start:end].permute(1, 2, 0)
            )
    emit(
        DispatchRecord(
            "kv_reshape_and_cache_back_flash",
            "torch-reference",
            device,
            str(dtype),
            "LMCache-compatible reverse Flash reshape reference",
        )
    )
    return True


def kv_load_and_dequantize_fp8_flash(
    key_cache_fp8: torch.Tensor,
    value_cache_fp8: torch.Tensor,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    k_scale: torch.Tensor | float,
    v_scale: torch.Tensor | float,
    *,
    max_seq_len: int | None = None,
    out_dtype: torch.dtype = torch.bfloat16,
    fp8_format: Literal["e4m3", "e5m2"] = "e4m3",
    out_k: torch.Tensor | None = None,
    out_v: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    device, dtype = tensor_device_dtype(key_cache_fp8)
    ensure_reference_allowed("kv_load_and_dequantize_fp8_flash")
    nb, nh, hd, bs = _validate_paged_shapes(key_cache_fp8, value_cache_fp8, name="kv_load_and_dequantize_fp8_flash")
    if block_tables.dim() != 2:
        raise ValueError("kv_load_and_dequantize_fp8_flash: block_tables must be [batch, max_blocks]")
    if seq_lens.dim() != 1 or seq_lens.shape[0] != block_tables.shape[0]:
        raise ValueError("kv_load_and_dequantize_fp8_flash: seq_lens must be [batch]")
    if key_cache_fp8.device != value_cache_fp8.device or block_tables.device != key_cache_fp8.device or seq_lens.device != key_cache_fp8.device:
        raise ValueError("kv_load_and_dequantize_fp8_flash: all tensors must be on the same device")
    if block_tables.dtype not in (torch.int16, torch.int32, torch.int64) or seq_lens.dtype not in (torch.int16, torch.int32, torch.int64):
        raise ValueError("kv_load_and_dequantize_fp8_flash: block_tables and seq_lens must be integer tensors")
    batch = int(block_tables.shape[0])
    seq_lens_cpu = seq_lens.to(device="cpu", dtype=torch.long)
    if seq_lens_cpu.numel() > 0 and int(seq_lens_cpu.min().item()) < 0:
        raise ValueError("kv_load_and_dequantize_fp8_flash: seq_lens must be non-negative")
    inferred_max = int(seq_lens_cpu.max().item()) if seq_lens_cpu.numel() > 0 else 0
    dense_len = inferred_max if max_seq_len is None else int(max_seq_len)
    if dense_len < inferred_max:
        raise ValueError("kv_load_and_dequantize_fp8_flash: max_seq_len is smaller than seq_lens max")
    if (dense_len + bs - 1) // bs > int(block_tables.shape[1]):
        raise ValueError("kv_load_and_dequantize_fp8_flash: block_tables does not contain enough columns")
    shape = (batch, dense_len, nh, hd)
    if out_k is None:
        out_k = torch.zeros(shape, dtype=out_dtype, device=key_cache_fp8.device)
    elif out_k.shape != shape or out_k.dtype is not out_dtype or out_k.device != key_cache_fp8.device:
        raise ValueError("kv_load_and_dequantize_fp8_flash: out_k must match [B, S, kv_heads, head_dim] dtype/device")
    else:
        out_k.zero_()
    if out_v is None:
        out_v = torch.zeros(shape, dtype=out_dtype, device=value_cache_fp8.device)
    elif out_v.shape != shape or out_v.dtype is not out_dtype or out_v.device != value_cache_fp8.device:
        raise ValueError("kv_load_and_dequantize_fp8_flash: out_v must match [B, S, kv_heads, head_dim] dtype/device")
    else:
        out_v.zero_()
    block_tables_cpu = block_tables.to(device="cpu", dtype=torch.long)
    used_native = False
    for seq_idx in range(batch):
        seq_len = int(seq_lens_cpu[seq_idx].item())
        slots = torch.full((dense_len,), -1, dtype=torch.long, device=key_cache_fp8.device)
        for pos in range(seq_len):
            logical_block = pos // bs
            offset = pos - logical_block * bs
            block_id = int(block_tables_cpu[seq_idx, logical_block].item())
            if block_id < 0 or block_id >= nb:
                raise ValueError(f"kv_load_and_dequantize_fp8_flash: invalid block id {block_id}")
            slots[pos] = block_id * bs + offset
        k_out, v_out = kv_paged_fp8_to_linear(key_cache_fp8, value_cache_fp8, slots, k_scale, v_scale, out_dtype=out_dtype, fp8_format=fp8_format)
        out_k[seq_idx].copy_(k_out)
        out_v[seq_idx].copy_(v_out)
        used_native = used_native or (_native_fp8_kv_cache_requested() and _is_musa_native_fp8_cache_tensor(key_cache_fp8))
    backend = "ctypes-musa-native-fp8" if used_native else "torch-reference"
    emit(DispatchRecord("kv_load_and_dequantize_fp8_flash", backend, device, str(dtype), f"FP8 {fp8_format} Flash-style load/dequantize"))
    return out_k, out_v


def _try_native_kv_copy_paged_blocks_fp8_vec(
    src_key_cache_fp8: torch.Tensor,
    src_value_cache_fp8: torch.Tensor,
    dst_key_cache_fp8: torch.Tensor,
    dst_value_cache_fp8: torch.Tensor,
    block_mapping: torch.Tensor,
    src_blocks: int,
    dst_blocks: int,
    multi_layer: bool,
) -> bool:
    if not (_native_fp8_kv_cache_requested() and _native_fp8_kv_vectorize_requested()):
        return False
    tensors = (src_key_cache_fp8, src_value_cache_fp8, dst_key_cache_fp8, dst_value_cache_fp8)
    if not all(_is_musa_native_fp8_cache_tensor(tensor) for tensor in tensors):
        return False
    if not (src_key_cache_fp8.dtype == src_value_cache_fp8.dtype == dst_key_cache_fp8.dtype == dst_value_cache_fp8.dtype):
        return False
    if not all(tensor.is_contiguous() for tensor in tensors):
        return False
    if str(block_mapping.device).split(":")[0] != "musa":
        return False
    mapping64 = block_mapping.to(dtype=torch.long).contiguous()
    if multi_layer:
        num_layers = int(src_key_cache_fp8.shape[0])
        num_kv_heads = int(src_key_cache_fp8.shape[2])
        head_dim = int(src_key_cache_fp8.shape[3])
        block_size = int(src_key_cache_fp8.shape[4])
    else:
        num_layers = 1
        num_kv_heads = int(src_key_cache_fp8.shape[1])
        head_dim = int(src_key_cache_fp8.shape[2])
        block_size = int(src_key_cache_fp8.shape[3])
    try:
        lib = load_native_fp8_kv_cache()
        if not has_symbol(lib, "musa_aiter_kv_copy_paged_blocks_fp8_vec8"):
            return False
        fn = lib.handle.musa_aiter_kv_copy_paged_blocks_fp8_vec8
    except Exception:
        return False
    fn.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_longlong,
        ctypes.c_longlong,
        ctypes.c_longlong,
        ctypes.c_longlong,
        ctypes.c_longlong,
        ctypes.c_longlong,
        ctypes.c_longlong,
        ctypes.c_void_p,
    ]
    fn.restype = None
    fn(
        ctypes.c_void_p(src_key_cache_fp8.data_ptr()),
        ctypes.c_void_p(src_value_cache_fp8.data_ptr()),
        ctypes.c_void_p(dst_key_cache_fp8.data_ptr()),
        ctypes.c_void_p(dst_value_cache_fp8.data_ptr()),
        ctypes.c_void_p(mapping64.data_ptr()),
        ctypes.c_longlong(block_mapping.shape[0]),
        ctypes.c_longlong(num_layers),
        ctypes.c_longlong(src_blocks),
        ctypes.c_longlong(dst_blocks),
        ctypes.c_longlong(num_kv_heads),
        ctypes.c_longlong(head_dim),
        ctypes.c_longlong(block_size),
        _current_musa_stream_ptr(),
    )
    return True


def kv_copy_paged_blocks_fp8(
    src_key_cache_fp8: torch.Tensor,
    src_value_cache_fp8: torch.Tensor,
    dst_key_cache_fp8: torch.Tensor,
    dst_value_cache_fp8: torch.Tensor,
    block_mapping: torch.Tensor,
) -> int:
    device, dtype = tensor_device_dtype(dst_key_cache_fp8)
    ensure_reference_allowed("kv_copy_paged_blocks_fp8")
    src_blocks, dst_blocks, multi_layer = _validate_block_copy_shapes(
        src_key_cache_fp8,
        src_value_cache_fp8,
        dst_key_cache_fp8,
        dst_value_cache_fp8,
        name="kv_copy_paged_blocks_fp8",
    )
    for tensor in (src_key_cache_fp8, src_value_cache_fp8, dst_key_cache_fp8, dst_value_cache_fp8):
        _validate_fp8_cache_tensor(tensor, name="kv_copy_paged_blocks_fp8")
    pairs = _ensure_block_mapping_pairs(block_mapping, src_blocks=src_blocks, dst_blocks=dst_blocks, name="kv_copy_paged_blocks_fp8")
    if _try_native_kv_copy_paged_blocks_fp8_vec(src_key_cache_fp8, src_value_cache_fp8, dst_key_cache_fp8, dst_value_cache_fp8, block_mapping, src_blocks, dst_blocks, multi_layer):
        emit(DispatchRecord("kv_copy_paged_blocks_fp8", "ctypes-musa-native-fp8-vec", device, str(dtype), "vectorized raw FP8 paged block copy"))
        return int(((pairs[:, 0] >= 0) & (pairs[:, 1] >= 0)).sum().item())
    copied = 0
    for src_block, dst_block in pairs.tolist():
        if src_block < 0 or dst_block < 0:
            continue
        if multi_layer:
            dst_key_cache_fp8[:, dst_block].copy_(src_key_cache_fp8[:, src_block])
            dst_value_cache_fp8[:, dst_block].copy_(src_value_cache_fp8[:, src_block])
        else:
            dst_key_cache_fp8[dst_block].copy_(src_key_cache_fp8[src_block])
            dst_value_cache_fp8[dst_block].copy_(src_value_cache_fp8[src_block])
        copied += 1
    backend = "ctypes-musa-native-fp8" if _native_fp8_kv_cache_requested() and all(_is_musa_native_fp8_cache_tensor(t) for t in (src_key_cache_fp8, src_value_cache_fp8, dst_key_cache_fp8, dst_value_cache_fp8)) else "torch-reference"
    emit(DispatchRecord("kv_copy_paged_blocks_fp8", backend, device, str(dtype), "raw FP8 paged block copy"))
    return copied


def _validate_block_copy_shapes(
    src_key_cache: torch.Tensor,
    src_value_cache: torch.Tensor,
    dst_key_cache: torch.Tensor,
    dst_value_cache: torch.Tensor,
    *,
    name: str,
) -> tuple[int, int, bool]:
    if src_key_cache.shape != src_value_cache.shape:
        raise ValueError(f"{name}: source key/value cache shapes must match")
    if dst_key_cache.shape != dst_value_cache.shape:
        raise ValueError(f"{name}: destination key/value cache shapes must match")
    if src_key_cache.dim() not in (4, 5) or dst_key_cache.dim() != src_key_cache.dim():
        raise ValueError(f"{name}: caches must be 4-D single-layer or 5-D multi-layer paged tensors")
    if src_key_cache.dtype != src_value_cache.dtype or dst_key_cache.dtype != dst_value_cache.dtype:
        raise ValueError(f"{name}: key/value dtype pairs must match")
    if src_key_cache.dtype != dst_key_cache.dtype:
        raise ValueError(f"{name}: source and destination dtypes must match")
    if src_key_cache.dim() == 4:
        if src_key_cache.shape[1:] != dst_key_cache.shape[1:]:
            raise ValueError(f"{name}: single-layer cache tails must match")
        return int(src_key_cache.shape[0]), int(dst_key_cache.shape[0]), False
    if src_key_cache.shape[0] != dst_key_cache.shape[0] or src_key_cache.shape[2:] != dst_key_cache.shape[2:]:
        raise ValueError(f"{name}: multi-layer cache layer count and tails must match")
    return int(src_key_cache.shape[1]), int(dst_key_cache.shape[1]), True


def _ensure_block_mapping_pairs(
    block_mapping: torch.Tensor,
    *,
    src_blocks: int,
    dst_blocks: int,
    name: str,
) -> torch.Tensor:
    if block_mapping.dim() != 2 or block_mapping.shape[1] != 2:
        raise ValueError(f"{name}: block_mapping must be [num_pairs, 2] with source and destination block ids")
    if block_mapping.dtype not in (torch.int16, torch.int32, torch.int64):
        raise ValueError(f"{name}: block_mapping must use an integer dtype")
    pairs = block_mapping.to(device="cpu", dtype=torch.long)
    if pairs.numel() == 0:
        return pairs
    valid = (pairs[:, 0] >= 0) & (pairs[:, 1] >= 0)
    if not bool(valid.any()):
        return pairs
    src_max = int(pairs[valid, 0].max().item())
    dst_max = int(pairs[valid, 1].max().item())
    if src_max >= src_blocks:
        raise ValueError(f"{name}: source block index {src_max} out of bounds for {src_blocks} blocks")
    if dst_max >= dst_blocks:
        raise ValueError(f"{name}: destination block index {dst_max} out of bounds for {dst_blocks} blocks")
    return pairs


def _try_native_kv_copy_paged_blocks(
    src_key_cache: torch.Tensor,
    src_value_cache: torch.Tensor,
    dst_key_cache: torch.Tensor,
    dst_value_cache: torch.Tensor,
    block_mapping: torch.Tensor,
    src_blocks: int,
    dst_blocks: int,
    multi_layer: bool,
) -> bool:
    if not _native_kv_transfer_requested():
        return False
    tensors = (src_key_cache, src_value_cache, dst_key_cache, dst_value_cache)
    if not all(_is_musa_native_kv_dtype_tensor(tensor) for tensor in tensors):
        return False
    if not (src_key_cache.dtype == src_value_cache.dtype == dst_key_cache.dtype == dst_value_cache.dtype):
        return False
    if not all(tensor.is_contiguous() for tensor in tensors):
        return False
    if str(block_mapping.device).split(":")[0] != "musa":
        return False
    mapping64 = block_mapping.to(dtype=torch.long).contiguous()
    if multi_layer:
        num_layers = int(src_key_cache.shape[0])
        num_kv_heads = int(src_key_cache.shape[2])
        head_dim = int(src_key_cache.shape[3])
        block_size = int(src_key_cache.shape[4])
    else:
        num_layers = 1
        num_kv_heads = int(src_key_cache.shape[1])
        head_dim = int(src_key_cache.shape[2])
        block_size = int(src_key_cache.shape[3])
    try:
        lib = load_native_kv_transfer()
        fn = _get_native_kv_function(lib, "musa_aiter_kv_copy_paged_blocks", src_key_cache.dtype)
        if fn is None:
            return False
    except Exception:
        return False
    fn.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_longlong,
        ctypes.c_longlong,
        ctypes.c_longlong,
        ctypes.c_longlong,
        ctypes.c_longlong,
        ctypes.c_longlong,
        ctypes.c_longlong,
        ctypes.c_void_p,
    ]
    fn.restype = None
    fn(
        ctypes.c_void_p(src_key_cache.data_ptr()),
        ctypes.c_void_p(src_value_cache.data_ptr()),
        ctypes.c_void_p(dst_key_cache.data_ptr()),
        ctypes.c_void_p(dst_value_cache.data_ptr()),
        ctypes.c_void_p(mapping64.data_ptr()),
        ctypes.c_longlong(block_mapping.shape[0]),
        ctypes.c_longlong(num_layers),
        ctypes.c_longlong(src_blocks),
        ctypes.c_longlong(dst_blocks),
        ctypes.c_longlong(num_kv_heads),
        ctypes.c_longlong(head_dim),
        ctypes.c_longlong(block_size),
        ctypes.c_void_p(0),
    )
    return True


def kv_copy_paged_blocks(
    src_key_cache: torch.Tensor,
    src_value_cache: torch.Tensor,
    dst_key_cache: torch.Tensor,
    dst_value_cache: torch.Tensor,
    block_mapping: torch.Tensor,
) -> int:
    """Copy selected paged KV cache blocks between same-layout caches."""
    device, dtype = tensor_device_dtype(dst_key_cache)
    ensure_reference_allowed("kv_copy_paged_blocks")
    src_blocks, dst_blocks, multi_layer = _validate_block_copy_shapes(
        src_key_cache,
        src_value_cache,
        dst_key_cache,
        dst_value_cache,
        name="kv_copy_paged_blocks",
    )
    pairs = _ensure_block_mapping_pairs(
        block_mapping,
        src_blocks=src_blocks,
        dst_blocks=dst_blocks,
        name="kv_copy_paged_blocks",
    )
    if _try_native_kv_copy_paged_blocks(src_key_cache, src_value_cache, dst_key_cache, dst_value_cache, block_mapping, src_blocks, dst_blocks, multi_layer):
        emit(
            DispatchRecord(
                "kv_copy_paged_blocks",
                "ctypes-musa-native",
                device,
                str(dtype),
                "Phase PKV native paged-block copy",
            )
        )
        return int(((pairs[:, 0] >= 0) & (pairs[:, 1] >= 0)).sum().item())

    emit(
        DispatchRecord(
            "kv_copy_paged_blocks",
            "torch-reference",
            device,
            str(dtype),
            "Phase PKV MUSA staged paged-block copy façade",
        )
    )
    emit_vendor_planning_note("kv_copy_paged_blocks")
    copied = 0
    for src_block, dst_block in pairs.tolist():
        if src_block < 0 or dst_block < 0:
            continue
        if multi_layer:
            dst_key_cache[:, dst_block].copy_(src_key_cache[:, src_block])
            dst_value_cache[:, dst_block].copy_(src_value_cache[:, src_block])
        else:
            dst_key_cache[dst_block].copy_(src_key_cache[src_block])
            dst_value_cache[dst_block].copy_(src_value_cache[src_block])
        copied += 1
    return copied


def _normalize_fp8_format(fp8_format: str) -> str:
    normalized = fp8_format.lower().replace("-", "_")
    if normalized in ("fp8", "fp8_e4m3", "fp8_e4m3fn", "e4m3", "e4m3fn"):
        return "e4m3"
    if normalized in ("fp8_e5m2", "e5m2"):
        return "e5m2"
    raise ValueError(f"unsupported FP8 KV cache format: {fp8_format}")


def _fp8_format_params(fp8_format: str) -> tuple[int, int, int, int]:
    normalized = _normalize_fp8_format(fp8_format)
    if normalized == "e4m3":
        return 4, 3, 7, 8
    return 5, 2, 15, 16


def _ensure_fp8_scale(scale: torch.Tensor | float, *, device: torch.device, name: str, num_heads: int, num_layers: int | None = None) -> torch.Tensor:
    scale_tensor = torch.as_tensor(scale, dtype=torch.float32, device=device)
    if not bool(torch.isfinite(scale_tensor).all()) or float(scale_tensor.min().item()) <= 0.0:
        raise ValueError(f"{name}: FP8 scale must be positive and finite")
    if scale_tensor.numel() == 1:
        return scale_tensor.reshape(())
    if scale_tensor.dim() == 1 and int(scale_tensor.shape[0]) == num_heads:
        return scale_tensor.reshape(num_heads)
    if num_layers is not None and scale_tensor.dim() == 2 and tuple(scale_tensor.shape) == (num_layers, num_heads):
        return scale_tensor.reshape(num_layers, num_heads)
    expected = "scalar or [kv_heads]" if num_layers is None else "scalar, [kv_heads], or [layers, kv_heads]"
    raise ValueError(f"{name}: FP8 scale must be {expected}")


def _fp8_scale_for_head(scale: torch.Tensor, *, head_dim: int) -> torch.Tensor:
    if scale.dim() == 0:
        return scale
    return scale.reshape(scale.shape[-1], 1).expand(scale.shape[-1], head_dim)


def _fp8_scale_for_layer_head(scale: torch.Tensor, layer: int, *, head_dim: int) -> torch.Tensor:
    if scale.dim() == 0:
        return scale
    if scale.dim() == 1:
        return _fp8_scale_for_head(scale, head_dim=head_dim)
    return scale[layer].reshape(scale.shape[-1], 1).expand(scale.shape[-1], head_dim)


def _fp8_cache_slot_bits(cache: torch.Tensor, block: int, offset: int) -> torch.Tensor:
    slot = cache[block, :, :, offset]
    if cache.dtype is torch.uint8:
        return slot
    return slot.view(torch.uint8)


def _fp8_stacked_cache_slot_bits(cache: torch.Tensor, layer: int, block: int, offset: int) -> torch.Tensor:
    slot = cache[layer, block, :, :, offset]
    if cache.dtype is torch.uint8:
        return slot
    return slot.view(torch.uint8)


def _validate_fp8_cache_tensor(tensor: torch.Tensor, *, name: str) -> None:
    if tensor.dtype is torch.uint8:
        return
    dtype_name = str(tensor.dtype)
    if dtype_name in ("torch.float8_e4m3fn", "torch.float8_e5m2"):
        return
    raise ValueError(f"{name}: FP8 cache storage must be torch.uint8 or a supported torch.float8 dtype")


def _fp8_quantize_to_uint8(tensor: torch.Tensor, scale: torch.Tensor, fp8_format: str) -> torch.Tensor:
    _, mantissa_bits, bias, max_unbiased_exp = _fp8_format_params(fp8_format)
    work = tensor.to(torch.float32) / scale
    sign = work < 0
    abs_work = work.abs()
    finite = torch.isfinite(abs_work)
    abs_work = torch.where(finite, abs_work, torch.zeros_like(abs_work))
    max_value = (2.0 - 2.0 ** (-mantissa_bits)) * (2.0 ** max_unbiased_exp)
    min_normal = 2.0 ** (1 - bias)
    clipped = abs_work.clamp(max=max_value)
    nonzero = clipped >= min_normal
    safe = torch.where(nonzero, clipped, torch.ones_like(clipped))
    exponent = torch.floor(torch.log2(safe)).to(torch.int32)
    exponent = exponent.clamp(min=1 - bias, max=max_unbiased_exp)
    base = torch.pow(torch.tensor(2.0, dtype=torch.float32, device=tensor.device), exponent.to(torch.float32))
    mantissa = torch.round((safe / base - 1.0) * (2**mantissa_bits)).to(torch.int32)
    overflow = mantissa == (2**mantissa_bits)
    exponent = torch.where(overflow, exponent + 1, exponent).clamp(max=max_unbiased_exp)
    mantissa = torch.where(overflow, torch.zeros_like(mantissa), mantissa).clamp(min=0, max=2**mantissa_bits - 1)
    exponent_field = torch.where(nonzero, exponent + bias, torch.zeros_like(exponent))
    mantissa = torch.where(nonzero, mantissa, torch.zeros_like(mantissa))
    bits = (sign.to(torch.int32) << 7) | (exponent_field << mantissa_bits) | mantissa
    return bits.to(torch.uint8)


def _fp8_dequantize_from_uint8(encoded: torch.Tensor, scale: torch.Tensor, fp8_format: str, out_dtype: torch.dtype) -> torch.Tensor:
    _, mantissa_bits, bias, _ = _fp8_format_params(fp8_format)
    bits = encoded.to(torch.int32)
    sign = ((bits >> 7) & 1).to(torch.float32)
    exponent_field = (bits >> mantissa_bits) & ((1 << (7 - mantissa_bits)) - 1)
    mantissa = bits & ((1 << mantissa_bits) - 1)
    nonzero = exponent_field != 0
    exponent = exponent_field.to(torch.float32) - float(bias)
    value = (1.0 + mantissa.to(torch.float32) / float(2**mantissa_bits)) * torch.pow(
        torch.tensor(2.0, dtype=torch.float32, device=encoded.device),
        exponent,
    )
    value = torch.where(nonzero, value, torch.zeros_like(value))
    value = torch.where(sign > 0, -value, value)
    return (value * scale).to(out_dtype)


def kv_linear_to_paged_fp8(
    key_linear: torch.Tensor,
    value_linear: torch.Tensor,
    key_cache_fp8: torch.Tensor,
    value_cache_fp8: torch.Tensor,
    slot_mapping: torch.Tensor,
    k_scale: torch.Tensor | float,
    v_scale: torch.Tensor | float,
    *,
    fp8_format: Literal["e4m3", "e5m2"] = "e4m3",
) -> None:
    device, dtype = tensor_device_dtype(key_linear)
    ensure_reference_allowed("kv_linear_to_paged_fp8")
    fp8_format = _normalize_fp8_format(fp8_format)
    _validate_fp8_cache_tensor(key_cache_fp8, name="kv_linear_to_paged_fp8 key_cache_fp8")
    _validate_fp8_cache_tensor(value_cache_fp8, name="kv_linear_to_paged_fp8 value_cache_fp8")
    nb, nh, hd, bs = _validate_paged_shapes(key_cache_fp8, value_cache_fp8, name="kv_linear_to_paged_fp8")
    if key_linear.shape != value_linear.shape or key_linear.dim() != 3:
        raise ValueError("kv_linear_to_paged_fp8: key_linear and value_linear must be [tokens, kv_heads, head_dim]")
    tcount = int(key_linear.shape[0])
    if tuple(key_linear.shape[1:]) != (nh, hd):
        raise ValueError("kv_linear_to_paged_fp8: linear shape does not match FP8 cache shape")
    if key_linear.device != value_linear.device or key_cache_fp8.device != key_linear.device or value_cache_fp8.device != key_linear.device:
        raise ValueError("kv_linear_to_paged_fp8: all tensors must be on the same device")
    slots = _ensure_slotmapping_vector(slot_mapping, expected_len=tcount, name="kv_linear_to_paged_fp8")
    if slots.device != key_linear.device:
        raise ValueError("kv_linear_to_paged_fp8: slot_mapping must be on the same device")
    valid = slots >= 0
    _check_slots_in_bounds(slots, valid, nb, bs, "kv_linear_to_paged_fp8")
    k_scale_tensor = _ensure_fp8_scale(k_scale, device=key_linear.device, name="k_scale", num_heads=nh)
    v_scale_tensor = _ensure_fp8_scale(v_scale, device=value_linear.device, name="v_scale", num_heads=nh)
    if k_scale_tensor.dim() in (0, 1) and v_scale_tensor.dim() in (0, 1) and k_scale_tensor.dim() == v_scale_tensor.dim() and _try_native_kv_linear_to_paged_fp8_vec(key_linear, value_linear, key_cache_fp8, value_cache_fp8, slots, k_scale_tensor, v_scale_tensor, fp8_format):
        return
    if k_scale_tensor.dim() == 0 and v_scale_tensor.dim() == 0 and _try_native_kv_linear_to_paged_fp8(key_linear, value_linear, key_cache_fp8, value_cache_fp8, slots, k_scale_tensor, v_scale_tensor, fp8_format):
        return
    emit(DispatchRecord("kv_linear_to_paged_fp8", "torch-reference", device, str(dtype), "Phase FP8-KV reference quantize scatter"))
    key_scale_work = _fp8_scale_for_head(k_scale_tensor, head_dim=hd)
    value_scale_work = _fp8_scale_for_head(v_scale_tensor, head_dim=hd)
    for token in range(tcount):
        slot = int(slots[token].item())
        if slot < 0:
            continue
        block, offset = divmod(slot, bs)
        _fp8_cache_slot_bits(key_cache_fp8, block, offset).copy_(_fp8_quantize_to_uint8(key_linear[token], key_scale_work, fp8_format))
        _fp8_cache_slot_bits(value_cache_fp8, block, offset).copy_(_fp8_quantize_to_uint8(value_linear[token], value_scale_work, fp8_format))


def kv_paged_fp8_to_linear(
    key_cache_fp8: torch.Tensor,
    value_cache_fp8: torch.Tensor,
    slot_mapping: torch.Tensor,
    k_scale: torch.Tensor | float,
    v_scale: torch.Tensor | float,
    *,
    out_dtype: torch.dtype = torch.bfloat16,
    fp8_format: Literal["e4m3", "e5m2"] = "e4m3",
    out_k: torch.Tensor | None = None,
    out_v: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    device, dtype = tensor_device_dtype(key_cache_fp8)
    ensure_reference_allowed("kv_paged_fp8_to_linear")
    fp8_format = _normalize_fp8_format(fp8_format)
    _validate_fp8_cache_tensor(key_cache_fp8, name="kv_paged_fp8_to_linear key_cache_fp8")
    _validate_fp8_cache_tensor(value_cache_fp8, name="kv_paged_fp8_to_linear value_cache_fp8")
    nb, nh, hd, bs = _validate_paged_shapes(key_cache_fp8, value_cache_fp8, name="kv_paged_fp8_to_linear")
    tcount = int(slot_mapping.numel())
    slots = _ensure_slotmapping_vector(slot_mapping, expected_len=tcount, name="kv_paged_fp8_to_linear")
    if slots.device != key_cache_fp8.device:
        raise ValueError("kv_paged_fp8_to_linear: slot_mapping must be on the same device")
    valid = slots >= 0
    _check_slots_in_bounds(slots, valid, nb, bs, "kv_paged_fp8_to_linear")
    k_scale_tensor = _ensure_fp8_scale(k_scale, device=key_cache_fp8.device, name="k_scale", num_heads=nh)
    v_scale_tensor = _ensure_fp8_scale(v_scale, device=value_cache_fp8.device, name="v_scale", num_heads=nh)
    native = _try_native_kv_paged_fp8_to_linear_hd128_bs16(key_cache_fp8, value_cache_fp8, slots, k_scale_tensor, v_scale_tensor, out_dtype, out_k, out_v, fp8_format) if k_scale_tensor.dim() == 0 and v_scale_tensor.dim() == 0 else None
    if native is None:
        native = _try_native_kv_paged_fp8_to_linear_vec(key_cache_fp8, value_cache_fp8, slots, k_scale_tensor, v_scale_tensor, out_dtype, out_k, out_v, fp8_format) if k_scale_tensor.dim() in (0, 1) and v_scale_tensor.dim() in (0, 1) and k_scale_tensor.dim() == v_scale_tensor.dim() else None
    if native is None:
        native = _try_native_kv_paged_fp8_to_linear(key_cache_fp8, value_cache_fp8, slots, k_scale_tensor, v_scale_tensor, out_dtype, out_k, out_v, fp8_format) if k_scale_tensor.dim() == 0 and v_scale_tensor.dim() == 0 else None
    if native is not None:
        return native
    out_k_work = out_k if out_k is not None else torch.zeros(tcount, nh, hd, dtype=out_dtype, device=key_cache_fp8.device)
    out_v_work = out_v if out_v is not None else torch.zeros(tcount, nh, hd, dtype=out_dtype, device=value_cache_fp8.device)
    if out_k_work.shape != (tcount, nh, hd) or out_v_work.shape != (tcount, nh, hd):
        raise ValueError("kv_paged_fp8_to_linear: output shape mismatch")
    if out_k_work.dtype is not out_dtype or out_v_work.dtype is not out_dtype:
        raise ValueError("kv_paged_fp8_to_linear: output dtype mismatch")
    out_k_work.zero_()
    out_v_work.zero_()
    emit(DispatchRecord("kv_paged_fp8_to_linear", "torch-reference", device, str(dtype), "Phase FP8-KV reference dequantize gather"))
    key_scale_work = _fp8_scale_for_head(k_scale_tensor, head_dim=hd)
    value_scale_work = _fp8_scale_for_head(v_scale_tensor, head_dim=hd)
    for token in range(tcount):
        slot = int(slots[token].item())
        if slot < 0:
            continue
        block, offset = divmod(slot, bs)
        out_k_work[token].copy_(_fp8_dequantize_from_uint8(_fp8_cache_slot_bits(key_cache_fp8, block, offset), key_scale_work, fp8_format, out_dtype))
        out_v_work[token].copy_(_fp8_dequantize_from_uint8(_fp8_cache_slot_bits(value_cache_fp8, block, offset), value_scale_work, fp8_format, out_dtype))
    return out_k_work, out_v_work


def kv_linear_to_paged_fp8_multi_layer(
    key_linear: torch.Tensor,
    value_linear: torch.Tensor,
    key_cache_fp8: torch.Tensor,
    value_cache_fp8: torch.Tensor,
    slot_mapping: torch.Tensor,
    k_scale: torch.Tensor | float,
    v_scale: torch.Tensor | float,
    *,
    fp8_format: Literal["e4m3", "e5m2"] = "e4m3",
) -> None:
    device, dtype = tensor_device_dtype(key_linear)
    ensure_reference_allowed("kv_linear_to_paged_fp8_multi_layer")
    fp8_format = _normalize_fp8_format(fp8_format)
    _validate_fp8_cache_tensor(key_cache_fp8, name="kv_linear_to_paged_fp8_multi_layer key_cache_fp8")
    _validate_fp8_cache_tensor(value_cache_fp8, name="kv_linear_to_paged_fp8_multi_layer value_cache_fp8")
    nl, nb, nh, hd, bs = _validate_stacked_paged_shapes(key_cache_fp8, value_cache_fp8, name="kv_linear_to_paged_fp8_multi_layer")
    if key_linear.shape != value_linear.shape or key_linear.dim() != 4:
        raise ValueError("kv_linear_to_paged_fp8_multi_layer: key_linear and value_linear must be [layers, tokens, kv_heads, head_dim]")
    if key_linear.shape[0] != nl or tuple(key_linear.shape[2:]) != (nh, hd):
        raise ValueError("kv_linear_to_paged_fp8_multi_layer: linear shape does not match FP8 cache shape")
    if key_linear.device != value_linear.device or key_cache_fp8.device != key_linear.device or value_cache_fp8.device != key_linear.device:
        raise ValueError("kv_linear_to_paged_fp8_multi_layer: all tensors must be on the same device")
    tcount = int(key_linear.shape[1])
    slots = _ensure_slotmapping_vector(slot_mapping, expected_len=tcount, name="kv_linear_to_paged_fp8_multi_layer")
    if slots.device != key_linear.device:
        raise ValueError("kv_linear_to_paged_fp8_multi_layer: slot_mapping must be on the same device")
    valid = slots >= 0
    _check_slots_in_bounds(slots, valid, nb, bs, "kv_linear_to_paged_fp8_multi_layer")
    k_scale_tensor = _ensure_fp8_scale(k_scale, device=key_linear.device, name="k_scale", num_heads=nh, num_layers=nl)
    v_scale_tensor = _ensure_fp8_scale(v_scale, device=value_linear.device, name="v_scale", num_heads=nh, num_layers=nl)
    if k_scale_tensor.dim() == v_scale_tensor.dim() and _try_native_kv_linear_to_paged_fp8_multi_layer(key_linear, value_linear, key_cache_fp8, value_cache_fp8, slots, k_scale_tensor, v_scale_tensor, fp8_format):
        return
    emit(DispatchRecord("kv_linear_to_paged_fp8_multi_layer", "torch-reference", device, str(dtype), "Phase FP8-KV reference multi-layer quantize scatter"))
    for layer in range(nl):
        key_scale_work = _fp8_scale_for_layer_head(k_scale_tensor, layer, head_dim=hd)
        value_scale_work = _fp8_scale_for_layer_head(v_scale_tensor, layer, head_dim=hd)
        for token in range(tcount):
            slot = int(slots[token].item())
            if slot < 0:
                continue
            block, offset = divmod(slot, bs)
            _fp8_stacked_cache_slot_bits(key_cache_fp8, layer, block, offset).copy_(_fp8_quantize_to_uint8(key_linear[layer, token], key_scale_work, fp8_format))
            _fp8_stacked_cache_slot_bits(value_cache_fp8, layer, block, offset).copy_(_fp8_quantize_to_uint8(value_linear[layer, token], value_scale_work, fp8_format))


def kv_paged_fp8_to_linear_multi_layer(
    key_cache_fp8: torch.Tensor,
    value_cache_fp8: torch.Tensor,
    slot_mapping: torch.Tensor,
    k_scale: torch.Tensor | float,
    v_scale: torch.Tensor | float,
    *,
    out_dtype: torch.dtype = torch.bfloat16,
    fp8_format: Literal["e4m3", "e5m2"] = "e4m3",
    out_k: torch.Tensor | None = None,
    out_v: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    device, dtype = tensor_device_dtype(key_cache_fp8)
    ensure_reference_allowed("kv_paged_fp8_to_linear_multi_layer")
    fp8_format = _normalize_fp8_format(fp8_format)
    _validate_fp8_cache_tensor(key_cache_fp8, name="kv_paged_fp8_to_linear_multi_layer key_cache_fp8")
    _validate_fp8_cache_tensor(value_cache_fp8, name="kv_paged_fp8_to_linear_multi_layer value_cache_fp8")
    nl, nb, nh, hd, bs = _validate_stacked_paged_shapes(key_cache_fp8, value_cache_fp8, name="kv_paged_fp8_to_linear_multi_layer")
    tcount = int(slot_mapping.numel())
    slots = _ensure_slotmapping_vector(slot_mapping, expected_len=tcount, name="kv_paged_fp8_to_linear_multi_layer")
    if slots.device != key_cache_fp8.device:
        raise ValueError("kv_paged_fp8_to_linear_multi_layer: slot_mapping must be on the same device")
    valid = slots >= 0
    _check_slots_in_bounds(slots, valid, nb, bs, "kv_paged_fp8_to_linear_multi_layer")
    k_scale_tensor = _ensure_fp8_scale(k_scale, device=key_cache_fp8.device, name="k_scale", num_heads=nh, num_layers=nl)
    v_scale_tensor = _ensure_fp8_scale(v_scale, device=value_cache_fp8.device, name="v_scale", num_heads=nh, num_layers=nl)
    native = _try_native_kv_paged_fp8_to_linear_multi_layer(key_cache_fp8, value_cache_fp8, slots, k_scale_tensor, v_scale_tensor, out_dtype, out_k, out_v, fp8_format) if k_scale_tensor.dim() == v_scale_tensor.dim() else None
    if native is not None:
        return native
    out_k_work = out_k if out_k is not None else torch.zeros(nl, tcount, nh, hd, dtype=out_dtype, device=key_cache_fp8.device)
    out_v_work = out_v if out_v is not None else torch.zeros(nl, tcount, nh, hd, dtype=out_dtype, device=value_cache_fp8.device)
    if out_k_work.shape != (nl, tcount, nh, hd) or out_v_work.shape != (nl, tcount, nh, hd):
        raise ValueError("kv_paged_fp8_to_linear_multi_layer: output shape mismatch")
    if out_k_work.dtype is not out_dtype or out_v_work.dtype is not out_dtype:
        raise ValueError("kv_paged_fp8_to_linear_multi_layer: output dtype mismatch")
    out_k_work.zero_()
    out_v_work.zero_()
    emit(DispatchRecord("kv_paged_fp8_to_linear_multi_layer", "torch-reference", device, str(dtype), "Phase FP8-KV reference multi-layer dequantize gather"))
    for layer in range(nl):
        key_scale_work = _fp8_scale_for_layer_head(k_scale_tensor, layer, head_dim=hd)
        value_scale_work = _fp8_scale_for_layer_head(v_scale_tensor, layer, head_dim=hd)
        for token in range(tcount):
            slot = int(slots[token].item())
            if slot < 0:
                continue
            block, offset = divmod(slot, bs)
            out_k_work[layer, token].copy_(_fp8_dequantize_from_uint8(_fp8_stacked_cache_slot_bits(key_cache_fp8, layer, block, offset), key_scale_work, fp8_format, out_dtype))
            out_v_work[layer, token].copy_(_fp8_dequantize_from_uint8(_fp8_stacked_cache_slot_bits(value_cache_fp8, layer, block, offset), value_scale_work, fp8_format, out_dtype))
    return out_k_work, out_v_work


def _try_native_kv_linear_to_paged_fp8_vec(
    key_linear: torch.Tensor,
    value_linear: torch.Tensor,
    key_cache_fp8: torch.Tensor,
    value_cache_fp8: torch.Tensor,
    slots_long: torch.Tensor,
    k_scale: torch.Tensor | float,
    v_scale: torch.Tensor | float,
    fp8_format: str,
) -> bool:
    if not (_native_fp8_kv_cache_requested() and _native_fp8_kv_vectorize_requested()):
        return False
    if not (_is_musa_native_fp8_cache_tensor(key_cache_fp8) and _is_musa_native_fp8_cache_tensor(value_cache_fp8)):
        return False
    dtype_code = _native_fp8_linear_dtype_code(key_linear.dtype)
    if dtype_code is None or value_linear.dtype != key_linear.dtype:
        return False
    if not (str(key_linear.device).split(":")[0] == "musa" and value_linear.device == key_linear.device):
        return False
    if key_cache_fp8.device != key_linear.device or value_cache_fp8.device != key_linear.device:
        return False
    if int(key_cache_fp8.shape[2]) % 4 != 0:
        return False
    if not key_cache_fp8.is_contiguous() or not value_cache_fp8.is_contiguous():
        return False
    key_work = key_linear if key_linear.is_contiguous() else key_linear.contiguous()
    value_work = value_linear if value_linear.is_contiguous() else value_linear.contiguous()
    slots64 = slots_long.contiguous()
    k_scale_pack = _as_musa_float32_scale_vec_ptr(k_scale, device=key_linear.device, num_heads=int(key_cache_fp8.shape[1]))
    v_scale_pack = _as_musa_float32_scale_vec_ptr(v_scale, device=value_linear.device, num_heads=int(value_cache_fp8.shape[1]))
    if k_scale_pack is None or v_scale_pack is None:
        return False
    k_scale_ptr, k_scale_mode = k_scale_pack
    v_scale_ptr, v_scale_mode = v_scale_pack
    if k_scale_mode != v_scale_mode:
        return False
    try:
        fmt = _normalize_fp8_format(fp8_format)
        lib = load_native_fp8_kv_cache()
        symbol = f"musa_aiter_kv_linear_to_paged_fp8_vec4_{fmt}"
        if not has_symbol(lib, symbol):
            return False
        fn = getattr(lib.handle, symbol)
    except Exception:
        return False
    fn.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_longlong,
        ctypes.c_longlong,
        ctypes.c_longlong,
        ctypes.c_longlong,
        ctypes.c_longlong,
        ctypes.c_int,
        ctypes.c_void_p,
        ctypes.c_int,
    ]
    fn.restype = None
    fn(
        ctypes.c_void_p(key_work.data_ptr()),
        ctypes.c_void_p(value_work.data_ptr()),
        ctypes.c_void_p(slots64.data_ptr()),
        ctypes.c_void_p(key_cache_fp8.data_ptr()),
        ctypes.c_void_p(value_cache_fp8.data_ptr()),
        ctypes.c_void_p(k_scale_ptr.data_ptr()),
        ctypes.c_void_p(v_scale_ptr.data_ptr()),
        ctypes.c_longlong(key_linear.shape[0]),
        ctypes.c_longlong(key_cache_fp8.shape[0]),
        ctypes.c_longlong(key_cache_fp8.shape[1]),
        ctypes.c_longlong(key_cache_fp8.shape[2]),
        ctypes.c_longlong(key_cache_fp8.shape[3]),
        ctypes.c_int(dtype_code),
        _current_musa_stream_ptr(),
        ctypes.c_int(k_scale_mode),
    )
    emit(DispatchRecord("kv_linear_to_paged_fp8", "ctypes-musa-native-fp8-vec", key_linear.device.type, str(key_linear.dtype), f"vectorized native {fmt.upper()} FP8 quantize scatter"))
    return True


def _try_native_kv_linear_to_paged_fp8(
    key_linear: torch.Tensor,
    value_linear: torch.Tensor,
    key_cache_fp8: torch.Tensor,
    value_cache_fp8: torch.Tensor,
    slots_long: torch.Tensor,
    k_scale: torch.Tensor | float,
    v_scale: torch.Tensor | float,
    fp8_format: str,
) -> bool:
    if not _native_fp8_kv_cache_requested():
        return False
    if not (_is_musa_native_fp8_cache_tensor(key_cache_fp8) and _is_musa_native_fp8_cache_tensor(value_cache_fp8)):
        return False
    dtype_code = _native_fp8_linear_dtype_code(key_linear.dtype)
    if dtype_code is None or value_linear.dtype != key_linear.dtype:
        return False
    if not (str(key_linear.device).split(":")[0] == "musa" and value_linear.device == key_linear.device):
        return False
    if key_cache_fp8.device != key_linear.device or value_cache_fp8.device != key_linear.device:
        return False
    if not key_cache_fp8.is_contiguous() or not value_cache_fp8.is_contiguous():
        return False
    key_work = key_linear if key_linear.is_contiguous() else key_linear.contiguous()
    value_work = value_linear if value_linear.is_contiguous() else value_linear.contiguous()
    slots64 = slots_long.contiguous()
    k_scale_ptr = _as_musa_float32_scale_ptr(k_scale, device=key_linear.device)
    v_scale_ptr = _as_musa_float32_scale_ptr(v_scale, device=value_linear.device)
    if k_scale_ptr is None or v_scale_ptr is None:
        return False
    try:
        fmt = _normalize_fp8_format(fp8_format)
        lib = load_native_fp8_kv_cache()
        symbol = f"musa_aiter_kv_linear_to_paged_fp8_{fmt}"
        if not has_symbol(lib, symbol):
            return False
        fn = getattr(lib.handle, symbol)
    except Exception:
        return False
    fn.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_longlong,
        ctypes.c_longlong,
        ctypes.c_longlong,
        ctypes.c_longlong,
        ctypes.c_longlong,
        ctypes.c_int,
        ctypes.c_void_p,
    ]
    fn.restype = None
    fn(
        ctypes.c_void_p(key_work.data_ptr()),
        ctypes.c_void_p(value_work.data_ptr()),
        ctypes.c_void_p(slots64.data_ptr()),
        ctypes.c_void_p(key_cache_fp8.data_ptr()),
        ctypes.c_void_p(value_cache_fp8.data_ptr()),
        ctypes.c_void_p(k_scale_ptr.data_ptr()),
        ctypes.c_void_p(v_scale_ptr.data_ptr()),
        ctypes.c_longlong(key_linear.shape[0]),
        ctypes.c_longlong(key_cache_fp8.shape[0]),
        ctypes.c_longlong(key_cache_fp8.shape[1]),
        ctypes.c_longlong(key_cache_fp8.shape[2]),
        ctypes.c_longlong(key_cache_fp8.shape[3]),
        ctypes.c_int(dtype_code),
        _current_musa_stream_ptr(),
    )
    emit(DispatchRecord("kv_linear_to_paged_fp8", "ctypes-musa-native-fp8", key_linear.device.type, str(key_linear.dtype), f"native {fmt.upper()} FP8 quantize scatter"))
    return True


def _try_native_kv_paged_fp8_to_linear_hd128_bs16(
    key_cache_fp8: torch.Tensor,
    value_cache_fp8: torch.Tensor,
    slots_long: torch.Tensor,
    k_scale: torch.Tensor | float,
    v_scale: torch.Tensor | float,
    out_dtype: torch.dtype,
    out_k: torch.Tensor | None,
    out_v: torch.Tensor | None,
    fp8_format: str,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    if not (_native_fp8_kv_cache_requested() and _native_fp8_kv_vectorize_requested()):
        return None
    if not (_is_musa_native_fp8_cache_tensor(key_cache_fp8) and _is_musa_native_fp8_cache_tensor(value_cache_fp8)):
        return None
    dtype_code = _native_fp8_linear_dtype_code(out_dtype)
    if dtype_code is None:
        return None
    tcount = int(slots_long.numel())
    _, nh, hd, bs = key_cache_fp8.shape
    if not ((int(bs) == 16 and int(hd) in (64, 128)) or (int(bs) == 32 and int(hd) == 128)):
        return None
    out_k_work = out_k if out_k is not None else torch.empty((tcount, nh, hd), dtype=out_dtype, device=key_cache_fp8.device)
    out_v_work = out_v if out_v is not None else torch.empty((tcount, nh, hd), dtype=out_dtype, device=value_cache_fp8.device)
    if out_k_work.dtype is not out_dtype or out_v_work.dtype is not out_dtype:
        return None
    if out_k_work.device != key_cache_fp8.device or out_v_work.device != key_cache_fp8.device:
        return None
    if not key_cache_fp8.is_contiguous() or not value_cache_fp8.is_contiguous() or not out_k_work.is_contiguous() or not out_v_work.is_contiguous():
        return None
    slots64 = slots_long.contiguous()
    k_scale_ptr = _as_musa_float32_scale_ptr(k_scale, device=key_cache_fp8.device)
    v_scale_ptr = _as_musa_float32_scale_ptr(v_scale, device=value_cache_fp8.device)
    if k_scale_ptr is None or v_scale_ptr is None:
        return None
    try:
        fmt = _normalize_fp8_format(fp8_format)
        lib = load_native_fp8_kv_cache()
        symbol = f"musa_aiter_kv_paged_fp8_to_linear_hd{int(hd)}_bs{int(bs)}_{fmt}"
        if not has_symbol(lib, symbol):
            return None
        fn = getattr(lib.handle, symbol)
    except Exception:
        return None
    fn.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_longlong,
        ctypes.c_longlong,
        ctypes.c_longlong,
        ctypes.c_int,
        ctypes.c_void_p,
    ]
    fn.restype = None
    fn(
        ctypes.c_void_p(key_cache_fp8.data_ptr()),
        ctypes.c_void_p(value_cache_fp8.data_ptr()),
        ctypes.c_void_p(slots64.data_ptr()),
        ctypes.c_void_p(k_scale_ptr.data_ptr()),
        ctypes.c_void_p(v_scale_ptr.data_ptr()),
        ctypes.c_void_p(out_k_work.data_ptr()),
        ctypes.c_void_p(out_v_work.data_ptr()),
        ctypes.c_longlong(tcount),
        ctypes.c_longlong(key_cache_fp8.shape[0]),
        ctypes.c_longlong(key_cache_fp8.shape[1]),
        ctypes.c_int(dtype_code),
        _current_musa_stream_ptr(),
    )
    emit(DispatchRecord("kv_paged_fp8_to_linear", "ctypes-musa-native-fp8-shape", key_cache_fp8.device.type, str(key_cache_fp8.dtype), f"shape-specialized native {fmt.upper()} FP8 dequantize gather hd{int(hd)} bs{int(bs)}"))
    return out_k_work, out_v_work


def _try_native_kv_paged_fp8_to_linear_vec(
    key_cache_fp8: torch.Tensor,
    value_cache_fp8: torch.Tensor,
    slots_long: torch.Tensor,
    k_scale: torch.Tensor | float,
    v_scale: torch.Tensor | float,
    out_dtype: torch.dtype,
    out_k: torch.Tensor | None,
    out_v: torch.Tensor | None,
    fp8_format: str,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    if not (_native_fp8_kv_cache_requested() and _native_fp8_kv_vectorize_requested()):
        return None
    if not (_is_musa_native_fp8_cache_tensor(key_cache_fp8) and _is_musa_native_fp8_cache_tensor(value_cache_fp8)):
        return None
    dtype_code = _native_fp8_linear_dtype_code(out_dtype)
    if dtype_code is None:
        return None
    tcount = int(slots_long.numel())
    _, nh, hd, _ = key_cache_fp8.shape
    if hd % 4 != 0:
        return None
    out_k_work = out_k if out_k is not None else torch.empty((tcount, nh, hd), dtype=out_dtype, device=key_cache_fp8.device)
    out_v_work = out_v if out_v is not None else torch.empty((tcount, nh, hd), dtype=out_dtype, device=value_cache_fp8.device)
    if out_k_work.dtype is not out_dtype or out_v_work.dtype is not out_dtype:
        return None
    if out_k_work.device != key_cache_fp8.device or out_v_work.device != key_cache_fp8.device:
        return None
    if not key_cache_fp8.is_contiguous() or not value_cache_fp8.is_contiguous() or not out_k_work.is_contiguous() or not out_v_work.is_contiguous():
        return None
    slots64 = slots_long.contiguous()
    k_scale_pack = _as_musa_float32_scale_vec_ptr(k_scale, device=key_cache_fp8.device, num_heads=int(nh))
    v_scale_pack = _as_musa_float32_scale_vec_ptr(v_scale, device=value_cache_fp8.device, num_heads=int(nh))
    if k_scale_pack is None or v_scale_pack is None:
        return None
    k_scale_ptr, k_scale_mode = k_scale_pack
    v_scale_ptr, v_scale_mode = v_scale_pack
    if k_scale_mode != v_scale_mode:
        return None
    try:
        fmt = _normalize_fp8_format(fp8_format)
        lib = load_native_fp8_kv_cache()
        symbol = f"musa_aiter_kv_paged_fp8_to_linear_vec4_{fmt}"
        if not has_symbol(lib, symbol):
            return None
        fn = getattr(lib.handle, symbol)
    except Exception:
        return None
    fn.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_longlong,
        ctypes.c_longlong,
        ctypes.c_longlong,
        ctypes.c_longlong,
        ctypes.c_longlong,
        ctypes.c_int,
        ctypes.c_void_p,
        ctypes.c_int,
    ]
    fn.restype = None
    fn(
        ctypes.c_void_p(key_cache_fp8.data_ptr()),
        ctypes.c_void_p(value_cache_fp8.data_ptr()),
        ctypes.c_void_p(slots64.data_ptr()),
        ctypes.c_void_p(k_scale_ptr.data_ptr()),
        ctypes.c_void_p(v_scale_ptr.data_ptr()),
        ctypes.c_void_p(out_k_work.data_ptr()),
        ctypes.c_void_p(out_v_work.data_ptr()),
        ctypes.c_longlong(tcount),
        ctypes.c_longlong(key_cache_fp8.shape[0]),
        ctypes.c_longlong(key_cache_fp8.shape[1]),
        ctypes.c_longlong(key_cache_fp8.shape[2]),
        ctypes.c_longlong(key_cache_fp8.shape[3]),
        ctypes.c_int(dtype_code),
        _current_musa_stream_ptr(),
        ctypes.c_int(k_scale_mode),
    )
    emit(DispatchRecord("kv_paged_fp8_to_linear", "ctypes-musa-native-fp8-vec", key_cache_fp8.device.type, str(key_cache_fp8.dtype), f"vectorized native {fmt.upper()} FP8 dequantize gather"))
    return out_k_work, out_v_work


def _try_native_kv_paged_fp8_to_linear(
    key_cache_fp8: torch.Tensor,
    value_cache_fp8: torch.Tensor,
    slots_long: torch.Tensor,
    k_scale: torch.Tensor | float,
    v_scale: torch.Tensor | float,
    out_dtype: torch.dtype,
    out_k: torch.Tensor | None,
    out_v: torch.Tensor | None,
    fp8_format: str,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    if not _native_fp8_kv_cache_requested():
        return None
    if not (_is_musa_native_fp8_cache_tensor(key_cache_fp8) and _is_musa_native_fp8_cache_tensor(value_cache_fp8)):
        return None
    dtype_code = _native_fp8_linear_dtype_code(out_dtype)
    if dtype_code is None:
        return None
    tcount = int(slots_long.numel())
    _, nh, hd, _ = key_cache_fp8.shape
    out_k_work = out_k if out_k is not None else torch.empty((tcount, nh, hd), dtype=out_dtype, device=key_cache_fp8.device)
    out_v_work = out_v if out_v is not None else torch.empty((tcount, nh, hd), dtype=out_dtype, device=value_cache_fp8.device)
    if out_k_work.dtype is not out_dtype or out_v_work.dtype is not out_dtype:
        return None
    if out_k_work.device != key_cache_fp8.device or out_v_work.device != key_cache_fp8.device:
        return None
    if not key_cache_fp8.is_contiguous() or not value_cache_fp8.is_contiguous() or not out_k_work.is_contiguous() or not out_v_work.is_contiguous():
        return None
    slots64 = slots_long.contiguous()
    k_scale_ptr = _as_musa_float32_scale_ptr(k_scale, device=key_cache_fp8.device)
    v_scale_ptr = _as_musa_float32_scale_ptr(v_scale, device=value_cache_fp8.device)
    if k_scale_ptr is None or v_scale_ptr is None:
        return None
    try:
        fmt = _normalize_fp8_format(fp8_format)
        lib = load_native_fp8_kv_cache()
        symbol = f"musa_aiter_kv_paged_fp8_to_linear_{fmt}"
        if not has_symbol(lib, symbol):
            return None
        fn = getattr(lib.handle, symbol)
    except Exception:
        return None
    fn.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_longlong,
        ctypes.c_longlong,
        ctypes.c_longlong,
        ctypes.c_longlong,
        ctypes.c_longlong,
        ctypes.c_int,
        ctypes.c_void_p,
    ]
    fn.restype = None
    fn(
        ctypes.c_void_p(key_cache_fp8.data_ptr()),
        ctypes.c_void_p(value_cache_fp8.data_ptr()),
        ctypes.c_void_p(slots64.data_ptr()),
        ctypes.c_void_p(k_scale_ptr.data_ptr()),
        ctypes.c_void_p(v_scale_ptr.data_ptr()),
        ctypes.c_void_p(out_k_work.data_ptr()),
        ctypes.c_void_p(out_v_work.data_ptr()),
        ctypes.c_longlong(tcount),
        ctypes.c_longlong(key_cache_fp8.shape[0]),
        ctypes.c_longlong(key_cache_fp8.shape[1]),
        ctypes.c_longlong(key_cache_fp8.shape[2]),
        ctypes.c_longlong(key_cache_fp8.shape[3]),
        ctypes.c_int(dtype_code),
        _current_musa_stream_ptr(),
    )
    emit(DispatchRecord("kv_paged_fp8_to_linear", "ctypes-musa-native-fp8", key_cache_fp8.device.type, str(key_cache_fp8.dtype), f"native {fmt.upper()} FP8 dequantize gather"))
    return out_k_work, out_v_work


def _try_native_kv_linear_to_paged_fp8_multi_layer(
    key_linear: torch.Tensor,
    value_linear: torch.Tensor,
    key_cache_fp8: torch.Tensor,
    value_cache_fp8: torch.Tensor,
    slots_long: torch.Tensor,
    k_scale: torch.Tensor | float,
    v_scale: torch.Tensor | float,
    fp8_format: str,
) -> bool:
    if not _native_fp8_kv_cache_requested():
        return False
    if not (_is_musa_native_fp8_cache_tensor(key_cache_fp8) and _is_musa_native_fp8_cache_tensor(value_cache_fp8)):
        return False
    dtype_code = _native_fp8_linear_dtype_code(key_linear.dtype)
    if dtype_code is None or value_linear.dtype != key_linear.dtype:
        return False
    if key_linear.dim() != 4 or key_cache_fp8.dim() != 5:
        return False
    if not (str(key_linear.device).split(":")[0] == "musa" and value_linear.device == key_linear.device):
        return False
    if key_cache_fp8.device != key_linear.device or value_cache_fp8.device != key_linear.device:
        return False
    if not key_cache_fp8.is_contiguous() or not value_cache_fp8.is_contiguous():
        return False
    key_work = key_linear if key_linear.is_contiguous() else key_linear.contiguous()
    value_work = value_linear if value_linear.is_contiguous() else value_linear.contiguous()
    slots64 = slots_long.contiguous()
    k_scale_pack = _as_musa_float32_scale_layer_head_ptr(k_scale, device=key_linear.device, num_layers=int(key_cache_fp8.shape[0]), num_heads=int(key_cache_fp8.shape[2]))
    v_scale_pack = _as_musa_float32_scale_layer_head_ptr(v_scale, device=value_linear.device, num_layers=int(value_cache_fp8.shape[0]), num_heads=int(value_cache_fp8.shape[2]))
    if k_scale_pack is None or v_scale_pack is None:
        return False
    k_scale_ptr, k_scale_mode = k_scale_pack
    v_scale_ptr, v_scale_mode = v_scale_pack
    if k_scale_mode != v_scale_mode:
        return False
    try:
        fmt = _normalize_fp8_format(fp8_format)
        lib = load_native_fp8_kv_cache()
        symbol = f"musa_aiter_kv_linear_to_paged_fp8_multi_layer_{fmt}"
        if not has_symbol(lib, symbol):
            return False
        fn = getattr(lib.handle, symbol)
    except Exception:
        return False
    fn.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_longlong,
        ctypes.c_longlong,
        ctypes.c_longlong,
        ctypes.c_longlong,
        ctypes.c_longlong,
        ctypes.c_longlong,
        ctypes.c_int,
        ctypes.c_void_p,
        ctypes.c_int,
    ]
    fn.restype = None
    fn(
        ctypes.c_void_p(key_work.data_ptr()),
        ctypes.c_void_p(value_work.data_ptr()),
        ctypes.c_void_p(slots64.data_ptr()),
        ctypes.c_void_p(key_cache_fp8.data_ptr()),
        ctypes.c_void_p(value_cache_fp8.data_ptr()),
        ctypes.c_void_p(k_scale_ptr.data_ptr()),
        ctypes.c_void_p(v_scale_ptr.data_ptr()),
        ctypes.c_longlong(key_linear.shape[0]),
        ctypes.c_longlong(key_linear.shape[1]),
        ctypes.c_longlong(key_cache_fp8.shape[1]),
        ctypes.c_longlong(key_cache_fp8.shape[2]),
        ctypes.c_longlong(key_cache_fp8.shape[3]),
        ctypes.c_longlong(key_cache_fp8.shape[4]),
        ctypes.c_int(dtype_code),
        _current_musa_stream_ptr(),
        ctypes.c_int(k_scale_mode),
    )
    emit(DispatchRecord("kv_linear_to_paged_fp8_multi_layer", "ctypes-musa-native-fp8", key_linear.device.type, str(key_linear.dtype), f"native multi-layer {fmt.upper()} FP8 quantize scatter"))
    return True


def _try_native_kv_paged_fp8_to_linear_multi_layer(
    key_cache_fp8: torch.Tensor,
    value_cache_fp8: torch.Tensor,
    slots_long: torch.Tensor,
    k_scale: torch.Tensor | float,
    v_scale: torch.Tensor | float,
    out_dtype: torch.dtype,
    out_k: torch.Tensor | None,
    out_v: torch.Tensor | None,
    fp8_format: str,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    if not _native_fp8_kv_cache_requested():
        return None
    if not (_is_musa_native_fp8_cache_tensor(key_cache_fp8) and _is_musa_native_fp8_cache_tensor(value_cache_fp8)):
        return None
    dtype_code = _native_fp8_linear_dtype_code(out_dtype)
    if dtype_code is None:
        return None
    if key_cache_fp8.dim() != 5:
        return None
    nl, _, nh, hd, _ = key_cache_fp8.shape
    tcount = int(slots_long.numel())
    out_k_work = out_k if out_k is not None else torch.empty((nl, tcount, nh, hd), dtype=out_dtype, device=key_cache_fp8.device)
    out_v_work = out_v if out_v is not None else torch.empty((nl, tcount, nh, hd), dtype=out_dtype, device=value_cache_fp8.device)
    if out_k_work.dtype is not out_dtype or out_v_work.dtype is not out_dtype:
        return None
    if out_k_work.shape != (nl, tcount, nh, hd) or out_v_work.shape != (nl, tcount, nh, hd):
        return None
    if out_k_work.device != key_cache_fp8.device or out_v_work.device != key_cache_fp8.device:
        return None
    if not key_cache_fp8.is_contiguous() or not value_cache_fp8.is_contiguous() or not out_k_work.is_contiguous() or not out_v_work.is_contiguous():
        return None
    slots64 = slots_long.contiguous()
    k_scale_pack = _as_musa_float32_scale_layer_head_ptr(k_scale, device=key_cache_fp8.device, num_layers=int(nl), num_heads=int(nh))
    v_scale_pack = _as_musa_float32_scale_layer_head_ptr(v_scale, device=value_cache_fp8.device, num_layers=int(nl), num_heads=int(nh))
    if k_scale_pack is None or v_scale_pack is None:
        return None
    k_scale_ptr, k_scale_mode = k_scale_pack
    v_scale_ptr, v_scale_mode = v_scale_pack
    if k_scale_mode != v_scale_mode:
        return None
    try:
        fmt = _normalize_fp8_format(fp8_format)
        lib = load_native_fp8_kv_cache()
        symbol = f"musa_aiter_kv_paged_fp8_to_linear_multi_layer_{fmt}"
        if not has_symbol(lib, symbol):
            return None
        fn = getattr(lib.handle, symbol)
    except Exception:
        return None
    fn.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_longlong,
        ctypes.c_longlong,
        ctypes.c_longlong,
        ctypes.c_longlong,
        ctypes.c_longlong,
        ctypes.c_longlong,
        ctypes.c_int,
        ctypes.c_void_p,
        ctypes.c_int,
    ]
    fn.restype = None
    fn(
        ctypes.c_void_p(key_cache_fp8.data_ptr()),
        ctypes.c_void_p(value_cache_fp8.data_ptr()),
        ctypes.c_void_p(slots64.data_ptr()),
        ctypes.c_void_p(k_scale_ptr.data_ptr()),
        ctypes.c_void_p(v_scale_ptr.data_ptr()),
        ctypes.c_void_p(out_k_work.data_ptr()),
        ctypes.c_void_p(out_v_work.data_ptr()),
        ctypes.c_longlong(key_cache_fp8.shape[0]),
        ctypes.c_longlong(tcount),
        ctypes.c_longlong(key_cache_fp8.shape[1]),
        ctypes.c_longlong(key_cache_fp8.shape[2]),
        ctypes.c_longlong(key_cache_fp8.shape[3]),
        ctypes.c_longlong(key_cache_fp8.shape[4]),
        ctypes.c_int(dtype_code),
        _current_musa_stream_ptr(),
        ctypes.c_int(k_scale_mode),
    )
    emit(DispatchRecord("kv_paged_fp8_to_linear_multi_layer", "ctypes-musa-native-fp8", key_cache_fp8.device.type, str(key_cache_fp8.dtype), f"native multi-layer {fmt.upper()} FP8 dequantize gather"))
    return out_k_work, out_v_work


def _native_kv_fallback_reason(op_name: str) -> str:
    if _native_kv_transfer_requested():
        return f"native {op_name} requested but kernel is not built yet; Phase PKV torch reference fallback"
    return f"Phase PKV {op_name} reference"


def _validate_paged_shapes(
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    *,
    name: str,
) -> tuple[int, int, int, int]:
    if key_cache.dim() != 4 or value_cache.dim() != 4:
        raise ValueError(f"{name}: caches must be 4-D [blocks, kv_heads, head_dim, block_size]")
    if key_cache.shape != value_cache.shape:
        raise ValueError(f"{name}: key_cache and value_cache shapes must match")
    nb, nh, hd, bs = map(int, key_cache.shape)
    if nb <= 0 or nh <= 0 or hd <= 0 or bs <= 0:
        raise ValueError(f"{name}: all cache extents must be positive")
    return nb, nh, hd, bs


def _ensure_slotmapping_vector(slot_mapping: torch.Tensor, *, expected_len: int, name: str) -> torch.Tensor:
    if slot_mapping.dim() != 1:
        raise ValueError(f"{name}: slot_mapping must be a 1-D tensor")
    if slot_mapping.numel() != expected_len:
        raise ValueError(
            f"{name}: slot_mapping length {slot_mapping.numel()} "
            f"does not match num_tokens={expected_len}",
        )
    if slot_mapping.dtype not in (
        torch.int32,
        torch.int64,
        torch.int16,
    ):
        raise ValueError(f"{name}: slot_mapping must be integer-valued (int16/int32/int64)")
    return slot_mapping.to(torch.long)


def _check_slots_in_bounds(slots_long: torch.Tensor, valid_mask: torch.Tensor, nb: int, bs: int, name: str) -> None:
    if not bool(valid_mask.any()):
        return
    s = slots_long[valid_mask]
    max_slot = int(s.max().item())
    if max_slot >= nb * bs:
        raise ValueError(f"{name}: slot index {max_slot} out of bounds for caches (max {nb * bs - 1})")
    if int(s.min().item()) < 0:
        raise ValueError(f"{name}: unexpected negative slot in valid subset")


def _validate_stacked_paged_shapes(
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    *,
    name: str,
) -> tuple[int, int, int, int, int]:
    if key_cache.dim() != 5 or value_cache.dim() != 5:
        raise ValueError(f"{name}: caches must be 5-D [layers, blocks, kv_heads, head_dim, block_size]")
    if key_cache.shape != value_cache.shape:
        raise ValueError(f"{name}: key_cache and value_cache shapes must match")
    nl, nb, nh, hd, bs = map(int, key_cache.shape)
    if nl <= 0 or nb <= 0 or nh <= 0 or hd <= 0 or bs <= 0:
        raise ValueError(f"{name}: all cache extents must be positive")
    return nl, nb, nh, hd, bs


def _kv_paged_to_linear_compute(
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    slots_long: torch.Tensor,
    valid: torch.Tensor,
    tcount: int,
    nh: int,
    hd: int,
    bs: int,
    *,
    fill_skipped: Literal["zeros", "preserve"],
    out_k: torch.Tensor | None,
    out_v: torch.Tensor | None,
    name: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    safe_slots = torch.where(valid, slots_long, torch.zeros_like(slots_long))
    bid = torch.div(safe_slots, bs, rounding_mode="floor")
    off = safe_slots - bid * bs
    gathered_k = key_cache[bid, :, :, off]
    gathered_v = value_cache[bid, :, :, off]
    mask = valid.view(tcount, 1, 1)

    def _finalize(gathered: torch.Tensor, out_: torch.Tensor | None) -> torch.Tensor:
        if out_ is not None:
            if out_.shape != (tcount, nh, hd):
                raise ValueError(f"{name}: out buffer shape mismatch {tuple(out_.shape)}")
            if out_.dtype != gathered.dtype:
                raise ValueError(f"{name}: out buffer dtype mismatch")
            if out_.device != gathered.device:
                raise ValueError(f"{name}: out buffer device mismatch")
            if fill_skipped == "zeros":
                out_.zero_()
                out_[valid] = gathered[valid]
                return out_
            assert fill_skipped == "preserve"
            out_copy = out_
            torch.where(mask, gathered, out_copy, out=out_copy)
            return out_copy
        if fill_skipped == "preserve":
            raise ValueError(f"{name}: fill_skipped=preserve requires out_k/out_v buffers")
        return torch.where(mask, gathered, torch.zeros_like(gathered))

    return _finalize(gathered_k, out_k), _finalize(gathered_v, out_v)


def _kv_linear_to_paged_scatter(
    key_linear: torch.Tensor,
    value_linear: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    slots_long: torch.Tensor,
    nb: int,
    nh: int,
    hd: int,
    bs: int,
    tcount: int,
) -> None:
    for t in range(tcount):
        s = int(slots_long[t].item())
        if s < 0:
            continue
        b, off = divmod(s, bs)
        if not (0 <= b < nb and 0 <= off < bs):
            raise RuntimeError(f"kv_linear_to_paged: invalid slot mapping {s} at token {t}")
        key_cache[b, :, :, off].copy_(key_linear[t])
        value_cache[b, :, :, off].copy_(value_linear[t])


def kv_paged_to_linear(
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    *,
    fill_skipped: Literal["zeros", "preserve"] = "zeros",
    out_k: torch.Tensor | None = None,
    out_v: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Gather paged KV at ``slot_mapping`` into contiguous token-major tensors.

    For each token index ``t`` with ``slot = slot_mapping[t] >= 0``, copies::

        cache[block, :, :, offset]

    where ``block = slot // block_size``, ``offset = slot % block_size``.

    Args:
        key_cache / value_cache: paged KV on the compute device / dtype policy.
        slot_mapping: int tensor of length ``T`` (= number of offload tokens).
        fill_skipped: ``\"zeros\"`` (default) or ``\"preserve\"``. If ``\"preserve\"``,
            callers must supply ``out_k`` / ``out_v`` buffers; skipped rows are untouched.
        out_k / out_v: optional pre-allocated ``[T, num_kv_heads, head_dim]`` outputs.

    Returns:
        ``(key_linear, value_linear)`` each ``[T, num_kv_heads, head_dim]``.
    """
    device, dtype = tensor_device_dtype(key_cache)
    ensure_reference_allowed("kv_paged_to_linear")

    nb, nh, hd, bs = _validate_paged_shapes(key_cache, value_cache, name="kv_paged_to_linear")
    tcount = slot_mapping.numel()
    slots = _ensure_slotmapping_vector(slot_mapping, expected_len=tcount, name="kv_paged_to_linear")

    if key_cache.device != value_cache.device or slots.device != key_cache.device:
        raise ValueError("kv_paged_to_linear: all tensors must be on the same device")

    valid = slots >= 0
    _check_slots_in_bounds(slots, valid, nb, bs, "kv_paged_to_linear")

    native = _try_native_kv_paged_to_linear(
        key_cache,
        value_cache,
        slots,
        valid,
        tcount,
        nh,
        hd,
        bs,
        fill_skipped=fill_skipped,
        out_k=out_k,
        out_v=out_v,
    )
    if native is not None:
        emit(
            DispatchRecord(
                "kv_paged_to_linear",
                "ctypes-musa-native",
                device,
                str(dtype),
                "native Phase PKV paged→linear gather opt-in",
            )
        )
        emit_vendor_planning_note("kv_paged_to_linear")
        return native

    emit(
        DispatchRecord(
            "kv_paged_to_linear",
            "torch-reference",
            device,
            str(dtype),
            _native_kv_fallback_reason("kv_paged_to_linear"),
        )
    )
    emit_vendor_planning_note("kv_paged_to_linear")

    k_lin, v_lin = _kv_paged_to_linear_compute(
        key_cache,
        value_cache,
        slots,
        valid,
        tcount,
        nh,
        hd,
        bs,
        fill_skipped=fill_skipped,
        out_k=out_k,
        out_v=out_v,
        name="kv_paged_to_linear",
    )
    return k_lin, v_lin


def kv_paged_to_linear_multi_layer(
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    *,
    fill_skipped: Literal["zeros", "preserve"] = "zeros",
    out_k: torch.Tensor | None = None,
    out_v: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Gather paged KV for **stacked layers** into ``[L, T, num_kv_heads, head_dim]``.

    ``key_cache`` / ``value_cache`` are ``[L, blocks, kv_heads, head_dim, block_size]``.
    The same ``slot_mapping`` applies to every layer (typical transformer block tiling).
    """
    device, dtype = tensor_device_dtype(key_cache)
    ensure_reference_allowed("kv_paged_to_linear_multi_layer")

    nl, nb, nh, hd, bs = _validate_stacked_paged_shapes(key_cache, value_cache, name="kv_paged_to_linear_multi_layer")
    tcount = slot_mapping.numel()
    slots = _ensure_slotmapping_vector(slot_mapping, expected_len=tcount, name="kv_paged_to_linear_multi_layer")

    if value_cache.device != key_cache.device or slots.device != key_cache.device:
        raise ValueError("kv_paged_to_linear_multi_layer: all tensors must be on the same device")

    valid = slots >= 0
    _check_slots_in_bounds(slots, valid, nb, bs, "kv_paged_to_linear_multi_layer")

    if out_k is not None:
        if out_k.shape != (nl, tcount, nh, hd):
            raise ValueError(
                "kv_paged_to_linear_multi_layer: out_k must be "
                f"[num_layers, T, kv_heads, head_dim], got {tuple(out_k.shape)}",
            )
    if out_v is not None:
        if out_v.shape != (nl, tcount, nh, hd):
            raise ValueError(
                "kv_paged_to_linear_multi_layer: out_v must be "
                f"[num_layers, T, kv_heads, head_dim], got {tuple(out_v.shape)}",
            )
    if fill_skipped == "preserve" and (out_k is None or out_v is None):
        raise ValueError("kv_paged_to_linear_multi_layer: fill_skipped=preserve requires out_k and out_v")

    native = _try_native_kv_paged_to_linear_multi_layer(
        key_cache,
        value_cache,
        slots,
        tcount,
        nl,
        nh,
        hd,
        bs,
        fill_skipped=fill_skipped,
        out_k=out_k,
        out_v=out_v,
    )
    if native is not None:
        emit(
            DispatchRecord(
                "kv_paged_to_linear_multi_layer",
                "ctypes-musa-native",
                device,
                str(dtype),
                "native Phase PKV paged→linear gather opt-in (multi-layer)",
            )
        )
        emit_vendor_planning_note("kv_paged_to_linear_multi_layer")
        return native

    emit(
        DispatchRecord(
            "kv_paged_to_linear_multi_layer",
            "torch-reference",
            device,
            str(dtype),
            _native_kv_fallback_reason("kv_paged_to_linear_multi_layer"),
        )
    )
    emit_vendor_planning_note("kv_paged_to_linear_multi_layer")

    ks: list[torch.Tensor] = []
    vs: list[torch.Tensor] = []
    for layer in range(nl):
        ok = out_k[layer] if out_k is not None else None
        ov = out_v[layer] if out_v is not None else None
        k_lin, v_lin = _kv_paged_to_linear_compute(
            key_cache[layer],
            value_cache[layer],
            slots,
            valid,
            tcount,
            nh,
            hd,
            bs,
            fill_skipped=fill_skipped,
            out_k=ok,
            out_v=ov,
            name="kv_paged_to_linear_multi_layer",
        )
        ks.append(k_lin)
        vs.append(v_lin)
    return torch.stack(ks, dim=0), torch.stack(vs, dim=0)


def kv_linear_to_paged(
    key_linear: torch.Tensor,
    value_linear: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
) -> None:
    """Scatter linear KV rows into ``key_cache`` / ``value_cache`` at ``slot_mapping`` (in place).

    Skips indices with ``slot_mapping[t] < 0``. Rows with duplicated slots overwrite in token order
    (last writer wins)—callers must avoid unintended sharing.
    """
    device, dtype = tensor_device_dtype(key_linear)
    ensure_reference_allowed("kv_linear_to_paged")

    if key_linear.shape != value_linear.shape or key_linear.dim() != 3:
        raise ValueError("kv_linear_to_paged: key_linear/value_linear must be 3-D and match shapes")
    tcount = key_linear.shape[0]
    nb, nh, hd, bs = _validate_paged_shapes(key_cache, value_cache, name="kv_linear_to_paged")

    slots = _ensure_slotmapping_vector(slot_mapping, expected_len=tcount, name="kv_linear_to_paged")

    if key_linear.shape[1:] != (nh, hd):
        raise ValueError(
            "kv_linear_to_paged: last dims must match caches [num_kv_heads, head_dim] "
            f"got tail {tuple(key_linear.shape[1:])} vs {(nh, hd)}",
        )
    parties = (
        key_linear,
        value_linear,
        key_cache,
        value_cache,
        slots,
    )
    dev = key_cache.device
    if any(x.device != dev for x in parties):
        raise ValueError("kv_linear_to_paged: all tensors must be on same device")

    valid = slots >= 0
    _check_slots_in_bounds(slots, valid, nb, bs, "kv_linear_to_paged")

    if _try_native_kv_linear_to_paged(key_linear, value_linear, key_cache, value_cache, slots):
        emit(
            DispatchRecord(
                "kv_linear_to_paged",
                "ctypes-musa-native",
                device,
                str(dtype),
                "native Phase PKV linear→paged scatter opt-in",
            )
        )
        emit_vendor_planning_note("kv_linear_to_paged")
        return

    emit(
        DispatchRecord(
            "kv_linear_to_paged",
            "torch-reference",
            device,
            str(dtype),
            _native_kv_fallback_reason("kv_linear_to_paged"),
        )
    )
    emit_vendor_planning_note("kv_linear_to_paged")

    _kv_linear_to_paged_scatter(
        key_linear, value_linear, key_cache, value_cache, slots, nb, nh, hd, bs, tcount
    )


def kv_linear_to_paged_multi_layer(
    key_linear: torch.Tensor,
    value_linear: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
) -> None:
    """Scatter ``[L, T, kv_heads, head_dim]`` linear KV into stacked paged caches (in place)."""
    device, dtype = tensor_device_dtype(key_linear)
    ensure_reference_allowed("kv_linear_to_paged_multi_layer")

    if key_linear.shape != value_linear.shape or key_linear.dim() != 4:
        raise ValueError(
            "kv_linear_to_paged_multi_layer: key_linear/value_linear must be 4-D "
            "[num_layers, num_tokens, num_kv_heads, head_dim] and match",
        )
    nl, tcount, nh_lin, hd_lin = map(int, key_linear.shape)
    nb, nh, hd, bs = _validate_stacked_paged_shapes(key_cache, value_cache, name="kv_linear_to_paged_multi_layer")[1:]

    if nh_lin != nh or hd_lin != hd:
        raise ValueError(
            "kv_linear_to_paged_multi_layer: linear tail dims must match caches "
            f"got {(nh_lin, hd_lin)} vs cache {(nh, hd)}",
        )
    if key_cache.shape[0] != nl:
        raise ValueError(
            "kv_linear_to_paged_multi_layer: layer count mismatch "
            f"linear L={nl} vs cache L={key_cache.shape[0]}",
        )

    slots = _ensure_slotmapping_vector(slot_mapping, expected_len=tcount, name="kv_linear_to_paged_multi_layer")
    parties = (key_linear, value_linear, key_cache, value_cache, slots)
    dev = key_cache.device
    if any(x.device != dev for x in parties):
        raise ValueError("kv_linear_to_paged_multi_layer: all tensors must be on same device")

    valid = slots >= 0
    _check_slots_in_bounds(slots, valid, nb, bs, "kv_linear_to_paged_multi_layer")

    if _try_native_kv_linear_to_paged_multi_layer(key_linear, value_linear, key_cache, value_cache, slots):
        emit(
            DispatchRecord(
                "kv_linear_to_paged_multi_layer",
                "ctypes-musa-native",
                device,
                str(dtype),
                "native Phase PKV linear→paged scatter opt-in (multi-layer)",
            )
        )
        emit_vendor_planning_note("kv_linear_to_paged_multi_layer")
        return

    emit(
        DispatchRecord(
            "kv_linear_to_paged_multi_layer",
            "torch-reference",
            device,
            str(dtype),
            _native_kv_fallback_reason("kv_linear_to_paged_multi_layer"),
        )
    )
    emit_vendor_planning_note("kv_linear_to_paged_multi_layer")

    for layer in range(nl):
        _kv_linear_to_paged_scatter(
            key_linear[layer],
            value_linear[layer],
            key_cache[layer],
            value_cache[layer],
            slots,
            nb,
            nh,
            hd,
            bs,
            tcount,
        )


def kv_paged_fp8_to_merged_linear(
    key_cache_fp8: torch.Tensor,
    value_cache_fp8: torch.Tensor,
    slot_mapping: torch.Tensor,
    k_scale: torch.Tensor | float,
    v_scale: torch.Tensor | float,
    *,
    out_dtype: torch.dtype = torch.bfloat16,
    fp8_format: Literal["e4m3", "e5m2"] = "e4m3",
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    device, dtype = tensor_device_dtype(key_cache_fp8)
    ensure_reference_allowed("kv_paged_fp8_to_merged_linear")
    _, nh, hd, _ = _validate_paged_shapes(key_cache_fp8, value_cache_fp8, name="kv_paged_fp8_to_merged_linear")
    tcount = int(slot_mapping.numel())
    if out is not None:
        if out.shape != (tcount, 2, nh, hd):
            raise ValueError("kv_paged_fp8_to_merged_linear: out must be [T, 2, kv_heads, head_dim]")
        if out.device != key_cache_fp8.device or out.dtype is not out_dtype:
            raise ValueError("kv_paged_fp8_to_merged_linear: out dtype/device mismatch")
    out_work = out if out is not None else torch.empty(tcount, 2, nh, hd, dtype=out_dtype, device=key_cache_fp8.device)
    k_out, v_out = kv_paged_fp8_to_linear(key_cache_fp8, value_cache_fp8, slot_mapping, k_scale, v_scale, out_dtype=out_dtype, fp8_format=fp8_format)
    out_work[:, 0].copy_(k_out)
    out_work[:, 1].copy_(v_out)
    backend = "ctypes-musa-native-fp8" if _native_fp8_kv_cache_requested() and _is_musa_native_fp8_cache_tensor(key_cache_fp8) else "torch-reference"
    emit(DispatchRecord("kv_paged_fp8_to_merged_linear", backend, device, str(dtype), f"merged FP8 {fp8_format} dequantize gather"))
    return out_work


def kv_merged_linear_to_paged_fp8(
    merged_linear: torch.Tensor,
    key_cache_fp8: torch.Tensor,
    value_cache_fp8: torch.Tensor,
    slot_mapping: torch.Tensor,
    k_scale: torch.Tensor | float,
    v_scale: torch.Tensor | float,
    *,
    fp8_format: Literal["e4m3", "e5m2"] = "e4m3",
) -> None:
    device, dtype = tensor_device_dtype(merged_linear)
    ensure_reference_allowed("kv_merged_linear_to_paged_fp8")
    if merged_linear.dim() != 4 or merged_linear.shape[1] != 2:
        raise ValueError("kv_merged_linear_to_paged_fp8: merged_linear must be [T, 2, kv_heads, head_dim]")
    kv_linear_to_paged_fp8(merged_linear[:, 0], merged_linear[:, 1], key_cache_fp8, value_cache_fp8, slot_mapping, k_scale, v_scale, fp8_format=fp8_format)
    backend = "ctypes-musa-native-fp8" if _native_fp8_kv_cache_requested() and _is_musa_native_fp8_cache_tensor(key_cache_fp8) else "torch-reference"
    emit(DispatchRecord("kv_merged_linear_to_paged_fp8", backend, device, str(dtype), f"merged FP8 {fp8_format} quantize scatter"))


def kv_paged_fp8_to_merged_linear_multi_layer(
    key_cache_fp8: torch.Tensor,
    value_cache_fp8: torch.Tensor,
    slot_mapping: torch.Tensor,
    k_scale: torch.Tensor | float,
    v_scale: torch.Tensor | float,
    *,
    out_dtype: torch.dtype = torch.bfloat16,
    fp8_format: Literal["e4m3", "e5m2"] = "e4m3",
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    device, dtype = tensor_device_dtype(key_cache_fp8)
    ensure_reference_allowed("kv_paged_fp8_to_merged_linear_multi_layer")
    nl, _, nh, hd, _ = _validate_stacked_paged_shapes(key_cache_fp8, value_cache_fp8, name="kv_paged_fp8_to_merged_linear_multi_layer")
    tcount = int(slot_mapping.numel())
    if out is not None:
        if out.shape != (nl, tcount, 2, nh, hd):
            raise ValueError("kv_paged_fp8_to_merged_linear_multi_layer: out must be [L, T, 2, kv_heads, head_dim]")
        if out.device != key_cache_fp8.device or out.dtype is not out_dtype:
            raise ValueError("kv_paged_fp8_to_merged_linear_multi_layer: out dtype/device mismatch")
    out_work = out if out is not None else torch.empty(nl, tcount, 2, nh, hd, dtype=out_dtype, device=key_cache_fp8.device)
    k_out, v_out = kv_paged_fp8_to_linear_multi_layer(key_cache_fp8, value_cache_fp8, slot_mapping, k_scale, v_scale, out_dtype=out_dtype, fp8_format=fp8_format)
    out_work[:, :, 0].copy_(k_out)
    out_work[:, :, 1].copy_(v_out)
    backend = "ctypes-musa-native-fp8" if _native_fp8_kv_cache_requested() and _is_musa_native_fp8_cache_tensor(key_cache_fp8) else "torch-reference"
    emit(DispatchRecord("kv_paged_fp8_to_merged_linear_multi_layer", backend, device, str(dtype), f"merged multi-layer FP8 {fp8_format} dequantize gather"))
    return out_work


def kv_merged_linear_to_paged_fp8_multi_layer(
    merged_linear: torch.Tensor,
    key_cache_fp8: torch.Tensor,
    value_cache_fp8: torch.Tensor,
    slot_mapping: torch.Tensor,
    k_scale: torch.Tensor | float,
    v_scale: torch.Tensor | float,
    *,
    fp8_format: Literal["e4m3", "e5m2"] = "e4m3",
) -> None:
    device, dtype = tensor_device_dtype(merged_linear)
    ensure_reference_allowed("kv_merged_linear_to_paged_fp8_multi_layer")
    if merged_linear.dim() != 5 or merged_linear.shape[2] != 2:
        raise ValueError("kv_merged_linear_to_paged_fp8_multi_layer: merged_linear must be [L, T, 2, kv_heads, head_dim]")
    kv_linear_to_paged_fp8_multi_layer(merged_linear[:, :, 0], merged_linear[:, :, 1], key_cache_fp8, value_cache_fp8, slot_mapping, k_scale, v_scale, fp8_format=fp8_format)
    backend = "ctypes-musa-native-fp8" if _native_fp8_kv_cache_requested() and _is_musa_native_fp8_cache_tensor(key_cache_fp8) else "torch-reference"
    emit(DispatchRecord("kv_merged_linear_to_paged_fp8_multi_layer", backend, device, str(dtype), f"merged multi-layer FP8 {fp8_format} quantize scatter"))


def kv_paged_to_merged_linear(
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    *,
    fill_skipped: Literal["zeros", "preserve"] = "zeros",
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Gather separate paged K/V caches into merged ``[T, 2, kv_heads, head_dim]`` linear layout."""
    device, dtype = tensor_device_dtype(key_cache)
    ensure_reference_allowed("kv_paged_to_merged_linear")
    nb, nh, hd, bs = _validate_paged_shapes(key_cache, value_cache, name="kv_paged_to_merged_linear")
    tcount = slot_mapping.numel()
    slots = _ensure_slotmapping_vector(slot_mapping, expected_len=tcount, name="kv_paged_to_merged_linear")
    if value_cache.device != key_cache.device or slots.device != key_cache.device:
        raise ValueError("kv_paged_to_merged_linear: all tensors must be on the same device")
    if out is not None:
        if out.shape != (tcount, 2, nh, hd):
            raise ValueError("kv_paged_to_merged_linear: out must be [T, 2, kv_heads, head_dim]")
        if out.device != key_cache.device or out.dtype != key_cache.dtype:
            raise ValueError("kv_paged_to_merged_linear: out dtype/device must match caches")
    if fill_skipped == "preserve" and out is None:
        raise ValueError("kv_paged_to_merged_linear: fill_skipped=preserve requires out")
    valid = slots >= 0
    _check_slots_in_bounds(slots, valid, nb, bs, "kv_paged_to_merged_linear")
    native = _try_native_kv_paged_to_merged_linear(
        key_cache,
        value_cache,
        slots,
        tcount,
        nh,
        hd,
        bs,
        fill_skipped=fill_skipped,
        out=out,
    )
    if native is not None:
        emit(
            DispatchRecord(
                "kv_paged_to_merged_linear",
                "ctypes-musa-native",
                device,
                str(dtype),
                "native Phase PKV paged→merged-linear gather opt-in",
            )
        )
        emit_vendor_planning_note("kv_paged_to_merged_linear")
        return native
    if out is not None:
        k_lin, v_lin = kv_paged_to_linear(
            key_cache,
            value_cache,
            slot_mapping,
            fill_skipped=fill_skipped,
            out_k=out[:, 0],
            out_v=out[:, 1],
        )
        if k_lin.data_ptr() != out[:, 0].data_ptr() or v_lin.data_ptr() != out[:, 1].data_ptr():
            out[:, 0].copy_(k_lin)
            out[:, 1].copy_(v_lin)
        return out
    k_lin, v_lin = kv_paged_to_linear(key_cache, value_cache, slot_mapping, fill_skipped=fill_skipped)
    return torch.stack((k_lin, v_lin), dim=1)


def kv_merged_linear_to_paged(
    merged_linear: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
) -> None:
    """Scatter merged ``[T, 2, kv_heads, head_dim]`` linear KV into separate paged caches."""
    device, dtype = tensor_device_dtype(merged_linear)
    ensure_reference_allowed("kv_merged_linear_to_paged")
    if merged_linear.dim() != 4 or merged_linear.shape[1] != 2:
        raise ValueError("kv_merged_linear_to_paged: merged_linear must be [T, 2, kv_heads, head_dim]")
    tcount = int(merged_linear.shape[0])
    nb, nh, hd, bs = _validate_paged_shapes(key_cache, value_cache, name="kv_merged_linear_to_paged")
    if tuple(merged_linear.shape[2:]) != (nh, hd):
        raise ValueError("kv_merged_linear_to_paged: merged tail dims must match caches")
    slots = _ensure_slotmapping_vector(slot_mapping, expected_len=tcount, name="kv_merged_linear_to_paged")
    parties = (merged_linear, key_cache, value_cache, slots)
    if any(x.device != key_cache.device for x in parties):
        raise ValueError("kv_merged_linear_to_paged: all tensors must be on same device")
    valid = slots >= 0
    _check_slots_in_bounds(slots, valid, nb, bs, "kv_merged_linear_to_paged")
    if _try_native_kv_merged_linear_to_paged(merged_linear, key_cache, value_cache, slots):
        emit(
            DispatchRecord(
                "kv_merged_linear_to_paged",
                "ctypes-musa-native",
                device,
                str(dtype),
                "native Phase PKV merged-linear→paged scatter opt-in",
            )
        )
        emit_vendor_planning_note("kv_merged_linear_to_paged")
        return
    kv_linear_to_paged(merged_linear[:, 0], merged_linear[:, 1], key_cache, value_cache, slot_mapping)


def kv_paged_to_merged_linear_multi_layer(
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    *,
    fill_skipped: Literal["zeros", "preserve"] = "zeros",
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Gather stacked paged K/V caches into merged ``[L, T, 2, kv_heads, head_dim]`` layout."""
    device, dtype = tensor_device_dtype(key_cache)
    ensure_reference_allowed("kv_paged_to_merged_linear_multi_layer")
    nl, nb, nh, hd, bs = _validate_stacked_paged_shapes(
        key_cache,
        value_cache,
        name="kv_paged_to_merged_linear_multi_layer",
    )
    tcount = slot_mapping.numel()
    slots = _ensure_slotmapping_vector(slot_mapping, expected_len=tcount, name="kv_paged_to_merged_linear_multi_layer")
    if value_cache.device != key_cache.device or slots.device != key_cache.device:
        raise ValueError("kv_paged_to_merged_linear_multi_layer: all tensors must be on the same device")
    if out is not None:
        if out.shape != (nl, tcount, 2, nh, hd):
            raise ValueError("kv_paged_to_merged_linear_multi_layer: out must be [L, T, 2, kv_heads, head_dim]")
        if out.device != key_cache.device or out.dtype != key_cache.dtype:
            raise ValueError("kv_paged_to_merged_linear_multi_layer: out dtype/device must match caches")
    if fill_skipped == "preserve" and out is None:
        raise ValueError("kv_paged_to_merged_linear_multi_layer: fill_skipped=preserve requires out")
    valid = slots >= 0
    _check_slots_in_bounds(slots, valid, nb, bs, "kv_paged_to_merged_linear_multi_layer")
    native = _try_native_kv_paged_to_merged_linear_multi_layer(
        key_cache,
        value_cache,
        slots,
        tcount,
        nl,
        nh,
        hd,
        bs,
        fill_skipped=fill_skipped,
        out=out,
    )
    if native is not None:
        emit(
            DispatchRecord(
                "kv_paged_to_merged_linear_multi_layer",
                "ctypes-musa-native",
                device,
                str(dtype),
                "native Phase PKV paged→merged-linear gather opt-in (multi-layer)",
            )
        )
        emit_vendor_planning_note("kv_paged_to_merged_linear_multi_layer")
        return native
    if out is not None:
        k_lin, v_lin = kv_paged_to_linear_multi_layer(
            key_cache,
            value_cache,
            slot_mapping,
            fill_skipped=fill_skipped,
            out_k=out[:, :, 0],
            out_v=out[:, :, 1],
        )
        if k_lin.data_ptr() != out[:, :, 0].data_ptr() or v_lin.data_ptr() != out[:, :, 1].data_ptr():
            out[:, :, 0].copy_(k_lin)
            out[:, :, 1].copy_(v_lin)
        return out
    k_lin, v_lin = kv_paged_to_linear_multi_layer(key_cache, value_cache, slot_mapping, fill_skipped=fill_skipped)
    return torch.stack((k_lin, v_lin), dim=2)


def kv_merged_linear_to_paged_multi_layer(
    merged_linear: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
) -> None:
    """Scatter merged ``[L, T, 2, kv_heads, head_dim]`` linear KV into stacked paged caches."""
    device, dtype = tensor_device_dtype(merged_linear)
    ensure_reference_allowed("kv_merged_linear_to_paged_multi_layer")
    if merged_linear.dim() != 5 or merged_linear.shape[2] != 2:
        raise ValueError(
            "kv_merged_linear_to_paged_multi_layer: merged_linear must be [L, T, 2, kv_heads, head_dim]",
        )
    nl, tcount, _, nh_lin, hd_lin = map(int, merged_linear.shape)
    nb, nh, hd, bs = _validate_stacked_paged_shapes(
        key_cache,
        value_cache,
        name="kv_merged_linear_to_paged_multi_layer",
    )[1:]
    if nh_lin != nh or hd_lin != hd or key_cache.shape[0] != nl:
        raise ValueError("kv_merged_linear_to_paged_multi_layer: merged dims must match caches")
    slots = _ensure_slotmapping_vector(slot_mapping, expected_len=tcount, name="kv_merged_linear_to_paged_multi_layer")
    parties = (merged_linear, key_cache, value_cache, slots)
    if any(x.device != key_cache.device for x in parties):
        raise ValueError("kv_merged_linear_to_paged_multi_layer: all tensors must be on same device")
    valid = slots >= 0
    _check_slots_in_bounds(slots, valid, nb, bs, "kv_merged_linear_to_paged_multi_layer")
    if _try_native_kv_merged_linear_to_paged_multi_layer(merged_linear, key_cache, value_cache, slots):
        emit(
            DispatchRecord(
                "kv_merged_linear_to_paged_multi_layer",
                "ctypes-musa-native",
                device,
                str(dtype),
                "native Phase PKV merged-linear→paged scatter opt-in (multi-layer)",
            )
        )
        emit_vendor_planning_note("kv_merged_linear_to_paged_multi_layer")
        return
    kv_linear_to_paged_multi_layer(merged_linear[:, :, 0], merged_linear[:, :, 1], key_cache, value_cache, slot_mapping)


def _validate_flat_paged_shapes(
    first_cache: torch.Tensor,
    second_cache: torch.Tensor,
    *,
    name: str,
) -> tuple[int, int, int, int]:
    if first_cache.dim() != 3 or second_cache.dim() != 3:
        raise ValueError(f"{name}: caches must be 3-D [blocks, block_size, hidden_dim]")
    if first_cache.shape[:2] != second_cache.shape[:2]:
        raise ValueError(f"{name}: caches must share [blocks, block_size]")
    nb, bs, first_dim = map(int, first_cache.shape)
    second_dim = int(second_cache.shape[2])
    if nb <= 0 or bs <= 0 or first_dim <= 0 or second_dim <= 0:
        raise ValueError(f"{name}: all cache extents must be positive")
    return nb, bs, first_dim, second_dim


def _flat_paged_to_linear_compute(
    first_cache: torch.Tensor,
    second_cache: torch.Tensor,
    slots_long: torch.Tensor,
    valid: torch.Tensor,
    tcount: int,
    bs: int,
    *,
    first_out: torch.Tensor | None,
    second_out: torch.Tensor | None,
    name: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    safe_slots = torch.where(valid, slots_long, torch.zeros_like(slots_long))
    bid = torch.div(safe_slots, bs, rounding_mode="floor")
    off = safe_slots - bid * bs
    first = first_cache[bid, off]
    second = second_cache[bid, off]
    first = torch.where(valid.view(tcount, 1), first, torch.zeros_like(first))
    second = torch.where(valid.view(tcount, 1), second, torch.zeros_like(second))
    if first_out is not None:
        if first_out.shape != first.shape:
            raise ValueError(f"{name}: first_out shape mismatch {tuple(first_out.shape)}")
        first_out.copy_(first)
        first = first_out
    if second_out is not None:
        if second_out.shape != second.shape:
            raise ValueError(f"{name}: second_out shape mismatch {tuple(second_out.shape)}")
        second_out.copy_(second)
        second = second_out
    return first, second


def _flat_linear_to_paged_scatter(
    first_linear: torch.Tensor,
    second_linear: torch.Tensor,
    first_cache: torch.Tensor,
    second_cache: torch.Tensor,
    slots_long: torch.Tensor,
    nb: int,
    bs: int,
    tcount: int,
    *,
    name: str,
) -> None:
    for t in range(tcount):
        s = int(slots_long[t].item())
        if s < 0:
            continue
        b, off = divmod(s, bs)
        if not (0 <= b < nb and 0 <= off < bs):
            raise RuntimeError(f"{name}: invalid slot mapping {s} at token {t}")
        first_cache[b, off].copy_(first_linear[t])
        second_cache[b, off].copy_(second_linear[t])


def kv_paged_to_mla_linear(
    kv_cache: torch.Tensor,
    k_pe_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    *,
    out_kv: torch.Tensor | None = None,
    out_k_pe: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Gather MLA-style flat paged caches into token-major asymmetric linear buffers."""
    device, dtype = tensor_device_dtype(kv_cache)
    ensure_reference_allowed("kv_paged_to_mla_linear")
    nb, bs, _, _ = _validate_flat_paged_shapes(kv_cache, k_pe_cache, name="kv_paged_to_mla_linear")
    tcount = slot_mapping.numel()
    slots = _ensure_slotmapping_vector(slot_mapping, expected_len=tcount, name="kv_paged_to_mla_linear")
    if k_pe_cache.device != kv_cache.device or slots.device != kv_cache.device:
        raise ValueError("kv_paged_to_mla_linear: all tensors must be on the same device")
    valid = slots >= 0
    _check_slots_in_bounds(slots, valid, nb, bs, "kv_paged_to_mla_linear")
    native = _try_native_kv_paged_to_mla_linear(
        kv_cache,
        k_pe_cache,
        slots,
        tcount,
        nb,
        bs,
        int(kv_cache.shape[2]),
        int(k_pe_cache.shape[2]),
        out_kv=out_kv,
        out_k_pe=out_k_pe,
    )
    if native is not None:
        emit(
            DispatchRecord(
                "kv_paged_to_mla_linear",
                "ctypes-musa-native",
                device,
                str(dtype),
                "native Phase PKV MLA/DSA asymmetric flat-cache gather opt-in",
            )
        )
        emit_vendor_planning_note("kv_paged_to_mla_linear")
        return native
    emit(
        DispatchRecord(
            "kv_paged_to_mla_linear",
            "torch-reference",
            device,
            str(dtype),
            "Phase PKV MLA/DSA asymmetric flat-cache gather reference",
        )
    )
    emit_vendor_planning_note("kv_paged_to_mla_linear")
    return _flat_paged_to_linear_compute(
        kv_cache,
        k_pe_cache,
        slots,
        valid,
        tcount,
        bs,
        first_out=out_kv,
        second_out=out_k_pe,
        name="kv_paged_to_mla_linear",
    )


def kv_mla_linear_to_paged(
    kv_linear: torch.Tensor,
    k_pe_linear: torch.Tensor,
    kv_cache: torch.Tensor,
    k_pe_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
) -> None:
    """Scatter token-major MLA-style asymmetric linear buffers into flat paged caches."""
    device, dtype = tensor_device_dtype(kv_linear)
    ensure_reference_allowed("kv_mla_linear_to_paged")
    if kv_linear.dim() != 2 or k_pe_linear.dim() != 2:
        raise ValueError("kv_mla_linear_to_paged: linear buffers must be [T, hidden_dim]")
    if kv_linear.shape[0] != k_pe_linear.shape[0]:
        raise ValueError("kv_mla_linear_to_paged: linear buffers must share token count")
    tcount = int(kv_linear.shape[0])
    nb, bs, kv_dim, pe_dim = _validate_flat_paged_shapes(kv_cache, k_pe_cache, name="kv_mla_linear_to_paged")
    if tuple(kv_linear.shape[1:]) != (kv_dim,) or tuple(k_pe_linear.shape[1:]) != (pe_dim,):
        raise ValueError("kv_mla_linear_to_paged: linear hidden dims must match caches")
    slots = _ensure_slotmapping_vector(slot_mapping, expected_len=tcount, name="kv_mla_linear_to_paged")
    parties = (kv_linear, k_pe_linear, kv_cache, k_pe_cache, slots)
    if any(x.device != kv_cache.device for x in parties):
        raise ValueError("kv_mla_linear_to_paged: all tensors must be on same device")
    valid = slots >= 0
    _check_slots_in_bounds(slots, valid, nb, bs, "kv_mla_linear_to_paged")
    if _try_native_kv_mla_linear_to_paged(kv_linear, k_pe_linear, kv_cache, k_pe_cache, slots):
        emit(
            DispatchRecord(
                "kv_mla_linear_to_paged",
                "ctypes-musa-native",
                device,
                str(dtype),
                "native Phase PKV MLA/DSA asymmetric flat-cache scatter opt-in",
            )
        )
        emit_vendor_planning_note("kv_mla_linear_to_paged")
        return
    emit(
        DispatchRecord(
            "kv_mla_linear_to_paged",
            "torch-reference",
            device,
            str(dtype),
            "Phase PKV MLA/DSA asymmetric flat-cache scatter reference",
        )
    )
    emit_vendor_planning_note("kv_mla_linear_to_paged")
    _flat_linear_to_paged_scatter(
        kv_linear,
        k_pe_linear,
        kv_cache,
        k_pe_cache,
        slots,
        nb,
        bs,
        tcount,
        name="kv_mla_linear_to_paged",
    )


def _validate_stacked_mla_shapes(
    kv_cache: torch.Tensor,
    k_pe_cache: torch.Tensor,
    *,
    name: str,
) -> tuple[int, int, int, int, int]:
    if kv_cache.dim() != 4 or k_pe_cache.dim() != 4:
        raise ValueError(f"{name}: caches must be [layers, blocks, block_size, hidden_dim]")
    if tuple(kv_cache.shape[:3]) != tuple(k_pe_cache.shape[:3]):
        raise ValueError(f"{name}: caches must share [layers, blocks, block_size]")
    nl, nb, bs, kv_dim = map(int, kv_cache.shape)
    pe_dim = int(k_pe_cache.shape[3])
    if nl <= 0 or nb <= 0 or bs <= 0 or kv_dim <= 0 or pe_dim <= 0:
        raise ValueError(f"{name}: all cache extents must be positive")
    return nl, nb, bs, kv_dim, pe_dim


def kv_paged_to_mla_linear_multi_layer(
    kv_cache: torch.Tensor,
    k_pe_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Gather stacked asymmetric MLA caches into layer-major linear buffers."""
    device, dtype = tensor_device_dtype(kv_cache)
    ensure_reference_allowed("kv_paged_to_mla_linear_multi_layer")
    nl, _, _, kv_dim, pe_dim = _validate_stacked_mla_shapes(
        kv_cache,
        k_pe_cache,
        name="kv_paged_to_mla_linear_multi_layer",
    )
    if k_pe_cache.dtype != kv_cache.dtype or k_pe_cache.device != kv_cache.device:
        raise ValueError("kv_paged_to_mla_linear_multi_layer: caches must share dtype and device")
    slots = _ensure_slotmapping_vector(
        slot_mapping,
        expected_len=int(slot_mapping.numel()),
        name="kv_paged_to_mla_linear_multi_layer",
    )
    if slots.device != kv_cache.device:
        raise ValueError("kv_paged_to_mla_linear_multi_layer: slot_mapping must share device")
    native = _try_native_kv_paged_to_mla_linear_multi_layer(
        kv_cache,
        k_pe_cache,
        slots,
        nl,
        kv_dim,
        pe_dim,
    )
    if native is not None:
        emit(
            DispatchRecord(
                "kv_paged_to_mla_linear_multi_layer",
                "ctypes-musa-native",
                device,
                str(dtype),
                "LMCache native multi-layer MLA gather",
            )
        )
        return native
    kv_out = torch.empty(
        (nl, slots.numel(), kv_dim),
        dtype=kv_cache.dtype,
        device=kv_cache.device,
    )
    pe_out = torch.empty(
        (nl, slots.numel(), pe_dim),
        dtype=k_pe_cache.dtype,
        device=k_pe_cache.device,
    )
    for layer in range(nl):
        layer_kv, layer_pe = kv_paged_to_mla_linear(
            kv_cache[layer],
            k_pe_cache[layer],
            slots,
        )
        kv_out[layer].copy_(layer_kv)
        pe_out[layer].copy_(layer_pe)
    emit(
        DispatchRecord(
            "kv_paged_to_mla_linear_multi_layer",
            "torch-reference",
            device,
            str(dtype),
            "stacked MLA/DSA asymmetric flat-cache gather reference",
        )
    )
    return kv_out, pe_out


def kv_mla_linear_to_paged_multi_layer(
    kv_linear: torch.Tensor,
    k_pe_linear: torch.Tensor,
    kv_cache: torch.Tensor,
    k_pe_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
) -> bool:
    """Scatter layer-major asymmetric MLA buffers into stacked flat caches."""
    device, dtype = tensor_device_dtype(kv_linear)
    ensure_reference_allowed("kv_mla_linear_to_paged_multi_layer")
    nl, _, _, kv_dim, pe_dim = _validate_stacked_mla_shapes(
        kv_cache,
        k_pe_cache,
        name="kv_mla_linear_to_paged_multi_layer",
    )
    if kv_linear.dim() != 3 or k_pe_linear.dim() != 3:
        raise ValueError("kv_mla_linear_to_paged_multi_layer: linear buffers must be [layers, tokens, hidden]")
    if kv_linear.shape[0] != nl or k_pe_linear.shape[0] != nl:
        raise ValueError("kv_mla_linear_to_paged_multi_layer: layer counts must match caches")
    if kv_linear.shape[1] != k_pe_linear.shape[1]:
        raise ValueError("kv_mla_linear_to_paged_multi_layer: token counts must match")
    if tuple(kv_linear.shape[2:]) != (kv_dim,) or tuple(k_pe_linear.shape[2:]) != (pe_dim,):
        raise ValueError("kv_mla_linear_to_paged_multi_layer: hidden dims must match caches")
    if kv_linear.dtype != kv_cache.dtype or k_pe_linear.dtype != k_pe_cache.dtype:
        raise ValueError("kv_mla_linear_to_paged_multi_layer: dtypes must match caches")
    if any(
        tensor.device != kv_cache.device
        for tensor in (kv_linear, k_pe_linear, kv_cache, k_pe_cache, slot_mapping)
    ):
        raise ValueError("kv_mla_linear_to_paged_multi_layer: all tensors must share device")
    slots = slot_mapping.to(torch.long)
    if _try_native_kv_mla_linear_to_paged_multi_layer(
        kv_linear,
        k_pe_linear,
        kv_cache,
        k_pe_cache,
        slots,
    ):
        emit(
            DispatchRecord(
                "kv_mla_linear_to_paged_multi_layer",
                "ctypes-musa-native",
                device,
                str(dtype),
                "LMCache native multi-layer MLA scatter",
            )
        )
        return True
    for layer in range(nl):
        kv_mla_linear_to_paged(
            kv_linear[layer],
            k_pe_linear[layer],
            kv_cache[layer],
            k_pe_cache[layer],
            slot_mapping,
        )
    emit(
        DispatchRecord(
            "kv_mla_linear_to_paged_multi_layer",
            "torch-reference",
            device,
            str(dtype),
            "stacked MLA/DSA asymmetric flat-cache scatter reference",
        )
    )
    return True


__all__ = [
    "build_native_kv_transfer",
    "build_native_fp8_kv_cache",
    "is_native_kv_transfer_available",
    "is_native_fp8_kv_cache_available",
    "is_native_fp8_kv_vectorized_available",
    "native_kv_transfer_abi_version",
    "native_fp8_kv_cache_abi_version",
    "kv_cachegen_decode",
    "kv_cachegen_encode",
    "kv_copy_paged_blocks",
    "kv_copy_paged_blocks_fp8",
    "kv_load_and_reshape_flash",
    "kv_load_and_dequantize_fp8_flash",
    "kv_paged_to_linear",
    "kv_paged_fp8_to_linear",
    "kv_paged_fp8_to_linear_multi_layer",
    "kv_paged_fp8_to_merged_linear",
    "kv_paged_fp8_to_merged_linear_multi_layer",
    "kv_paged_to_linear_multi_layer",
    "kv_linear_to_paged",
    "kv_linear_to_paged_fp8",
    "kv_linear_to_paged_fp8_multi_layer",
    "kv_merged_linear_to_paged_fp8",
    "kv_merged_linear_to_paged_fp8_multi_layer",
    "kv_linear_to_paged_multi_layer",
    "kv_paged_to_merged_linear",
    "kv_paged_to_merged_linear_multi_layer",
    "kv_merged_linear_to_paged",
    "kv_merged_linear_to_paged_multi_layer",
    "kv_paged_to_mla_linear",
    "kv_mla_linear_to_paged",
    "kv_paged_to_mla_linear_multi_layer",
    "kv_mla_linear_to_paged_multi_layer",
    "kv_copy_paged_rows",
    "kv_copy_paged_rows_layer_table",
    "kv_copy_paged_rows_multi_pool",
    "InterleavedFp8Layout",
    "PoolSpec",
    "clear_dsv4_ptr_table_cache",
]


# ---------------------------------------------------------------------------
# DSV4 flat byte-row gather/scatter (REQ-1/2/3, kuae-ssd req3musaiter-dsv4)
#
# DSV4-Flash host pools store KV as opaque uint8 byte rows
# (kv_buffer[L, pages, item_bytes]) with interleaved K/V/fp8-scale regions.
# These ops copy whole page rows by index, with no KV semantics.
# ---------------------------------------------------------------------------


def _is_musa_uint8_rows_tensor(tensor: torch.Tensor) -> bool:
    return (
        tensor.dtype is torch.uint8
        and tensor.dim() == 2
        and tensor.is_contiguous()
        and str(tensor.device).split(":")[0] == "musa"
    )


def _dsv4_native_indices(indices: torch.Tensor) -> torch.Tensor | None:
    if str(indices.device).split(":")[0] != "musa":
        return None
    if indices.dtype is torch.int64 and indices.is_contiguous():
        return indices
    return indices.to(device=indices.device, dtype=torch.long).contiguous()


_DSV4_NATIVE_FNS: dict = {}


def _dsv4_native_fn(name: str, argtypes: list):
    """Cache ctypes function objects with argtypes bound once (per-call
    argtypes assignment costs ~17us and dominates small transfers)."""
    fn = _DSV4_NATIVE_FNS.get(name)
    if fn is not None:
        return fn
    lib = load_native_kv_transfer()
    if not has_symbol(lib, name):
        return None
    fn = getattr(lib.handle, name)
    fn.argtypes = argtypes
    fn.restype = None
    _DSV4_NATIVE_FNS[name] = fn
    return fn


def _try_native_kv_copy_paged_rows(
    src: torch.Tensor,
    dst: torch.Tensor,
    src_indices: torch.Tensor,
    dst_indices: torch.Tensor,
) -> bool:
    if not _native_kv_transfer_requested():
        return False
    if not (_is_musa_uint8_rows_tensor(src) and _is_musa_uint8_rows_tensor(dst)):
        return False
    src_rows = _dsv4_native_indices(src_indices)
    dst_rows = _dsv4_native_indices(dst_indices)
    if src_rows is None or dst_rows is None:
        return False
    try:
        fn = _dsv4_native_fn(
            "musa_aiter_kv_copy_paged_rows_u8",
            [
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_longlong,
                ctypes.c_longlong,
                ctypes.c_longlong,
                ctypes.c_longlong,
                ctypes.c_void_p,
            ],
        )
        if fn is None:
            return False
    except Exception:
        return False
    fn(
        ctypes.c_void_p(src.data_ptr()),
        ctypes.c_void_p(dst.data_ptr()),
        ctypes.c_void_p(src_rows.data_ptr()),
        ctypes.c_void_p(dst_rows.data_ptr()),
        ctypes.c_longlong(src_rows.numel()),
        ctypes.c_longlong(src.shape[1]),
        ctypes.c_longlong(src.shape[0]),
        ctypes.c_longlong(dst.shape[0]),
        _current_musa_stream_ptr(),
    )
    return True


# P1-3: ptr-table cache. Host pool ptr tables are invariant for the pool lifetime.
# Cache them keyed by (tuple of tensor ids, device) to avoid per-call H2D upload.
_DSV4_PTR_TABLE_CACHE: dict[tuple[int, int, str], tuple[torch.Tensor, torch.Tensor]] = {}


def _dsv4_ptr_table_cache_key(src_ptrs, dst_ptrs, device: str):
    # Use object identity (id()) because the same pool tensors are reused.
    src_key = tuple(id(t) for t in src_ptrs)
    dst_key = tuple(id(t) for t in dst_ptrs)
    return (src_key, dst_key, device)


def _dsv4_get_or_create_ptr_table(src_ptrs, dst_ptrs, device):
    key = _dsv4_ptr_table_cache_key(src_ptrs, dst_ptrs, str(device))
    if key in _DSV4_PTR_TABLE_CACHE:
        return _DSV4_PTR_TABLE_CACHE[key]
    ptr_table_src = torch.tensor([t.data_ptr() for t in src_ptrs], dtype=torch.int64, device=device)
    ptr_table_dst = torch.tensor([t.data_ptr() for t in dst_ptrs], dtype=torch.int64, device=device)
    _DSV4_PTR_TABLE_CACHE[key] = (ptr_table_src, ptr_table_dst)
    # LRU-style cap to avoid unbounded growth if the user cycles many pools.
    if len(_DSV4_PTR_TABLE_CACHE) > 64:
        # Evict first half (oldest insertion-order entries; dict preserves insertion order since 3.7).
        for k in list(_DSV4_PTR_TABLE_CACHE.keys())[:32]:
            _DSV4_PTR_TABLE_CACHE.pop(k, None)
    return ptr_table_src, ptr_table_dst


def clear_dsv4_ptr_table_cache() -> None:
    """Drop cached device-side pointer tables (call if pool tensors are freed/recreated)."""
    _DSV4_PTR_TABLE_CACHE.clear()


def _layers_are_unbind_of_contiguous(src_ptrs, dst_ptrs) -> bool:
    """True when both layer lists are unbind(0) views of contiguous [L,P,IB]
    tensors with uniform stride (the sglang DSV4 pool layout)."""
    src0 = src_ptrs[0]
    dst0 = dst_ptrs[0]
    if src0.stride(0) != src0.shape[1] or dst0.stride(0) != dst0.shape[1]:
        return False
    src_base = src0.data_ptr()
    dst_base = dst0.data_ptr()
    src_stride = src0.stride(0)
    dst_stride = dst0.stride(0)
    for layer, (s, d) in enumerate(zip(src_ptrs, dst_ptrs)):
        if (
            s.data_ptr() != src_base + layer * src_stride
            or d.data_ptr() != dst_base + layer * dst_stride
            or s.shape != src0.shape
            or d.shape != dst0.shape
        ):
            return False
    return True


def _try_native_kv_copy_paged_rows_contig_layers(
    src_ptrs,
    dst_ptrs,
    src_rows: torch.Tensor,
    dst_rows: torch.Tensor,
    device,
) -> bool:
    if not _layers_are_unbind_of_contiguous(src_ptrs, dst_ptrs):
        return False
    try:
        fn = _dsv4_native_fn(
            "musa_aiter_kv_copy_paged_rows_contig_layers_u8",
            [
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_longlong,
                ctypes.c_longlong,
                ctypes.c_longlong,
                ctypes.c_longlong,
                ctypes.c_longlong,
                ctypes.c_void_p,
            ],
        )
        if fn is None:
            return False
    except Exception:
        return False
    fn(
        ctypes.c_void_p(src_ptrs[0].data_ptr()),
        ctypes.c_void_p(dst_ptrs[0].data_ptr()),
        ctypes.c_void_p(src_rows.data_ptr()),
        ctypes.c_void_p(dst_rows.data_ptr()),
        ctypes.c_longlong(src_rows.numel()),
        ctypes.c_longlong(src_ptrs[0].shape[1]),
        ctypes.c_longlong(len(src_ptrs)),
        ctypes.c_longlong(src_ptrs[0].shape[0]),
        ctypes.c_longlong(dst_ptrs[0].shape[0]),
        _current_musa_stream_ptr(),
    )
    return True


def _try_native_kv_copy_paged_rows_layer_table(
    src_ptrs,
    dst_ptrs,
    src_indices: torch.Tensor,
    dst_indices: torch.Tensor,
) -> bool:
    if not _native_kv_transfer_requested():
        return False
    if not all(_is_musa_uint8_rows_tensor(t) for t in list(src_ptrs) + list(dst_ptrs)):
        return False
    src_rows = _dsv4_native_indices(src_indices)
    dst_rows = _dsv4_native_indices(dst_indices)
    if src_rows is None or dst_rows is None:
        return False
    device = src_ptrs[0].device
    # P1'' fast path: layers from unbind(0) of one contiguous [L, P, IB]
    # tensor have uniform stride; use the pointer-free contig kernel.
    if _try_native_kv_copy_paged_rows_contig_layers(src_ptrs, dst_ptrs, src_rows, dst_rows, device):
        return True
    ptr_table_src, ptr_table_dst = _dsv4_get_or_create_ptr_table(src_ptrs, dst_ptrs, device)
    try:
        fn = _dsv4_native_fn(
            "musa_aiter_kv_copy_paged_rows_layer_table_u8",
            [
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_longlong,
                ctypes.c_longlong,
                ctypes.c_longlong,
                ctypes.c_longlong,
                ctypes.c_longlong,
                ctypes.c_void_p,
            ],
        )
        if fn is None:
            return False
    except Exception:
        return False
    fn(
        ctypes.c_void_p(ptr_table_src.data_ptr()),
        ctypes.c_void_p(ptr_table_dst.data_ptr()),
        ctypes.c_void_p(src_rows.data_ptr()),
        ctypes.c_void_p(dst_rows.data_ptr()),
        ctypes.c_longlong(src_rows.numel()),
        ctypes.c_longlong(src_ptrs[0].shape[1]),
        ctypes.c_longlong(len(src_ptrs)),
        ctypes.c_longlong(src_ptrs[0].shape[0]),
        ctypes.c_longlong(dst_ptrs[0].shape[0]),
        _current_musa_stream_ptr(),
    )
    return True

from dataclasses import dataclass  # noqa: E402


@dataclass(frozen=True)
class InterleavedFp8Layout:
    """REQ-2: descriptor for DSV4 interleaved fp8 block-scale page rows.

    A page row is ``[value region][scale region]``; constants must come from
    the pool (e.g. ``c4_kv_pool.bytes_per_page_padded``), never hardcoded.
    """

    value_bytes: int
    scale_bytes: int
    scale_granularity: int

    def __post_init__(self) -> None:
        if self.value_bytes <= 0 or self.scale_bytes <= 0:
            raise ValueError("InterleavedFp8Layout: value_bytes and scale_bytes must be positive")
        if self.scale_granularity <= 0:
            raise ValueError("InterleavedFp8Layout: scale_granularity must be positive")

    @property
    def item_bytes(self) -> int:
        return self.value_bytes + self.scale_bytes

    def as_kv_layout_spec(
        self,
        *,
        num_layers: int,
        num_blocks: int,
        block_size: int,
        dtype_name: str = "uint8",
    ) -> KVLayoutSpec:
        """Return the common DSV4 blocked-scale descriptor for this row layout."""
        return KVLayoutSpec.dsv4(
            num_layers=num_layers,
            num_blocks=num_blocks,
            block_size=block_size,
            row_bytes=self.item_bytes,
            block_stride_bytes=self.item_bytes * block_size,
            dtype_name=dtype_name,
            scale_kind=KVScaleKind.BLOCKED,
            value_bytes=self.value_bytes,
            scale_bytes=self.scale_bytes,
        )


@dataclass
class PoolSpec:
    """REQ-3: one pool of the DSV4 anchor-shared-index multi-pool batch copy.

    ``item_bytes`` is validated against each layer tensor's last dim.
    """

    src_ptrs: list  # Sequence[Tensor uint8 [pages, item_bytes]]
    dst_ptrs: list
    item_bytes: int


def _validate_kv_copy_paged_rows_args(
    src: torch.Tensor,
    dst: torch.Tensor,
    src_indices: torch.Tensor,
    dst_indices: torch.Tensor,
    *,
    name: str,
) -> tuple[int, int, int]:
    if not isinstance(src, torch.Tensor) or not isinstance(dst, torch.Tensor):
        raise TypeError(f"{name}: src and dst must be torch.Tensor")
    if src.dtype is not torch.uint8 or dst.dtype is not torch.uint8:
        raise ValueError(f"{name}: src and dst must be uint8 byte rows, got src={src.dtype}, dst={dst.dtype}")
    if src.dim() != 2 or dst.dim() != 2:
        raise ValueError(f"{name}: src and dst must be [rows, item_bytes]")
    item_bytes = int(src.shape[1])
    if int(dst.shape[1]) != item_bytes:
        raise ValueError(
            f"{name}: src and dst item_bytes must match, got src={item_bytes}, dst={int(dst.shape[1])}"
        )
    if item_bytes <= 0:
        raise ValueError(f"{name}: item_bytes must be positive")
    if not _is_integer_indices(src_indices) or not _is_integer_indices(dst_indices):
        raise ValueError(f"{name}: indices must use an integer dtype")
    if src_indices.numel() != dst_indices.numel():
        raise ValueError(f"{name}: src_indices and dst_indices must have the same length")
    _check_row_indices(src_indices, int(src.shape[0]), f"{name}: src")
    _check_row_indices(dst_indices, int(dst.shape[0]), f"{name}: dst")
    return item_bytes, int(src.shape[0]), int(dst.shape[0])


def _is_integer_indices(indices: torch.Tensor) -> bool:
    return isinstance(indices, torch.Tensor) and indices.dim() == 1 and indices.dtype in (
        torch.int16, torch.int32, torch.int64,
    )


def _check_row_indices(indices: torch.Tensor, num_rows: int, name: str) -> None:
    if indices.numel() == 0:
        return
    # GPU-side indices: skip eager bounds check (avoid sync; kernel handles).
    if str(indices.device).split(":")[0] != "cpu":
        return
    idx = indices.to(dtype=torch.long)
    lo = int(idx.min().item())
    hi = int(idx.max().item())
    if lo < 0:
        raise IndexError(f"{name} row index {lo} is negative")
    if hi >= num_rows:
        raise IndexError(f"{name} row index {hi} out of bounds for {num_rows} rows")


def kv_copy_paged_rows(
    src: torch.Tensor,
    dst: torch.Tensor,
    src_indices: torch.Tensor,
    dst_indices: torch.Tensor,
    *,
    layout: "InterleavedFp8Layout | None" = None,
) -> None:
    """REQ-1: per-layer opaque byte-row gather/scatter.

    Copies ``src[src_indices[i], :] -> dst[dst_indices[i], :]`` for each i.
    Semantics align with sglang ``transfer_kv_per_layer_mla`` but the rows
    are uint8 bytes with no KV/float semantics; item_bytes is derived from
    the tensor shape (observed range 1728~37440).

    ``layout`` (REQ-2) is accepted to pin the API contract; the copy itself
    is layout-agnostic (whole-row) and ignores it.

    Out-of-bounds index semantics (device-dependent, by design to avoid a
    GPU->CPU sync on every call):
      - CPU indices: eager bounds check, raises ``IndexError``.
      - MUSA (GPU) indices: checked inside the native kernel; out-of-bounds
        rows are **silently skipped** (no exception, no partial copy).
    """
    item_bytes, _, _ = _validate_kv_copy_paged_rows_args(
        src, dst, src_indices, dst_indices, name="kv_copy_paged_rows"
    )
    if layout is not None and layout.item_bytes != item_bytes:
        raise ValueError(
            f"kv_copy_paged_rows: layout.item_bytes={layout.item_bytes} != tensor row width {item_bytes}"
        )
    device, dtype = tensor_device_dtype(src)
    ensure_reference_allowed("kv_copy_paged_rows", native_available=_try_native_kv_copy_paged_rows(src, dst, src_indices, dst_indices))
    src_rows = src_indices.to(dtype=torch.long).contiguous()
    dst_rows = dst_indices.to(dtype=torch.long).contiguous()
    if _try_native_kv_copy_paged_rows(src, dst, src_indices, dst_indices):
        emit(DispatchRecord("kv_copy_paged_rows", "ctypes-musa-native-u8", device, str(dtype), "DSV4 opaque page-row copy (per-layer, native)"))
        return
    _timed_copy_rows(src, dst, src_rows, dst_rows)
    emit(DispatchRecord("kv_copy_paged_rows", "torch-reference", device, str(dtype), "DSV4 opaque page-row copy (per-layer)"))


def _validate_layer_table(
    src_ptrs,
    dst_ptrs,
    src_indices: torch.Tensor,
    dst_indices: torch.Tensor,
    *,
    name: str,
) -> int:
    if not isinstance(src_ptrs, (list, tuple)) or not isinstance(dst_ptrs, (list, tuple)):
        raise TypeError(f"{name}: src_ptrs and dst_ptrs must be lists of tensors")
    if len(src_ptrs) == 0 or len(dst_ptrs) == 0:
        raise ValueError(f"{name}: ptr-tables must be non-empty")
    if len(src_ptrs) != len(dst_ptrs):
        raise ValueError(
            f"{name}: src/dst layer counts must match, got {len(src_ptrs)} vs {len(dst_ptrs)}"
        )
    item_bytes = int(src_ptrs[0].shape[1])
    for layer, (src, dst) in enumerate(zip(src_ptrs, dst_ptrs)):
        _validate_kv_copy_paged_rows_args(
            src, dst, src_indices, dst_indices, name=f"{name}[layer{layer}]"
        )
        if int(dst.shape[1]) != item_bytes:
            raise ValueError(
                f"{name}: layer {layer} item_bytes mismatch, expected {item_bytes}, got {int(dst.shape[1])}"
            )
    return item_bytes


def kv_copy_paged_rows_layer_table(
    src_ptrs,
    dst_ptrs,
    src_indices: torch.Tensor,
    dst_indices: torch.Tensor,
) -> None:
    """REQ-1: all-layer ptr-table opaque byte-row gather/scatter.

    One logical launch covering every layer: each ``src_ptrs[l][src_rows]``
    row is copied to ``dst_ptrs[l][dst_rows]``. Semantics align with sglang
    ``transfer_kv_all_layer_mla`` ptr-table form.

    Out-of-bounds index semantics: CPU indices raise ``IndexError`` eagerly;
    MUSA (GPU) indices are guarded inside the native kernel and out-of-bounds
    rows are silently skipped (avoids per-call GPU->CPU sync).
    """
    item_bytes = _validate_layer_table(
        src_ptrs, dst_ptrs, src_indices, dst_indices, name="kv_copy_paged_rows_layer_table"
    )
    device, dtype = tensor_device_dtype(src_ptrs[0])
    ensure_reference_allowed("kv_copy_paged_rows_layer_table")
    src_rows = src_indices.to(dtype=torch.long).contiguous()
    dst_rows = dst_indices.to(dtype=torch.long).contiguous()
    if _try_native_kv_copy_paged_rows_layer_table(src_ptrs, dst_ptrs, src_indices, dst_indices):
        emit(DispatchRecord("kv_copy_paged_rows_layer_table", "ctypes-musa-native-u8", device, str(dtype), f"DSV4 opaque page-row copy (all-layer native, {len(src_ptrs)} layers, item_bytes={item_bytes})"))
        return
    for src, dst in zip(src_ptrs, dst_ptrs):
        _timed_copy_rows(src, dst, src_rows, dst_rows)
    emit(DispatchRecord("kv_copy_paged_rows_layer_table", "torch-reference", device, str(dtype), f"DSV4 opaque page-row copy (all-layer, {len(src_ptrs)} layers, item_bytes={item_bytes})"))


def kv_copy_paged_rows_multi_pool(
    pools: "list[PoolSpec]",
    indices: torch.Tensor,
    *,
    direction: Literal["H2D", "D2H"],
) -> None:
    """REQ-3: anchor-shared-index multi-pool batch copy.

    The DSV4 KV/SWA/C4/C128/indexer/state pools share one set of page row
    indices; each pool copies its own rows. Indices are consumed once.

    Out-of-bounds index semantics: CPU indices raise ``IndexError`` eagerly;
    MUSA (GPU) indices are guarded inside the native kernel and out-of-bounds
    rows are silently skipped (avoids per-call GPU->CPU sync).
    """
    if direction not in ("H2D", "D2H"):
        raise ValueError(f"kv_copy_paged_rows_multi_pool: direction must be 'H2D' or 'D2H', got {direction!r}")
    if not isinstance(pools, (list, tuple)) or len(pools) == 0:
        raise ValueError("kv_copy_paged_rows_multi_pool: pools must be a non-empty list of PoolSpec")
    for pool_idx, pool in enumerate(pools):
        if not isinstance(pool, PoolSpec):
            raise TypeError(f"kv_copy_paged_rows_multi_pool: pools[{pool_idx}] must be PoolSpec")
        item_bytes = _validate_layer_table(
            pool.src_ptrs,
            pool.dst_ptrs,
            indices,
            indices,
            name=f"kv_copy_paged_rows_multi_pool[pool{pool_idx}]",
        )
        if pool.item_bytes != item_bytes:
            raise ValueError(
                f"kv_copy_paged_rows_multi_pool: pools[{pool_idx}].item_bytes={pool.item_bytes} "
                f"!= tensor row width {item_bytes}"
            )
    rows = indices.to(dtype=torch.long).contiguous()
    device = str(pools[0].dst_ptrs[0].device) if direction == "H2D" else str(pools[0].src_ptrs[0].device)
    dtype = str(pools[0].src_ptrs[0].dtype)
    ensure_reference_allowed("kv_copy_paged_rows_multi_pool")
    for pool in pools:
        for src, dst in zip(pool.src_ptrs, pool.dst_ptrs):
            _timed_copy_rows(src, dst, rows, rows)
    emit(DispatchRecord("kv_copy_paged_rows_multi_pool", "torch-reference", device, dtype, f"DSV4 multi-pool page-row copy ({len(pools)} pools, direction={direction})"))


# ---------------------------------------------------------------------------
# REQ-4: unpinned host memory support + transfer telemetry.
#
# MUSA HiCache deployments cannot pin every large host pool (VA exhaustion,
# rc=712). The copy ops above transparently work on unpinned pageable host
# memory: torch copy_ between pageable CPU tensors and MUSA tensors already
# routes through the driver's pageable path. Telemetry below records
# pinned/unpinned call counts and observed byte volumes so deployments can
# evaluate the gap (microbench T3.2).
# ---------------------------------------------------------------------------

from collections import defaultdict  # noqa: E402
import threading  # noqa: E402
import time  # noqa: E402

_kv_copy_paged_rows_stats_lock = threading.Lock()
_kv_copy_paged_rows_stats: dict = {
    "pinned_calls": 0,
    "unpinned_calls": 0,
    "pinned_bytes": 0,
    "unpinned_bytes": 0,
    "pinned_ns_total": 0,
    "unpinned_ns_total": 0,
}


def kv_transfer_stats() -> dict:
    """REQ-4 telemetry: pinned vs unpinned call counts / bytes / time.

    Returns a plain dict snapshot; bandwidth can be derived as
    bytes / (ns_total / 1e9).
    """
    with _kv_copy_paged_rows_stats_lock:
        return dict(_kv_copy_paged_rows_stats)


def reset_kv_transfer_stats() -> None:
    with _kv_copy_paged_rows_stats_lock:
        for key in _kv_copy_paged_rows_stats:
            _kv_copy_paged_rows_stats[key] = 0


def _is_pinned_tensor(tensor: torch.Tensor) -> bool:
    try:
        return bool(tensor.is_pinned())
    except Exception:
        return False


def _record_copy_stats(src: torch.Tensor, dst: torch.Tensor, num_rows: int, item_bytes: int, elapsed_ns: int) -> None:
    moved = num_rows * item_bytes
    # A host-involved copy is "pinned" only if the host side is pinned.
    host_side = src if str(src.device).split(":")[0] == "cpu" else (dst if str(dst.device).split(":")[0] == "cpu" else None)
    pinned = host_side is None or _is_pinned_tensor(host_side)
    with _kv_copy_paged_rows_stats_lock:
        if pinned:
            _kv_copy_paged_rows_stats["pinned_calls"] += 1
            _kv_copy_paged_rows_stats["pinned_bytes"] += moved
            _kv_copy_paged_rows_stats["pinned_ns_total"] += elapsed_ns
        else:
            _kv_copy_paged_rows_stats["unpinned_calls"] += 1
            _kv_copy_paged_rows_stats["unpinned_bytes"] += moved
            _kv_copy_paged_rows_stats["unpinned_ns_total"] += elapsed_ns


def _timed_copy_rows(src: torch.Tensor, dst: torch.Tensor, src_rows: torch.Tensor, dst_rows: torch.Tensor) -> None:
    start = time.perf_counter_ns()
    if str(src.device) != str(dst.device):
        # Cross-device (host <-> musa): stage gathered rows on dst device.
        gathered = src.index_select(0, src_rows.to(src.device))
        dst.index_copy_(0, dst_rows.to(dst.device), gathered.to(dst.device))
    else:
        dst.index_copy_(0, dst_rows.to(dst.device), src.index_select(0, src_rows.to(src.device)))
    _record_copy_stats(src, dst, int(src_rows.numel()), int(src.shape[1]), time.perf_counter_ns() - start)
