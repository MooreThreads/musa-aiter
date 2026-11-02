from __future__ import annotations

import ctypes
import math
import os

import torch

from musa_aiter.dispatch import DispatchRecord, emit, tensor_device_dtype
from musa_aiter._paths import project_data_path
from musa_aiter.jit.core import build_shared_object
from musa_aiter.jit.ctypes_loader import has_symbol, load_library
from musa_aiter.policy import ensure_reference_allowed
from musa_aiter.vendors.planning import emit_vendor_planning_note

_PAGED_SPARSE_SOURCE = project_data_path("csrc", "attention", "paged_sparse_decode_attention.mu")

_PAGED_SPARSE_NATIVE_ATTEMPTED = False
_PAGED_SPARSE_NATIVE_RESULT = None
_PAGED_SPARSE_NATIVE_LIB = None
_PAGED_SPARSE_TILED_SCRATCH_CACHE: dict[tuple[str, int, int, int, int, str], tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}


def build_native_paged_sparse_decode_attention():
    global _PAGED_SPARSE_NATIVE_ATTEMPTED, _PAGED_SPARSE_NATIVE_RESULT
    if not _PAGED_SPARSE_NATIVE_ATTEMPTED:
        _PAGED_SPARSE_NATIVE_RESULT = build_shared_object(name="paged_sparse_decode_attention", sources=[_PAGED_SPARSE_SOURCE])
        _PAGED_SPARSE_NATIVE_ATTEMPTED = True
    return _PAGED_SPARSE_NATIVE_RESULT


def load_native_paged_sparse_decode_attention():
    global _PAGED_SPARSE_NATIVE_LIB
    result = build_native_paged_sparse_decode_attention()
    if not result.ok or result.library_path is None:
        raise RuntimeError(result.message)
    if _PAGED_SPARSE_NATIVE_LIB is None:
        _PAGED_SPARSE_NATIVE_LIB = load_library(result.library_path)
    return _PAGED_SPARSE_NATIVE_LIB


def native_paged_sparse_decode_attention_abi_version() -> int | None:
    try:
        lib = load_native_paged_sparse_decode_attention()
        if not has_symbol(lib, "musa_aiter_paged_sparse_attention_abi_version"):
            return None
        fn = lib.handle.musa_aiter_paged_sparse_attention_abi_version
        fn.argtypes = []
        fn.restype = ctypes.c_int
        return int(fn())
    except Exception:
        return None


def is_native_paged_sparse_decode_attention_available() -> bool:
    result = build_native_paged_sparse_decode_attention()
    if not bool(result and result.ok and result.library_path):
        return False
    try:
        lib = load_native_paged_sparse_decode_attention()
        return (
            has_symbol(lib, "musa_aiter_paged_sparse_decode_attention_float")
            and has_symbol(lib, "musa_aiter_paged_sparse_decode_attention_half")
            and has_symbol(lib, "musa_aiter_paged_sparse_decode_attention_bfloat16")
        )
    except Exception:
        return False


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


def _native_dtype_suffix(dtype: torch.dtype) -> str | None:
    if dtype is torch.float32:
        return "float"
    if dtype is torch.float16:
        return "half"
    if dtype is torch.bfloat16:
        return "bfloat16"
    return None


def _is_musa_attention_tensor(tensor: torch.Tensor) -> bool:
    return str(tensor.device).split(":")[0] == "musa" and tensor.dtype in (torch.float32, torch.float16, torch.bfloat16)


def _is_musa_integer_tensor(tensor: torch.Tensor) -> bool:
    return str(tensor.device).split(":")[0] == "musa" and tensor.dtype in (torch.int16, torch.int32, torch.int64)


def _is_tiled_paged_sparse_enabled() -> bool:
    return os.environ.get("MUSA_AITER_PAGED_SPARSE_ATTENTION_TILED", "0") == "1"


def _paged_sparse_tile_size() -> int:
    raw = os.environ.get("MUSA_AITER_PAGED_SPARSE_ATTENTION_TILE_SIZE", "512")
    try:
        value = int(raw)
    except ValueError:
        value = 512
    return max(1, value)


def _is_tiled_compact_scratch_enabled() -> bool:
    return os.environ.get("MUSA_AITER_PAGED_SPARSE_ATTENTION_COMPACT_SCRATCH", "1") != "0"


def _get_tiled_scratch(
    *,
    device: torch.device,
    batch: int,
    num_q_heads: int,
    head_dim: int,
    num_tiles: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    partial_out_dtype = torch.bfloat16
    key = (str(device), batch, num_q_heads, head_dim, num_tiles, str(partial_out_dtype))
    cached = _PAGED_SPARSE_TILED_SCRATCH_CACHE.get(key)
    if cached is not None:
        return cached
    partial_shape = (batch * num_q_heads * num_tiles,)
    partial_max = torch.empty(partial_shape, dtype=torch.float32, device=device)
    partial_sum = torch.empty(partial_shape, dtype=torch.float32, device=device)
    partial_out = torch.empty((batch * num_q_heads * num_tiles, head_dim), dtype=partial_out_dtype, device=device)
    cached = (partial_max, partial_sum, partial_out)
    _PAGED_SPARSE_TILED_SCRATCH_CACHE[key] = cached
    return cached


def clear_paged_sparse_tiled_scratch_cache() -> None:
    _PAGED_SPARSE_TILED_SCRATCH_CACHE.clear()


def _try_native_paged_sparse_decode_attention_tiled_bfloat16(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    kv_indptr: torch.Tensor,
    kv_indices: torch.Tensor,
    out: torch.Tensor,
    scale: float,
    max_seq_len: int | None,
    tile_size: int | None,
) -> bool:
    if not _is_tiled_paged_sparse_enabled() or not _is_tiled_compact_scratch_enabled():
        return False
    if q.dtype is not torch.bfloat16:
        return False
    if not (_is_musa_attention_tensor(q) and _is_musa_attention_tensor(k) and _is_musa_attention_tensor(v) and _is_musa_attention_tensor(out)):
        return False
    if not (_is_musa_integer_tensor(kv_indptr) and _is_musa_integer_tensor(kv_indices)):
        return False
    if not (q.dtype == k.dtype == v.dtype == out.dtype):
        return False
    if not (q.is_contiguous() and k.is_contiguous() and v.is_contiguous() and out.is_contiguous() and kv_indptr.is_contiguous() and kv_indices.is_contiguous()):
        return False
    try:
        lib = load_native_paged_sparse_decode_attention()
        symbol = "musa_aiter_paged_sparse_decode_attention_bfloat16_tiled"
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
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_longlong,
        ctypes.c_longlong,
        ctypes.c_longlong,
        ctypes.c_longlong,
        ctypes.c_longlong,
        ctypes.c_longlong,
        ctypes.c_longlong,
        ctypes.c_longlong,
        ctypes.c_float,
        ctypes.c_void_p,
    ]
    fn.restype = None
    batch, num_q_heads, head_dim = map(int, q.shape)
    num_tokens = int(k.shape[0])
    num_kv_heads = int(k.shape[1])
    if max_seq_len is None:
        return False
    max_seq_len_value = int(max_seq_len)
    if max_seq_len_value <= 0:
        return False
    tile_size_value = int(tile_size) if tile_size is not None else _paged_sparse_tile_size()
    if tile_size_value <= 0:
        return False
    num_tiles = max(1, (max_seq_len_value + tile_size_value - 1) // tile_size_value)
    partial_max, partial_sum, partial_out = _get_tiled_scratch(device=q.device, batch=batch, num_q_heads=num_q_heads, head_dim=head_dim, num_tiles=num_tiles)
    fn(
        ctypes.c_void_p(q.data_ptr()),
        ctypes.c_void_p(k.data_ptr()),
        ctypes.c_void_p(v.data_ptr()),
        ctypes.c_void_p(kv_indptr.data_ptr()),
        ctypes.c_void_p(kv_indices.data_ptr()),
        ctypes.c_void_p(out.data_ptr()),
        ctypes.c_void_p(partial_max.data_ptr()),
        ctypes.c_void_p(partial_sum.data_ptr()),
        ctypes.c_void_p(partial_out.data_ptr()),
        ctypes.c_longlong(batch),
        ctypes.c_longlong(kv_indices.numel()),
        ctypes.c_longlong(num_tokens),
        ctypes.c_longlong(num_q_heads),
        ctypes.c_longlong(num_kv_heads),
        ctypes.c_longlong(head_dim),
        ctypes.c_longlong(tile_size_value),
        ctypes.c_longlong(num_tiles),
        ctypes.c_float(float(scale)),
        _current_musa_stream_ptr(),
    )
    return True


def _try_native_paged_sparse_decode_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    kv_indptr: torch.Tensor,
    kv_indices: torch.Tensor,
    out: torch.Tensor,
    scale: float,
) -> bool:
    if not (_is_musa_attention_tensor(q) and _is_musa_attention_tensor(k) and _is_musa_attention_tensor(v) and _is_musa_attention_tensor(out)):
        return False
    if not (_is_musa_integer_tensor(kv_indptr) and _is_musa_integer_tensor(kv_indices)):
        return False
    if not (q.dtype == k.dtype == v.dtype == out.dtype):
        return False
    if not (q.is_contiguous() and k.is_contiguous() and v.is_contiguous() and out.is_contiguous() and kv_indptr.is_contiguous() and kv_indices.is_contiguous()):
        return False
    suffix = _native_dtype_suffix(q.dtype)
    if suffix is None:
        return False
    try:
        lib = load_native_paged_sparse_decode_attention()
        symbol = f"musa_aiter_paged_sparse_decode_attention_{suffix}"
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
        ctypes.c_longlong,
        ctypes.c_longlong,
        ctypes.c_longlong,
        ctypes.c_longlong,
        ctypes.c_longlong,
        ctypes.c_longlong,
        ctypes.c_float,
        ctypes.c_void_p,
    ]
    fn.restype = None
    batch, num_q_heads, head_dim = map(int, q.shape)
    num_tokens = int(k.shape[0])
    num_kv_heads = int(k.shape[1])
    fn(
        ctypes.c_void_p(q.data_ptr()),
        ctypes.c_void_p(k.data_ptr()),
        ctypes.c_void_p(v.data_ptr()),
        ctypes.c_void_p(kv_indptr.data_ptr()),
        ctypes.c_void_p(kv_indices.data_ptr()),
        ctypes.c_void_p(out.data_ptr()),
        ctypes.c_longlong(batch),
        ctypes.c_longlong(kv_indices.numel()),
        ctypes.c_longlong(num_tokens),
        ctypes.c_longlong(num_q_heads),
        ctypes.c_longlong(num_kv_heads),
        ctypes.c_longlong(head_dim),
        ctypes.c_float(float(scale)),
        _current_musa_stream_ptr(),
    )
    return True


def paged_sparse_decode_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    kv_indptr: torch.Tensor,
    kv_indices: torch.Tensor,
    *,
    softmax_scale: float | None = None,
    out: torch.Tensor | None = None,
    max_seq_len: int | None = None,
    tile_size: int | None = None,
) -> torch.Tensor:
    """Decode attention for flat paged KV indices without dense KV materialization."""
    device, dtype = tensor_device_dtype(q)
    ensure_reference_allowed("paged_sparse_decode_attention")
    if q.dim() != 3:
        raise ValueError("paged_sparse_decode_attention: q must be [batch, heads, head_dim]")
    if k.dim() != 3 or v.dim() != 3:
        raise ValueError("paged_sparse_decode_attention: k/v must be [tokens, heads, head_dim]")
    if k.shape != v.shape or k.dtype != v.dtype or k.device != v.device:
        raise ValueError("paged_sparse_decode_attention: k/v shape, dtype, and device must match")
    if q.device != k.device or q.dtype != k.dtype:
        raise ValueError("paged_sparse_decode_attention: q/k/v device and dtype must match")
    if kv_indptr.dim() != 1 or kv_indices.dim() != 1:
        raise ValueError("paged_sparse_decode_attention: kv_indptr and kv_indices must be 1-D")
    if kv_indptr.shape[0] != q.shape[0] + 1:
        raise ValueError("paged_sparse_decode_attention: kv_indptr must be [batch + 1]")
    if kv_indptr.device != q.device or kv_indices.device != q.device:
        raise ValueError("paged_sparse_decode_attention: index tensors must be on the same device as q")
    if kv_indptr.dtype not in (torch.int16, torch.int32, torch.int64) or kv_indices.dtype not in (torch.int16, torch.int32, torch.int64):
        raise ValueError("paged_sparse_decode_attention: index tensors must use integer dtype")
    if q.shape[2] != k.shape[2] or q.shape[1] % k.shape[1] != 0:
        raise ValueError("paged_sparse_decode_attention: q shape must match KV head_dim and q heads must be divisible by KV heads")
    scale = float(softmax_scale) if softmax_scale is not None else 1.0 / math.sqrt(int(q.shape[-1]))
    shape = tuple(q.shape)
    if out is None:
        out = torch.empty(shape, dtype=q.dtype, device=q.device)
    elif out.shape != shape or out.dtype != q.dtype or out.device != q.device or not out.is_contiguous():
        raise ValueError("paged_sparse_decode_attention: out must match q shape/dtype/device and be contiguous")

    q_work = q if q.is_contiguous() else q.contiguous()
    k_work = k if k.is_contiguous() else k.contiguous()
    v_work = v if v.is_contiguous() else v.contiguous()
    indptr64 = kv_indptr.to(dtype=torch.long).contiguous()
    indices64 = kv_indices.to(dtype=torch.long).contiguous()
    if _try_native_paged_sparse_decode_attention_tiled_bfloat16(q_work, k_work, v_work, indptr64, indices64, out, scale, max_seq_len, tile_size):
        emit(DispatchRecord("paged_sparse_decode_attention", "ctypes-musa-native-tiled-bfloat16", device, str(dtype), "native MUSA tiled paged sparse decode attention"))
        return out
    if _try_native_paged_sparse_decode_attention(q_work, k_work, v_work, indptr64, indices64, out, scale):
        emit(DispatchRecord("paged_sparse_decode_attention", "ctypes-musa-native", device, str(dtype), "native MUSA paged sparse decode attention"))
        return out

    emit(DispatchRecord("paged_sparse_decode_attention", "torch-reference", device, str(dtype), "paged sparse decode attention reference"))
    emit_vendor_planning_note("paged_sparse_decode_attention")
    dense_rows = []
    for batch_idx in range(int(q.shape[0])):
        start = int(kv_indptr[batch_idx].item())
        end = int(kv_indptr[batch_idx + 1].item())
        slots = kv_indices[start:end].to(dtype=torch.long)
        dense_rows.append(k.index_select(0, slots))
    max_len = max((int(row.shape[0]) for row in dense_rows), default=0)
    dense_k = k.new_zeros((int(q.shape[0]), max_len, int(k.shape[1]), int(k.shape[2])))
    dense_v = v.new_zeros((int(q.shape[0]), max_len, int(v.shape[1]), int(v.shape[2])))
    for batch_idx, row in enumerate(dense_rows):
        dense_k[batch_idx, : int(row.shape[0])].copy_(row)
        start = int(kv_indptr[batch_idx].item())
        end = int(kv_indptr[batch_idx + 1].item())
        slots = kv_indices[start:end].to(dtype=torch.long)
        dense_v[batch_idx, : int(row.shape[0])].copy_(v.index_select(0, slots))
    if dense_k.shape[2] != q.shape[1]:
        repeat = q.shape[1] // dense_k.shape[2]
        dense_k = dense_k.repeat_interleave(repeat, dim=2)
        dense_v = dense_v.repeat_interleave(repeat, dim=2)
    scores = torch.einsum("bhd,bshd->bhs", q.float(), dense_k.float()) * scale
    probs = torch.softmax(scores, dim=-1)
    result = torch.einsum("bhs,bshd->bhd", probs, dense_v.float()).to(dtype=q.dtype)
    out.copy_(result)
    return out
