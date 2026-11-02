from __future__ import annotations

import ctypes
import math

import torch

from musa_aiter.dispatch import DispatchRecord, emit, tensor_device_dtype
from musa_aiter._paths import project_data_path
from musa_aiter.jit.core import build_shared_object
from musa_aiter.jit.ctypes_loader import has_symbol, load_library
from musa_aiter.policy import ensure_reference_allowed
from musa_aiter.vendors.planning import emit_vendor_planning_note

_ATTENTION_SOURCE = project_data_path("csrc", "attention", "dense_decode_attention.mu")

_ATTENTION_NATIVE_ATTEMPTED = False
_ATTENTION_NATIVE_RESULT = None
_ATTENTION_NATIVE_LIB = None


def build_native_dense_decode_attention():
    global _ATTENTION_NATIVE_ATTEMPTED, _ATTENTION_NATIVE_RESULT
    if not _ATTENTION_NATIVE_ATTEMPTED:
        _ATTENTION_NATIVE_RESULT = build_shared_object(name="dense_decode_attention", sources=[_ATTENTION_SOURCE])
        _ATTENTION_NATIVE_ATTEMPTED = True
    return _ATTENTION_NATIVE_RESULT


def load_native_dense_decode_attention():
    global _ATTENTION_NATIVE_LIB
    result = build_native_dense_decode_attention()
    if not result.ok or result.library_path is None:
        raise RuntimeError(result.message)
    if _ATTENTION_NATIVE_LIB is None:
        _ATTENTION_NATIVE_LIB = load_library(result.library_path)
    return _ATTENTION_NATIVE_LIB


def native_dense_decode_attention_abi_version() -> int | None:
    try:
        lib = load_native_dense_decode_attention()
        if not has_symbol(lib, "musa_aiter_attention_abi_version"):
            return None
        fn = lib.handle.musa_aiter_attention_abi_version
        fn.argtypes = []
        fn.restype = ctypes.c_int
        return int(fn())
    except Exception:
        return None


def is_native_dense_decode_attention_available() -> bool:
    result = build_native_dense_decode_attention()
    if not bool(result and result.ok and result.library_path):
        return False
    try:
        lib = load_native_dense_decode_attention()
        return (
            has_symbol(lib, "musa_aiter_dense_decode_attention_float")
            and has_symbol(lib, "musa_aiter_dense_decode_attention_half")
            and has_symbol(lib, "musa_aiter_dense_decode_attention_bfloat16")
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


def _native_attention_dtype_suffix(dtype: torch.dtype) -> str | None:
    if dtype is torch.float32:
        return "float"
    if dtype is torch.float16:
        return "half"
    if dtype is torch.bfloat16:
        return "bfloat16"
    return None


def _is_musa_attention_tensor(tensor: torch.Tensor) -> bool:
    return str(tensor.device).split(":")[0] == "musa" and tensor.dtype in (torch.float32, torch.float16, torch.bfloat16)


def _try_native_dense_decode_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    scale: float,
) -> bool:
    if not (_is_musa_attention_tensor(q) and _is_musa_attention_tensor(k) and _is_musa_attention_tensor(v) and _is_musa_attention_tensor(out)):
        return False
    if not (q.dtype == k.dtype == v.dtype == out.dtype):
        return False
    if not (q.is_contiguous() and k.is_contiguous() and v.is_contiguous() and out.is_contiguous()):
        return False
    suffix = _native_attention_dtype_suffix(q.dtype)
    if suffix is None:
        return False
    try:
        lib = load_native_dense_decode_attention()
        symbol = f"musa_aiter_dense_decode_attention_{suffix}"
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
    seq_len = int(k.shape[1])
    num_kv_heads = int(k.shape[2])
    fn(
        ctypes.c_void_p(q.data_ptr()),
        ctypes.c_void_p(k.data_ptr()),
        ctypes.c_void_p(v.data_ptr()),
        ctypes.c_void_p(out.data_ptr()),
        ctypes.c_longlong(batch),
        ctypes.c_longlong(seq_len),
        ctypes.c_longlong(num_q_heads),
        ctypes.c_longlong(num_kv_heads),
        ctypes.c_longlong(head_dim),
        ctypes.c_float(float(scale)),
        _current_musa_stream_ptr(),
    )
    return True


def dense_decode_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    softmax_scale: float | None = None,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Decode attention for ``q=[B,H,D]`` and dense KV ``[B,S,H,D]``."""
    device, dtype = tensor_device_dtype(q)
    ensure_reference_allowed("dense_decode_attention")
    if q.dim() != 3:
        raise ValueError("dense_decode_attention: q must be [batch, heads, head_dim]")
    if k.dim() != 4 or v.dim() != 4:
        raise ValueError("dense_decode_attention: k/v must be [batch, seq, heads, head_dim]")
    if k.shape != v.shape or k.dtype != v.dtype or k.device != v.device:
        raise ValueError("dense_decode_attention: k/v shape, dtype, and device must match")
    if q.device != k.device or q.dtype != k.dtype:
        raise ValueError("dense_decode_attention: q/k/v device and dtype must match")
    if q.shape[0] != k.shape[0] or q.shape[2] != k.shape[3] or q.shape[1] % k.shape[2] != 0:
        raise ValueError("dense_decode_attention: q shape must match dense KV batch/head_dim and q heads must be divisible by KV heads")
    scale = float(softmax_scale) if softmax_scale is not None else 1.0 / math.sqrt(int(q.shape[-1]))
    shape = tuple(q.shape)
    if out is None:
        out = torch.empty(shape, dtype=q.dtype, device=q.device)
    elif out.shape != shape or out.dtype != q.dtype or out.device != q.device or not out.is_contiguous():
        raise ValueError("dense_decode_attention: out must match q shape/dtype/device and be contiguous")

    q_work = q if q.is_contiguous() else q.contiguous()
    k_work = k if k.is_contiguous() else k.contiguous()
    v_work = v if v.is_contiguous() else v.contiguous()
    if _try_native_dense_decode_attention(q_work, k_work, v_work, out, scale):
        emit(DispatchRecord("dense_decode_attention", "ctypes-musa-native", device, str(dtype), "native MUSA dense decode attention"))
        return out

    emit(DispatchRecord("dense_decode_attention", "torch-reference", device, str(dtype), "dense decode attention reference"))
    emit_vendor_planning_note("dense_decode_attention")
    if k.shape[2] != q.shape[1]:
        repeat = q.shape[1] // k.shape[2]
        k_ref = k.repeat_interleave(repeat, dim=2)
        v_ref = v.repeat_interleave(repeat, dim=2)
    else:
        k_ref = k
        v_ref = v
    scores = torch.einsum("bhd,bshd->bhs", q.float(), k_ref.float()) * scale
    probs = torch.softmax(scores, dim=-1)
    result = torch.einsum("bhs,bshd->bhd", probs, v_ref.float()).to(dtype=q.dtype)
    out.copy_(result)
    return out
