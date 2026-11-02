"""Phase P2 — ctypes/JIT parallel reduction smoke (KERNEL_IMPLEMENTATION_PLAN §7 P2).

Single-block GPU sum teaching kernel; Torch reference fallback.
"""

from __future__ import annotations

import ctypes
import os

import torch

from musa_aiter.dispatch import DispatchRecord, emit, tensor_device_dtype
from musa_aiter._paths import project_data_path
from musa_aiter.jit.core import build_shared_object
from musa_aiter.jit.ctypes_loader import has_symbol, load_library
from musa_aiter.vendors.planning import emit_vendor_planning_note

_SUM_SOURCE = project_data_path("csrc", "reduction", "sum_float.mu")

_REDUCE_ATTEMPTED = False
_REDUCE_RESULT = None
_REDUCE_LIB = None


def build_native_reduce_sum():
    global _REDUCE_ATTEMPTED, _REDUCE_RESULT
    if not _REDUCE_ATTEMPTED:
        _REDUCE_RESULT = build_shared_object(name="smoke_reduce_sum", sources=[_SUM_SOURCE])
        _REDUCE_ATTEMPTED = True
    return _REDUCE_RESULT


def load_native_reduce_sum():
    global _REDUCE_LIB
    result = build_native_reduce_sum()
    if not result.ok or result.library_path is None:
        raise RuntimeError(result.message)
    if _REDUCE_LIB is None:
        _REDUCE_LIB = load_library(result.library_path)
    return _REDUCE_LIB


def native_smoke_reduce_abi_version() -> int | None:
    try:
        lib = load_native_reduce_sum()
        if not has_symbol(lib, "musa_aiter_smoke_reduce_abi_version"):
            return None
        fn = lib.handle.musa_aiter_smoke_reduce_abi_version
        fn.argtypes = []
        fn.restype = ctypes.c_int
        return int(fn())
    except Exception:
        return None


def is_native_reduce_sum_available() -> bool:
    result = build_native_reduce_sum()
    if not bool(result and result.ok and result.library_path):
        return False
    try:
        lib = load_native_reduce_sum()
        return has_symbol(lib, "musa_aiter_reduce_sum_float")
    except Exception:
        return False


def _try_native_reduce_sum(x: torch.Tensor) -> torch.Tensor | None:
    if os.environ.get("MUSA_AITER_NATIVE_SMOKE", "0") != "1":
        return None
    if str(x.device).split(":")[0] != "musa":
        return None
    if x.dtype is not torch.float32:
        return None
    x = x.reshape(-1)
    if not x.is_contiguous():
        x = x.contiguous()
    lib = load_native_reduce_sum()
    out = torch.empty((), device=x.device, dtype=torch.float32)
    fn = lib.handle.musa_aiter_reduce_sum_float
    fn.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_longlong, ctypes.c_void_p]
    fn.restype = None
    fn(
        ctypes.c_void_p(x.data_ptr()),
        ctypes.c_void_p(out.data_ptr()),
        ctypes.c_longlong(x.numel()),
        ctypes.c_void_p(0),
    )
    return out


def reduce_sum(x: torch.Tensor, *, prefer_native: bool = True) -> torch.Tensor:
    """Sum all elements of *x* (returns 0-dim tensor). Native path: MUSA float32 only."""

    device, dtype = tensor_device_dtype(x)
    if prefer_native:
        native = _try_native_reduce_sum(x)
        if native is not None:
            emit(
                DispatchRecord(
                    "reduce_sum",
                    "ctypes-musa-native",
                    device,
                    dtype,
                    "native smoke reduce opt-in",
                )
            )
            emit_vendor_planning_note("reduce_sum")
            return native
        build_native_reduce_sum()
    emit(
        DispatchRecord(
            "reduce_sum",
            "torch-reference",
            device,
            dtype,
            "native unavailable or not requested",
        )
    )
    emit_vendor_planning_note("reduce_sum")
    return torch.sum(x)
