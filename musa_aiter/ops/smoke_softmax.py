"""Phase P3 smoke — last-dim softmax on MUSA (Flash/MLA-style stable reduce pattern)."""

from __future__ import annotations

import ctypes
import math
import os

import torch

from musa_aiter.dispatch import DispatchRecord, emit, tensor_device_dtype
from musa_aiter._paths import project_data_path
from musa_aiter.jit.core import build_shared_object
from musa_aiter.jit.ctypes_loader import has_symbol, load_library
from musa_aiter.vendors.planning import emit_vendor_planning_note

_SOFTMAX_SOURCE = project_data_path("csrc", "attention", "softmax_lastdim_float.mu")

# One CTA per row; tuned for smoke—not production tile sizes.
_SOFTMAX_LD_MAX_DIM = 1024

_SOFTMAX_ATTEMPTED = False
_SOFTMAX_RESULT = None
_SOFTMAX_LIB = None


def build_native_softmax_last_dim():
    global _SOFTMAX_ATTEMPTED, _SOFTMAX_RESULT
    if not _SOFTMAX_ATTEMPTED:
        _SOFTMAX_RESULT = build_shared_object(name="smoke_softmax_ld", sources=[_SOFTMAX_SOURCE])
        _SOFTMAX_ATTEMPTED = True
    return _SOFTMAX_RESULT


def load_native_softmax_last_dim():
    global _SOFTMAX_LIB
    result = build_native_softmax_last_dim()
    if not result.ok or result.library_path is None:
        raise RuntimeError(result.message)
    if _SOFTMAX_LIB is None:
        _SOFTMAX_LIB = load_library(result.library_path)
    return _SOFTMAX_LIB


def native_smoke_softmax_abi_version() -> int | None:
    try:
        lib = load_native_softmax_last_dim()
        if not has_symbol(lib, "musa_aiter_smoke_softmax_abi_version"):
            return None
        fn = lib.handle.musa_aiter_smoke_softmax_abi_version
        fn.argtypes = []
        fn.restype = ctypes.c_int
        return int(fn())
    except Exception:
        return None


def is_native_softmax_last_dim_available() -> bool:
    result = build_native_softmax_last_dim()
    if not bool(result and result.ok and result.library_path):
        return False
    try:
        lib = load_native_softmax_last_dim()
        return has_symbol(lib, "musa_aiter_softmax_lastdim_float")
    except Exception:
        return False


def _normalized_dim(ndim: int, dim: int) -> int:
    if dim >= 0:
        return dim
    return ndim + dim


def softmax_last_dim(
    x: torch.Tensor,
    *,
    dim: int = -1,
    prefer_native: bool = True,
) -> torch.Tensor:
    """``torch.softmax`` on ``dim``, float32 contiguous last axis; native on MUSA when enabled.

    Native path requires ``dim`` to refer to the last dimension, ``float32``, contiguous memory,
    and inner size ≤ :data:`SOFTMAX_LD_MAX_DIM`.
    """

    nd = x.ndim
    if nd < 1:
        raise ValueError("softmax_last_dim expects at least one dimension")

    device, dtype = tensor_device_dtype(x)
    axis = _normalized_dim(nd, dim)
    if axis != nd - 1:
        if prefer_native and os.environ.get("MUSA_AITER_NATIVE_SMOKE", "0") == "1":
            emit(
                DispatchRecord(
                    "softmax_last_dim",
                    "torch-reference",
                    device,
                    dtype,
                    "native softmax smoke requires the last dimension",
                )
            )
        return torch.softmax(x, dim=dim)

    inner = int(x.shape[-1])
    if inner > _SOFTMAX_LD_MAX_DIM:
        emit(
            DispatchRecord(
                "softmax_last_dim",
                "torch-reference",
                device,
                dtype,
                f"dim {inner} exceeds native smoke max {_SOFTMAX_LD_MAX_DIM}",
            )
        )
        emit_vendor_planning_note("softmax_last_dim")
        return torch.softmax(x, dim=dim)

    work = x if x.is_contiguous() else x.contiguous()
    outer = int(math.prod(work.shape[:-1])) if nd > 1 else 1
    xv = work.reshape(outer, inner)

    device, dtype = tensor_device_dtype(xv)
    if prefer_native:
        native = _try_native_softmax_lastdim(xv)
        if native is not None:
            emit(
                DispatchRecord(
                    "softmax_last_dim",
                    "ctypes-musa-native",
                    device,
                    dtype,
                    "native smoke softmax_ld opt-in",
                )
            )
            emit_vendor_planning_note("softmax_last_dim")
            return native.reshape(work.shape)

    build_native_softmax_last_dim()
    emit(
        DispatchRecord(
            "softmax_last_dim",
            "torch-reference",
            device,
            dtype,
            "native unavailable or not requested",
        )
    )
    emit_vendor_planning_note("softmax_last_dim")
    return torch.softmax(work, dim=-1)


def _try_native_softmax_lastdim(x2d: torch.Tensor) -> torch.Tensor | None:
    if os.environ.get("MUSA_AITER_NATIVE_SMOKE", "0") != "1":
        return None
    if str(x2d.device).split(":")[0] != "musa":
        return None
    if x2d.dtype is not torch.float32:
        return None
    if x2d.ndim != 2:
        return None

    outer, inner = int(x2d.shape[0]), int(x2d.shape[1])
    if inner > _SOFTMAX_LD_MAX_DIM or outer <= 0 or inner <= 0:
        return None

    lib = load_native_softmax_last_dim()
    out = torch.empty_like(x2d)
    fn = lib.handle.musa_aiter_softmax_lastdim_float
    fn.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int32,
        ctypes.c_int32,
        ctypes.c_void_p,
    ]
    fn.restype = None
    fn(
        ctypes.c_void_p(x2d.data_ptr()),
        ctypes.c_void_p(out.data_ptr()),
        ctypes.c_int32(outer),
        ctypes.c_int32(inner),
        ctypes.c_void_p(0),
    )
    return out


SOFTMAX_LD_MAX_DIM = _SOFTMAX_LD_MAX_DIM
