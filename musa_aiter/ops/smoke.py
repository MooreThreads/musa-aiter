from __future__ import annotations

import ctypes
import os
from pathlib import Path

import torch

from musa_aiter.dispatch import DispatchRecord, emit, tensor_device_dtype
from musa_aiter._paths import project_data_path
from musa_aiter.jit.core import build_shared_object
from musa_aiter.jit.ctypes_loader import has_symbol, load_library
from musa_aiter.vendors.planning import emit_vendor_planning_note

_NATIVE_ATTEMPTED = False
_NATIVE_RESULT = None
_NATIVE_LIB = None

_MUL_NATIVE_ATTEMPTED = False
_MUL_NATIVE_RESULT = None
_MUL_NATIVE_LIB = None


def _add_source_path() -> Path:
    return project_data_path("csrc", "smoke", "add.mu")


def _mul_source_path() -> Path:
    return project_data_path("csrc", "smoke", "mul_float.mu")


def build_native_add():
    global _NATIVE_ATTEMPTED, _NATIVE_RESULT
    if not _NATIVE_ATTEMPTED:
        _NATIVE_RESULT = build_shared_object(name="smoke_add", sources=[_add_source_path()])
        _NATIVE_ATTEMPTED = True
    return _NATIVE_RESULT


def build_native_mul():
    global _MUL_NATIVE_ATTEMPTED, _MUL_NATIVE_RESULT
    if not _MUL_NATIVE_ATTEMPTED:
        _MUL_NATIVE_RESULT = build_shared_object(name="smoke_mul", sources=[_mul_source_path()])
        _MUL_NATIVE_ATTEMPTED = True
    return _MUL_NATIVE_RESULT


def load_native_add():
    global _NATIVE_LIB
    result = build_native_add()
    if not result.ok or result.library_path is None:
        raise RuntimeError(result.message)
    if _NATIVE_LIB is None:
        _NATIVE_LIB = load_library(result.library_path)
    return _NATIVE_LIB


def load_native_mul():
    global _MUL_NATIVE_LIB
    result = build_native_mul()
    if not result.ok or result.library_path is None:
        raise RuntimeError(result.message)
    if _MUL_NATIVE_LIB is None:
        _MUL_NATIVE_LIB = load_library(result.library_path)
    return _MUL_NATIVE_LIB


def native_smoke_abi_version() -> int | None:
    try:
        lib = load_native_add()
        if not has_symbol(lib, "musa_aiter_smoke_abi_version"):
            return None
        fn = lib.handle.musa_aiter_smoke_abi_version
        fn.argtypes = []
        fn.restype = ctypes.c_int
        return int(fn())
    except Exception:
        return None


def native_smoke_mul_abi_version() -> int | None:
    try:
        lib = load_native_mul()
        if not has_symbol(lib, "musa_aiter_smoke_mul_abi_version"):
            return None
        fn = lib.handle.musa_aiter_smoke_mul_abi_version
        fn.argtypes = []
        fn.restype = ctypes.c_int
        return int(fn())
    except Exception:
        return None


def is_native_add_available() -> bool:
    result = build_native_add()
    if not bool(result and result.ok and result.library_path):
        return False
    try:
        lib = load_native_add()
        return has_symbol(lib, "musa_aiter_add_float")
    except Exception:
        return False


def is_native_mul_available() -> bool:
    result = build_native_mul()
    if not bool(result and result.ok and result.library_path):
        return False
    try:
        lib = load_native_mul()
        return has_symbol(lib, "musa_aiter_mul_float")
    except Exception:
        return False


def _validate_binary_elwise(a: torch.Tensor, b: torch.Tensor) -> None:
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: {tuple(a.shape)} != {tuple(b.shape)}")
    if a.dtype != b.dtype:
        raise ValueError(f"dtype mismatch: {a.dtype} != {b.dtype}")
    if a.device != b.device:
        raise ValueError(f"device mismatch: {a.device} != {b.device}")


def _try_native_add(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor | None:
    """Opt-in native smoke execution.

    This is guarded by MUSA_AITER_NATIVE_SMOKE=1 and only attempts MUSA tensors.
    The current environment lacks torch_musa, so CI normally does not enter here.
    """
    if os.environ.get("MUSA_AITER_NATIVE_SMOKE", "0") != "1":
        return None
    if str(a.device).split(":")[0] != "musa":
        return None
    if a.dtype is not torch.float32:
        return None
    if not a.is_contiguous() or not b.is_contiguous():
        a = a.contiguous()
        b = b.contiguous()
    out = torch.empty_like(a)
    lib = load_native_add()
    fn = lib.handle.musa_aiter_add_float
    fn.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_longlong, ctypes.c_void_p]
    fn.restype = None
    # Stream extraction varies by TorchMUSA version. Use default stream nullptr for Phase 1 smoke.
    fn(
        ctypes.c_void_p(a.data_ptr()),
        ctypes.c_void_p(b.data_ptr()),
        ctypes.c_void_p(out.data_ptr()),
        ctypes.c_longlong(a.numel()),
        ctypes.c_void_p(0),
    )
    return out


def _try_native_mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor | None:
    if os.environ.get("MUSA_AITER_NATIVE_SMOKE", "0") != "1":
        return None
    if str(a.device).split(":")[0] != "musa":
        return None
    if a.dtype is not torch.float32:
        return None
    if not a.is_contiguous() or not b.is_contiguous():
        a = a.contiguous()
        b = b.contiguous()
    out = torch.empty_like(a)
    lib = load_native_mul()
    fn = lib.handle.musa_aiter_mul_float
    fn.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_longlong, ctypes.c_void_p]
    fn.restype = None
    fn(
        ctypes.c_void_p(a.data_ptr()),
        ctypes.c_void_p(b.data_ptr()),
        ctypes.c_void_p(out.data_ptr()),
        ctypes.c_longlong(a.numel()),
        ctypes.c_void_p(0),
    )
    return out


def add(a: torch.Tensor, b: torch.Tensor, *, prefer_native: bool = True) -> torch.Tensor:
    """Phase-0/1 smoke add operator.

    Default execution is torch fallback. Native execution is opt-in via
    MUSA_AITER_NATIVE_SMOKE=1 and requires MUSA float32 tensors.
    """
    _validate_binary_elwise(a, b)

    device, dtype = tensor_device_dtype(a)
    if prefer_native:
        native = _try_native_add(a, b)
        if native is not None:
            emit(DispatchRecord("add", "ctypes-musa-native", device, dtype, "native smoke opt-in"))
            emit_vendor_planning_note("add")
            return native
        build_native_add()
    emit(DispatchRecord("add", "torch-reference", device, dtype, "native unavailable or not requested"))
    emit_vendor_planning_note("add")
    return torch.add(a, b)


def multiply(a: torch.Tensor, b: torch.Tensor, *, prefer_native: bool = True) -> torch.Tensor:
    """Elementwise multiply (Phase-1 teaching kernel; same JIT/ctypes path as ``add``)."""

    _validate_binary_elwise(a, b)

    device, dtype = tensor_device_dtype(a)
    if prefer_native:
        native = _try_native_mul(a, b)
        if native is not None:
            emit(
                DispatchRecord(
                    "multiply",
                    "ctypes-musa-native",
                    device,
                    dtype,
                    "native smoke mul opt-in",
                )
            )
            emit_vendor_planning_note("multiply")
            return native
        build_native_mul()
    emit(DispatchRecord("multiply", "torch-reference", device, dtype, "native unavailable or not requested"))
    emit_vendor_planning_note("multiply")
    return torch.mul(a, b)
