"""JIT/ctypes RMSNorm smoke kernel (KERNEL plan — norm path toward attention stacks)."""

from __future__ import annotations

import ctypes
import math
import os

import torch

from musa_aiter._paths import project_data_path
from musa_aiter.jit.core import build_shared_object
from musa_aiter.jit.ctypes_loader import has_symbol, load_library

_RMSNORM_SOURCE = project_data_path("csrc", "norm", "rmsnorm_float.mu")

_RMS_ATTEMPTED = False
_RMS_RESULT = None
_RMS_LIB = None

_RMSNORM_NATIVE_MAX_DIM = 4096


def build_native_rmsnorm():
    global _RMS_ATTEMPTED, _RMS_RESULT
    if not _RMS_ATTEMPTED:
        _RMS_RESULT = build_shared_object(name="smoke_rmsnorm", sources=[_RMSNORM_SOURCE])
        _RMS_ATTEMPTED = True
    return _RMS_RESULT


def load_native_rmsnorm():
    global _RMS_LIB
    result = build_native_rmsnorm()
    if not result.ok or result.library_path is None:
        raise RuntimeError(result.message)
    if _RMS_LIB is None:
        _RMS_LIB = load_library(result.library_path)
    return _RMS_LIB


def native_smoke_rmsnorm_abi_version() -> int | None:
    try:
        lib = load_native_rmsnorm()
        if not has_symbol(lib, "musa_aiter_smoke_rmsnorm_abi_version"):
            return None
        fn = lib.handle.musa_aiter_smoke_rmsnorm_abi_version
        fn.argtypes = []
        fn.restype = ctypes.c_int
        return int(fn())
    except Exception:
        return None


def is_native_rmsnorm_available() -> bool:
    result = build_native_rmsnorm()
    if not bool(result and result.ok and result.library_path):
        return False
    try:
        lib = load_native_rmsnorm()
        return has_symbol(lib, "musa_aiter_rmsnorm_float")
    except Exception:
        return False


def try_native_rmsnorm(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> torch.Tensor | None:
    if os.environ.get("MUSA_AITER_NATIVE_SMOKE", "0") != "1":
        return None
    if str(x.device).split(":")[0] != "musa":
        return None
    if x.dtype is not torch.float32 or weight.dtype is not torch.float32:
        return None

    dim = int(x.shape[-1])
    if dim > _RMSNORM_NATIVE_MAX_DIM or dim < 1:
        return None
    # Smoke ABI: strictly 1-D weight[d] matching reference layout used in kernels.
    if weight.ndim != 1 or weight.shape[0] != dim:
        return None

    if not x.is_contiguous():
        x = x.contiguous()
    if not weight.is_contiguous():
        weight = weight.contiguous()

    nd = x.ndim
    outer = int(math.prod(x.shape[:-1])) if nd > 1 else 1
    x2 = x.reshape(outer, dim)

    lib = load_native_rmsnorm()
    out = torch.empty_like(x2)
    fn = lib.handle.musa_aiter_rmsnorm_float
    fn.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int32,
        ctypes.c_int32,
        ctypes.c_float,
        ctypes.c_void_p,
    ]
    fn.restype = None
    fn(
        ctypes.c_void_p(x2.data_ptr()),
        ctypes.c_void_p(weight.data_ptr()),
        ctypes.c_void_p(out.data_ptr()),
        ctypes.c_int32(outer),
        ctypes.c_int32(dim),
        ctypes.c_float(float(eps)),
        ctypes.c_void_p(0),
    )
    return out.reshape(x.shape)


RMSNORM_NATIVE_MAX_DIM = _RMSNORM_NATIVE_MAX_DIM
