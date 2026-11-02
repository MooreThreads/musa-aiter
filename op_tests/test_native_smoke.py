import shutil

import pytest

pytestmark = [
    pytest.mark.native,
    pytest.mark.skipif(shutil.which("mcc") is None, reason="mcc compiler is not available"),
]

from musa_aiter.jit.ctypes_loader import has_symbol, load_library
from musa_aiter.ops.smoke import (
    build_native_add,
    build_native_mul,
    native_smoke_abi_version,
    native_smoke_mul_abi_version,
)

from musa_aiter.ops.smoke_reduce import (
    build_native_reduce_sum,
    native_smoke_reduce_abi_version,
)
from musa_aiter.ops.smoke_softmax import (
    build_native_softmax_last_dim,
    native_smoke_softmax_abi_version,
)
from musa_aiter.ops.smoke_rmsnorm import (
    build_native_rmsnorm,
    native_smoke_rmsnorm_abi_version,
)


def test_build_smoke_add_abi_symbol():
    result = build_native_add()
    assert result.ok, result.message
    assert result.library_path is not None
    lib = load_library(result.library_path)
    assert has_symbol(lib, "musa_aiter_smoke_abi_version")
    assert native_smoke_abi_version() == 1


def test_build_smoke_mul_abi_symbol():
    result = build_native_mul()
    assert result.ok, result.message
    assert result.library_path is not None
    lib = load_library(result.library_path)
    assert has_symbol(lib, "musa_aiter_smoke_mul_abi_version")
    assert native_smoke_mul_abi_version() == 1


def test_build_smoke_reduce_abi_symbol():
    result = build_native_reduce_sum()
    assert result.ok, result.message
    assert result.library_path is not None
    lib = load_library(result.library_path)
    assert has_symbol(lib, "musa_aiter_smoke_reduce_abi_version")
    assert native_smoke_reduce_abi_version() == 1


def test_build_smoke_softmax_ld_abi_symbol():
    result = build_native_softmax_last_dim()
    assert result.ok, result.message
    assert result.library_path is not None
    lib = load_library(result.library_path)
    assert has_symbol(lib, "musa_aiter_smoke_softmax_abi_version")
    assert native_smoke_softmax_abi_version() == 1


def test_build_smoke_rmsnorm_abi_symbol():
    result = build_native_rmsnorm()
    assert result.ok, result.message
    assert result.library_path is not None
    lib = load_library(result.library_path)
    assert has_symbol(lib, "musa_aiter_smoke_rmsnorm_abi_version")
    assert native_smoke_rmsnorm_abi_version() == 1
