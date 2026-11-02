from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

import musa_aiter.ops.kv_transfer as kv_transfer
import musa_aiter.ops.lmcache_kv_transfer as lmcache_transfer


def test_lmcache_float32_list_uses_float_abi(monkeypatch):
    memory = torch.zeros(2, 2, 3, 4, dtype=torch.float32)
    caches = [torch.zeros(2, 2, 2, 1, 2, dtype=torch.float32) for _ in range(2)]
    slots = torch.tensor([0, 1, 2], dtype=torch.int64)
    calls = []

    monkeypatch.setattr(lmcache_transfer, "_can_native_list", lambda *args: True)
    monkeypatch.setattr(lmcache_transfer, "is_native_lmcache_kv_transfer_available", lambda: True)
    monkeypatch.setattr(
        lmcache_transfer,
        "_try_cpp_extension_non_mla_list_to_paged",
        lambda *args: False,
    )
    monkeypatch.setattr(
        lmcache_transfer,
        "_call_non_mla_list_symbol",
        lambda symbol, *args: calls.append(symbol) or True,
    )

    assert lmcache_transfer._try_native_non_mla_list_to_paged(
        memory, caches, slots, 0, 2, 1, 2
    )
    assert calls == ["lmcache_kv_buffer_to_paged_list_float"] * 2


def test_lmcache_float32_two_x_nl_uses_float_abi(monkeypatch):
    memory = torch.zeros(2, 2, 3, 2, dtype=torch.float32)
    caches = [
        [torch.zeros(2, 2, 1, 2, dtype=torch.float32) for _ in range(2)]
        for _ in range(2)
    ]
    slots = torch.tensor([0, 1, 2], dtype=torch.int64)
    calls = []

    monkeypatch.setattr(lmcache_transfer, "_can_native_list", lambda *args: True)
    monkeypatch.setattr(
        lmcache_transfer,
        "is_native_lmcache_two_x_nl_kv_transfer_available",
        lambda: True,
    )
    monkeypatch.setattr(
        lmcache_transfer,
        "_try_cpp_extension_two_x_nl_to_paged",
        lambda *args: False,
    )
    monkeypatch.setattr(
        lmcache_transfer,
        "_call_two_x_nl_symbol",
        lambda symbol, *args: calls.append(symbol) or True,
    )

    assert lmcache_transfer._try_native_two_x_nl_to_paged(
        memory, caches, slots, 0, 2, 1, 2
    )
    assert calls == ["lmcache_kv_buffer_to_paged_two_x_nl_float"] * 4


def test_lmcache_float32_list_mla_uses_float_abi(monkeypatch):
    memory = torch.zeros(1, 2, 3, 4, dtype=torch.float32)
    caches = [torch.zeros(2, 2, 4, dtype=torch.float32) for _ in range(2)]
    slots = torch.tensor([0, 1, 2], dtype=torch.int64)
    calls = []

    monkeypatch.setattr(lmcache_transfer, "_can_native_list", lambda *args: True)
    monkeypatch.setattr(
        lmcache_transfer,
        "_call_mla_list_symbol",
        lambda symbol, *args: calls.append(symbol) or True,
    )

    assert lmcache_transfer._try_native_mla_list_to_paged(
        memory, caches, slots, 0, 2, 4
    )
    assert calls == ["lmcache_mla_buffer_to_paged_list_float"] * 2


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_merged_native_selects_dtype_specific_symbol(monkeypatch, dtype):
    key_cache = torch.zeros(2, 1, 4, 2, dtype=dtype)
    value_cache = torch.zeros_like(key_cache)
    slots = torch.tensor([0, 1], dtype=torch.int64)
    out = torch.zeros(2, 2, 1, 4, dtype=dtype)
    calls = []

    monkeypatch.setenv("MUSA_AITER_NATIVE_KV_TRANSFER", "1")
    monkeypatch.setattr(
        kv_transfer,
        "_is_musa_native_kv_dtype_tensor",
        lambda tensor: tensor.dtype in (torch.float16, torch.bfloat16, torch.float32),
    )
    monkeypatch.setattr(
        kv_transfer,
        "_get_native_kv_function",
        lambda _lib, base, actual_dtype: calls.append((base, actual_dtype)) or (lambda *args: None),
    )
    monkeypatch.setattr(kv_transfer, "load_native_kv_transfer", lambda: SimpleNamespace())

    result = kv_transfer.kv_paged_to_merged_linear(
        key_cache,
        value_cache,
        slots,
        out=out,
    )

    assert result is out
    assert calls == [("musa_aiter_kv_paged_to_merged_linear", dtype)]
