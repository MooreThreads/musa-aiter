from __future__ import annotations

import torch

import musa_aiter.ops.hicache_transfer as hicache
import musa_aiter.ops.lmcache_kv_transfer as lmcache
from musa_aiter.ops.native_gate import decide_native_gate, native_requested


def test_native_gate_accepts_family_alias_and_reports_reference_mode(monkeypatch):
    monkeypatch.delenv("MUSA_AITER_NATIVE_HICACHE_TRANSFER", raising=False)
    monkeypatch.setenv("MUSA_AITER_NATIVE_KV_TRANSFER", "1")
    assert native_requested(
        "MUSA_AITER_NATIVE_HICACHE_TRANSFER",
        "MUSA_AITER_NATIVE_KV_TRANSFER",
    )
    decision = decide_native_gate(
        "hicache_transfer",
        (
            "MUSA_AITER_NATIVE_HICACHE_TRANSFER",
            "MUSA_AITER_NATIVE_KV_TRANSFER",
        ),
        available=False,
    )
    assert decision.mode == "fallback"
    assert decision.reason == "native_unavailable"


def test_native_gate_off_is_reference_even_when_symbol_is_available(monkeypatch):
    monkeypatch.delenv("MUSA_AITER_NATIVE_KV_TRANSFER", raising=False)
    decision = decide_native_gate(
        "lmcache",
        ("MUSA_AITER_NATIVE_KV_TRANSFER",),
        available=True,
    )
    assert decision.mode == "reference"
    assert decision.reason == "native_gate_off"
    monkeypatch.setenv("MUSA_AITER_NATIVE_KV_TRANSFER", "1")
    required = decide_native_gate(
        "lmcache",
        ("MUSA_AITER_NATIVE_KV_TRANSFER",),
        available=False,
        native_required=True,
    )
    assert required.mode == "error"


def test_strided_block_transfer_does_not_load_native_symbol_when_gate_is_off(monkeypatch):
    monkeypatch.delenv("MUSA_AITER_NATIVE_LMCACHE_KV_TRANSFER", raising=False)
    monkeypatch.delenv("LMCACHE_MUSA_NATIVE_KV_TRANSFER", raising=False)
    monkeypatch.delenv("MUSA_AITER_NATIVE_KV_TRANSFER", raising=False)
    calls = []
    monkeypatch.setattr(lmcache, "_cached_native_fn", lambda *args: calls.append(args) or object())
    block_ids = torch.tensor([0], dtype=torch.int64)

    assert not lmcache.lmcache_kv_strided_block_transfer(
        torch.tensor([1], dtype=torch.uint8),
        [1],
        block_ids,
        0,
        1,
        0,
        1,
        0,
        1,
        1,
        4,
        4,
        0,
    )
    assert calls == []


def test_family_gate_helpers_share_the_canonical_policy(monkeypatch):
    monkeypatch.setenv("MUSA_AITER_NATIVE_KV_TRANSFER", "1")
    assert hicache._native_requested()
    assert lmcache._native_requested()


def test_native_capability_cache_can_be_cleared_after_rebuild():
    lmcache._NATIVE_AVAILABLE_CACHE = True
    lmcache._NATIVE_FN_CACHE["stale"] = object()
    lmcache.clear_native_lmcache_kv_transfer_cache()
    assert lmcache._NATIVE_AVAILABLE_CACHE is None
    assert lmcache._NATIVE_FN_CACHE == {}
