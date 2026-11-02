from __future__ import annotations

import torch

import musa_aiter.integration.sglang_kv as sglang_kv


def test_cpu_host_staging_uses_batch_copy_only_for_pinned_native_route(monkeypatch):
    destination = torch.zeros(16, dtype=torch.float32)
    source = torch.ones_like(destination)
    calls = []

    monkeypatch.setattr(
        sglang_kv,
        "is_sglang_native_cpu_host_kv_transfer_enabled",
        lambda: True,
    )
    monkeypatch.setattr(sglang_kv, "_is_cpu_tensor", lambda tensor: tensor is source)
    monkeypatch.setattr(sglang_kv, "_is_musa_tensor", lambda tensor: tensor is destination)
    monkeypatch.setattr(sglang_kv, "_host_tensor_is_pinned", lambda _: True)
    monkeypatch.setattr(
        sglang_kv,
        "lmcache_musa_memcpy_batch_async",
        lambda destinations, sources, sizes, **kwargs: calls.append(
            (destinations, sources, sizes, kwargs)
        )
        or True,
    )

    assert sglang_kv._try_native_cpu_host_staging_copy(destination, source, op="test_h2d")
    assert calls[0][2] == [source.numel() * source.element_size()]
    assert calls[0][3]["require_pinned_host"] is True


def test_cpu_host_staging_falls_back_when_host_is_pageable_or_gate_is_off(monkeypatch):
    destination = torch.zeros(4, dtype=torch.float32)
    source = torch.ones_like(destination)
    batch_calls = []
    monkeypatch.setattr(sglang_kv, "lmcache_musa_memcpy_batch_async", lambda *args, **kwargs: batch_calls.append(1))
    monkeypatch.setattr(sglang_kv, "_is_cpu_tensor", lambda tensor: tensor is source)
    monkeypatch.setattr(sglang_kv, "_is_musa_tensor", lambda tensor: tensor is destination)

    monkeypatch.setattr(
        sglang_kv,
        "is_sglang_native_cpu_host_kv_transfer_enabled",
        lambda: False,
    )
    assert not sglang_kv._try_native_cpu_host_staging_copy(destination, source, op="gate_off")

    monkeypatch.setattr(
        sglang_kv,
        "is_sglang_native_cpu_host_kv_transfer_enabled",
        lambda: True,
    )
    monkeypatch.setattr(sglang_kv, "_host_tensor_is_pinned", lambda _: False)
    assert not sglang_kv._try_native_cpu_host_staging_copy(destination, source, op="pageable")
    assert batch_calls == []
