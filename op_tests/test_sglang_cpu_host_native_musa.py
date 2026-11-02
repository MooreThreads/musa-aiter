from __future__ import annotations

import logging
import os

import pytest
import torch

import musa_aiter
from musa_aiter.integration.sglang_kv import (
    try_sglang_hicache_layer_first_cpu_host_transfer,
    try_sglang_hicache_layer_first_cpu_host_transfer_multi_layer,
)


def _require_host_native() -> None:
    if not hasattr(torch, "musa") or not torch.musa.is_available():
        pytest.skip("TorchMUSA with a visible MUSA device is required")
    if os.environ.get("MUSA_AITER_SGLANG_ADAPTER", "0") != "1":
        pytest.skip("set MUSA_AITER_SGLANG_ADAPTER=1")
    if os.environ.get("MUSA_AITER_NATIVE_CPU_HOST_KV_TRANSFER", "0") != "1":
        pytest.skip("set MUSA_AITER_NATIVE_CPU_HOST_KV_TRANSFER=1")
    if not musa_aiter.is_native_lmcache_musa_batch_copy_available():
        pytest.skip("MUSA SDK batch-copy extension is unavailable")


def _enable_native_route(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MUSA_AITER_LMCACHE_BATCH_COPY", "1")
    # _log_route / _profile_route only emit when route logging is enabled, and the
    # native-evidence assertion below reads the route log.
    monkeypatch.setenv("MUSA_AITER_SGLANG_KV_LOG", "1")


def _layer_first_cpu_cache(shape: tuple[int, ...], offset: float) -> torch.Tensor:
    numel = 1
    for dim in shape:
        numel *= dim
    return (torch.arange(numel, dtype=torch.float32).reshape(shape) + offset).pin_memory()


@pytest.mark.musa
@pytest.mark.native
@pytest.mark.parametrize("direction", ["host_to_device", "device_to_host"])
def test_sglang_cpu_host_staging_native_single_layer(
    direction: str, caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    """3-D single-layer entry: host/device caches are [tokens, kv_heads, head_dim]."""
    _require_host_native()
    _enable_native_route(monkeypatch)
    caplog.set_level(logging.INFO)

    tokens, heads, head_dim, page_size = 4, 1, 4, 2
    shape = (tokens, heads, head_dim)
    host_indices = torch.arange(tokens, dtype=torch.int64)
    device_indices = host_indices.to("musa")

    if direction == "host_to_device":
        host_key = _layer_first_cpu_cache(shape, 0.0)
        host_value = _layer_first_cpu_cache(shape, 100.0)
        device_key = torch.zeros(shape, dtype=torch.float32, device="musa")
        device_value = torch.zeros(shape, dtype=torch.float32, device="musa")
        expected_key, expected_value = host_key.clone(), host_value.clone()
    else:
        device_key = _layer_first_cpu_cache(shape, 0.0).to("musa")
        device_value = _layer_first_cpu_cache(shape, 100.0).to("musa")
        expected_key, expected_value = device_key.cpu().clone(), device_value.cpu().clone()
        host_key = torch.zeros(shape, dtype=torch.float32).pin_memory()
        host_value = torch.zeros(shape, dtype=torch.float32).pin_memory()

    routed = try_sglang_hicache_layer_first_cpu_host_transfer(
        host_key,
        host_value,
        device_key,
        device_value,
        host_indices,
        device_indices,
        page_size=page_size,
        direction=direction,
    )
    assert routed, "native single-layer CPU-host route was not taken:\n" + caplog.text
    torch.musa.synchronize()

    if direction == "host_to_device":
        torch.testing.assert_close(device_key.cpu(), expected_key)
        torch.testing.assert_close(device_value.cpu(), expected_value)
    else:
        torch.testing.assert_close(host_key, expected_key)
        torch.testing.assert_close(host_value, expected_value)
    assert "backend=musa-sdk-batch-copy" in caplog.text


@pytest.mark.musa
@pytest.mark.native
@pytest.mark.parametrize("direction", ["host_to_device", "device_to_host"])
def test_sglang_cpu_host_staging_native_multi_layer(
    direction: str, caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    """4-D multi-layer entry: host/device caches are [layers, tokens, kv_heads, head_dim]."""
    _require_host_native()
    _enable_native_route(monkeypatch)
    caplog.set_level(logging.INFO)

    layers, tokens, heads, head_dim, page_size = 2, 4, 1, 4, 2
    shape = (layers, tokens, heads, head_dim)
    host_indices = torch.arange(tokens, dtype=torch.int64)
    device_indices = host_indices.to("musa")

    if direction == "host_to_device":
        host_key = _layer_first_cpu_cache(shape, 0.0)
        host_value = _layer_first_cpu_cache(shape, 100.0)
        device_key = torch.zeros(shape, dtype=torch.float32, device="musa")
        device_value = torch.zeros(shape, dtype=torch.float32, device="musa")
        expected_key, expected_value = host_key.clone(), host_value.clone()
    else:
        device_key = _layer_first_cpu_cache(shape, 0.0).to("musa")
        device_value = _layer_first_cpu_cache(shape, 100.0).to("musa")
        expected_key, expected_value = device_key.cpu().clone(), device_value.cpu().clone()
        host_key = torch.zeros(shape, dtype=torch.float32).pin_memory()
        host_value = torch.zeros(shape, dtype=torch.float32).pin_memory()

    routed = try_sglang_hicache_layer_first_cpu_host_transfer_multi_layer(
        host_key,
        host_value,
        device_key,
        device_value,
        host_indices,
        device_indices,
        page_size=page_size,
        direction=direction,
    )
    assert routed, "native multi-layer CPU-host route was not taken:\n" + caplog.text
    torch.musa.synchronize()

    if direction == "host_to_device":
        torch.testing.assert_close(device_key.cpu(), expected_key)
        torch.testing.assert_close(device_value.cpu(), expected_value)
    else:
        torch.testing.assert_close(host_key, expected_key)
        torch.testing.assert_close(host_value, expected_value)
    assert "backend=musa-sdk-batch-copy" in caplog.text
