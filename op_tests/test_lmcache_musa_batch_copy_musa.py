from __future__ import annotations

import os

import pytest
import torch

import musa_aiter


def _require_native_batch_copy() -> None:
    if not hasattr(torch, "musa") or not torch.musa.is_available():
        pytest.skip("TorchMUSA with a visible MUSA device is required")
    if os.environ.get("MUSA_AITER_LMCACHE_BATCH_COPY", "0") != "1":
        pytest.skip("set MUSA_AITER_LMCACHE_BATCH_COPY=1")
    if not musa_aiter.is_native_lmcache_musa_batch_copy_available():
        pytest.skip("MUSA SDK 5.1+ batch-copy extension is unavailable")


@pytest.mark.musa
@pytest.mark.native
@pytest.mark.parametrize("direction", ["h2d", "d2h"])
def test_native_batch_copy_roundtrip(direction: str) -> None:
    _require_native_batch_copy()
    host = torch.arange(256, dtype=torch.uint8, pin_memory=True)
    device = torch.zeros(256, dtype=torch.uint8, device="musa")

    if direction == "h2d":
        destinations, sources = [device], [host]
    else:
        device.copy_(host)
        host.zero_()
        destinations, sources = [host], [device]

    assert musa_aiter.lmcache_musa_memcpy_batch_async(
        destinations,
        sources,
        [96],
        destination_offsets=[64],
        source_offsets=[32],
    )
    torch.musa.synchronize()

    if direction == "h2d":
        torch.testing.assert_close(device[64:160].cpu(), host[32:128])
    else:
        torch.testing.assert_close(host[64:160], torch.arange(32, 128, dtype=torch.uint8))


@pytest.mark.musa
@pytest.mark.native
def test_native_batch_copy_uses_current_non_default_stream() -> None:
    _require_native_batch_copy()
    host = torch.arange(128, dtype=torch.uint8, pin_memory=True)
    device = torch.zeros(128, dtype=torch.uint8, device="musa")
    stream = torch.musa.Stream()

    with torch.musa.stream(stream):
        assert musa_aiter.lmcache_musa_memcpy_batch_async([device], [host], [128])
        device.add_(1)
    stream.synchronize()

    torch.testing.assert_close(device.cpu(), host + 1)


@pytest.mark.musa
@pytest.mark.native
def test_native_batch_copy_observes_non_default_stream_producer() -> None:
    _require_native_batch_copy()
    host = torch.zeros(128, dtype=torch.uint8, pin_memory=True)
    device = torch.zeros(128, dtype=torch.uint8, device="musa")
    stream = torch.musa.Stream()

    with torch.musa.stream(stream):
        device.fill_(37)
        assert musa_aiter.lmcache_musa_memcpy_batch_async([host], [device], [128])
    stream.synchronize()

    torch.testing.assert_close(host, torch.full_like(host, 37))


@pytest.mark.musa
@pytest.mark.native
def test_native_batch_copy_splits_at_host_registration_boundary() -> None:
    _require_native_batch_copy()
    host = torch.arange(64, dtype=torch.uint8, pin_memory=True)
    device = torch.zeros(64, dtype=torch.uint8, device="musa")

    assert musa_aiter.lmcache_musa_memcpy_batch_async(
        [device],
        [host],
        [20],
        destination_offsets=[10],
        source_offsets=[6],
        host_registration_offsets=[6],
        pin_chunk_size=8,
    )
    torch.musa.synchronize()

    torch.testing.assert_close(device[10:30].cpu(), host[6:26])


@pytest.mark.musa
@pytest.mark.native
def test_native_batch_copy_chunks_large_entry_tables_in_stream_order() -> None:
    _require_native_batch_copy()
    versions = musa_aiter.native_lmcache_musa_batch_copy_runtime_versions()
    assert versions is not None
    assert versions[0] >= 5010
    assert versions[1] >= 5010

    entry_count = 1300
    host = torch.arange(entry_count, dtype=torch.uint8, pin_memory=True)
    device = torch.zeros(entry_count, dtype=torch.uint8, device="musa")
    offsets = list(range(entry_count))

    assert musa_aiter.lmcache_musa_memcpy_batch_async(
        [device] * entry_count,
        [host] * entry_count,
        [1] * entry_count,
        destination_offsets=offsets,
        source_offsets=offsets,
        max_batch_entries=512,
    )
    torch.musa.synchronize()

    torch.testing.assert_close(device.cpu(), host)
