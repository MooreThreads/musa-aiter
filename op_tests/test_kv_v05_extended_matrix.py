from __future__ import annotations

import os

import pytest
import torch

import musa_aiter
from musa_aiter.ops import kv_transfer as kv_transfer_impl


def test_multi_layer_mla_roundtrip_supports_asymmetric_hidden_sizes():
    layers, blocks, block_size = 2, 3, 4
    kv_dim, pe_dim = 5, 3
    key_cache = torch.arange(
        layers * blocks * block_size * kv_dim, dtype=torch.float32
    ).reshape(layers, blocks, block_size, kv_dim)
    pe_cache = key_cache[..., :pe_dim].clone()
    slots = torch.tensor([0, 5, 8, -1], dtype=torch.int64)
    kv_linear = torch.randn(layers, slots.numel(), kv_dim)
    pe_linear = torch.randn(layers, slots.numel(), pe_dim)

    assert musa_aiter.kv_mla_linear_to_paged_multi_layer(
        kv_linear,
        pe_linear,
        key_cache,
        pe_cache,
        slots,
    )
    recovered_kv, recovered_pe = musa_aiter.kv_paged_to_mla_linear_multi_layer(
        key_cache,
        pe_cache,
        slots,
    )

    torch.testing.assert_close(recovered_kv[:, :3], kv_linear[:, :3])
    torch.testing.assert_close(recovered_pe[:, :3], pe_linear[:, :3])
    assert torch.count_nonzero(recovered_kv[:, 3]) == 0
    assert torch.count_nonzero(recovered_pe[:, 3]) == 0


def test_interleaved_fp8_layout_exposes_kv_format_facts():
    layout = musa_aiter.InterleavedFp8Layout(
        value_bytes=128,
        scale_bytes=4,
        scale_granularity=16,
    )

    facts = layout.as_kv_layout_spec(
        num_layers=2,
        num_blocks=8,
        block_size=16,
    ).format_facts()

    assert facts.value_bytes == 128
    assert facts.scale_bytes == 4
    assert facts.requires_transform is True


def test_multi_layer_mla_uses_native_hooks_when_available(monkeypatch):
    gather_calls = []
    scatter_calls = []

    def fake_gather(*args, **kwargs):
        gather_calls.append((args, kwargs))
        return (
            torch.zeros(2, 3, 5),
            torch.zeros(2, 3, 3),
        )

    def fake_scatter(*args, **kwargs):
        scatter_calls.append((args, kwargs))
        return True

    monkeypatch.setattr(
        kv_transfer_impl,
        "_try_native_kv_paged_to_mla_linear_multi_layer",
        fake_gather,
    )
    monkeypatch.setattr(
        kv_transfer_impl,
        "_try_native_kv_mla_linear_to_paged_multi_layer",
        fake_scatter,
    )

    key_cache = torch.zeros(2, 3, 4, 5)
    pe_cache = torch.zeros(2, 3, 4, 3)
    slots = torch.tensor([0, 5, -1])
    kv_linear = torch.randn(2, 3, 5)
    pe_linear = torch.randn(2, 3, 3)

    assert musa_aiter.kv_mla_linear_to_paged_multi_layer(
        kv_linear,
        pe_linear,
        key_cache,
        pe_cache,
        slots,
    )
    recovered_kv, recovered_pe = musa_aiter.kv_paged_to_mla_linear_multi_layer(
        key_cache,
        pe_cache,
        slots,
    )

    assert len(scatter_calls) == 1
    assert len(gather_calls) == 1
    assert recovered_kv.shape == (2, 3, 5)
    assert recovered_pe.shape == (2, 3, 3)


@pytest.mark.musa
@pytest.mark.native
def test_multi_layer_mla_native_roundtrip_when_enabled():
    if not hasattr(torch, "musa") or not torch.musa.is_available():
        pytest.skip("TorchMUSA with a visible MUSA device is required")
    if os.environ.get("MUSA_AITER_NATIVE_KV_TRANSFER", "0") != "1":
        pytest.skip("set MUSA_AITER_NATIVE_KV_TRANSFER=1")
    if not musa_aiter.is_native_lmcache_kv_transfer_available():
        pytest.skip("native LMCache KV transfer .so is unavailable")

    layers, blocks, block_size = 2, 3, 4
    kv_dim, pe_dim, tokens = 5, 3, 4
    key_cache = torch.zeros(layers, blocks, block_size, kv_dim, device="musa")
    pe_cache = torch.zeros(layers, blocks, block_size, pe_dim, device="musa")
    slots = torch.tensor([0, 5, 8, -1], dtype=torch.int64, device="musa")
    kv_linear = torch.randn(layers, tokens, kv_dim, device="musa")
    pe_linear = torch.randn(layers, tokens, pe_dim, device="musa")

    assert musa_aiter.kv_mla_linear_to_paged_multi_layer(
        kv_linear,
        pe_linear,
        key_cache,
        pe_cache,
        slots,
    )
    recovered_kv, recovered_pe = musa_aiter.kv_paged_to_mla_linear_multi_layer(
        key_cache,
        pe_cache,
        slots,
    )
    torch.musa.synchronize()
    valid = torch.tensor([True, True, True, False])
    torch.testing.assert_close(recovered_kv[:, valid].cpu(), kv_linear[:, valid].cpu())
    torch.testing.assert_close(recovered_pe[:, valid].cpu(), pe_linear[:, valid].cpu())
    assert recovered_kv[:, 3].abs().max().item() == 0
    assert recovered_pe[:, 3].abs().max().item() == 0
