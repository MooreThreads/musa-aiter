from __future__ import annotations

import os

import pytest
import torch

import musa_aiter
from musa_aiter.ops import kv_transfer as kv_transfer_impl


def test_flash_reshape_reverse_roundtrip_writes_partial_pages():
    nb, kvh, head_dim, block_size = 4, 2, 3, 4
    dense_k = torch.arange(2 * 7 * kvh * head_dim, dtype=torch.float32).reshape(
        2, 7, kvh, head_dim
    )
    dense_v = dense_k + 1000
    key_cache = torch.zeros(nb, kvh, head_dim, block_size)
    value_cache = torch.zeros_like(key_cache)
    block_tables = torch.tensor([[1, 3], [0, 2]], dtype=torch.int32)
    seq_lens = torch.tensor([6, 3], dtype=torch.int32)

    assert musa_aiter.kv_reshape_and_cache_back_flash(
        dense_k,
        dense_v,
        key_cache,
        value_cache,
        block_tables,
        seq_lens,
    )

    recovered_k, recovered_v = musa_aiter.kv_load_and_reshape_flash(
        key_cache,
        value_cache,
        block_tables,
        seq_lens,
        max_seq_len=7,
    )

    torch.testing.assert_close(recovered_k[0, :6], dense_k[0, :6])
    torch.testing.assert_close(recovered_v[0, :6], dense_v[0, :6])
    torch.testing.assert_close(recovered_k[1, :3], dense_k[1, :3])
    torch.testing.assert_close(recovered_v[1, :3], dense_v[1, :3])
    assert recovered_k[0, 6].abs().max() == 0
    assert recovered_k[1, 3:].abs().max() == 0


def test_flash_reshape_reverse_rejects_invalid_dense_shape():
    with pytest.raises(ValueError, match="dense"):
        musa_aiter.kv_reshape_and_cache_back_flash(
            torch.zeros(2, 7, 3),
            torch.zeros(2, 7, 3),
            torch.zeros(4, 2, 3, 4),
            torch.zeros(4, 2, 3, 4),
            torch.tensor([[0, 1], [2, 3]], dtype=torch.int32),
            torch.tensor([4, 4], dtype=torch.int32),
        )


def test_flash_reshape_reverse_uses_native_hook_when_available(monkeypatch):
    calls = []

    def fake_native(*args, **kwargs):
        calls.append((args, kwargs))
        return True

    monkeypatch.setattr(
        kv_transfer_impl,
        "_try_native_kv_reshape_and_cache_back_flash",
        fake_native,
    )
    dense_k = torch.zeros(1, 2, 1, 2, dtype=torch.float32)
    dense_v = torch.zeros_like(dense_k)
    key_cache = torch.zeros(2, 1, 2, 2, dtype=torch.float32)
    value_cache = torch.zeros_like(key_cache)
    block_tables = torch.tensor([[0]], dtype=torch.int32)
    seq_lens = torch.tensor([2], dtype=torch.int32)

    assert musa_aiter.kv_reshape_and_cache_back_flash(
        dense_k,
        dense_v,
        key_cache,
        value_cache,
        block_tables,
        seq_lens,
    )
    assert len(calls) == 1


@pytest.mark.musa
@pytest.mark.native
def test_flash_reshape_reverse_native_roundtrip_when_enabled():
    if not hasattr(torch, "musa") or not torch.musa.is_available():
        pytest.skip("TorchMUSA with a visible MUSA device is required")
    if os.environ.get("MUSA_AITER_NATIVE_KV_TRANSFER", "0") != "1":
        pytest.skip("set MUSA_AITER_NATIVE_KV_TRANSFER=1")
    if not musa_aiter.is_native_kv_transfer_available():
        pytest.skip("native KV transfer .so is unavailable")

    batch, dense_len, kvh, head_dim, block_size = 2, 7, 2, 3, 4
    dense_k = torch.randn(batch, dense_len, kvh, head_dim, device="musa")
    dense_v = torch.randn_like(dense_k)
    key_cache = torch.zeros(4, kvh, head_dim, block_size, device="musa")
    value_cache = torch.zeros_like(key_cache)
    block_tables = torch.tensor([[1, 3], [0, 2]], dtype=torch.int32, device="musa")
    seq_lens = torch.tensor([6, 3], dtype=torch.int32, device="musa")

    assert musa_aiter.kv_reshape_and_cache_back_flash(
        dense_k,
        dense_v,
        key_cache,
        value_cache,
        block_tables,
        seq_lens,
    )
    recovered_k, recovered_v = musa_aiter.kv_load_and_reshape_flash(
        key_cache,
        value_cache,
        block_tables,
        seq_lens,
        max_seq_len=dense_len,
    )
    torch.musa.synchronize()
    torch.testing.assert_close(recovered_k[:, :6].cpu(), dense_k[:, :6].cpu())
    torch.testing.assert_close(recovered_v[:, :6].cpu(), dense_v[:, :6].cpu())
