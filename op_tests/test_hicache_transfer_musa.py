from __future__ import annotations

import os

import pytest
import torch

import musa_aiter


def _require_native_hicache() -> None:
    if not hasattr(torch, "musa") or not torch.musa.is_available():
        pytest.skip("TorchMUSA with a visible MUSA device is required")
    if os.environ.get("MUSA_AITER_NATIVE_HICACHE_TRANSFER", "0") != "1":
        pytest.skip("set MUSA_AITER_NATIVE_HICACHE_TRANSFER=1")
    if not musa_aiter.is_native_hicache_transfer_available():
        pytest.skip("HiCache native MUSA extension is unavailable")


@pytest.mark.musa
@pytest.mark.native
def test_native_hicache_page_first_to_layer_first_per_layer() -> None:
    _require_native_hicache()
    layers, pages, item = 2, 4, 8
    src_k = torch.arange(pages * layers * item, dtype=torch.float16, device="musa").reshape(
        pages, layers, item
    )
    src_v = src_k + 1000
    dst_k = torch.zeros((layers, pages, item), dtype=torch.float16, device="musa")
    dst_v = torch.zeros_like(dst_k)
    src_indices = torch.tensor([3, 1], dtype=torch.int64, device="musa")
    dst_indices = torch.tensor([0, 2], dtype=torch.int64, device="musa")

    assert musa_aiter.hicache_transfer_kv_per_layer_pf_lf(
        src_k,
        dst_k,
        src_v,
        dst_v,
        src_indices,
        dst_indices,
        layer_id=1,
    )
    torch.musa.synchronize()

    torch.testing.assert_close(dst_k[1, 0], src_k[3, 1])
    torch.testing.assert_close(dst_k[1, 2], src_k[1, 1])
    torch.testing.assert_close(dst_v[1, 0], src_v[3, 1])


@pytest.mark.musa
@pytest.mark.native
def test_native_hicache_all_layer_lf_to_pf_uses_current_stream() -> None:
    _require_native_hicache()
    layers, pages, item = 2, 4, 8
    src_k = torch.arange(layers * pages * item, dtype=torch.float16, device="musa").reshape(
        layers, pages, item
    )
    src_v = src_k + 2000
    dst_k = torch.zeros((pages, layers, item), dtype=torch.float16, device="musa")
    dst_v = torch.zeros_like(dst_k)
    src_indices = torch.tensor([0, 2], dtype=torch.int64, device="musa")
    dst_indices = torch.tensor([1, 3], dtype=torch.int64, device="musa")
    stream = torch.musa.Stream()

    with torch.musa.stream(stream):
        assert musa_aiter.hicache_transfer_kv_all_layer_lf_pf(
            src_k,
            dst_k,
            src_v,
            dst_v,
            src_indices,
            dst_indices,
        )
    stream.synchronize()

    torch.testing.assert_close(dst_k[1, 0], src_k[0, 0])
    torch.testing.assert_close(dst_k[1, 1], src_k[1, 0])
    torch.testing.assert_close(dst_k[3, 0], src_k[0, 2])
    torch.testing.assert_close(dst_v[3, 1], src_v[1, 2])


@pytest.mark.musa
@pytest.mark.native
def test_native_hicache_all_layer_mla_lf_to_pf_uses_current_stream() -> None:
    _require_native_hicache()
    layers, pages, item = 2, 4, 8
    src = torch.arange(layers * pages * item, dtype=torch.float16, device="musa").reshape(
        layers, pages, item
    )
    dst = torch.zeros((pages, layers, item), dtype=torch.float16, device="musa")
    src_indices = torch.tensor([-1, 1, 3], dtype=torch.int64, device="musa")
    dst_indices = torch.tensor([0, 2, 1], dtype=torch.int64, device="musa")
    stream = torch.musa.Stream()

    with torch.musa.stream(stream):
        assert musa_aiter.hicache_transfer_kv_all_layer_mla_lf_pf(
            src, dst, src_indices, dst_indices
        )
    stream.synchronize()

    torch.testing.assert_close(dst[2], src[:, 1])
    torch.testing.assert_close(dst[1], src[:, 3])
    assert torch.count_nonzero(dst[0]) == 0


@pytest.mark.musa
@pytest.mark.native
def test_native_hicache_direct_all_three_contracts() -> None:
    _require_native_hicache()
    page_size, layers, item = 2, 2, 4
    src_tokens = torch.arange(layers * 6 * item, dtype=torch.float16, device="musa").reshape(
        layers, 6, item
    )
    dst_tokens = torch.zeros_like(src_tokens)
    src_indices = torch.tensor([0, 1, 4, 5], dtype=torch.int64, device="musa")
    dst_indices = torch.tensor([2, 3, 0, 1], dtype=torch.int64, device="musa")
    assert musa_aiter.hicache_transfer_kv_direct(
        [src_tokens[0], src_tokens[1]],
        [dst_tokens[0], dst_tokens[1]],
        src_indices,
        dst_indices,
        page_size,
    )

    src_pf = torch.arange(3 * layers * page_size * item, dtype=torch.float16, device="musa").reshape(
        3, layers, page_size, item
    )
    dst_lf = torch.zeros(layers, 6, item, dtype=torch.float16, device="musa")
    assert musa_aiter.hicache_transfer_kv_per_layer_direct_pf_lf(
        [src_pf],
        [dst_lf[1]],
        torch.tensor([0, 1], dtype=torch.int64, device="musa"),
        torch.tensor([2, 3], dtype=torch.int64, device="musa"),
        1,
        page_size,
    )

    dst_pf = torch.zeros(3, layers, page_size, item, dtype=torch.float16, device="musa")
    assert musa_aiter.hicache_transfer_kv_all_layer_direct_lf_pf(
        [src_tokens[0], src_tokens[1]],
        [dst_pf],
        torch.tensor([0, 1], dtype=torch.int64, device="musa"),
        torch.tensor([2, 3], dtype=torch.int64, device="musa"),
        page_size,
    )
    torch.musa.synchronize()
    torch.testing.assert_close(dst_tokens[:, 2:4], src_tokens[:, 0:2])
    torch.testing.assert_close(dst_lf[1, 2:4], src_pf[0, 1])
    torch.testing.assert_close(dst_pf[1], src_tokens[:, 0:2])
