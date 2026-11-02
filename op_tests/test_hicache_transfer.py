from __future__ import annotations

import pytest
import torch

from musa_aiter.ops.hicache_transfer import (
    hicache_transfer_kv_all_layer,
    hicache_transfer_kv_all_layer_direct_lf_pf,
    hicache_transfer_kv_all_layer_lf_pf,
    hicache_transfer_kv_all_layer_mla_lf_pf,
    hicache_transfer_kv_direct,
    hicache_transfer_kv_per_layer,
    hicache_transfer_kv_per_layer_direct_pf_lf,
    hicache_transfer_kv_per_layer_mla,
    hicache_transfer_kv_per_layer_mla_pf_lf,
    hicache_transfer_kv_per_layer_pf_lf,
    is_native_hicache_transfer_available,
)


def _kv_layer_first(layers: int = 2, pages: int = 4, item: int = 8):
    key = torch.arange(layers * pages * item, dtype=torch.float32).reshape(layers, pages, item)
    value = key + 10000
    return key, value


def test_per_layer_layer_first_transfer_uses_two_index_tables():
    src_k, src_v = _kv_layer_first()
    dst_k = torch.zeros_like(src_k)
    dst_v = torch.zeros_like(src_v)

    result = hicache_transfer_kv_per_layer(
        src_k,
        dst_k,
        src_v,
        dst_v,
        torch.tensor([3, 1], dtype=torch.int64),
        torch.tensor([0, 2], dtype=torch.int64),
        layer_id=1,
    )

    assert result is True
    assert torch.equal(dst_k[1, 0], src_k[1, 3])
    assert torch.equal(dst_k[1, 2], src_k[1, 1])
    assert torch.count_nonzero(dst_k[0]) == 0
    assert torch.equal(dst_v[1, 0], src_v[1, 3])


def test_per_layer_page_first_to_layer_first_moves_selected_layer():
    layers, pages, item = 2, 4, 8
    src_k = torch.arange(pages * layers * item, dtype=torch.float32).reshape(pages, layers, item)
    src_v = src_k + 5000
    dst_k = torch.zeros(layers, pages, item, dtype=torch.float32)
    dst_v = torch.zeros_like(dst_k)

    assert hicache_transfer_kv_per_layer_pf_lf(
        src_k,
        dst_k,
        src_v,
        dst_v,
        torch.tensor([2, 0], dtype=torch.int64),
        torch.tensor([1, 3], dtype=torch.int64),
        layer_id=0,
    )

    assert torch.equal(dst_k[0, 1], src_k[2, 0])
    assert torch.equal(dst_k[0, 3], src_k[0, 0])
    assert torch.count_nonzero(dst_k[1]) == 0
    assert torch.equal(dst_v[0, 1], src_v[2, 0])


def test_all_layer_layer_first_to_page_first_preserves_layer_order():
    src_k, src_v = _kv_layer_first()
    dst_k = torch.zeros(4, 2, 8, dtype=torch.float32)
    dst_v = torch.zeros_like(dst_k)

    assert hicache_transfer_kv_all_layer_lf_pf(
        src_k,
        dst_k,
        src_v,
        dst_v,
        torch.tensor([0, 2], dtype=torch.int64),
        torch.tensor([1, 3], dtype=torch.int64),
    )

    assert torch.equal(dst_k[1, 0], src_k[0, 0])
    assert torch.equal(dst_k[1, 1], src_k[1, 0])
    assert torch.equal(dst_k[3, 0], src_k[0, 2])
    assert torch.equal(dst_k[3, 1], src_k[1, 2])


def test_all_layer_layer_first_same_layout_and_negative_index_skip():
    src_k, src_v = _kv_layer_first()
    dst_k = torch.full_like(src_k, -1)
    dst_v = torch.full_like(src_v, -1)

    assert hicache_transfer_kv_all_layer(
        src_k,
        dst_k,
        src_v,
        dst_v,
        torch.tensor([-1, 1], dtype=torch.int64),
        torch.tensor([0, 2], dtype=torch.int64),
    )

    assert torch.equal(dst_k[:, 2], src_k[:, 1])
    assert torch.equal(dst_v[:, 2], src_v[:, 1])
    assert torch.all(dst_k[:, 0] == -1)


def test_mla_per_layer_page_first_to_layer_first_has_no_value_plane():
    layers, pages, item = 2, 3, 6
    src = torch.arange(pages * layers * item, dtype=torch.float32).reshape(pages, layers, item)
    dst = torch.zeros(layers, pages, item, dtype=torch.float32)

    assert hicache_transfer_kv_per_layer_mla_pf_lf(
        src,
        dst,
        torch.tensor([1], dtype=torch.int64),
        torch.tensor([2], dtype=torch.int64),
        layer_id=1,
    )
    assert torch.equal(dst[1, 2], src[1, 1])
    assert torch.count_nonzero(dst[0]) == 0


def test_mla_all_layer_layer_first_to_page_first_preserves_all_layers():
    layers, pages, item = 2, 4, 6
    src = torch.arange(layers * pages * item, dtype=torch.float32).reshape(layers, pages, item)
    dst = torch.zeros(pages, layers, item, dtype=torch.float32)

    assert hicache_transfer_kv_all_layer_mla_lf_pf(
        src,
        dst,
        torch.tensor([-1, 1, 3], dtype=torch.int64),
        torch.tensor([0, 2, 1], dtype=torch.int64),
    )

    assert torch.equal(dst[2], src[:, 1])
    assert torch.equal(dst[1], src[:, 3])
    assert torch.all(dst[0] == 0)


def test_mla_per_layer_rejects_value_plane_arguments():
    src_k, src_v = _kv_layer_first()
    with pytest.raises(ValueError, match="MLA"):
        hicache_transfer_kv_per_layer_mla(
            src_k,
            torch.zeros_like(src_k),
            torch.tensor([0], dtype=torch.int64),
            torch.tensor([0], dtype=torch.int64),
            value=src_v,
        )


def test_direct_transfer_coalesces_contiguous_token_ranges_for_all_layers():
    layers, tokens, item = 2, 8, 4
    src_k = torch.arange(layers * tokens * item, dtype=torch.float32).reshape(layers, tokens, item)
    src_v = src_k + 1000
    dst_k = torch.zeros_like(src_k)
    dst_v = torch.zeros_like(src_v)

    assert hicache_transfer_kv_direct(
        [src_k[0], src_k[1], src_v[0], src_v[1]],
        [dst_k[0], dst_k[1], dst_v[0], dst_v[1]],
        torch.tensor([0, 1, 4, 5], dtype=torch.int64),
        torch.tensor([2, 3, 0, 1], dtype=torch.int64),
        page_size=2,
    )

    assert torch.equal(dst_k[:, 2:4], src_k[:, 0:2])
    assert torch.equal(dst_k[:, 0:2], src_k[:, 4:6])
    assert torch.equal(dst_v[:, 2:4], src_v[:, 0:2])


def test_direct_per_layer_page_first_to_layer_first_copies_mha_planes():
    pages, layers, page_size, item = 3, 2, 2, 4
    src_k = torch.arange(pages * layers * page_size * item, dtype=torch.float32).reshape(
        pages, layers, page_size, item
    )
    src_v = src_k + 5000
    dst_k = torch.zeros(layers, pages * page_size, item)
    dst_v = torch.zeros_like(dst_k)

    assert hicache_transfer_kv_per_layer_direct_pf_lf(
        [src_k, src_v],
        [dst_k[1], dst_v[1]],
        torch.tensor([0, 1, 4, 5], dtype=torch.int64),
        torch.tensor([2, 3, 0, 1], dtype=torch.int64),
        layer_id=1,
        page_size=2,
    )

    assert torch.equal(dst_k[1, 2:4], src_k[0, 1])
    assert torch.equal(dst_k[1, 0:2], src_k[2, 1])
    assert torch.equal(dst_v[1, 2:4], src_v[0, 1])
    assert torch.count_nonzero(dst_k[0]) == 0


def test_direct_per_layer_page_first_to_layer_first_copies_mla_plane():
    pages, layers, page_size, item = 3, 2, 2, 4
    src = torch.arange(pages * layers * page_size * item, dtype=torch.float32).reshape(
        pages, layers, page_size, item
    )
    dst = torch.zeros(layers, pages * page_size, item)

    assert hicache_transfer_kv_per_layer_direct_pf_lf(
        [src],
        [dst[0]],
        torch.tensor([2, 3], dtype=torch.int64),
        torch.tensor([4, 5], dtype=torch.int64),
        layer_id=0,
        page_size=2,
    )

    assert torch.equal(dst[0, 4:6], src[1, 0])


def test_direct_all_layer_layer_first_to_page_first_copies_mha_and_mla():
    layers, tokens, pages, page_size, item = 2, 6, 3, 2, 4
    src_k = torch.arange(layers * tokens * item, dtype=torch.float32).reshape(layers, tokens, item)
    src_v = src_k + 7000
    dst_k = torch.zeros(pages, layers, page_size, item)
    dst_v = torch.zeros_like(dst_k)

    assert hicache_transfer_kv_all_layer_direct_lf_pf(
        [src_k[0], src_k[1], src_v[0], src_v[1]],
        [dst_k, dst_v],
        torch.tensor([0, 1, 4, 5], dtype=torch.int64),
        torch.tensor([2, 3, 0, 1], dtype=torch.int64),
        page_size=2,
    )

    assert torch.equal(dst_k[1, :, :, :], src_k[:, 0:2])
    assert torch.equal(dst_k[0, :, :, :], src_k[:, 4:6])
    assert torch.equal(dst_v[1], src_v[:, 0:2])

    src_mla = [src_k[0], src_k[1]]
    dst_mla = torch.zeros(pages, layers, page_size, item)
    assert hicache_transfer_kv_all_layer_direct_lf_pf(
        src_mla,
        [dst_mla],
        torch.tensor([0, 1], dtype=torch.int64),
        torch.tensor([2, 3], dtype=torch.int64),
        page_size=2,
    )
    assert torch.equal(dst_mla[1], src_k[:, 0:2])


def test_direct_transfer_rejects_non_page_aligned_index_count():
    src = torch.zeros(4, 4)
    dst = torch.zeros_like(src)
    with pytest.raises(ValueError, match="divisible"):
        hicache_transfer_kv_direct(
            [src],
            [dst],
            torch.tensor([0, 1, 2], dtype=torch.int64),
            torch.tensor([0, 1, 2], dtype=torch.int64),
            page_size=2,
        )


def test_hicache_native_probe_is_safe_without_musa_toolchain():
    assert isinstance(is_native_hicache_transfer_available(), bool)
