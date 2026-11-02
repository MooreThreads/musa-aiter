from __future__ import annotations

import pytest

from musa_aiter.ops.kv_layout import (
    KVLayoutKind,
    KVLayoutSpec,
    KVOrganization,
    KVScaleKind,
)


def test_standard_paged_layout_exposes_hidden_size_and_kind():
    spec = KVLayoutSpec.standard_paged(
        num_layers=2,
        num_blocks=4,
        block_size=16,
        num_heads=8,
        head_size=64,
        dtype_name="float16",
        row_bytes=128,
        block_stride_bytes=2048,
    )

    assert spec.kind is KVLayoutKind.STANDARD_PAGED
    assert spec.organization is KVOrganization.SEPARATE
    assert spec.hidden_size == 512
    assert spec.is_paged is True


def test_layout_spec_rejects_non_positive_row_bytes():
    with pytest.raises(ValueError, match="row_bytes"):
        KVLayoutSpec.standard_paged(
            num_layers=1,
            num_blocks=4,
            block_size=16,
            num_heads=2,
            head_size=8,
            dtype_name="float16",
            row_bytes=0,
            block_stride_bytes=32,
        )


def test_layout_spec_rejects_block_stride_smaller_than_row():
    with pytest.raises(ValueError, match="block_stride_bytes"):
        KVLayoutSpec.standard_paged(
            num_layers=1,
            num_blocks=4,
            block_size=16,
            num_heads=2,
            head_size=8,
            dtype_name="float16",
            row_bytes=32,
            block_stride_bytes=16,
        )


def test_blocked_scale_requires_value_and_scale_bytes():
    with pytest.raises(ValueError, match="scale_bytes"):
        KVLayoutSpec.dsv4(
            num_layers=1,
            num_blocks=4,
            block_size=16,
            row_bytes=132,
            block_stride_bytes=8640,
            dtype_name="uint8",
            scale_kind=KVScaleKind.BLOCKED,
            value_bytes=128,
            scale_bytes=0,
        )


def test_page_first_constructor_preserves_non_contiguous_layout_fact():
    spec = KVLayoutSpec.page_first(
        num_layers=2,
        num_blocks=8,
        block_size=16,
        num_heads=4,
        head_size=32,
        dtype_name="bfloat16",
        row_bytes=64,
        block_stride_bytes=1024,
    )

    assert spec.kind is KVLayoutKind.PAGE_FIRST
    assert spec.is_contiguous is False
    assert spec.requires_layout_transform is True


def test_dsv4_blocked_scale_facts_preserve_byte_geometry():
    spec = KVLayoutSpec.dsv4(
        num_layers=1,
        num_blocks=8,
        block_size=64,
        row_bytes=132,
        block_stride_bytes=8640,
        dtype_name="uint8",
        scale_kind=KVScaleKind.BLOCKED,
        value_bytes=128,
        scale_bytes=4,
    )

    assert spec.kind is KVLayoutKind.DSV4_BLOCKED_SCALE
    assert spec.format_facts().value_bytes == 128
    assert spec.format_facts().scale_bytes == 4
    assert spec.format_facts().requires_transform is True


def test_public_package_exports_layout_contract():
    import musa_aiter

    assert musa_aiter.KVLayoutSpec is KVLayoutSpec
    assert musa_aiter.KVLayoutKind is KVLayoutKind
