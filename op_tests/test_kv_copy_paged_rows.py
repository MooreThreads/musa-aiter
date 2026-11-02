"""
L1 API contract tests for DSV4 flat byte-row gather/scatter (REQ-1/2/3).

These tests validate API signatures and input validation without requiring
a MUSA device (pure host, no kernel execution). All tests should pass
once the API signatures are implemented (they validate contracts, not
correctness of results).
"""

import pytest
import torch


class TestKvCopyPagedRowsContract:
    """REQ-1 per-layer and all-layer ptr-table API contract tests."""

    def test_rejects_non_uint8_dtype(self):
        """T1.1: kv_copy_paged_rows must reject non-uint8 input."""
        pytest.importorskip("musa_aiter")
        import musa_aiter

        src = torch.zeros(16, 1728, dtype=torch.float32)
        dst = torch.zeros(16, 1728, dtype=torch.float32)
        src_indices = torch.tensor([0, 1, 2], dtype=torch.int64)
        dst_indices = torch.tensor([5, 6, 7], dtype=torch.int64)

        with pytest.raises((ValueError, TypeError), match="uint8"):
            musa_aiter.kv_copy_paged_rows(src, dst, src_indices, dst_indices)

    def test_rejects_inconsistent_item_bytes(self):
        """T1.1: src and dst must have matching item_bytes (last dim)."""
        pytest.importorskip("musa_aiter")
        import musa_aiter

        src = torch.zeros(16, 1728, dtype=torch.uint8)
        dst = torch.zeros(16, 2048, dtype=torch.uint8)  # different item_bytes
        src_indices = torch.tensor([0, 1, 2], dtype=torch.int64)
        dst_indices = torch.tensor([5, 6, 7], dtype=torch.int64)

        with pytest.raises(ValueError, match="item_bytes|shape"):
            musa_aiter.kv_copy_paged_rows(src, dst, src_indices, dst_indices)

    def test_indices_out_of_bounds_raises(self):
        """T1.2: indices beyond tensor bounds must raise clear exception."""
        pytest.importorskip("musa_aiter")
        import musa_aiter

        src = torch.zeros(10, 1728, dtype=torch.uint8)
        dst = torch.zeros(10, 1728, dtype=torch.uint8)
        src_indices = torch.tensor([0, 1, 99], dtype=torch.int64)  # 99 out of bounds
        dst_indices = torch.tensor([5, 6, 7], dtype=torch.int64)

        with pytest.raises((ValueError, IndexError), match="bound|index"):
            musa_aiter.kv_copy_paged_rows(src, dst, src_indices, dst_indices)

    def test_accepts_valid_inputs(self):
        """Contract test: valid inputs should not raise (kernel may be stub)."""
        pytest.importorskip("musa_aiter")
        import musa_aiter

        src = torch.zeros(16, 1728, dtype=torch.uint8)
        dst = torch.zeros(16, 1728, dtype=torch.uint8)
        src_indices = torch.tensor([0, 1, 2], dtype=torch.int64)
        dst_indices = torch.tensor([5, 6, 7], dtype=torch.int64)

        # Should not raise ValueError/TypeError for valid inputs
        # (may raise NotImplementedError if kernel not implemented)
        try:
            musa_aiter.kv_copy_paged_rows(src, dst, src_indices, dst_indices)
        except NotImplementedError:
            pass  # Acceptable before kernel implementation

    def test_layer_table_requires_ptr_table(self):
        """T1.4: kv_copy_paged_rows_layer_table requires Sequence[Tensor]."""
        pytest.importorskip("musa_aiter")
        import musa_aiter

        src_ptrs = [torch.zeros(16, 1728, dtype=torch.uint8) for _ in range(3)]
        dst_ptrs = [torch.zeros(16, 1728, dtype=torch.uint8) for _ in range(3)]
        src_indices = torch.tensor([0, 1, 2], dtype=torch.int64)
        dst_indices = torch.tensor([5, 6, 7], dtype=torch.int64)

        # Should accept valid ptr-table
        try:
            musa_aiter.kv_copy_paged_rows_layer_table(
                src_ptrs, dst_ptrs, src_indices, dst_indices
            )
        except NotImplementedError:
            pass

    def test_layer_table_validates_item_bytes_consistency(self):
        """T1.4: all layers in ptr-table must have matching item_bytes."""
        pytest.importorskip("musa_aiter")
        import musa_aiter

        src_ptrs = [
            torch.zeros(16, 1728, dtype=torch.uint8),
            torch.zeros(16, 2048, dtype=torch.uint8),  # mismatch
        ]
        dst_ptrs = [
            torch.zeros(16, 1728, dtype=torch.uint8),
            torch.zeros(16, 2048, dtype=torch.uint8),
        ]
        src_indices = torch.tensor([0, 1, 2], dtype=torch.int64)
        dst_indices = torch.tensor([5, 6, 7], dtype=torch.int64)

        with pytest.raises(ValueError, match="item_bytes|shape"):
            musa_aiter.kv_copy_paged_rows_layer_table(
                src_ptrs, dst_ptrs, src_indices, dst_indices
            )


class TestInterleavedFp8LayoutContract:
    """REQ-2: InterleavedFp8Layout descriptor contract tests."""

    def test_layout_item_bytes_must_match_pool_row(self):
        """T1.3: layout.item_bytes must equal actual pool row width."""
        pytest.importorskip("musa_aiter")
        import musa_aiter

        from musa_aiter.ops.kv_transfer import InterleavedFp8Layout

        # This should pass: value_bytes + scale_bytes = item_bytes
        layout = InterleavedFp8Layout(
            value_bytes=64, scale_bytes=64, scale_granularity=16
        )
        assert layout.item_bytes == 128

    def test_layout_rejects_negative_bytes(self):
        """Layout must reject negative value/scale bytes."""
        pytest.importorskip("musa_aiter")
        import musa_aiter

        from musa_aiter.ops.kv_transfer import InterleavedFp8Layout

        with pytest.raises((ValueError, AssertionError)):
            InterleavedFp8Layout(
                value_bytes=-1, scale_bytes=64, scale_granularity=16
            )


class TestMultiPoolContract:
    """REQ-3: kv_copy_paged_rows_multi_pool contract tests."""

    def test_multi_pool_requires_pool_spec_list(self):
        """T1.4: multi_pool requires list of PoolSpec."""
        pytest.importorskip("musa_aiter")
        import musa_aiter

        from musa_aiter.ops.kv_transfer import PoolSpec

        pools = [
            PoolSpec(
                src_ptrs=[torch.zeros(16, 1728, dtype=torch.uint8) for _ in range(3)],
                dst_ptrs=[torch.zeros(16, 1728, dtype=torch.uint8) for _ in range(3)],
                item_bytes=1728,
            )
        ]
        indices = torch.tensor([0, 1, 2], dtype=torch.int64)

        try:
            musa_aiter.kv_copy_paged_rows_multi_pool(
                pools, indices, direction="H2D"
            )
        except NotImplementedError:
            pass

    def test_multi_pool_validates_item_bytes(self):
        """T1.4: PoolSpec.item_bytes must match first layer shape."""
        pytest.importorskip("musa_aiter")
        import musa_aiter

        from musa_aiter.ops.kv_transfer import PoolSpec

        pools = [
            PoolSpec(
                src_ptrs=[torch.zeros(16, 1728, dtype=torch.uint8)],
                dst_ptrs=[torch.zeros(16, 1728, dtype=torch.uint8)],
                item_bytes=2048,  # mismatch with actual 1728
            )
        ]
        indices = torch.tensor([0, 1, 2], dtype=torch.int64)

        with pytest.raises(ValueError, match="item_bytes|shape"):
            musa_aiter.kv_copy_paged_rows_multi_pool(
                pools, indices, direction="H2D"
            )

    def test_multi_pool_rejects_invalid_direction(self):
        """T1.5: direction must be H2D or D2H."""
        pytest.importorskip("musa_aiter")
        import musa_aiter

        from musa_aiter.ops.kv_transfer import PoolSpec

        pools = [
            PoolSpec(
                src_ptrs=[torch.zeros(16, 1728, dtype=torch.uint8)],
                dst_ptrs=[torch.zeros(16, 1728, dtype=torch.uint8)],
                item_bytes=1728,
            )
        ]
        indices = torch.tensor([0, 1, 2], dtype=torch.int64)

        with pytest.raises(ValueError, match="direction"):
            musa_aiter.kv_copy_paged_rows_multi_pool(
                pools, indices, direction="INVALID"
            )