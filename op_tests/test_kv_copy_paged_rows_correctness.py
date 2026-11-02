"""
L2 correctness tests for DSV4 flat byte-row gather/scatter (REQ-1/3).

Reference semantics: torch index_copy_ on byte rows. Tests run on CPU
always; a MUSA parametrization exercises the same assertions when a
device is available. Bit-exactness is required (uint8 bytes).
"""

import pytest
import torch

import musa_aiter
from musa_aiter.ops.kv_transfer import InterleavedFp8Layout, PoolSpec


def _musa_if_available(tensor, use_musa):
    return tensor.to("musa") if use_musa else tensor


def _devices():
    return [pytest.param(False, id="cpu")]


try:
    if torch.musa.is_available():
        _devices().append(pytest.param(True, id="musa"))
except AttributeError:
    pass


def _rand_rows(rows, item_bytes, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, 256, (rows, item_bytes), dtype=torch.uint8, generator=g)


def _rand_indices(n, bound, seed=1):
    g = torch.Generator().manual_seed(seed)
    return torch.randperm(bound, generator=g)[:n].to(torch.int64)


@pytest.mark.parametrize("use_musa", _devices())
class TestKvCopyPagedRowsCorrectness:
    def test_small_rows_bit_exact(self, use_musa):
        """T2.1: n=16 rows, item_bytes=1728, round-trip both directions."""
        src = _rand_rows(32, 1728, seed=42)
        dst = torch.zeros(32, 1728, dtype=torch.uint8)
        src_idx = _rand_indices(16, 32, seed=1)
        dst_idx = _rand_indices(16, 32, seed=2)

        src, dst = _musa_if_available(src, use_musa), _musa_if_available(dst, use_musa)
        src_idx, dst_idx = _musa_if_available(src_idx, use_musa), _musa_if_available(dst_idx, use_musa)

        musa_aiter.kv_copy_paged_rows(src, dst, src_idx, dst_idx)

        dst_ref = torch.zeros(32, 1728, dtype=torch.uint8)
        src_cpu = src.cpu()
        dst_ref.index_copy_(0, dst_idx.cpu(), src_cpu.index_select(0, src_idx.cpu()))
        assert torch.equal(dst.cpu(), dst_ref)  # bit-exact

    @pytest.mark.parametrize("item_bytes", [1729, 37439, 1, 37440])
    def test_unaligned_item_bytes_bit_exact(self, use_musa, item_bytes):
        """T2.2: non-vector-aligned item_bytes (fp8 tail scale region)."""
        src = _rand_rows(16, item_bytes, seed=7)
        dst = torch.zeros(16, item_bytes, dtype=torch.uint8)
        src_idx = _rand_indices(8, 16, seed=3)
        dst_idx = _rand_indices(8, 16, seed=4)

        src, dst = _musa_if_available(src, use_musa), _musa_if_available(dst, use_musa)
        src_idx, dst_idx = _musa_if_available(src_idx, use_musa), _musa_if_available(dst_idx, use_musa)

        musa_aiter.kv_copy_paged_rows(src, dst, src_idx, dst_idx)

        dst_ref = torch.zeros(16, item_bytes, dtype=torch.uint8)
        dst_ref.index_copy_(0, dst_idx.cpu(), src.cpu().index_select(0, src_idx.cpu()))
        assert torch.equal(dst.cpu(), dst_ref)

    def test_duplicate_dst_indices_last_write_wins(self, use_musa):
        """T2.3: repeated dst indices behave like index_copy_ (last wins, deterministic)."""
        src = _rand_rows(8, 256, seed=5)
        dst = torch.zeros(4, 256, dtype=torch.uint8)
        src_idx = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7], dtype=torch.int64)
        dst_idx = torch.tensor([1, 1, 2, 2, 3, 3, 0, 0], dtype=torch.int64)

        src, dst = _musa_if_available(src, use_musa), _musa_if_available(dst, use_musa)
        src_idx, dst_idx = _musa_if_available(src_idx, use_musa), _musa_if_available(dst_idx, use_musa)

        musa_aiter.kv_copy_paged_rows(src, dst, src_idx, dst_idx)

        dst_ref = torch.zeros(4, 256, dtype=torch.uint8)
        dst_ref.index_copy_(0, dst_idx.cpu(), src.cpu().index_select(0, src_idx.cpu()))
        assert torch.equal(dst.cpu(), dst_ref)

    def test_layout_passthrough_copy(self, use_musa):
        """REQ-2: layout param accepted, copy still whole-row bit-exact."""
        layout = InterleavedFp8Layout(value_bytes=192, scale_bytes=64, scale_granularity=64)
        src = _rand_rows(8, layout.item_bytes, seed=11)
        dst = torch.zeros(8, layout.item_bytes, dtype=torch.uint8)
        idx = torch.arange(8, dtype=torch.int64)

        src, dst, idx = (_musa_if_available(t, use_musa) for t in (src, dst, idx))
        musa_aiter.kv_copy_paged_rows(src, dst, idx, idx, layout=layout)
        assert torch.equal(dst.cpu(), src.cpu())


@pytest.mark.parametrize("use_musa", _devices())
class TestLayerTableCorrectness:
    def test_43_layers_all_exact(self, use_musa):
        """T2.4: 43-layer ptr-table (DSV4 scale), bit-exact per layer."""
        layers, pages, item_bytes = 43, 64, 1728
        src_ptrs = [_rand_rows(pages, item_bytes, seed=100 + i) for i in range(layers)]
        dst_ptrs = [torch.zeros(pages, item_bytes, dtype=torch.uint8) for _ in range(layers)]
        src_idx = _rand_indices(16, pages, seed=21)
        dst_idx = _rand_indices(16, pages, seed=22)

        src_ptrs = [_musa_if_available(t, use_musa) for t in src_ptrs]
        dst_ptrs = [_musa_if_available(t, use_musa) for t in dst_ptrs]
        src_idx, dst_idx = _musa_if_available(src_idx, use_musa), _musa_if_available(dst_idx, use_musa)

        musa_aiter.kv_copy_paged_rows_layer_table(src_ptrs, dst_ptrs, src_idx, dst_idx)

        for layer in range(layers):
            ref = torch.zeros(pages, item_bytes, dtype=torch.uint8)
            ref.index_copy_(0, dst_idx.cpu(), src_ptrs[layer].cpu().index_select(0, src_idx.cpu()))
            assert torch.equal(dst_ptrs[layer].cpu(), ref), f"layer {layer} mismatch"


@pytest.mark.parametrize("use_musa", _devices())
class TestMultiPoolCorrectness:
    def test_six_pools_shared_indices(self, use_musa):
        """T2.5: six pools with distinct item_bytes, one shared index set."""
        pages = 32
        pool_specs_raw = [
            (2, 37440),  # SWA/latent-like large rows
            (3, 1728),   # C4 sidecar
            (1, 4096),
            (4, 1024),
            (1, 512),
            (2, 256),
        ]
        pools = []
        for num_layers, item_bytes in pool_specs_raw:
            src_ptrs = [_rand_rows(pages, item_bytes, seed=item_bytes + i) for i in range(num_layers)]
            dst_ptrs = [torch.zeros(pages, item_bytes, dtype=torch.uint8) for _ in range(num_layers)]
            pools.append(
                PoolSpec(
                    src_ptrs=[_musa_if_available(t, use_musa) for t in src_ptrs],
                    dst_ptrs=[_musa_if_available(t, use_musa) for t in dst_ptrs],
                    item_bytes=item_bytes,
                )
            )
        indices = _rand_indices(10, pages, seed=99)
        indices = _musa_if_available(indices, use_musa)

        musa_aiter.kv_copy_paged_rows_multi_pool(pools, indices, direction="H2D")

        for pool_idx, (num_layers, item_bytes) in enumerate(pool_specs_raw):
            pool = pools[pool_idx]
            for layer in range(num_layers):
                ref = torch.zeros(pages, item_bytes, dtype=torch.uint8)
                ref.index_copy_(
                    0, indices.cpu(), pool.src_ptrs[layer].cpu().index_select(0, indices.cpu())
                )
                assert torch.equal(pool.dst_ptrs[layer].cpu(), ref), (
                    f"pool {pool_idx} layer {layer} mismatch"
                )


class TestUnpinnedStagingCorrectness:
    """T2.6: unpinned host memory produces bit-identical results (REQ-4)."""

    def test_unpinned_cpu_matches_pinned_semantics(self):
        """A pageable (unpinned) CPU src copies bit-exact and is counted as unpinned."""
        from musa_aiter.ops.kv_transfer import kv_transfer_stats, reset_kv_transfer_stats

        reset_kv_transfer_stats()
        src = _rand_rows(16, 1728, seed=13)  # plain CPU tensor = pageable
        dst = torch.zeros(16, 1728, dtype=torch.uint8)
        idx = torch.arange(16, dtype=torch.int64)

        musa_aiter.kv_copy_paged_rows(src, dst, idx, idx)

        assert torch.equal(dst, src)  # bit-exact
        stats = kv_transfer_stats()
        assert stats["unpinned_calls"] == 1
        assert stats["unpinned_bytes"] == 16 * 1728

    def test_pinned_cpu_counted_as_pinned(self):
        """A pinned CPU src is counted under pinned telemetry."""
        from musa_aiter.ops.kv_transfer import kv_transfer_stats, reset_kv_transfer_stats

        try:
            pinned_src = torch.empty(8, 256, dtype=torch.uint8).pin_memory()
        except RuntimeError:
            pytest.skip("pin_memory unavailable on this host (no accelerator driver)")
        pinned_src.copy_(_rand_rows(8, 256, seed=17))
        dst = torch.zeros(8, 256, dtype=torch.uint8)
        idx = torch.arange(8, dtype=torch.int64)

        reset_kv_transfer_stats()
        musa_aiter.kv_copy_paged_rows(pinned_src, dst, idx, idx)

        assert torch.equal(dst, pinned_src)
        stats = kv_transfer_stats()
        assert stats["pinned_calls"] == 1
        assert stats["unpinned_calls"] == 0


class TestPtrTableCache:
    """P1-3: device ptr-table upload is cached across layer_table calls."""

    def test_cache_reuses_entry_and_clear_works(self):
        from musa_aiter.ops.kv_transfer import (
            _DSV4_PTR_TABLE_CACHE,
            _dsv4_get_or_create_ptr_table,
            clear_dsv4_ptr_table_cache,
        )

        src_ptrs = [torch.zeros(8, 256, dtype=torch.uint8) for _ in range(3)]
        dst_ptrs = [torch.zeros(8, 256, dtype=torch.uint8) for _ in range(3)]
        clear_dsv4_ptr_table_cache()
        t1_src, t1_dst = _dsv4_get_or_create_ptr_table(src_ptrs, dst_ptrs, "cpu")
        t2_src, t2_dst = _dsv4_get_or_create_ptr_table(src_ptrs, dst_ptrs, "cpu")
        assert t1_src is t2_src and t1_dst is t2_dst  # cached (same objects)
        assert len(_DSV4_PTR_TABLE_CACHE) == 1
        # different tensors -> separate entry
        other = [torch.zeros(8, 256, dtype=torch.uint8) for _ in range(3)]
        _dsv4_get_or_create_ptr_table(other, other, "cpu")
        assert len(_DSV4_PTR_TABLE_CACHE) == 2
        # correct contents
        assert t1_src.tolist() == [t.data_ptr() for t in src_ptrs]
        clear_dsv4_ptr_table_cache()
        assert len(_DSV4_PTR_TABLE_CACHE) == 0


class TestGpuIndicesOutOfBoundsSemantics:
    """P1'': GPU-side out-of-bounds indices are silently skipped (documented
    semantics to avoid per-call GPU->CPU sync); CPU indices raise."""

    def test_gpu_oob_silently_skipped_no_crash(self, monkeypatch):
        try:
            if not (hasattr(torch, "musa") and torch.musa.is_available()):
                pytest.skip("no MUSA device")
        except AttributeError:
            pytest.skip("no MUSA device")
        # OOB-skip semantics are guaranteed by the native kernel; enable it.
        monkeypatch.setenv("MUSA_AITER_NATIVE_KV_TRANSFER", "1")
        src = _rand_rows(16, 256, seed=31).to("musa")
        dst = torch.zeros(16, 256, dtype=torch.uint8, device="musa")
        # mixed: one valid, one OOB row, plus negative
        src_idx = torch.tensor([0, 999, -1, 3], dtype=torch.int64).clamp(min=0).to("musa")
        src_idx[1] = 999  # re-set OOB after clamp
        dst_idx = torch.tensor([5, 6, 7, 8], dtype=torch.int64).to("musa")

        musa_aiter.kv_copy_paged_rows(src, dst, src_idx, dst_idx)  # must not raise / crash
        torch.musa.synchronize()
        # valid rows copied; OOB row untouched; clamped -1 -> src row 0 is valid
        assert torch.equal(dst[5].cpu(), src[0].cpu())
        assert torch.equal(dst[8].cpu(), src[3].cpu())
        assert torch.equal(dst[6].cpu(), torch.zeros(256, dtype=torch.uint8))
        assert torch.equal(dst[7].cpu(), src[0].cpu())

    def test_cpu_oob_still_raises(self):
        src = _rand_rows(8, 256, seed=32)
        dst = torch.zeros(8, 256, dtype=torch.uint8)
        idx_bad = torch.tensor([0, 99], dtype=torch.int64)
        idx_ok = torch.tensor([1, 2], dtype=torch.int64)
        with pytest.raises(IndexError):
            musa_aiter.kv_copy_paged_rows(src, dst, idx_bad, idx_ok)
