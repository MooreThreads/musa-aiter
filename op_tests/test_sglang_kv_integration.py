import pytest
import torch

from musa_aiter.ops.dense_decode_attention import dense_decode_attention, is_native_dense_decode_attention_available
from musa_aiter.ops.kv_transfer import kv_triton_indexed_to_dense

from musa_aiter.integration.sglang_kv import (
    check_kv_copy_paged_blocks,
    check_sglang_attention_fp8_kv_preload_contract,
    check_kv_load_and_dequantize_fp8_flash,
    check_kv_load_and_reshape_flash,
    clear_sglang_cpu_host_staging_cache,
    is_sglang_cpu_host_staging_cache_enabled,
    is_sglang_kv_route_logging_enabled,
    is_sglang_kv_route_profile_enabled,
    is_sglang_attention_preload_probe_enabled,
    is_sglang_attention_preload_route_enabled,
    is_sglang_attention_preload_replace_enabled,
    is_sglang_paged_sparse_attention_enabled,
    sglang_paged_sparse_attention_max_seq_len,
    sglang_paged_sparse_attention_scratch_max_bytes,
    _estimate_tiled_paged_sparse_scratch_bytes,
    _is_triton_paged_sparse_replacement_within_scratch_gate,
    is_sglang_musa_aiter_adapter_enabled,
    is_sglang_native_cpu_host_kv_transfer_enabled,
    _is_musa_graph_capture_active,
    resolve_hicache_io_backend_for_musa,
    sglang_integration_import_status,
    try_kv_load_and_reshape_flash,
    probe_sglang_attention_preload,
    try_sglang_attention_fp8_kv_preload,
    try_sglang_attention_kv_preload,
    try_sglang_attention_kv_preload_route,
    try_sglang_attention_dense_kv_replacement,
    try_sglang_triton_attention_dense_kv_replacement,
    try_sglang_triton_paged_sparse_attention_replacement,
    resolve_sglang_attention_preload_replacement,
    _contiguous_index_range,
    _contiguous_device_index_range,
    _copy_host_slots,
    _copy_layer_first_contiguous_h2d,
    _select_host_slots,
    _select_layer_first_contiguous_d2h,
    try_sglang_hicache_layer_first_cpu_host_transfer,
    try_sglang_hicache_layer_first_cpu_host_transfer_multi_layer,
    try_sglang_hicache_layer_first_transfer,
    try_sglang_hicache_layout_transfer,
)


def test_resolve_hicache_io_backend_for_musa_keeps_kernel_public_backend():
    decision = resolve_hicache_io_backend_for_musa(
        "kernel",
        is_musa=True,
        adapter_available=True,
        layout_supported=True,
    )
    assert decision.backend == "kernel"
    assert decision.use_musa_aiter_adapter is True


def test_hicache_layout_route_rejects_cpu_tensors(monkeypatch):
    monkeypatch.setenv("MUSA_AITER_SGLANG_ADAPTER", "1")
    key = torch.zeros((2, 4, 8), dtype=torch.float32)
    value = torch.zeros_like(key)
    indices = torch.tensor([0], dtype=torch.int64)

    assert (
        try_sglang_hicache_layout_transfer(
            key,
            key.clone(),
            value,
            value.clone(),
            indices,
            indices,
            source_layout="layer_first",
            destination_layout="layer_first",
        )
        is False
    )


def test_hicache_layout_route_is_exported_from_integration_package():
    from musa_aiter.integration import try_sglang_hicache_layout_transfer as exported

    assert exported is try_sglang_hicache_layout_transfer



def test_sglang_adapter_env_flag(monkeypatch):
    monkeypatch.delenv("MUSA_AITER_SGLANG_ADAPTER", raising=False)
    assert is_sglang_musa_aiter_adapter_enabled() is False
    monkeypatch.setenv("MUSA_AITER_SGLANG_ADAPTER", "1")
    assert is_sglang_musa_aiter_adapter_enabled() is True


def test_sglang_kv_route_logging_env_flag(monkeypatch):
    monkeypatch.delenv("MUSA_AITER_SGLANG_KV_LOG", raising=False)
    assert is_sglang_kv_route_logging_enabled() is False
    monkeypatch.setenv("MUSA_AITER_SGLANG_KV_LOG", "1")
    assert is_sglang_kv_route_logging_enabled() is True


def test_sglang_kv_route_profile_env_flag(monkeypatch):
    monkeypatch.delenv("MUSA_AITER_SGLANG_KV_PROFILE", raising=False)
    assert is_sglang_kv_route_profile_enabled() is False
    monkeypatch.setenv("MUSA_AITER_SGLANG_KV_PROFILE", "1")
    assert is_sglang_kv_route_profile_enabled() is True


def test_sglang_attention_preload_probe_env_flag(monkeypatch):
    monkeypatch.delenv("MUSA_AITER_SGLANG_ATTENTION_PRELOAD_PROBE", raising=False)
    assert is_sglang_attention_preload_probe_enabled() is False
    monkeypatch.setenv("MUSA_AITER_SGLANG_ATTENTION_PRELOAD_PROBE", "1")
    assert is_sglang_attention_preload_probe_enabled() is True


def test_sglang_attention_preload_probe_disabled_by_default(monkeypatch):
    monkeypatch.setenv("MUSA_AITER_SGLANG_ADAPTER", "1")
    monkeypatch.delenv("MUSA_AITER_SGLANG_ATTENTION_PRELOAD_PROBE", raising=False)
    key = torch.zeros((2, 1, 4, 2), dtype=torch.float32)
    value = torch.zeros_like(key)
    block_tables = torch.tensor([[0, 1]], dtype=torch.long)
    seq_lens = torch.tensor([4], dtype=torch.long)

    decision = probe_sglang_attention_preload(key, value, block_tables, seq_lens)
    assert decision.enabled is False
    assert "PROBE" in decision.reason


def test_sglang_attention_preload_probe_reports_cpu_rejection_when_enabled(monkeypatch):
    monkeypatch.setenv("MUSA_AITER_SGLANG_ADAPTER", "1")
    monkeypatch.setenv("MUSA_AITER_SGLANG_ATTENTION_PRELOAD_PROBE", "1")
    key = torch.zeros((2, 1, 4, 2), dtype=torch.float32)
    value = torch.zeros_like(key)
    block_tables = torch.tensor([[0, 1]], dtype=torch.long)
    seq_lens = torch.tensor([4], dtype=torch.long)

    decision = probe_sglang_attention_preload(key, value, block_tables, seq_lens)
    assert decision.enabled is False
    assert "MUSA" in decision.reason


def test_sglang_attention_preload_route_env_flag(monkeypatch):
    monkeypatch.delenv("MUSA_AITER_SGLANG_ATTENTION_PRELOAD", raising=False)
    assert is_sglang_attention_preload_route_enabled() is False
    monkeypatch.setenv("MUSA_AITER_SGLANG_ATTENTION_PRELOAD", "1")
    assert is_sglang_attention_preload_route_enabled() is True


def test_sglang_attention_preload_route_disabled_by_default(monkeypatch):
    monkeypatch.setenv("MUSA_AITER_SGLANG_ADAPTER", "1")
    monkeypatch.delenv("MUSA_AITER_SGLANG_ATTENTION_PRELOAD", raising=False)
    key = torch.zeros((2, 1, 4, 2), dtype=torch.float32)
    value = torch.zeros_like(key)
    block_tables = torch.tensor([[0, 1]], dtype=torch.long)
    seq_lens = torch.tensor([4], dtype=torch.long)

    result = try_sglang_attention_kv_preload_route(key, value, block_tables, seq_lens)
    assert result is None


def test_sglang_attention_preload_route_enabled_reports_cpu_rejection(monkeypatch):
    monkeypatch.setenv("MUSA_AITER_SGLANG_ADAPTER", "1")
    monkeypatch.setenv("MUSA_AITER_SGLANG_ATTENTION_PRELOAD", "1")
    key = torch.zeros((2, 1, 4, 2), dtype=torch.float32)
    value = torch.zeros_like(key)
    block_tables = torch.tensor([[0, 1]], dtype=torch.long)
    seq_lens = torch.tensor([4], dtype=torch.long)

    result = try_sglang_attention_kv_preload_route(key, value, block_tables, seq_lens)
    assert result is None


def test_sglang_attention_preload_replace_env_flag(monkeypatch):
    monkeypatch.delenv("MUSA_AITER_SGLANG_ATTENTION_PRELOAD_REPLACE", raising=False)
    assert is_sglang_attention_preload_replace_enabled() is False
    monkeypatch.setenv("MUSA_AITER_SGLANG_ATTENTION_PRELOAD_REPLACE", "1")
    assert is_sglang_attention_preload_replace_enabled() is True


def test_sglang_attention_preload_replacement_blocked_without_dense_mate_contract(monkeypatch):
    monkeypatch.setenv("MUSA_AITER_SGLANG_ATTENTION_PRELOAD", "1")
    monkeypatch.setenv("MUSA_AITER_SGLANG_ATTENTION_PRELOAD_REPLACE", "1")

    decision = resolve_sglang_attention_preload_replacement(
        mate_accepts_dense_kv=False,
        parity_validated=False,
    )
    assert decision.enabled is False
    assert "dense" in decision.reason or "parity" in decision.reason


def test_sglang_attention_preload_replacement_requires_parity(monkeypatch):
    monkeypatch.setenv("MUSA_AITER_SGLANG_ATTENTION_PRELOAD", "1")
    monkeypatch.setenv("MUSA_AITER_SGLANG_ATTENTION_PRELOAD_REPLACE", "1")

    decision = resolve_sglang_attention_preload_replacement(
        mate_accepts_dense_kv=True,
        parity_validated=False,
    )
    assert decision.enabled is False
    assert "parity" in decision.reason


def test_sglang_native_cpu_host_kv_transfer_env_flag(monkeypatch):
    monkeypatch.delenv("MUSA_AITER_NATIVE_CPU_HOST_KV_TRANSFER", raising=False)
    assert is_sglang_native_cpu_host_kv_transfer_enabled() is False
    monkeypatch.setenv("MUSA_AITER_NATIVE_CPU_HOST_KV_TRANSFER", "1")
    assert is_sglang_native_cpu_host_kv_transfer_enabled() is True


def test_sglang_cpu_host_staging_cache_env_flag(monkeypatch):
    monkeypatch.delenv("MUSA_AITER_CPU_HOST_STAGING_CACHE", raising=False)
    assert is_sglang_cpu_host_staging_cache_enabled() is True
    monkeypatch.setenv("MUSA_AITER_CPU_HOST_STAGING_CACHE", "0")
    assert is_sglang_cpu_host_staging_cache_enabled() is False
    monkeypatch.setenv("MUSA_AITER_CPU_HOST_STAGING_CACHE", "1")
    assert is_sglang_cpu_host_staging_cache_enabled() is True
    clear_sglang_cpu_host_staging_cache()


def test_sglang_cpu_host_contiguous_index_range_and_copy_fast_path():
    indices = torch.tensor([2, 3, 4], dtype=torch.long)
    slots, index_range = _contiguous_index_range(indices)
    assert slots.tolist() == [2, 3, 4]
    assert index_range == (2, 5)

    cache = torch.arange(20, dtype=torch.float32).reshape(10, 2)
    _, selected, mode = _select_host_slots(cache, 0, indices)
    assert mode == "slice"
    assert torch.equal(selected, cache[2:5])

    dst = torch.zeros_like(cache)
    copy_mode = _copy_host_slots(dst, 0, indices, selected)
    assert copy_mode == "slice_copy"
    assert torch.equal(dst[2:5], selected)


def test_sglang_cpu_host_noncontiguous_index_range_uses_index_ops():
    indices = torch.tensor([2, 4, 5], dtype=torch.long)
    _, index_range = _contiguous_index_range(indices)
    assert index_range is None

    cache = torch.arange(20, dtype=torch.float32).reshape(10, 2)
    _, selected, mode = _select_host_slots(cache, 0, indices)
    assert mode == "index_select"
    assert torch.equal(selected, cache.index_select(0, indices))

    dst = torch.zeros_like(cache)
    copy_mode = _copy_host_slots(dst, 0, indices, selected)
    assert copy_mode == "index_copy"
    assert torch.equal(dst.index_select(0, indices), selected)


def test_sglang_layer_first_contiguous_device_range_cpu_reference():
    indices = torch.tensor([1, 2, 3], dtype=torch.long)
    assert _contiguous_device_index_range(indices) == (1, 4)

    key_linear = torch.arange(6, dtype=torch.float32).reshape(3, 1, 2)
    value_linear = key_linear + 100
    device_key = torch.zeros((6, 1, 2), dtype=torch.float32)
    device_value = torch.zeros_like(device_key)
    assert _copy_layer_first_contiguous_h2d(key_linear, value_linear, device_key, device_value, indices) is True
    assert torch.equal(device_key[1:4], key_linear)
    assert torch.equal(device_value[1:4], value_linear)

    selected = _select_layer_first_contiguous_d2h(device_key, device_value, indices)
    assert selected is not None
    selected_key, selected_value = selected
    assert torch.equal(selected_key, key_linear)
    assert torch.equal(selected_value, value_linear)


def test_sglang_layer_first_noncontiguous_device_range_rejects_fast_path():
    indices = torch.tensor([1, 3, 4], dtype=torch.long)
    assert _contiguous_device_index_range(indices) is None
    key_linear = torch.zeros((3, 1, 2), dtype=torch.float32)
    value_linear = torch.zeros_like(key_linear)
    device_key = torch.zeros((6, 1, 2), dtype=torch.float32)
    device_value = torch.zeros_like(device_key)
    assert _copy_layer_first_contiguous_h2d(key_linear, value_linear, device_key, device_value, indices) is False
    assert _select_layer_first_contiguous_d2h(device_key, device_value, indices) is None


def test_sglang_integration_import_status_shape():
    status = sglang_integration_import_status()
    assert "sglang" in status
    assert "musa_aiter" in status
    assert all(isinstance(value, bool) for value in status.values())


def test_check_kv_load_and_reshape_requires_musa_when_enabled(monkeypatch):
    monkeypatch.setenv("MUSA_AITER_SGLANG_ADAPTER", "1")
    key = torch.zeros((2, 1, 4, 2), dtype=torch.float32)
    value = torch.zeros_like(key)
    block_tables = torch.tensor([[0, 1]], dtype=torch.long)
    seq_lens = torch.tensor([4], dtype=torch.long)

    decision = check_kv_load_and_reshape_flash(key, value, block_tables, seq_lens)
    assert decision.enabled is False
    assert "MUSA" in decision.reason


def test_try_kv_load_and_reshape_returns_none_when_env_disabled(monkeypatch):
    monkeypatch.delenv("MUSA_AITER_SGLANG_ADAPTER", raising=False)
    key = torch.zeros((2, 1, 4, 2), dtype=torch.float32)
    value = torch.zeros_like(key)
    block_tables = torch.tensor([[0, 1]], dtype=torch.long)
    seq_lens = torch.tensor([4], dtype=torch.long)

    result = try_kv_load_and_reshape_flash(key, value, block_tables, seq_lens)
    assert result is None


def test_check_kv_load_and_reshape_accepts_reference_shape_without_env_gate():
    key = torch.zeros((2, 1, 4, 2), dtype=torch.float32)
    value = torch.zeros_like(key)
    block_tables = torch.tensor([[0, 1]], dtype=torch.long)
    seq_lens = torch.tensor([4], dtype=torch.long)

    decision = check_kv_load_and_reshape_flash(key, value, block_tables, seq_lens, require_env=False)
    assert decision.enabled is False
    assert "MUSA" in decision.reason


def test_check_kv_copy_paged_blocks_validates_mapping_shape(monkeypatch):
    monkeypatch.setenv("MUSA_AITER_SGLANG_ADAPTER", "1")
    src = torch.zeros((2, 1, 4, 2), dtype=torch.float32)
    dst = torch.zeros_like(src)
    mapping = torch.tensor([0, 1], dtype=torch.long)

    decision = check_kv_copy_paged_blocks(src, src, dst, dst, mapping, require_env=False)
    assert decision.enabled is False
    assert "MUSA" in decision.reason


def test_check_fp8_load_rejects_non_fp8_cache_after_base_checks(monkeypatch):
    monkeypatch.setenv("MUSA_AITER_SGLANG_ADAPTER", "1")
    key = torch.zeros((2, 1, 4, 2), dtype=torch.float32)
    value = torch.zeros_like(key)
    block_tables = torch.tensor([[0, 1]], dtype=torch.long)
    seq_lens = torch.tensor([4], dtype=torch.long)

    decision = check_kv_load_and_dequantize_fp8_flash(
        key,
        value,
        block_tables,
        seq_lens,
        1.0,
        1.0,
        require_env=False,
    )
    assert decision.enabled is False
    assert "MUSA" in decision.reason


def test_sglang_hicache_layer_first_route_returns_false_for_cpu(monkeypatch):
    monkeypatch.setenv("MUSA_AITER_SGLANG_ADAPTER", "1")
    host_key = torch.arange(8, dtype=torch.float32).reshape(4, 1, 2)
    host_value = host_key + 100
    device_key = torch.zeros_like(host_key)
    device_value = torch.zeros_like(host_key)
    host_indices = torch.tensor([0, 1, 2, 3], dtype=torch.long)
    device_indices = torch.tensor([0, 1, 2, 3], dtype=torch.long)

    routed = try_sglang_hicache_layer_first_transfer(
        host_key,
        host_value,
        device_key,
        device_value,
        host_indices,
        device_indices,
        page_size=2,
        direction="host_to_device",
    )
    assert routed is False


def test_sglang_cpu_host_route_requires_native_env(monkeypatch):
    monkeypatch.setenv("MUSA_AITER_SGLANG_ADAPTER", "1")
    monkeypatch.delenv("MUSA_AITER_NATIVE_CPU_HOST_KV_TRANSFER", raising=False)
    host_key = torch.arange(8, dtype=torch.float32).reshape(4, 1, 2)
    host_value = host_key + 100
    device_key = torch.zeros_like(host_key)
    device_value = torch.zeros_like(host_key)
    host_indices = torch.tensor([0, 1, 2, 3], dtype=torch.long)
    device_indices = torch.tensor([0, 1, 2, 3], dtype=torch.long)

    routed = try_sglang_hicache_layer_first_cpu_host_transfer(
        host_key,
        host_value,
        device_key,
        device_value,
        host_indices,
        device_indices,
        page_size=2,
        direction="host_to_device",
    )
    assert routed is False


def test_sglang_cpu_host_route_rejects_cpu_device_cache(monkeypatch):
    monkeypatch.setenv("MUSA_AITER_SGLANG_ADAPTER", "1")
    monkeypatch.setenv("MUSA_AITER_NATIVE_CPU_HOST_KV_TRANSFER", "1")
    host_key = torch.arange(8, dtype=torch.float32).reshape(4, 1, 2)
    host_value = host_key + 100
    device_key = torch.zeros_like(host_key)
    device_value = torch.zeros_like(host_key)
    host_indices = torch.tensor([0, 1, 2, 3], dtype=torch.long)
    device_indices = torch.tensor([0, 1, 2, 3], dtype=torch.long)

    routed = try_sglang_hicache_layer_first_cpu_host_transfer(
        host_key,
        host_value,
        device_key,
        device_value,
        host_indices,
        device_indices,
        page_size=2,
        direction="host_to_device",
    )
    assert routed is False


def test_sglang_cpu_host_multi_layer_route_rejects_cpu_device_cache(monkeypatch):
    monkeypatch.setenv("MUSA_AITER_SGLANG_ADAPTER", "1")
    monkeypatch.setenv("MUSA_AITER_NATIVE_CPU_HOST_KV_TRANSFER", "1")
    host_key = torch.arange(16, dtype=torch.float32).reshape(2, 4, 1, 2)
    host_value = host_key + 100
    device_key = torch.zeros_like(host_key)
    device_value = torch.zeros_like(host_key)
    host_indices = torch.tensor([0, 1, 2, 3], dtype=torch.long)
    device_indices = torch.tensor([0, 1, 2, 3], dtype=torch.long)

    routed = try_sglang_hicache_layer_first_cpu_host_transfer_multi_layer(
        host_key,
        host_value,
        device_key,
        device_value,
        host_indices,
        device_indices,
        page_size=2,
        direction="device_to_host",
    )
    assert routed is False


def test_sglang_attention_preload_returns_none_for_cpu(monkeypatch):
    monkeypatch.setenv("MUSA_AITER_SGLANG_ADAPTER", "1")
    key = torch.zeros((2, 1, 4, 2), dtype=torch.float32)
    value = torch.zeros_like(key)
    block_tables = torch.tensor([[0, 1]], dtype=torch.long)
    seq_lens = torch.tensor([4], dtype=torch.long)

    result = try_sglang_attention_kv_preload(key, value, block_tables, seq_lens)
    assert result is None


def test_sglang_attention_fp8_contract_rejects_unsupported_scale_shape_after_base_checks(monkeypatch):
    monkeypatch.setenv("MUSA_AITER_SGLANG_ADAPTER", "1")
    key = torch.zeros((2, 2, 4, 2), dtype=torch.uint8)
    value = torch.zeros_like(key)
    block_tables = torch.tensor([[0, 1]], dtype=torch.long)
    seq_lens = torch.tensor([4], dtype=torch.long)
    bad_scale = torch.ones((2, 2, 1), dtype=torch.float32)

    decision = check_sglang_attention_fp8_kv_preload_contract(
        key,
        value,
        block_tables,
        seq_lens,
        bad_scale,
        bad_scale,
        require_env=False,
    )
    assert decision.enabled is False
    assert "scale" in decision.reason


def test_sglang_attention_fp8_contract_accepts_scalar_float_scale_until_musa_device_check(monkeypatch):
    monkeypatch.setenv("MUSA_AITER_SGLANG_ADAPTER", "1")
    key = torch.zeros((2, 2, 4, 2), dtype=torch.uint8)
    value = torch.zeros_like(key)
    block_tables = torch.tensor([[0, 1]], dtype=torch.long)
    seq_lens = torch.tensor([4], dtype=torch.long)

    decision = check_sglang_attention_fp8_kv_preload_contract(
        key,
        value,
        block_tables,
        seq_lens,
        1.0,
        1.0,
        require_env=False,
    )
    assert decision.enabled is False
    assert "MUSA" in decision.reason


def test_sglang_attention_fp8_contract_accepts_per_head_scale_shape_until_musa_device_check(monkeypatch):
    monkeypatch.setenv("MUSA_AITER_SGLANG_ADAPTER", "1")
    key = torch.zeros((2, 2, 4, 2), dtype=torch.uint8)
    value = torch.zeros_like(key)
    block_tables = torch.tensor([[0, 1]], dtype=torch.long)
    seq_lens = torch.tensor([4], dtype=torch.long)
    scale = torch.ones((2,), dtype=torch.float32)

    decision = check_sglang_attention_fp8_kv_preload_contract(
        key,
        value,
        block_tables,
        seq_lens,
        scale,
        scale,
        require_env=False,
    )
    assert decision.enabled is False
    assert "MUSA" in decision.reason


def test_sglang_attention_fp8_contract_accepts_per_layer_head_scale_shape_until_musa_device_check(monkeypatch):
    monkeypatch.setenv("MUSA_AITER_SGLANG_ADAPTER", "1")
    key = torch.zeros((3, 2, 2, 4, 2), dtype=torch.uint8)
    value = torch.zeros_like(key)
    block_tables = torch.tensor([[0, 1]], dtype=torch.long)
    seq_lens = torch.tensor([4], dtype=torch.long)
    scale = torch.ones((3, 2), dtype=torch.float32)

    decision = check_sglang_attention_fp8_kv_preload_contract(
        key,
        value,
        block_tables,
        seq_lens,
        scale,
        scale,
        require_env=False,
    )
    assert decision.enabled is False
    assert "MUSA" in decision.reason


def _require_musa_device() -> torch.device:
    if not hasattr(torch, "musa") or not torch.musa.is_available():
        pytest.skip("TorchMUSA is not available")
    return torch.device("musa:0")


def _reference_flash_load(cache: torch.Tensor, block_tables: torch.Tensor, seq_lens: torch.Tensor, *, max_seq_len: int | None = None) -> torch.Tensor:
    nb, nh, hd, bs = map(int, cache.shape)
    batch = int(block_tables.shape[0])
    dense_len = int(seq_lens.max().item()) if max_seq_len is None else int(max_seq_len)
    out = torch.zeros((batch, dense_len, nh, hd), dtype=cache.dtype)
    for seq_idx in range(batch):
        seq_len = int(seq_lens[seq_idx].item())
        for pos in range(seq_len):
            block_id = int(block_tables[seq_idx, pos // bs].item())
            offset = pos % bs
            assert 0 <= block_id < nb
            out[seq_idx, pos].copy_(cache[block_id, :, :, offset])
    return out


def test_sglang_attention_preload_musa_bfloat16_matches_reference(monkeypatch):
    device = _require_musa_device()
    monkeypatch.setenv("MUSA_AITER_SGLANG_ADAPTER", "1")
    key_cpu = torch.arange(2 * 2 * 4 * 2, dtype=torch.float32).reshape(2, 2, 4, 2).to(torch.bfloat16)
    value_cpu = key_cpu + 100
    block_tables_cpu = torch.tensor([[0, 1]], dtype=torch.long)
    seq_lens_cpu = torch.tensor([4], dtype=torch.long)

    result = try_sglang_attention_kv_preload(
        key_cpu.to(device),
        value_cpu.to(device),
        block_tables_cpu.to(device),
        seq_lens_cpu.to(device),
        require_env=True,
    )
    assert result is not None
    out_k, out_v = result
    assert torch.equal(out_k.cpu(), _reference_flash_load(key_cpu, block_tables_cpu, seq_lens_cpu))
    assert torch.equal(out_v.cpu(), _reference_flash_load(value_cpu, block_tables_cpu, seq_lens_cpu))


def test_sglang_attention_dense_kv_replacement_musa_matches_cpu(monkeypatch):
    device = _require_musa_device()
    monkeypatch.setenv("MUSA_AITER_SGLANG_ADAPTER", "1")
    monkeypatch.setenv("MUSA_AITER_SGLANG_ATTENTION_PRELOAD", "1")
    monkeypatch.setenv("MUSA_AITER_SGLANG_ATTENTION_PRELOAD_REPLACE", "1")
    dtype = torch.bfloat16
    query_cpu = torch.randn((1, 4, 2, 8), dtype=dtype)
    key_cpu = torch.randn((2, 2, 8, 2), dtype=dtype)
    value_cpu = torch.randn((2, 2, 8, 2), dtype=dtype)
    block_tables_cpu = torch.tensor([[0, 1]], dtype=torch.long)
    seq_lens_cpu = torch.tensor([4], dtype=torch.long)
    dense_key_cpu = _reference_flash_load(key_cpu, block_tables_cpu, seq_lens_cpu)
    dense_value_cpu = _reference_flash_load(value_cpu, block_tables_cpu, seq_lens_cpu)

    import musa_aiter

    expected = musa_aiter.flash_attn_func(query_cpu, dense_key_cpu, dense_value_cpu, causal=False)
    actual = try_sglang_attention_dense_kv_replacement(
        query_cpu.to(device),
        key_cpu.to(device),
        value_cpu.to(device),
        block_tables_cpu.to(device),
        seq_lens_cpu.to(device),
        causal=False,
        mate_accepts_dense_kv=True,
        parity_validated=True,
        require_env=True,
    )
    assert actual is not None
    torch.testing.assert_close(actual.cpu(), expected, rtol=8e-2, atol=8e-2)


def test_sglang_attention_dense_kv_replacement_musa_decode_query_matches_cpu(monkeypatch):
    device = _require_musa_device()
    monkeypatch.setenv("MUSA_AITER_SGLANG_ADAPTER", "1")
    monkeypatch.setenv("MUSA_AITER_SGLANG_ATTENTION_PRELOAD", "1")
    monkeypatch.setenv("MUSA_AITER_SGLANG_ATTENTION_PRELOAD_REPLACE", "1")
    dtype = torch.bfloat16
    query_cpu = torch.randn((1, 2, 8), dtype=dtype)
    key_cpu = torch.randn((2, 2, 8, 2), dtype=dtype)
    value_cpu = torch.randn((2, 2, 8, 2), dtype=dtype)
    block_tables_cpu = torch.tensor([[0, 1]], dtype=torch.long)
    seq_lens_cpu = torch.tensor([4], dtype=torch.long)
    dense_key_cpu = _reference_flash_load(key_cpu, block_tables_cpu, seq_lens_cpu)
    dense_value_cpu = _reference_flash_load(value_cpu, block_tables_cpu, seq_lens_cpu)

    import musa_aiter

    expected = musa_aiter.flash_attn_func(query_cpu.unsqueeze(1), dense_key_cpu, dense_value_cpu, causal=False).squeeze(1)
    actual = try_sglang_attention_dense_kv_replacement(
        query_cpu.to(device),
        key_cpu.to(device),
        value_cpu.to(device),
        block_tables_cpu.to(device),
        seq_lens_cpu.to(device),
        causal=False,
        mate_accepts_dense_kv=True,
        parity_validated=True,
        require_env=True,
    )
    assert actual is not None
    assert tuple(actual.shape) == tuple(query_cpu.shape)
    torch.testing.assert_close(actual.cpu(), expected, rtol=8e-2, atol=8e-2)


def _reference_triton_indexed_load(cache: torch.Tensor, kv_indptr: torch.Tensor, kv_indices: torch.Tensor) -> torch.Tensor:
    batch = int(kv_indptr.numel() - 1)
    max_len = max(int(kv_indptr[i + 1].item() - kv_indptr[i].item()) for i in range(batch))
    _, heads, head_dim = map(int, cache.shape)
    out = torch.zeros((batch, max_len, heads, head_dim), dtype=cache.dtype)
    for batch_idx in range(batch):
        start = int(kv_indptr[batch_idx].item())
        end = int(kv_indptr[batch_idx + 1].item())
        slots = kv_indices[start:end].to(dtype=torch.long)
        out[batch_idx, : int(slots.numel())].copy_(cache.index_select(0, slots))
    return out


def test_dense_decode_attention_musa_matches_reference(monkeypatch):
    device = _require_musa_device()
    assert is_native_dense_decode_attention_available() is True
    dtype = torch.bfloat16
    query_cpu = torch.randn((2, 4, 8), dtype=dtype)
    key_cpu = torch.randn((2, 5, 2, 8), dtype=dtype)
    value_cpu = torch.randn((2, 5, 2, 8), dtype=dtype)
    dense_key_cpu = key_cpu.repeat_interleave(2, dim=2)
    dense_value_cpu = value_cpu.repeat_interleave(2, dim=2)
    scale = 1.0 / (8**0.5)
    scores = torch.einsum("bhd,bshd->bhs", query_cpu.float(), dense_key_cpu.float()) * scale
    expected = torch.einsum("bhs,bshd->bhd", torch.softmax(scores, dim=-1), dense_value_cpu.float()).to(dtype)

    actual = dense_decode_attention(query_cpu.to(device), key_cpu.to(device), value_cpu.to(device), softmax_scale=scale)

    assert tuple(actual.shape) == tuple(query_cpu.shape)
    torch.testing.assert_close(actual.cpu(), expected, rtol=5e-2, atol=5e-2)


def test_kv_triton_indexed_to_dense_musa_matches_reference(monkeypatch):
    device = _require_musa_device()
    monkeypatch.setenv("MUSA_AITER_NATIVE_KV_TRANSFER", "1")
    dtype = torch.bfloat16
    key_cpu = torch.arange(8 * 2 * 4, dtype=torch.float32).reshape(8, 2, 4).to(dtype)
    value_cpu = (torch.arange(8 * 2 * 4, dtype=torch.float32).reshape(8, 2, 4) + 100).to(dtype)
    kv_indptr_cpu = torch.tensor([0, 3, 7], dtype=torch.long)
    kv_indices_cpu = torch.tensor([0, 2, 4, 1, 3, 5, 7], dtype=torch.long)
    expected_k = _reference_triton_indexed_load(key_cpu, kv_indptr_cpu, kv_indices_cpu)
    expected_v = _reference_triton_indexed_load(value_cpu, kv_indptr_cpu, kv_indices_cpu)

    actual_k, actual_v = kv_triton_indexed_to_dense(
        key_cpu.to(device),
        value_cpu.to(device),
        kv_indptr_cpu.to(device),
        kv_indices_cpu.to(device),
        max_seq_len=4,
    )

    assert tuple(actual_k.shape) == (2, 4, 2, 4)
    torch.testing.assert_close(actual_k.cpu(), expected_k, rtol=0, atol=0)
    torch.testing.assert_close(actual_v.cpu(), expected_v, rtol=0, atol=0)


def test_sglang_triton_attention_dense_kv_replacement_musa_matches_cpu(monkeypatch):
    device = _require_musa_device()
    monkeypatch.setenv("MUSA_AITER_SGLANG_ADAPTER", "1")
    monkeypatch.setenv("MUSA_AITER_SGLANG_ATTENTION_PRELOAD", "1")
    monkeypatch.setenv("MUSA_AITER_SGLANG_ATTENTION_PRELOAD_REPLACE", "1")
    dtype = torch.bfloat16
    query_cpu = torch.randn((2, 4, 8), dtype=dtype)
    key_cpu = torch.randn((8, 2, 8), dtype=dtype)
    value_cpu = torch.randn((8, 2, 8), dtype=dtype)
    kv_indptr_cpu = torch.tensor([0, 3, 7], dtype=torch.long)
    kv_indices_cpu = torch.tensor([0, 2, 4, 1, 3, 5, 7], dtype=torch.long)
    dense_key_cpu = _reference_triton_indexed_load(key_cpu, kv_indptr_cpu, kv_indices_cpu).repeat_interleave(2, dim=2)
    dense_value_cpu = _reference_triton_indexed_load(value_cpu, kv_indptr_cpu, kv_indices_cpu).repeat_interleave(2, dim=2)

    import musa_aiter

    expected = musa_aiter.flash_attn_func(query_cpu.unsqueeze(1), dense_key_cpu, dense_value_cpu, causal=False).squeeze(1)
    actual = try_sglang_triton_attention_dense_kv_replacement(
        query_cpu.to(device),
        key_cpu.to(device),
        value_cpu.to(device),
        kv_indptr_cpu.to(device),
        kv_indices_cpu.to(device),
        causal=False,
        softmax_scale=None,
        mate_accepts_dense_kv=True,
        parity_validated=True,
        require_env=True,
    )
    assert actual is not None
    assert tuple(actual.shape) == tuple(query_cpu.shape)
    torch.testing.assert_close(actual.cpu(), expected, rtol=8e-2, atol=8e-2)


def test_sglang_triton_attention_dense_kv_replacement_capture_native_decode(monkeypatch):
    device = _require_musa_device()
    monkeypatch.setenv("MUSA_AITER_SGLANG_ADAPTER", "1")
    monkeypatch.setenv("MUSA_AITER_SGLANG_ATTENTION_PRELOAD", "1")
    monkeypatch.setenv("MUSA_AITER_SGLANG_ATTENTION_PRELOAD_REPLACE", "1")
    monkeypatch.setenv("MUSA_AITER_NATIVE_KV_TRANSFER", "1")
    monkeypatch.setattr("musa_aiter.integration.sglang_kv._is_musa_graph_capture_active", lambda: True)
    dtype = torch.bfloat16
    query_cpu = torch.randn((2, 4, 8), dtype=dtype)
    key_cpu = torch.randn((8, 2, 8), dtype=dtype)
    value_cpu = torch.randn((8, 2, 8), dtype=dtype)
    kv_indptr_cpu = torch.tensor([0, 3, 7], dtype=torch.long)
    kv_indices_cpu = torch.tensor([0, 2, 4, 1, 3, 5, 7], dtype=torch.long)
    dense_key_cpu = _reference_triton_indexed_load(key_cpu, kv_indptr_cpu, kv_indices_cpu)
    dense_value_cpu = _reference_triton_indexed_load(value_cpu, kv_indptr_cpu, kv_indices_cpu)
    dense_key_cpu = dense_key_cpu.repeat_interleave(2, dim=2)
    dense_value_cpu = dense_value_cpu.repeat_interleave(2, dim=2)
    scale = 1.0 / (8**0.5)
    scores = torch.einsum("bhd,bshd->bhs", query_cpu.float(), dense_key_cpu.float()) * scale
    expected = torch.einsum("bhs,bshd->bhd", torch.softmax(scores, dim=-1), dense_value_cpu.float()).to(dtype)

    actual = try_sglang_triton_attention_dense_kv_replacement(
        query_cpu.to(device),
        key_cpu.to(device),
        value_cpu.to(device),
        kv_indptr_cpu.to(device),
        kv_indices_cpu.to(device),
        causal=False,
        softmax_scale=scale,
        max_seq_len=4,
        mate_accepts_dense_kv=True,
        parity_validated=True,
        require_env=True,
    )

    assert actual is not None
    torch.testing.assert_close(actual.cpu(), expected, rtol=5e-2, atol=5e-2)


def test_sglang_triton_attention_dense_kv_replacement_rejects_cpu(monkeypatch):
    monkeypatch.setenv("MUSA_AITER_SGLANG_ADAPTER", "1")
    monkeypatch.setenv("MUSA_AITER_SGLANG_ATTENTION_PRELOAD", "1")
    monkeypatch.setenv("MUSA_AITER_SGLANG_ATTENTION_PRELOAD_REPLACE", "1")
    query = torch.zeros((1, 2, 8), dtype=torch.bfloat16)
    key = torch.zeros((4, 2, 8), dtype=torch.bfloat16)
    value = torch.zeros_like(key)
    kv_indptr = torch.tensor([0, 2], dtype=torch.long)
    kv_indices = torch.tensor([0, 1], dtype=torch.long)

    assert try_sglang_triton_attention_dense_kv_replacement(
        query,
        key,
        value,
        kv_indptr,
        kv_indices,
        mate_accepts_dense_kv=True,
        parity_validated=True,
    ) is None


def test_sglang_triton_attention_dense_kv_replacement_skips_graph_capture(monkeypatch):
    monkeypatch.setenv("MUSA_AITER_SGLANG_ADAPTER", "1")
    monkeypatch.setenv("MUSA_AITER_SGLANG_ATTENTION_PRELOAD", "1")
    monkeypatch.setenv("MUSA_AITER_SGLANG_ATTENTION_PRELOAD_REPLACE", "1")
    monkeypatch.setattr("musa_aiter.integration.sglang_kv._is_musa_graph_capture_active", lambda: True)
    monkeypatch.setattr("musa_aiter.integration.sglang_kv._is_musa_tensor", lambda tensor: isinstance(tensor, torch.Tensor))
    query = torch.zeros((1, 2, 8), dtype=torch.bfloat16)
    key = torch.zeros((4, 2, 8), dtype=torch.bfloat16)
    value = torch.zeros_like(key)
    kv_indptr = torch.tensor([0, 2], dtype=torch.long)
    kv_indices = torch.tensor([0, 1], dtype=torch.long)

    assert _is_musa_graph_capture_active() is False
    assert try_sglang_triton_attention_dense_kv_replacement(
        query,
        key,
        value,
        kv_indptr,
        kv_indices,
        mate_accepts_dense_kv=True,
        parity_validated=True,
    ) is None


def test_sglang_triton_attention_dense_kv_replacement_perf_gate_skips_long_context(monkeypatch):
    monkeypatch.setenv("MUSA_AITER_SGLANG_ADAPTER", "1")
    monkeypatch.setenv("MUSA_AITER_SGLANG_ATTENTION_PRELOAD", "1")
    monkeypatch.setenv("MUSA_AITER_SGLANG_ATTENTION_PRELOAD_REPLACE", "1")
    monkeypatch.setenv("MUSA_AITER_SGLANG_ATTENTION_PRELOAD_MAX_SEQ_LEN", "1")
    monkeypatch.setattr("musa_aiter.integration.sglang_kv._is_musa_tensor", lambda tensor: isinstance(tensor, torch.Tensor))
    query = torch.zeros((1, 2, 8), dtype=torch.bfloat16)
    key = torch.zeros((4, 2, 8), dtype=torch.bfloat16)
    value = torch.zeros_like(key)
    kv_indptr = torch.tensor([0, 2], dtype=torch.long)
    kv_indices = torch.tensor([0, 1], dtype=torch.long)

    assert try_sglang_triton_attention_dense_kv_replacement(
        query,
        key,
        value,
        kv_indptr,
        kv_indices,
        max_seq_len=2,
        mate_accepts_dense_kv=True,
        parity_validated=True,
    ) is None


def test_sglang_attention_fp8_contract_musa_per_head_scale_enabled(monkeypatch):
    device = _require_musa_device()
    monkeypatch.setenv("MUSA_AITER_SGLANG_ADAPTER", "1")
    key = torch.zeros((2, 2, 4, 2), dtype=torch.uint8, device=device)
    value = torch.zeros_like(key)
    block_tables = torch.tensor([[0, 1]], dtype=torch.long, device=device)
    seq_lens = torch.tensor([4], dtype=torch.long, device=device)
    scale = torch.ones((2,), dtype=torch.float32, device=device)

    decision = check_sglang_attention_fp8_kv_preload_contract(
        key,
        value,
        block_tables,
        seq_lens,
        scale,
        scale,
        require_env=True,
    )
    assert decision.enabled is True


def test_sglang_attention_fp8_preload_musa_scalar_scale_smoke(monkeypatch):
    device = _require_musa_device()
    monkeypatch.setenv("MUSA_AITER_SGLANG_ADAPTER", "1")
    key = torch.zeros((2, 1, 4, 2), dtype=torch.uint8, device=device)
    value = torch.zeros_like(key)
    block_tables = torch.tensor([[0, 1]], dtype=torch.long, device=device)
    seq_lens = torch.tensor([4], dtype=torch.long, device=device)

    result = try_sglang_attention_fp8_kv_preload(
        key,
        value,
        block_tables,
        seq_lens,
        1.0,
        1.0,
        out_dtype=torch.bfloat16,
        require_env=True,
    )
    assert result is not None
    out_k, out_v = result
    assert tuple(out_k.shape) == (1, 4, 1, 4)
    assert tuple(out_v.shape) == (1, 4, 1, 4)
    assert out_k.dtype is torch.bfloat16
    assert out_v.dtype is torch.bfloat16
    assert torch.equal(out_k.cpu(), torch.zeros_like(out_k.cpu()))
    assert torch.equal(out_v.cpu(), torch.zeros_like(out_v.cpu()))


def test_sglang_paged_sparse_attention_env_flag(monkeypatch):
    monkeypatch.delenv("MUSA_AITER_SGLANG_PAGED_SPARSE_ATTENTION", raising=False)
    assert is_sglang_paged_sparse_attention_enabled() is False
    monkeypatch.setenv("MUSA_AITER_SGLANG_PAGED_SPARSE_ATTENTION", "1")
    assert is_sglang_paged_sparse_attention_enabled() is True


def test_sglang_paged_sparse_attention_max_seq_len_env(monkeypatch):
    monkeypatch.delenv("MUSA_AITER_SGLANG_PAGED_SPARSE_ATTENTION_MAX_SEQ_LEN", raising=False)
    assert sglang_paged_sparse_attention_max_seq_len() == 4096
    monkeypatch.setenv("MUSA_AITER_SGLANG_PAGED_SPARSE_ATTENTION_MAX_SEQ_LEN", "8192")
    assert sglang_paged_sparse_attention_max_seq_len() == 8192
    monkeypatch.setenv("MUSA_AITER_SGLANG_PAGED_SPARSE_ATTENTION_MAX_SEQ_LEN", "0")
    assert sglang_paged_sparse_attention_max_seq_len() is None
    monkeypatch.setenv("MUSA_AITER_SGLANG_PAGED_SPARSE_ATTENTION_MAX_SEQ_LEN", "bad")
    assert sglang_paged_sparse_attention_max_seq_len() == 4096


def test_sglang_paged_sparse_attention_scratch_max_bytes_env(monkeypatch):
    monkeypatch.delenv("MUSA_AITER_SGLANG_PAGED_SPARSE_ATTENTION_SCRATCH_MAX_BYTES", raising=False)
    assert sglang_paged_sparse_attention_scratch_max_bytes() == 512 * 1024 * 1024
    monkeypatch.setenv("MUSA_AITER_SGLANG_PAGED_SPARSE_ATTENTION_SCRATCH_MAX_BYTES", "1024")
    assert sglang_paged_sparse_attention_scratch_max_bytes() == 1024
    monkeypatch.setenv("MUSA_AITER_SGLANG_PAGED_SPARSE_ATTENTION_SCRATCH_MAX_BYTES", "0")
    assert sglang_paged_sparse_attention_scratch_max_bytes() is None
    monkeypatch.setenv("MUSA_AITER_SGLANG_PAGED_SPARSE_ATTENTION_SCRATCH_MAX_BYTES", "bad")
    assert sglang_paged_sparse_attention_scratch_max_bytes() == 512 * 1024 * 1024


def test_sglang_paged_sparse_attention_scratch_gate_estimate(monkeypatch):
    monkeypatch.setenv("MUSA_AITER_PAGED_SPARSE_ATTENTION_TILED", "1")
    monkeypatch.delenv("MUSA_AITER_PAGED_SPARSE_ATTENTION_TILE_SIZE", raising=False)
    monkeypatch.delenv("MUSA_AITER_PAGED_SPARSE_ATTENTION_COMPACT_SCRATCH", raising=False)
    assert _estimate_tiled_paged_sparse_scratch_bytes(batch=1, num_q_heads=64, head_dim=128, max_seq_len=81920) == 64 * 160 * (2 * 4 + 128 * 2)
    assert _estimate_tiled_paged_sparse_scratch_bytes(batch=256, num_q_heads=64, head_dim=128, max_seq_len=40960) == 256 * 64 * 80 * (2 * 4 + 128 * 2)
    assert _is_triton_paged_sparse_replacement_within_scratch_gate(batch=1, num_q_heads=64, head_dim=128, max_seq_len=81920) is True
    assert _is_triton_paged_sparse_replacement_within_scratch_gate(batch=256, num_q_heads=64, head_dim=128, max_seq_len=40960) is True
    monkeypatch.setenv("MUSA_AITER_PAGED_SPARSE_ATTENTION_COMPACT_SCRATCH", "0")
    assert _estimate_tiled_paged_sparse_scratch_bytes(batch=256, num_q_heads=64, head_dim=128, max_seq_len=40960) == 256 * 64 * 80 * (2 * 4 + 128 * 4)
    assert _is_triton_paged_sparse_replacement_within_scratch_gate(batch=256, num_q_heads=64, head_dim=128, max_seq_len=40960) is False


def test_sglang_triton_paged_sparse_attention_replacement_scratch_gate(monkeypatch):
    monkeypatch.setenv("MUSA_AITER_SGLANG_ADAPTER", "1")
    monkeypatch.setenv("MUSA_AITER_SGLANG_ATTENTION_PRELOAD", "1")
    monkeypatch.setenv("MUSA_AITER_SGLANG_ATTENTION_PRELOAD_REPLACE", "1")
    monkeypatch.setenv("MUSA_AITER_SGLANG_PAGED_SPARSE_ATTENTION", "1")
    monkeypatch.setenv("MUSA_AITER_SGLANG_PAGED_SPARSE_ATTENTION_MAX_SEQ_LEN", "0")
    monkeypatch.setenv("MUSA_AITER_SGLANG_PAGED_SPARSE_ATTENTION_SCRATCH_MAX_BYTES", str(512 * 1024 * 1024))
    monkeypatch.setenv("MUSA_AITER_PAGED_SPARSE_ATTENTION_TILED", "1")
    monkeypatch.setenv("MUSA_AITER_PAGED_SPARSE_ATTENTION_COMPACT_SCRATCH", "0")
    monkeypatch.setattr("musa_aiter.integration.sglang_kv._is_musa_tensor", lambda tensor: isinstance(tensor, torch.Tensor))
    monkeypatch.setattr(
        "musa_aiter.integration.sglang_kv.paged_sparse_decode_attention",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("native should be scratch gated")),
    )
    query = torch.zeros((256, 64, 128), dtype=torch.bfloat16)
    key = torch.zeros((40960, 8, 128), dtype=torch.bfloat16)
    value = torch.zeros_like(key)
    kv_indptr = torch.arange(257, dtype=torch.long) * 40960
    kv_indices = torch.arange(256 * 40960, dtype=torch.long)

    assert try_sglang_triton_paged_sparse_attention_replacement(
        query,
        key,
        value,
        kv_indptr,
        kv_indices,
        max_seq_len=40960,
        mate_accepts_dense_kv=True,
        parity_validated=True,
    ) is None


def test_sglang_triton_paged_sparse_attention_replacement_perf_gate(monkeypatch):
    monkeypatch.setenv("MUSA_AITER_SGLANG_ADAPTER", "1")
    monkeypatch.setenv("MUSA_AITER_SGLANG_ATTENTION_PRELOAD", "1")
    monkeypatch.setenv("MUSA_AITER_SGLANG_ATTENTION_PRELOAD_REPLACE", "1")
    monkeypatch.setenv("MUSA_AITER_SGLANG_PAGED_SPARSE_ATTENTION", "1")
    monkeypatch.delenv("MUSA_AITER_SGLANG_PAGED_SPARSE_ATTENTION_MAX_SEQ_LEN", raising=False)
    monkeypatch.setattr("musa_aiter.integration.sglang_kv._is_musa_tensor", lambda tensor: isinstance(tensor, torch.Tensor))
    monkeypatch.setattr(
        "musa_aiter.integration.sglang_kv.paged_sparse_decode_attention",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("native should be gated")),
    )
    query = torch.zeros((1, 2, 8), dtype=torch.bfloat16)
    key = torch.zeros((4, 2, 8), dtype=torch.bfloat16)
    value = torch.zeros_like(key)
    kv_indptr = torch.tensor([0, 2], dtype=torch.long)
    kv_indices = torch.tensor([0, 1], dtype=torch.long)

    assert try_sglang_triton_paged_sparse_attention_replacement(
        query,
        key,
        value,
        kv_indptr,
        kv_indices,
        max_seq_len=81920,
        mate_accepts_dense_kv=True,
        parity_validated=True,
    ) is None


def test_sglang_triton_paged_sparse_attention_replacement_disabled_by_default(monkeypatch):
    monkeypatch.setenv("MUSA_AITER_SGLANG_ADAPTER", "1")
    monkeypatch.setenv("MUSA_AITER_SGLANG_ATTENTION_PRELOAD", "1")
    monkeypatch.setenv("MUSA_AITER_SGLANG_ATTENTION_PRELOAD_REPLACE", "1")
    monkeypatch.delenv("MUSA_AITER_SGLANG_PAGED_SPARSE_ATTENTION", raising=False)
    monkeypatch.setattr("musa_aiter.integration.sglang_kv._is_musa_tensor", lambda tensor: isinstance(tensor, torch.Tensor))
    query = torch.zeros((1, 2, 8), dtype=torch.bfloat16)
    key = torch.zeros((4, 2, 8), dtype=torch.bfloat16)
    value = torch.zeros_like(key)
    kv_indptr = torch.tensor([0, 2], dtype=torch.long)
    kv_indices = torch.tensor([0, 1], dtype=torch.long)

    assert try_sglang_triton_paged_sparse_attention_replacement(
        query,
        key,
        value,
        kv_indptr,
        kv_indices,
        mate_accepts_dense_kv=True,
        parity_validated=True,
    ) is None


def test_sglang_triton_paged_sparse_attention_replacement_musa_smoke(monkeypatch):
    device = _require_musa_device()
    monkeypatch.setenv("MUSA_AITER_SGLANG_ADAPTER", "1")
    monkeypatch.setenv("MUSA_AITER_SGLANG_ATTENTION_PRELOAD", "1")
    monkeypatch.setenv("MUSA_AITER_SGLANG_ATTENTION_PRELOAD_REPLACE", "1")
    monkeypatch.setenv("MUSA_AITER_SGLANG_PAGED_SPARSE_ATTENTION", "1")
    dtype = torch.bfloat16
    query_cpu = torch.randn((1, 4, 8), dtype=dtype)
    key_cpu = torch.randn((4, 2, 8), dtype=dtype)
    value_cpu = torch.randn((4, 2, 8), dtype=dtype)
    kv_indptr_cpu = torch.tensor([0, 4], dtype=torch.long)
    kv_indices_cpu = torch.tensor([0, 2, 1, 3], dtype=torch.long)
    dense_key_cpu = key_cpu.index_select(0, kv_indices_cpu).unsqueeze(0).repeat_interleave(2, dim=2)
    dense_value_cpu = value_cpu.index_select(0, kv_indices_cpu).unsqueeze(0).repeat_interleave(2, dim=2)
    scale = 1.0 / (8**0.5)
    scores = torch.einsum("bhd,bshd->bhs", query_cpu.float(), dense_key_cpu.float()) * scale
    expected = torch.einsum("bhs,bshd->bhd", torch.softmax(scores, dim=-1), dense_value_cpu.float()).to(dtype)

    actual = try_sglang_triton_paged_sparse_attention_replacement(
        query_cpu.to(device),
        key_cpu.to(device),
        value_cpu.to(device),
        kv_indptr_cpu.to(device),
        kv_indices_cpu.to(device),
        softmax_scale=scale,
        mate_accepts_dense_kv=True,
        parity_validated=True,
    )

    assert actual is not None
    torch.testing.assert_close(actual.cpu(), expected, rtol=5e-2, atol=5e-2)
