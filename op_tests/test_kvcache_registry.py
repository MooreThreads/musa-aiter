from musa_aiter.registry import get, list_capabilities


def test_kvcache_registry_family():
    caps = list_capabilities(family="kvcache")
    names = {c.name for c in caps}
    assert names == {
        "kv_cachegen_decode",
        "kv_cachegen_encode",
        "kv_copy_paged_blocks",
        "kv_copy_paged_blocks_fp8",
        "kv_load_and_dequantize_fp8_flash",
        "kv_load_and_reshape_flash",
        "kv_reshape_and_cache_back_flash",
        "kv_linear_to_paged",
        "kv_linear_to_paged_fp8",
        "kv_linear_to_paged_fp8_multi_layer",
        "kv_linear_to_paged_multi_layer",
        "kv_merged_linear_to_paged",
        "kv_merged_linear_to_paged_fp8",
        "kv_merged_linear_to_paged_fp8_multi_layer",
        "kv_merged_linear_to_paged_multi_layer",
        "kv_mla_linear_to_paged",
        "kv_mla_linear_to_paged_multi_layer",
        "kv_paged_to_linear",
        "kv_paged_fp8_to_linear",
        "kv_paged_fp8_to_linear_multi_layer",
        "kv_paged_fp8_to_merged_linear",
        "kv_paged_fp8_to_merged_linear_multi_layer",
        "kv_paged_to_linear_multi_layer",
        "kv_paged_to_merged_linear",
        "kv_paged_to_merged_linear_multi_layer",
        "kv_paged_to_mla_linear",
        "kv_paged_to_mla_linear_multi_layer",
        "hicache_transfer_kv_all_layer",
        "hicache_transfer_kv_all_layer_lf_pf",
        "hicache_transfer_kv_per_layer",
        "hicache_transfer_kv_per_layer_mla",
        "hicache_transfer_kv_per_layer_mla_pf_lf",
        "hicache_transfer_kv_per_layer_pf_lf",
        "lmcache_kv_buffer_to_paged_two_x_nl",
        "lmcache_kv_paged_to_buffer_two_x_nl",
        "lmcache_musa_memcpy_batch_async",
    }


def test_lmcache_two_x_nl_registry_lists_native_backend_order():
    expected = (
        "torch-reference",
        "torchmusa-cpp-extension-coalesced-u16",
        "ctypes-musa-native-u16",
    )
    for name in (
        "lmcache_kv_buffer_to_paged_two_x_nl",
        "lmcache_kv_paged_to_buffer_two_x_nl",
    ):
        assert get(name).backends == expected
