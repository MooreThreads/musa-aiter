from musa_aiter.benchmarking import benchmark_callable
from musa_aiter.diagnostics.benchmark_presets import benchmark_ops, make_benchmark_workload


def test_benchmark_callable():
    result = benchmark_callable("noop", lambda: 1 + 1, iterations=2, warmup=1)
    assert result.name == "noop"
    assert result.iterations == 2
    assert result.avg_ms >= 0


def test_kv_benchmark_presets_include_new_94_paths():
    expected = {
        "kv_load_and_reshape_flash",
        "kv_copy_paged_blocks",
        "kv_cachegen_encode",
        "kv_cachegen_decode",
        "apply_rope_to_key_cache",
    }
    assert expected.issubset(set(benchmark_ops()))
    for op in expected:
        fn, dtype = make_benchmark_workload(op, "cpu")
        result = fn()
        assert dtype
        assert result is not None

