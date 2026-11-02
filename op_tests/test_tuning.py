import pytest

from musa_aiter.tuning import (
    TunedConfig,
    load_bundled_example_tuned_configs,
    load_json_configs,
    select_config,
    write_json_configs,
)


def test_tuning_json_roundtrip(tmp_path):
    cfg = TunedConfig(
        op="grouped_gemm",
        arch="mp31",
        dtype="float16",
        shape="m=1,k=2,n=3",
        backend="torch-reference",
        kernel="ref",
        latency_us=1.0,
        extra={"x": 1},
    )
    path = write_json_configs([cfg], tmp_path / "cfg.json")
    loaded = load_json_configs(path)
    assert loaded[0].op == "grouped_gemm"
    selected = select_config(loaded, op="grouped_gemm", arch="mp31", dtype="float16", shape="m=1,k=2,n=3")
    assert selected is not None
    assert selected.kernel == "ref"


def test_load_phase2_reference_configs():
    configs = load_json_configs("configs/model_configs/phase2_reference_configs.json")
    assert len(configs) >= 2


def test_bundled_example_csv_select():
    cfgs = load_bundled_example_tuned_configs()
    if not cfgs:
        pytest.skip("example_mp31.csv not found (run from repo root)")
    sel = select_config(
        cfgs,
        op="gemm",
        arch="mp31",
        dtype="float16",
        shape="m=256,n=256,k=256",
    )
    assert sel is not None
    assert sel.kernel == "ref_matmul"
