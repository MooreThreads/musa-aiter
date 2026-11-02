from musa_aiter.registry import get, list_capabilities, as_dict


def test_registry_contains_phase1_ops():
    names = {cap.name for cap in list_capabilities()}
    assert "add" in names
    assert "multiply" in names
    assert "reduce_sum" in names
    assert "softmax_last_dim" in names
    assert "kv_paged_to_linear" in names
    assert "kv_linear_to_paged" in names
    assert "kv_paged_to_linear_multi_layer" in names
    assert "kv_linear_to_paged_multi_layer" in names
    assert "rmsnorm" in names
    assert "flash_attn_func" in names


def test_registry_get():
    cap = get("rmsnorm")
    assert cap.family == "norm"
    assert "torch-reference" in cap.backends


def test_registry_as_dict():
    data = as_dict()
    assert data["flash_attn_func"]["family"] == "attention"
