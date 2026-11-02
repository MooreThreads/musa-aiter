from musa_aiter.registry import get, list_capabilities


def test_phase2_registry_entries():
    names = {cap.name for cap in list_capabilities()}
    expected = {
        "topk_softmax", "topk_sigmoid", "moe_sorting", "fused_moe",
        "quantize_per_token", "dequantize_per_token",
        "quantize_per_tensor", "dequantize_per_tensor",
        "rmsnorm_quant", "gated_rmsnorm_quant",
        "mla_decode_fwd", "mla_prefill_fwd",
    }
    assert expected.issubset(names)


def test_phase2_family_groups():
    moe_ops = list_capabilities(family="moe")
    assert len(moe_ops) == 4
    quant_ops = list_capabilities(family="quant")
    assert len(quant_ops) == 6
    mla_ops = list_capabilities(family="mla")
    assert len(mla_ops) == 2
