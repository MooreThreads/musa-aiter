import pytest

from musa_aiter.integration.vllm_kv import resolve_vllm_kv_transfer_ops, vllm_kv_connector_import_status


def test_resolve_vllm_kv_transfer_ops():
    separate = resolve_vllm_kv_transfer_ops("separate")
    assert separate.gather_op == "kv_paged_to_linear"
    assert separate.scatter_op == "kv_linear_to_paged"

    merged = resolve_vllm_kv_transfer_ops("merged", multi_layer=True)
    assert merged.gather_op == "kv_paged_to_merged_linear_multi_layer"
    assert merged.scatter_op == "kv_merged_linear_to_paged_multi_layer"

    mla = resolve_vllm_kv_transfer_ops("mla")
    assert mla.gather_op == "kv_paged_to_mla_linear"
    assert mla.scatter_op == "kv_mla_linear_to_paged"


def test_resolve_vllm_kv_transfer_ops_supports_mla_multi_layer():
    mla = resolve_vllm_kv_transfer_ops("mla", multi_layer=True)

    assert mla.gather_op == "kv_paged_to_mla_linear_multi_layer"
    assert mla.scatter_op == "kv_mla_linear_to_paged_multi_layer"


def test_vllm_kv_connector_import_status_shape():
    status = vllm_kv_connector_import_status()
    assert "vllm.distributed.kv_transfer.kv_connector.factory" in status
    assert all(isinstance(value, bool) for value in status.values())
