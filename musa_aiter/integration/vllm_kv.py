from __future__ import annotations

from dataclasses import dataclass
from importlib.util import find_spec
from typing import Literal

KVLayout = Literal["separate", "merged", "mla"]


@dataclass(frozen=True)
class VLLMKVTransferOps:
    layout: KVLayout
    multi_layer: bool
    gather_op: str
    scatter_op: str
    paged_shape: str
    linear_shape: str
    vllm_save_hook: str = "save_kv_layer"
    vllm_load_hook: str = "wait_for_layer_load"


def resolve_vllm_kv_transfer_ops(layout: str, *, multi_layer: bool = False) -> VLLMKVTransferOps:
    if layout == "separate":
        if multi_layer:
            return VLLMKVTransferOps(
                layout,
                True,
                "kv_paged_to_linear_multi_layer",
                "kv_linear_to_paged_multi_layer",
                "[L, blocks, kv_heads, head_dim, block_size]",
                "[L, T, kv_heads, head_dim]",
            )
        return VLLMKVTransferOps(
            layout,
            False,
            "kv_paged_to_linear",
            "kv_linear_to_paged",
            "[blocks, kv_heads, head_dim, block_size]",
            "[T, kv_heads, head_dim]",
        )
    if layout == "merged":
        if multi_layer:
            return VLLMKVTransferOps(
                layout,
                True,
                "kv_paged_to_merged_linear_multi_layer",
                "kv_merged_linear_to_paged_multi_layer",
                "[L, blocks, kv_heads, head_dim, block_size]",
                "[L, T, 2, kv_heads, head_dim]",
            )
        return VLLMKVTransferOps(
            layout,
            False,
            "kv_paged_to_merged_linear",
            "kv_merged_linear_to_paged",
            "[blocks, kv_heads, head_dim, block_size]",
            "[T, 2, kv_heads, head_dim]",
        )
    if layout == "mla":
        if multi_layer:
            return VLLMKVTransferOps(
                layout,
                True,
                "kv_paged_to_mla_linear_multi_layer",
                "kv_mla_linear_to_paged_multi_layer",
                "[L, blocks, block_size, kv_hidden_dim] + [L, blocks, block_size, pe_hidden_dim]",
                "[L, T, kv_hidden_dim] + [L, T, pe_hidden_dim]",
            )
        return VLLMKVTransferOps(
            layout,
            False,
            "kv_paged_to_mla_linear",
            "kv_mla_linear_to_paged",
            "[blocks, block_size, kv_hidden_dim] + [blocks, block_size, pe_hidden_dim]",
            "[T, kv_hidden_dim] + [T, pe_hidden_dim]",
        )
    raise ValueError(f"unknown vLLM KV layout: {layout!r}")


def vllm_kv_connector_import_status() -> dict[str, bool]:
    modules = (
        "vllm.distributed.kv_transfer.kv_connector.factory",
        "vllm.distributed.kv_transfer.kv_connector.v1.simple_cpu_offload_connector",
        "vllm.model_executor.layers.attention.kv_transfer_utils",
        "vllm.model_executor.layers.attention.mla_attention",
    )
    status: dict[str, bool] = {}
    for module in modules:
        try:
            status[module] = find_spec(module) is not None
        except (ImportError, ModuleNotFoundError, ValueError):
            status[module] = False
    return status
