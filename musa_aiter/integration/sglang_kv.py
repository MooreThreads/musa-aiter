from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from importlib.util import find_spec
from typing import Literal

import torch

from musa_aiter.ops.dense_decode_attention import dense_decode_attention
from musa_aiter.ops.paged_sparse_attention import paged_sparse_decode_attention
from musa_aiter.ops.kv_transfer import (
    kv_copy_paged_blocks,
    kv_copy_paged_blocks_fp8,
    kv_copy_paged_rows,
    kv_copy_paged_rows_layer_table,
    kv_linear_to_paged,
    kv_linear_to_paged_multi_layer,
    kv_load_and_dequantize_fp8_flash,
    kv_load_and_reshape_flash,
    kv_triton_indexed_to_dense,
    kv_paged_to_linear,
    kv_paged_to_linear_multi_layer,
)
from musa_aiter.ops.lmcache_musa_batch_copy import lmcache_musa_memcpy_batch_async
from musa_aiter.ops.native_gate import native_requested
from musa_aiter.ops.hicache_transfer import (
    hicache_transfer_kv_all_layer,
    hicache_transfer_kv_all_layer_direct_lf_pf,
    hicache_transfer_kv_all_layer_lf_pf,
    hicache_transfer_kv_all_layer_mla_lf_pf,
    hicache_transfer_kv_per_layer,
    hicache_transfer_kv_per_layer_direct_pf_lf,
    hicache_transfer_kv_per_layer_mla,
    hicache_transfer_kv_per_layer_mla_pf_lf,
    hicache_transfer_kv_per_layer_pf_lf,
    hicache_transfer_kv_direct,
)

HiCacheIOBackend = Literal["direct", "kernel"]

logger = logging.getLogger(__name__)


def is_sglang_kv_route_logging_enabled() -> bool:
    return os.environ.get("MUSA_AITER_SGLANG_KV_LOG", "0") == "1"


def is_sglang_kv_route_profile_enabled() -> bool:
    return os.environ.get("MUSA_AITER_SGLANG_KV_PROFILE", "0") == "1"


def is_sglang_attention_preload_probe_enabled() -> bool:
    return os.environ.get("MUSA_AITER_SGLANG_ATTENTION_PRELOAD_PROBE", "0") == "1"


def is_sglang_attention_preload_route_enabled() -> bool:
    return os.environ.get("MUSA_AITER_SGLANG_ATTENTION_PRELOAD", "0") == "1"


def is_sglang_attention_preload_replace_enabled() -> bool:
    return os.environ.get("MUSA_AITER_SGLANG_ATTENTION_PRELOAD_REPLACE", "0") == "1"


def is_sglang_paged_sparse_attention_enabled() -> bool:
    return os.environ.get("MUSA_AITER_SGLANG_PAGED_SPARSE_ATTENTION", "0") == "1"


def sglang_paged_sparse_attention_max_seq_len() -> int | None:
    raw = os.environ.get("MUSA_AITER_SGLANG_PAGED_SPARSE_ATTENTION_MAX_SEQ_LEN", "4096")
    try:
        value = int(raw)
    except ValueError:
        return 4096
    return value if value > 0 else None


def sglang_paged_sparse_attention_scratch_max_bytes() -> int | None:
    raw = os.environ.get("MUSA_AITER_SGLANG_PAGED_SPARSE_ATTENTION_SCRATCH_MAX_BYTES", str(512 * 1024 * 1024))
    try:
        value = int(raw)
    except ValueError:
        return 512 * 1024 * 1024
    return value if value > 0 else None


def _sglang_paged_sparse_tile_size() -> int:
    raw = os.environ.get("MUSA_AITER_PAGED_SPARSE_ATTENTION_TILE_SIZE", "512")
    try:
        value = int(raw)
    except ValueError:
        return 512
    return max(1, value)


def _is_sglang_paged_sparse_compact_scratch_enabled() -> bool:
    return os.environ.get("MUSA_AITER_PAGED_SPARSE_ATTENTION_COMPACT_SCRATCH", "1") != "0"


def _estimate_tiled_paged_sparse_scratch_bytes(*, batch: int, num_q_heads: int, head_dim: int, max_seq_len: int | None) -> int | None:
    if os.environ.get("MUSA_AITER_PAGED_SPARSE_ATTENTION_TILED", "0") != "1":
        return 0
    if max_seq_len is None:
        return None
    tile_size = _sglang_paged_sparse_tile_size()
    num_tiles = max(1, (int(max_seq_len) + tile_size - 1) // tile_size)
    partial_count = int(batch) * int(num_q_heads) * num_tiles
    partial_out_bytes = int(head_dim) * (2 if _is_sglang_paged_sparse_compact_scratch_enabled() else 4)
    return partial_count * (2 * 4 + partial_out_bytes)


def _is_triton_paged_sparse_replacement_within_scratch_gate(*, batch: int, num_q_heads: int, head_dim: int, max_seq_len: int | None) -> bool:
    limit = sglang_paged_sparse_attention_scratch_max_bytes()
    if limit is None:
        return True
    estimate = _estimate_tiled_paged_sparse_scratch_bytes(batch=batch, num_q_heads=num_q_heads, head_dim=head_dim, max_seq_len=max_seq_len)
    return estimate is None or estimate <= limit


def sglang_attention_preload_max_seq_len() -> int | None:
    raw = os.environ.get("MUSA_AITER_SGLANG_ATTENTION_PRELOAD_MAX_SEQ_LEN", "128")
    try:
        value = int(raw)
    except ValueError:
        return 128
    return value if value > 0 else None


def _tensor_summary(tensor: torch.Tensor) -> str:
    return f"shape={tuple(tensor.shape)} dtype={tensor.dtype} device={tensor.device} contiguous={tensor.is_contiguous()}"


def _log_route(message: str, *args) -> None:
    if is_sglang_kv_route_logging_enabled():
        logger.info("musa_aiter SGLang KV route: " + message, *args)


def _profile_route(message: str, *args) -> None:
    if is_sglang_kv_route_logging_enabled() or is_sglang_kv_route_profile_enabled():
        logger.info("musa_aiter SGLang KV profile: " + message, *args)


def _sync_if_musa(tensor: torch.Tensor) -> None:
    if not is_sglang_kv_route_logging_enabled() or not isinstance(tensor, torch.Tensor) or not _is_musa_tensor(tensor):
        return
    try:
        torch.musa.synchronize(tensor.device)
    except Exception:
        pass


def _is_musa_graph_capture_active() -> bool:
    if not hasattr(torch, "musa"):
        return False
    try:
        capture_status = getattr(torch.musa, "is_current_stream_capturing", None)
        if callable(capture_status):
            return bool(capture_status())
    except Exception:
        return False
    try:
        graph_module = getattr(torch.musa, "graph", None)
        capture_status = getattr(graph_module, "is_current_stream_capturing", None)
        if callable(capture_status):
            return bool(capture_status())
    except Exception:
        return False
    return False


@dataclass(frozen=True)
class SGLangAdapterDecision:
    enabled: bool
    reason: str
    op: str


@dataclass(frozen=True)
class SGLangHiCacheBackendDecision:
    backend: HiCacheIOBackend
    use_musa_aiter_adapter: bool
    reason: str


def is_sglang_musa_aiter_adapter_enabled() -> bool:
    return os.environ.get("MUSA_AITER_SGLANG_ADAPTER", "0") == "1"


def is_sglang_native_cpu_host_kv_transfer_enabled() -> bool:
    return native_requested(
        "MUSA_AITER_NATIVE_CPU_HOST_KV_TRANSFER",
        "MUSA_AITER_NATIVE_KV_TRANSFER",
    )


def is_sglang_cpu_host_staging_cache_enabled() -> bool:
    return os.environ.get("MUSA_AITER_CPU_HOST_STAGING_CACHE", "1") != "0"


def sglang_integration_import_status() -> dict[str, bool]:
    modules = (
        "sglang",
        "sglang.srt.mem_cache.memory_pool_host",
        "sglang.srt.managers.cache_controller",
        "sglang.srt.layers.attention.flashattention_backend",
        "musa_aiter",
    )
    status: dict[str, bool] = {}
    for module in modules:
        try:
            status[module] = find_spec(module) is not None
        except (ImportError, ModuleNotFoundError, ValueError):
            status[module] = False
    return status


def resolve_hicache_io_backend_for_musa(
    backend: str,
    *,
    is_musa: bool,
    adapter_available: bool,
    layout_supported: bool,
) -> SGLangHiCacheBackendDecision:
    if backend not in ("direct", "kernel"):
        return SGLangHiCacheBackendDecision("direct", False, f"unsupported backend {backend!r}")
    typed_backend = backend
    if typed_backend == "direct":
        return SGLangHiCacheBackendDecision("direct", False, "direct is fallback/debug transfer path")
    if not is_musa:
        return SGLangHiCacheBackendDecision("kernel", False, "not a MUSA runtime")
    if not adapter_available:
        return SGLangHiCacheBackendDecision("kernel", False, "musa_aiter adapter is not available")
    if not layout_supported:
        return SGLangHiCacheBackendDecision("kernel", False, "layout is not supported by musa_aiter adapter")
    return SGLangHiCacheBackendDecision("kernel", True, "use internal musa_aiter adapter under kernel backend")


def try_sglang_hicache_layout_transfer(
    src_k: torch.Tensor,
    dst_k: torch.Tensor,
    src_v: torch.Tensor | None,
    dst_v: torch.Tensor | None,
    src_indices: torch.Tensor,
    dst_indices: torch.Tensor,
    *,
    source_layout: Literal["layer_first", "page_first"],
    destination_layout: Literal["layer_first", "page_first"],
    layer_id: int | None = None,
    mla: bool = False,
    require_env: bool = True,
) -> bool:
    """Route exact HiCache layout tensors to the K3 transfer façade.

    This adapter intentionally accepts only MUSA tensors. CPU/reference callers
    should use the operator module directly so framework policy does not alter
    the reference test contract.
    """
    op = f"sglang_hicache_{source_layout}_to_{destination_layout}"
    if require_env and not is_sglang_musa_aiter_adapter_enabled():
        _log_route("fallback op=%s reason=adapter_disabled", op)
        return False
    tensors = (src_k, dst_k, src_indices, dst_indices)
    if src_v is not None:
        tensors += (src_v,)
    if dst_v is not None:
        tensors += (dst_v,)
    if not all(isinstance(tensor, torch.Tensor) for tensor in tensors):
        _log_route("fallback op=%s reason=non_tensor_input", op)
        return False
    if not all(_is_musa_tensor(tensor) for tensor in tensors):
        _log_route("fallback op=%s reason=all_inputs_must_be_musa", op)
        return False
    if source_layout not in ("layer_first", "page_first") or destination_layout not in (
        "layer_first",
        "page_first",
    ):
        _log_route("fallback op=%s reason=unsupported_layout", op)
        return False
    if source_layout == "page_first" and destination_layout == "layer_first":
        if mla:
            return hicache_transfer_kv_per_layer_mla_pf_lf(
                src_k, dst_k, src_indices, dst_indices, layer_id=layer_id or 0
            )
        if layer_id is None:
            _log_route("fallback op=%s reason=page_first_all_layer_requires_explicit_route", op)
            return False
        return hicache_transfer_kv_per_layer_pf_lf(
            src_k,
            dst_k,
            src_v,
            dst_v,
            src_indices,
            dst_indices,
            layer_id=layer_id,
        )
    if source_layout == "layer_first" and destination_layout == "page_first":
        if mla and layer_id is None:
            return hicache_transfer_kv_all_layer_mla_lf_pf(
                src_k, dst_k, src_indices, dst_indices
            )
        if mla or layer_id is not None:
            _log_route("fallback op=%s reason=unsupported_layer_first_to_page_first_contract", op)
            return False
        return hicache_transfer_kv_all_layer_lf_pf(
            src_k, dst_k, src_v, dst_v, src_indices, dst_indices
        )
    if source_layout == "layer_first" and destination_layout == "layer_first":
        if mla:
            if layer_id is None:
                _log_route("fallback op=%s reason=MLA_all_layer_not_exposed_by_adapter", op)
                return False
            return hicache_transfer_kv_per_layer_mla(
                src_k, dst_k, src_indices, dst_indices, layer_id=layer_id
            )
        if layer_id is None:
            return hicache_transfer_kv_all_layer(
                src_k, dst_k, src_v, dst_v, src_indices, dst_indices
            )
        return hicache_transfer_kv_per_layer(
            src_k,
            dst_k,
            src_v,
            dst_v,
            src_indices,
            dst_indices,
            layer_id=layer_id,
        )
    _log_route("fallback op=%s reason=layout_route_not_implemented", op)
    return False


def try_sglang_hicache_direct_transfer(
    src_layers: list[torch.Tensor],
    dst_layers: list[torch.Tensor],
    src_indices: torch.Tensor,
    dst_indices: torch.Tensor,
    page_size: int,
    *,
    mode: Literal["direct", "per_layer_pf_lf", "all_layer_lf_pf"] = "direct",
    layer_id: int = 0,
    require_env: bool = True,
) -> bool:
    """Execute one of SGLang's three direct HiCache transfer contracts."""

    op = f"sglang_hicache_{mode}"
    if require_env and not is_sglang_musa_aiter_adapter_enabled():
        _log_route("fallback op=%s reason=adapter_disabled", op)
        return False
    tensors = list(src_layers) + list(dst_layers) + [src_indices, dst_indices]
    if not all(isinstance(tensor, torch.Tensor) for tensor in tensors):
        _log_route("fallback op=%s reason=non_tensor_input", op)
        return False
    if mode == "direct":
        return hicache_transfer_kv_direct(
            src_layers, dst_layers, src_indices, dst_indices, page_size
        )
    if mode == "per_layer_pf_lf":
        return hicache_transfer_kv_per_layer_direct_pf_lf(
            src_layers,
            dst_layers,
            src_indices,
            dst_indices,
            layer_id,
            page_size,
        )
    if mode == "all_layer_lf_pf":
        return hicache_transfer_kv_all_layer_direct_lf_pf(
            src_layers, dst_layers, src_indices, dst_indices, page_size
        )
    raise ValueError(f"unsupported SGLang direct mode={mode!r}")


def _is_musa_tensor(tensor: torch.Tensor) -> bool:
    return str(tensor.device).split(":")[0] == "musa"


def _is_integer_tensor(tensor: torch.Tensor) -> bool:
    return tensor.dtype in (torch.int16, torch.int32, torch.int64)


def _indices_to_slots(indices: torch.Tensor) -> torch.Tensor:
    return indices.to(dtype=torch.long).contiguous()


def _as_layer_first_cache(cache: torch.Tensor, *, page_size: int) -> torch.Tensor | None:
    if cache.dim() != 3 or page_size <= 0:
        return None
    tokens, heads, head_dim = map(int, cache.shape)
    if tokens % page_size != 0:
        return None
    return cache.reshape(tokens // page_size, page_size, heads, head_dim).permute(0, 2, 3, 1).contiguous()


def _copy_layer_first_cache_from_paged(cache: torch.Tensor, paged: torch.Tensor) -> None:
    tokens = int(cache.shape[0])
    page_size = int(paged.shape[3])
    cache.copy_(paged.permute(0, 3, 1, 2).reshape(tokens, int(cache.shape[1]), int(cache.shape[2])))


def _can_route_sglang_layer_first_hicache(
    host_cache: torch.Tensor,
    device_cache: torch.Tensor,
    host_indices: torch.Tensor,
    device_indices: torch.Tensor,
    *,
    page_size: int,
    op: str,
    require_env: bool,
) -> SGLangAdapterDecision:
    env = _env_decision(op, require_env)
    if env is not None:
        return env
    tensors = (host_cache, device_cache, host_indices, device_indices)
    if not all(isinstance(tensor, torch.Tensor) for tensor in tensors):
        return SGLangAdapterDecision(False, "all inputs must be torch.Tensor", op)
    if not _is_musa_tensor(host_cache) or not _is_musa_tensor(device_cache):
        return SGLangAdapterDecision(False, "host and device cache tensors must both be MUSA tensors", op)
    if not _is_musa_tensor(host_indices) or not _is_musa_tensor(device_indices):
        return SGLangAdapterDecision(False, "host/device indices must both be MUSA tensors", op)
    if host_cache.dtype != device_cache.dtype:
        return SGLangAdapterDecision(False, "host and device cache dtypes must match", op)
    if host_cache.dtype not in (torch.float32, torch.float16, torch.bfloat16):
        return SGLangAdapterDecision(False, "cache dtype must be float32, float16, or bfloat16", op)
    if host_cache.dim() != 3 or device_cache.dim() != 3:
        return SGLangAdapterDecision(False, "layer cache tensors must be [tokens, kv_heads, head_dim]", op)
    if host_cache.shape[1:] != device_cache.shape[1:]:
        return SGLangAdapterDecision(False, "host/device cache inner shapes must match", op)
    if page_size <= 0:
        return SGLangAdapterDecision(False, "page_size must be positive", op)
    if int(host_cache.shape[0]) % page_size != 0 or int(device_cache.shape[0]) % page_size != 0:
        return SGLangAdapterDecision(False, "token dimension must be divisible by page_size", op)
    if not _is_integer_tensor(host_indices) or not _is_integer_tensor(device_indices):
        return SGLangAdapterDecision(False, "host/device indices must use integer dtype", op)
    if host_indices.numel() != device_indices.numel():
        return SGLangAdapterDecision(False, "host/device indices must have the same length", op)
    if host_indices.dim() != 1 or device_indices.dim() != 1:
        return SGLangAdapterDecision(False, "host/device indices must be 1-D", op)
    return SGLangAdapterDecision(True, "supported", op)


def try_sglang_hicache_layer_first_transfer(
    host_key_cache: torch.Tensor,
    host_value_cache: torch.Tensor,
    device_key_cache: torch.Tensor,
    device_value_cache: torch.Tensor,
    host_indices: torch.Tensor,
    device_indices: torch.Tensor,
    *,
    page_size: int,
    direction: Literal["host_to_device", "device_to_host"],
    require_env: bool = True,
) -> bool:
    op = f"sglang_hicache_layer_first_{direction}"
    _log_route(
        "attempt op=%s direction=%s page_size=%s host_key=(%s) device_key=(%s) host_indices=(%s) device_indices=(%s)",
        op,
        direction,
        page_size,
        _tensor_summary(host_key_cache) if isinstance(host_key_cache, torch.Tensor) else type(host_key_cache).__name__,
        _tensor_summary(device_key_cache) if isinstance(device_key_cache, torch.Tensor) else type(device_key_cache).__name__,
        _tensor_summary(host_indices) if isinstance(host_indices, torch.Tensor) else type(host_indices).__name__,
        _tensor_summary(device_indices) if isinstance(device_indices, torch.Tensor) else type(device_indices).__name__,
    )
    if direction not in ("host_to_device", "device_to_host"):
        _log_route("fallback op=%s reason=unsupported_direction direction=%s", op, direction)
        return False
    key_decision = _can_route_sglang_layer_first_hicache(
        host_key_cache,
        device_key_cache,
        host_indices,
        device_indices,
        page_size=page_size,
        op=op,
        require_env=require_env,
    )
    if not key_decision.enabled:
        _log_route("fallback op=%s stage=key_check reason=%s", op, key_decision.reason)
        return False
    value_decision = _can_route_sglang_layer_first_hicache(
        host_value_cache,
        device_value_cache,
        host_indices,
        device_indices,
        page_size=page_size,
        op=op,
        require_env=False,
    )
    if not value_decision.enabled:
        _log_route("fallback op=%s stage=value_check reason=%s", op, value_decision.reason)
        return False
    if host_key_cache.shape != host_value_cache.shape or device_key_cache.shape != device_value_cache.shape:
        _log_route(
            "fallback op=%s stage=kv_shape_check reason=kv_shape_mismatch host_key_shape=%s host_value_shape=%s device_key_shape=%s device_value_shape=%s",
            op,
            tuple(host_key_cache.shape),
            tuple(host_value_cache.shape),
            tuple(device_key_cache.shape),
            tuple(device_value_cache.shape),
        )
        return False
    host_key_paged = _as_layer_first_cache(host_key_cache, page_size=page_size)
    host_value_paged = _as_layer_first_cache(host_value_cache, page_size=page_size)
    device_key_paged = _as_layer_first_cache(device_key_cache, page_size=page_size)
    device_value_paged = _as_layer_first_cache(device_value_cache, page_size=page_size)
    if host_key_paged is None or host_value_paged is None or device_key_paged is None or device_value_paged is None:
        _log_route("fallback op=%s stage=reshape reason=layer_first_to_paged_failed page_size=%s", op, page_size)
        return False
    _log_route(
        "dispatch op=%s direction=%s tokens=%s host_blocks=%s device_blocks=%s kv_heads=%s head_dim=%s dtype=%s",
        op,
        direction,
        int(host_indices.numel()),
        int(host_key_paged.shape[0]),
        int(device_key_paged.shape[0]),
        int(host_key_paged.shape[1]),
        int(host_key_paged.shape[2]),
        host_key_cache.dtype,
    )
    route_start = time.perf_counter()
    if direction == "host_to_device":
        gather_start = time.perf_counter()
        key_linear, value_linear = kv_paged_to_linear(
            host_key_paged,
            host_value_paged,
            _indices_to_slots(host_indices),
        )
        _sync_if_musa(host_key_cache)
        gather_ms = (time.perf_counter() - gather_start) * 1000.0
        scatter_start = time.perf_counter()
        kv_linear_to_paged(
            key_linear,
            value_linear,
            device_key_paged,
            device_value_paged,
            _indices_to_slots(device_indices),
        )
        _copy_layer_first_cache_from_paged(device_key_cache, device_key_paged)
        _copy_layer_first_cache_from_paged(device_value_cache, device_value_paged)
        _sync_if_musa(device_key_cache)
        scatter_ms = (time.perf_counter() - scatter_start) * 1000.0
        total_ms = (time.perf_counter() - route_start) * 1000.0
        _log_route(
            "success op=%s direction=%s tokens=%s gather_ms=%.3f scatter_ms=%.3f total_ms=%.3f linear_key=(%s) linear_value=(%s)",
            op,
            direction,
            int(host_indices.numel()),
            gather_ms,
            scatter_ms,
            total_ms,
            _tensor_summary(key_linear),
            _tensor_summary(value_linear),
        )
        return True
    gather_start = time.perf_counter()
    key_linear, value_linear = kv_paged_to_linear(
        device_key_paged,
        device_value_paged,
        _indices_to_slots(device_indices),
    )
    _sync_if_musa(device_key_cache)
    gather_ms = (time.perf_counter() - gather_start) * 1000.0
    scatter_start = time.perf_counter()
    kv_linear_to_paged(
        key_linear,
        value_linear,
        host_key_paged,
        host_value_paged,
        _indices_to_slots(host_indices),
    )
    _copy_layer_first_cache_from_paged(host_key_cache, host_key_paged)
    _copy_layer_first_cache_from_paged(host_value_cache, host_value_paged)
    _sync_if_musa(host_key_cache)
    scatter_ms = (time.perf_counter() - scatter_start) * 1000.0
    total_ms = (time.perf_counter() - route_start) * 1000.0
    _log_route(
        "success op=%s direction=%s tokens=%s gather_ms=%.3f scatter_ms=%.3f total_ms=%.3f linear_key=(%s) linear_value=(%s)",
        op,
        direction,
        int(host_indices.numel()),
        gather_ms,
        scatter_ms,
        total_ms,
        _tensor_summary(key_linear),
        _tensor_summary(value_linear),
    )
    return True


def _env_decision(op: str, require_env: bool) -> SGLangAdapterDecision | None:
    if require_env and not is_sglang_musa_aiter_adapter_enabled():
        return SGLangAdapterDecision(False, "MUSA_AITER_SGLANG_ADAPTER is not enabled", op)
    return None


def _cpu_host_env_decision(op: str, require_env: bool) -> SGLangAdapterDecision | None:
    env = _env_decision(op, require_env)
    if env is not None:
        return env
    if require_env and not is_sglang_native_cpu_host_kv_transfer_enabled():
        return SGLangAdapterDecision(False, "MUSA_AITER_NATIVE_CPU_HOST_KV_TRANSFER is not enabled", op)
    return None


def _can_route_sglang_flat_paged_hicache(
    host_pool,
    device_pool,
    *,
    op: str,
    require_env: bool,
) -> SGLangAdapterDecision:
    """REQ-5: DSV4 flat uint8 byte-row pools (kv_buffer[L, pages, item_bytes])."""
    env = _env_decision(op, require_env)
    if env is not None:
        return env
    if host_pool is None or device_pool is None:
        return SGLangAdapterDecision(False, "host/device pools must be objects", op)
    if getattr(host_pool, "layout", None) != "layer_first":
        return SGLangAdapterDecision(False, "host pool layout must be layer_first", op)
    host_kv = getattr(host_pool, "kv_buffer", None)
    device_kv = getattr(device_pool, "kv_buffer", None)
    if not isinstance(host_kv, torch.Tensor) or not isinstance(device_kv, torch.Tensor):
        return SGLangAdapterDecision(False, "host/device pools must expose kv_buffer tensors", op)
    if host_kv.dtype is not torch.uint8 or device_kv.dtype is not torch.uint8:
        return SGLangAdapterDecision(
            False, f"kv_buffer must be uint8 byte rows, got host={host_kv.dtype}, device={device_kv.dtype}", op
        )
    if host_kv.dim() != 3 or device_kv.dim() != 3:
        return SGLangAdapterDecision(False, "kv_buffer must be [layers, pages, item_bytes]", op)
    if host_kv.shape[0] != device_kv.shape[0] or host_kv.shape[2] != device_kv.shape[2]:
        return SGLangAdapterDecision(
            False,
            f"kv_buffer layer/item_bytes mismatch: host={tuple(host_kv.shape)}, device={tuple(device_kv.shape)}",
            op,
        )
    return SGLangAdapterDecision(True, "supported", op)


def try_sglang_hicache_flat_paged_transfer(
    host_pool,
    device_pool,
    host_indices: torch.Tensor,
    device_indices: torch.Tensor,
    *,
    direction: Literal["load_to_device", "backup_from_device"],
    require_env: bool = True,
) -> bool:
    """REQ-5: route DSV4 flat byte-row pools to kv_copy_paged_rows ops.

    Matches pools whose kv_buffer is uint8 [layers, pages, item_bytes] with
    layer_first layout; no K/V float semantics — whole page rows are copied
    with the REQ-1 kernel path. The existing MHA/MLA routes are unaffected.
    """
    op = f"sglang_hicache_flat_paged_{direction}"
    _log_route(
        "attempt op=%s host_pool=%s device_pool=%s host_indices=(%s) device_indices=(%s)",
        op,
        type(host_pool).__name__,
        type(device_pool).__name__,
        _tensor_summary(host_indices) if isinstance(host_indices, torch.Tensor) else type(host_indices).__name__,
        _tensor_summary(device_indices) if isinstance(device_indices, torch.Tensor) else type(device_indices).__name__,
    )
    if direction not in ("load_to_device", "backup_from_device"):
        _log_route("fallback op=%s reason=unsupported_direction direction=%s", op, direction)
        return False
    decision = _can_route_sglang_flat_paged_hicache(host_pool, device_pool, op=op, require_env=require_env)
    if not decision.enabled:
        _log_route("fallback op=%s reason=%s", op, decision.reason)
        return False
    if not _is_integer_tensor(host_indices) or not _is_integer_tensor(device_indices):
        _log_route("fallback op=%s reason=indices_not_integer", op)
        return False
    if host_indices.numel() != device_indices.numel() or host_indices.dim() != 1:
        _log_route("fallback op=%s reason=indices_shape", op)
        return False
    host_kv = host_pool.kv_buffer
    device_kv = device_pool.kv_buffer
    num_layers = int(host_kv.shape[0])
    src = host_kv.unbind(0) if direction == "load_to_device" else device_kv.unbind(0)
    dst = device_kv.unbind(0) if direction == "load_to_device" else host_kv.unbind(0)
    src_indices, dst_indices = (
        (host_indices, device_indices) if direction == "load_to_device" else (device_indices, host_indices)
    )
    route_start = time.perf_counter()
    kv_copy_paged_rows_layer_table(list(src), list(dst), src_indices, dst_indices)
    _sync_if_musa(dst[0])
    total_ms = (time.perf_counter() - route_start) * 1000.0
    _log_route(
        "success op=%s route_type=flat_paged direction=%s layers=%s rows=%s item_bytes=%s total_ms=%.3f",
        op,
        direction,
        num_layers,
        int(host_indices.numel()),
        int(host_kv.shape[2]),
        total_ms,
    )
    return True


def _is_cpu_tensor(tensor: torch.Tensor) -> bool:
    return str(tensor.device).split(":")[0] == "cpu"


def _indices_to_cpu_long(indices: torch.Tensor) -> torch.Tensor:
    return indices.to(device="cpu", dtype=torch.long).contiguous()


def _indices_to_musa_long(indices: torch.Tensor, device: torch.device) -> torch.Tensor:
    return indices.to(device=device, dtype=torch.long).contiguous()


_STAGING_CACHE_MAX_ENTRIES = 32
_sglang_cpu_host_staging_cache: dict[tuple[str, str, tuple[int, ...], str], torch.Tensor] = {}


def clear_sglang_cpu_host_staging_cache() -> None:
    _sglang_cpu_host_staging_cache.clear()


def _staging_cache_key(name: str, *, shape: tuple[int, ...], dtype: torch.dtype, device: torch.device) -> tuple[str, str, tuple[int, ...], str]:
    return (name, str(dtype), shape, str(device))


def _get_staging_tensor(
    name: str,
    *,
    like: torch.Tensor,
    shape: tuple[int, ...],
    device: torch.device,
    pin_memory: bool | None = None,
) -> torch.Tensor:
    if pin_memory is None:
        pin_memory = device.type == "cpu" and _is_musa_tensor(like)
    if not is_sglang_cpu_host_staging_cache_enabled():
        return torch.empty(shape, dtype=like.dtype, device=device, pin_memory=pin_memory)
    key = _staging_cache_key(name, shape=shape, dtype=like.dtype, device=device)
    cached = _sglang_cpu_host_staging_cache.get(key)
    if cached is not None:
        return cached
    if len(_sglang_cpu_host_staging_cache) >= _STAGING_CACHE_MAX_ENTRIES:
        _sglang_cpu_host_staging_cache.clear()
    cached = torch.empty(shape, dtype=like.dtype, device=device, pin_memory=pin_memory)
    _sglang_cpu_host_staging_cache[key] = cached
    return cached


def _host_tensor_is_pinned(tensor: torch.Tensor) -> bool:
    try:
        return bool(tensor.is_pinned())
    except (AttributeError, RuntimeError):
        return False


def _try_native_cpu_host_staging_copy(
    destination: torch.Tensor,
    source: torch.Tensor,
    *,
    op: str,
) -> bool:
    if not is_sglang_native_cpu_host_kv_transfer_enabled():
        _log_route("fallback op=%s stage=host_copy reason=native_host_gate_off", op)
        return False
    if _is_cpu_tensor(source) and _is_musa_tensor(destination):
        host_tensor = source
    elif _is_musa_tensor(source) and _is_cpu_tensor(destination):
        host_tensor = destination
    else:
        _log_route("fallback op=%s stage=host_copy reason=expected_cpu_musa_pair", op)
        return False
    if not _host_tensor_is_pinned(host_tensor):
        _log_route("fallback op=%s stage=host_copy reason=host_tensor_not_pinned", op)
        return False
    size_bytes = int(source.numel()) * int(source.element_size())
    try:
        submitted = lmcache_musa_memcpy_batch_async(
            [destination],
            [source],
            [size_bytes],
            require_pinned_host=True,
        )
    except (RuntimeError, TypeError, ValueError) as exc:
        _log_route("fallback op=%s stage=host_copy reason=batch_copy_error error=%s", op, exc)
        return False
    if not submitted:
        _log_route("fallback op=%s stage=host_copy reason=batch_copy_unavailable", op)
        return False
    _log_route(
        "native op=%s stage=host_copy backend=musa-sdk-batch-copy direction=%s bytes=%s",
        op,
        "h2d" if _is_cpu_tensor(source) else "d2h",
        size_bytes,
    )
    return True


def _copy_to_staging(name: str, source: torch.Tensor, *, device: torch.device) -> torch.Tensor:
    staging = _get_staging_tensor(name, like=source, shape=tuple(source.shape), device=device)
    if not _try_native_cpu_host_staging_copy(staging, source, op=name):
        staging.copy_(source, non_blocking=True)
    return staging


def _copy_from_staging(name: str, source: torch.Tensor, *, like: torch.Tensor) -> torch.Tensor:
    staging = _get_staging_tensor(
        name,
        like=like,
        shape=tuple(source.shape),
        device=torch.device("cpu"),
        pin_memory=True,
    )
    if not _try_native_cpu_host_staging_copy(staging, source, op=name):
        staging.copy_(source, non_blocking=True)
    return staging


def _contiguous_index_range(indices: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int] | None]:
    slots = _indices_to_cpu_long(indices)
    count = int(slots.numel())
    if count == 0:
        return slots, (0, 0)
    start = int(slots[0].item())
    if count == 1:
        return slots, (start, start + 1)
    if bool((slots[1:] - slots[:-1]).eq(1).all().item()):
        return slots, (start, start + count)
    return slots, None


def _select_host_slots(cache: torch.Tensor, dim: int, indices: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, str]:
    slots, index_range = _contiguous_index_range(indices)
    if index_range is None:
        return slots, cache.index_select(dim, slots), "index_select"
    start, end = index_range
    return slots, cache.narrow(dim, start, end - start), "slice"


def _copy_host_slots(cache: torch.Tensor, dim: int, indices: torch.Tensor, source: torch.Tensor) -> str:
    slots, index_range = _contiguous_index_range(indices)
    if index_range is None:
        cache.index_copy_(dim, slots, source)
        return "index_copy"
    start, end = index_range
    cache.narrow(dim, start, end - start).copy_(source)
    return "slice_copy"


def _contiguous_device_index_range(indices: torch.Tensor) -> tuple[int, int] | None:
    slots = indices.to(dtype=torch.long).contiguous()
    count = int(slots.numel())
    if count == 0:
        return (0, 0)
    start = int(slots[0].item())
    if count == 1:
        return (start, start + 1)
    diff = slots[1:] - slots[:-1]
    if _is_musa_tensor(diff):
        try:
            contiguous = bool(diff.eq(1).all().item())
        except Exception:
            return None
    else:
        contiguous = bool(diff.eq(1).all().item())
    if contiguous:
        return (start, start + count)
    return None


def _copy_layer_first_contiguous_h2d(
    key_linear: torch.Tensor,
    value_linear: torch.Tensor,
    device_key_cache: torch.Tensor,
    device_value_cache: torch.Tensor,
    device_indices: torch.Tensor,
) -> bool:
    index_range = _contiguous_device_index_range(device_indices)
    if index_range is None:
        return False
    start, end = index_range
    device_key_cache.narrow(0, start, end - start).copy_(key_linear)
    device_value_cache.narrow(0, start, end - start).copy_(value_linear)
    return True


def _select_layer_first_contiguous_d2h(
    device_key_cache: torch.Tensor,
    device_value_cache: torch.Tensor,
    device_indices: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    index_range = _contiguous_device_index_range(device_indices)
    if index_range is None:
        return None
    start, end = index_range
    return device_key_cache.narrow(0, start, end - start), device_value_cache.narrow(0, start, end - start)


def _can_route_sglang_layer_first_cpu_host_hicache(
    host_cache: torch.Tensor,
    device_cache: torch.Tensor,
    host_indices: torch.Tensor,
    device_indices: torch.Tensor,
    *,
    page_size: int,
    op: str,
    require_env: bool,
) -> SGLangAdapterDecision:
    env = _cpu_host_env_decision(op, require_env)
    if env is not None:
        return env
    tensors = (host_cache, device_cache, host_indices, device_indices)
    if not all(isinstance(tensor, torch.Tensor) for tensor in tensors):
        return SGLangAdapterDecision(False, "all inputs must be torch.Tensor", op)
    if not _is_cpu_tensor(host_cache):
        return SGLangAdapterDecision(False, "host cache tensor must be a CPU tensor", op)
    if not _is_musa_tensor(device_cache):
        return SGLangAdapterDecision(False, "device cache tensor must be a MUSA tensor", op)
    if not _is_integer_tensor(host_indices) or not _is_integer_tensor(device_indices):
        return SGLangAdapterDecision(False, "host/device indices must use integer dtype", op)
    if host_cache.dtype != device_cache.dtype:
        return SGLangAdapterDecision(False, "host and device cache dtypes must match", op)
    if host_cache.dtype not in (torch.float32, torch.float16, torch.bfloat16):
        return SGLangAdapterDecision(False, "cache dtype must be float32, float16, or bfloat16", op)
    if host_cache.dim() != 3 or device_cache.dim() != 3:
        return SGLangAdapterDecision(False, "layer cache tensors must be [tokens, kv_heads, head_dim]", op)
    if host_cache.shape[1:] != device_cache.shape[1:]:
        return SGLangAdapterDecision(False, "host/device cache inner shapes must match", op)
    if page_size <= 0:
        return SGLangAdapterDecision(False, "page_size must be positive", op)
    if int(host_cache.shape[0]) % page_size != 0 or int(device_cache.shape[0]) % page_size != 0:
        return SGLangAdapterDecision(False, "token dimension must be divisible by page_size", op)
    if host_indices.numel() != device_indices.numel():
        return SGLangAdapterDecision(False, "host/device indices must have the same length", op)
    if host_indices.dim() != 1 or device_indices.dim() != 1:
        return SGLangAdapterDecision(False, "host/device indices must be 1-D", op)
    return SGLangAdapterDecision(True, "supported", op)


def _can_route_sglang_layer_first_cpu_host_hicache_multi_layer(
    host_key_cache: torch.Tensor,
    host_value_cache: torch.Tensor,
    device_key_cache: torch.Tensor,
    device_value_cache: torch.Tensor,
    host_indices: torch.Tensor,
    device_indices: torch.Tensor,
    *,
    page_size: int,
    op: str,
    require_env: bool,
) -> SGLangAdapterDecision:
    env = _cpu_host_env_decision(op, require_env)
    if env is not None:
        return env
    tensors = (host_key_cache, host_value_cache, device_key_cache, device_value_cache, host_indices, device_indices)
    if not all(isinstance(tensor, torch.Tensor) for tensor in tensors):
        return SGLangAdapterDecision(False, "all inputs must be torch.Tensor", op)
    if host_key_cache.dim() != 4 or host_value_cache.dim() != 4 or device_key_cache.dim() != 4 or device_value_cache.dim() != 4:
        return SGLangAdapterDecision(False, "multi-layer caches must be [layers, tokens, kv_heads, head_dim]", op)
    if host_key_cache.shape != host_value_cache.shape or device_key_cache.shape != device_value_cache.shape:
        return SGLangAdapterDecision(False, "key/value cache shapes must match", op)
    if host_key_cache.shape[0] != device_key_cache.shape[0] or host_key_cache.shape[2:] != device_key_cache.shape[2:]:
        return SGLangAdapterDecision(False, "host/device multi-layer cache layer count and inner shapes must match", op)
    key_decision = _can_route_sglang_layer_first_cpu_host_hicache(
        host_key_cache[0],
        device_key_cache[0],
        host_indices,
        device_indices,
        page_size=page_size,
        op=op,
        require_env=False,
    )
    if not key_decision.enabled:
        return key_decision
    value_decision = _can_route_sglang_layer_first_cpu_host_hicache(
        host_value_cache[0],
        device_value_cache[0],
        host_indices,
        device_indices,
        page_size=page_size,
        op=op,
        require_env=False,
    )
    if not value_decision.enabled:
        return value_decision
    return SGLangAdapterDecision(True, "supported", op)


def _layer_first_multi_to_paged(cache: torch.Tensor, *, page_size: int) -> torch.Tensor | None:
    if cache.dim() != 4 or page_size <= 0:
        return None
    layers, tokens, heads, head_dim = map(int, cache.shape)
    if tokens % page_size != 0:
        return None
    return cache.reshape(layers, tokens // page_size, page_size, heads, head_dim).permute(0, 1, 3, 4, 2).contiguous()


def _copy_layer_first_multi_from_paged(cache: torch.Tensor, paged: torch.Tensor) -> None:
    layers = int(cache.shape[0])
    tokens = int(cache.shape[1])
    page_size = int(paged.shape[4])
    cache.copy_(paged.permute(0, 1, 4, 2, 3).reshape(layers, tokens, int(cache.shape[2]), int(cache.shape[3])))


def try_sglang_hicache_layer_first_cpu_host_transfer_multi_layer(
    host_key_cache: torch.Tensor,
    host_value_cache: torch.Tensor,
    device_key_cache: torch.Tensor,
    device_value_cache: torch.Tensor,
    host_indices: torch.Tensor,
    device_indices: torch.Tensor,
    *,
    page_size: int,
    direction: Literal["host_to_device", "device_to_host"],
    require_env: bool = True,
) -> bool:
    op = f"sglang_hicache_layer_first_cpu_host_multi_layer_{direction}"
    if direction not in ("host_to_device", "device_to_host"):
        _log_route("fallback op=%s route_type=cpu_host_staged_multi_layer reason=unsupported_direction direction=%s", op, direction)
        return False
    decision = _can_route_sglang_layer_first_cpu_host_hicache_multi_layer(
        host_key_cache,
        host_value_cache,
        device_key_cache,
        device_value_cache,
        host_indices,
        device_indices,
        page_size=page_size,
        op=op,
        require_env=require_env,
    )
    if not decision.enabled:
        _log_route("fallback op=%s route_type=cpu_host_staged_multi_layer reason=%s", op, decision.reason)
        return False
    device_key_paged = _layer_first_multi_to_paged(device_key_cache, page_size=page_size)
    device_value_paged = _layer_first_multi_to_paged(device_value_cache, page_size=page_size)
    if device_key_paged is None or device_value_paged is None:
        _log_route("fallback op=%s route_type=cpu_host_staged_multi_layer reason=device_layer_first_to_paged_failed", op)
        return False
    route_start = time.perf_counter()
    if direction == "host_to_device":
        host_slots_cpu, key_linear_cpu, cpu_index_mode = _select_host_slots(host_key_cache, 1, host_indices)
        _, value_linear_cpu, value_cpu_index_mode = _select_host_slots(host_value_cache, 1, host_indices)
        key_linear = _copy_to_staging("cpu_host_h2d_key_multi", key_linear_cpu, device=device_key_cache.device)
        value_linear = _copy_to_staging("cpu_host_h2d_value_multi", value_linear_cpu, device=device_value_cache.device)
        kv_linear_to_paged_multi_layer(
            key_linear,
            value_linear,
            device_key_paged,
            device_value_paged,
            _indices_to_musa_long(device_indices, device_key_cache.device),
        )
        _copy_layer_first_multi_from_paged(device_key_cache, device_key_paged)
        _copy_layer_first_multi_from_paged(device_value_cache, device_value_paged)
        _log_route(
            "success op=%s route_type=cpu_host_staged_multi_layer direction=%s layers=%s tokens=%s staging_cache=%s cpu_index_mode=%s value_cpu_index_mode=%s total_ms=%.3f",
            op,
            direction,
            int(host_key_cache.shape[0]),
            int(host_indices.numel()),
            is_sglang_cpu_host_staging_cache_enabled(),
            cpu_index_mode,
            value_cpu_index_mode,
            (time.perf_counter() - route_start) * 1000.0,
        )
        return True
    key_linear, value_linear = kv_paged_to_linear_multi_layer(
        device_key_paged,
        device_value_paged,
        _indices_to_musa_long(device_indices, device_key_cache.device),
    )
    key_linear_cpu = _copy_from_staging("cpu_host_d2h_key_multi", key_linear, like=host_key_cache[:, : int(device_indices.numel())])
    value_linear_cpu = _copy_from_staging("cpu_host_d2h_value_multi", value_linear, like=host_value_cache[:, : int(device_indices.numel())])
    cpu_index_mode = _copy_host_slots(host_key_cache, 1, host_indices, key_linear_cpu)
    value_cpu_index_mode = _copy_host_slots(host_value_cache, 1, host_indices, value_linear_cpu)
    _log_route(
        "success op=%s route_type=cpu_host_staged_multi_layer direction=%s layers=%s tokens=%s staging_cache=%s cpu_index_mode=%s value_cpu_index_mode=%s total_ms=%.3f",
        op,
        direction,
        int(host_key_cache.shape[0]),
        int(host_indices.numel()),
        is_sglang_cpu_host_staging_cache_enabled(),
        cpu_index_mode,
        value_cpu_index_mode,
        (time.perf_counter() - route_start) * 1000.0,
    )
    return True


def try_sglang_hicache_layer_first_cpu_host_transfer(
    host_key_cache: torch.Tensor,
    host_value_cache: torch.Tensor,
    device_key_cache: torch.Tensor,
    device_value_cache: torch.Tensor,
    host_indices: torch.Tensor,
    device_indices: torch.Tensor,
    *,
    page_size: int,
    direction: Literal["host_to_device", "device_to_host"],
    require_env: bool = True,
) -> bool:
    op = f"sglang_hicache_layer_first_cpu_host_{direction}"
    _log_route(
        "attempt op=%s route_type=cpu_host_staged direction=%s page_size=%s host_key=(%s) device_key=(%s) host_indices=(%s) device_indices=(%s)",
        op,
        direction,
        page_size,
        _tensor_summary(host_key_cache) if isinstance(host_key_cache, torch.Tensor) else type(host_key_cache).__name__,
        _tensor_summary(device_key_cache) if isinstance(device_key_cache, torch.Tensor) else type(device_key_cache).__name__,
        _tensor_summary(host_indices) if isinstance(host_indices, torch.Tensor) else type(host_indices).__name__,
        _tensor_summary(device_indices) if isinstance(device_indices, torch.Tensor) else type(device_indices).__name__,
    )
    if direction not in ("host_to_device", "device_to_host"):
        _log_route("fallback op=%s route_type=cpu_host_staged reason=unsupported_direction direction=%s", op, direction)
        return False
    key_decision = _can_route_sglang_layer_first_cpu_host_hicache(
        host_key_cache,
        device_key_cache,
        host_indices,
        device_indices,
        page_size=page_size,
        op=op,
        require_env=require_env,
    )
    if not key_decision.enabled:
        _log_route("fallback op=%s route_type=cpu_host_staged stage=key_check reason=%s", op, key_decision.reason)
        return False
    value_decision = _can_route_sglang_layer_first_cpu_host_hicache(
        host_value_cache,
        device_value_cache,
        host_indices,
        device_indices,
        page_size=page_size,
        op=op,
        require_env=False,
    )
    if not value_decision.enabled:
        _log_route("fallback op=%s route_type=cpu_host_staged stage=value_check reason=%s", op, value_decision.reason)
        return False
    if host_key_cache.shape != host_value_cache.shape or device_key_cache.shape != device_value_cache.shape:
        _log_route(
            "fallback op=%s route_type=cpu_host_staged stage=kv_shape_check reason=kv_shape_mismatch host_key_shape=%s host_value_shape=%s device_key_shape=%s device_value_shape=%s",
            op,
            tuple(host_key_cache.shape),
            tuple(host_value_cache.shape),
            tuple(device_key_cache.shape),
            tuple(device_value_cache.shape),
        )
        return False
    device_key_paged = _as_layer_first_cache(device_key_cache, page_size=page_size)
    device_value_paged = _as_layer_first_cache(device_value_cache, page_size=page_size)
    if device_key_paged is None or device_value_paged is None:
        _log_route("fallback op=%s route_type=cpu_host_staged stage=reshape reason=device_layer_first_to_paged_failed page_size=%s", op, page_size)
        return False
    route_start = time.perf_counter()
    if direction == "host_to_device":
        cpu_index_start = time.perf_counter()
        host_slots_cpu, key_linear_cpu, cpu_index_mode = _select_host_slots(host_key_cache, 0, host_indices)
        _, value_linear_cpu, value_cpu_index_mode = _select_host_slots(host_value_cache, 0, host_indices)
        cpu_index_ms = (time.perf_counter() - cpu_index_start) * 1000.0
        h2d_start = time.perf_counter()
        key_linear = _copy_to_staging("cpu_host_h2d_key", key_linear_cpu, device=device_key_cache.device)
        value_linear = _copy_to_staging("cpu_host_h2d_value", value_linear_cpu, device=device_value_cache.device)
        h2d_copy_ms = (time.perf_counter() - h2d_start) * 1000.0
        h2d_sync_start = time.perf_counter()
        _sync_if_musa(device_key_cache)
        h2d_sync_ms = (time.perf_counter() - h2d_sync_start) * 1000.0
        h2d_ms = h2d_copy_ms + h2d_sync_ms
        device_layout_start = time.perf_counter()
        layout_mode = "layer_first_slice" if _copy_layer_first_contiguous_h2d(
            key_linear,
            value_linear,
            device_key_cache,
            device_value_cache,
            device_indices,
        ) else "paged_kernel"
        if layout_mode == "paged_kernel":
            kv_linear_to_paged(
                key_linear,
                value_linear,
                device_key_paged,
                device_value_paged,
                _indices_to_musa_long(device_indices, device_key_cache.device),
            )
            _copy_layer_first_cache_from_paged(device_key_cache, device_key_paged)
            _copy_layer_first_cache_from_paged(device_value_cache, device_value_paged)
        layout_op_ms = (time.perf_counter() - device_layout_start) * 1000.0
        layout_sync_start = time.perf_counter()
        _sync_if_musa(device_key_cache)
        layout_sync_ms = (time.perf_counter() - layout_sync_start) * 1000.0
        device_layout_ms = layout_op_ms + layout_sync_ms
        total_ms = (time.perf_counter() - route_start) * 1000.0
        _profile_route(
            "success op=%s route_type=cpu_host_staged direction=%s host_pinned=%s staging_cache=%s tokens=%s cpu_index_mode=%s value_cpu_index_mode=%s layout_mode=%s cpu_index_ms=%.3f h2d_ms=%.3f h2d_copy_ms=%.3f h2d_sync_ms=%.3f device_layout_ms=%.3f layout_op_ms=%.3f layout_sync_ms=%.3f total_ms=%.3f linear_key=(%s) linear_value=(%s)",
            op,
            direction,
            bool(host_key_cache.is_pinned()) if hasattr(host_key_cache, "is_pinned") else False,
            is_sglang_cpu_host_staging_cache_enabled(),
            int(host_indices.numel()),
            cpu_index_mode,
            value_cpu_index_mode,
            layout_mode,
            cpu_index_ms,
            h2d_ms,
            h2d_copy_ms,
            h2d_sync_ms,
            device_layout_ms,
            layout_op_ms,
            layout_sync_ms,
            total_ms,
            _tensor_summary(key_linear),
            _tensor_summary(value_linear),
        )
        return True
    device_layout_start = time.perf_counter()
    direct_d2h = _select_layer_first_contiguous_d2h(device_key_cache, device_value_cache, device_indices)
    if direct_d2h is not None:
        key_linear, value_linear = direct_d2h
        layout_mode = "layer_first_slice"
    else:
        key_linear, value_linear = kv_paged_to_linear(
            device_key_paged,
            device_value_paged,
            _indices_to_musa_long(device_indices, device_key_cache.device),
        )
        layout_mode = "paged_kernel"
    layout_op_ms = (time.perf_counter() - device_layout_start) * 1000.0
    layout_sync_start = time.perf_counter()
    _sync_if_musa(device_key_cache)
    layout_sync_ms = (time.perf_counter() - layout_sync_start) * 1000.0
    device_layout_ms = layout_op_ms + layout_sync_ms
    d2h_start = time.perf_counter()
    key_linear_cpu = _copy_from_staging("cpu_host_d2h_key", key_linear, like=host_key_cache)
    value_linear_cpu = _copy_from_staging("cpu_host_d2h_value", value_linear, like=host_value_cache)
    d2h_copy_ms = (time.perf_counter() - d2h_start) * 1000.0
    d2h_ms = d2h_copy_ms
    cpu_index_start = time.perf_counter()
    cpu_index_mode = _copy_host_slots(host_key_cache, 0, host_indices, key_linear_cpu)
    value_cpu_index_mode = _copy_host_slots(host_value_cache, 0, host_indices, value_linear_cpu)
    cpu_index_ms = (time.perf_counter() - cpu_index_start) * 1000.0
    total_ms = (time.perf_counter() - route_start) * 1000.0
    _profile_route(
        "success op=%s route_type=cpu_host_staged direction=%s host_pinned=%s staging_cache=%s tokens=%s layout_mode=%s device_layout_ms=%.3f layout_op_ms=%.3f layout_sync_ms=%.3f d2h_ms=%.3f d2h_copy_ms=%.3f cpu_index_mode=%s value_cpu_index_mode=%s cpu_index_ms=%.3f total_ms=%.3f linear_key=(%s) linear_value=(%s)",
        op,
        direction,
        bool(host_key_cache.is_pinned()) if hasattr(host_key_cache, "is_pinned") else False,
        is_sglang_cpu_host_staging_cache_enabled(),
        int(host_indices.numel()),
        layout_mode,
        device_layout_ms,
        layout_op_ms,
        layout_sync_ms,
        d2h_ms,
        d2h_copy_ms,
        cpu_index_mode,
        value_cpu_index_mode,
        cpu_index_ms,
        total_ms,
        _tensor_summary(key_linear),
        _tensor_summary(value_linear),
    )
    return True


def check_kv_load_and_reshape_flash(
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    *,
    require_env: bool = True,
) -> SGLangAdapterDecision:
    op = "kv_load_and_reshape_flash"
    env = _env_decision(op, require_env)
    if env is not None:
        return env
    tensors = (key_cache, value_cache, block_tables, seq_lens)
    if not all(isinstance(tensor, torch.Tensor) for tensor in tensors):
        return SGLangAdapterDecision(False, "all inputs must be torch.Tensor", op)
    if not all(_is_musa_tensor(tensor) for tensor in tensors):
        return SGLangAdapterDecision(False, "all inputs must be MUSA tensors", op)
    if key_cache.dim() != 4 or value_cache.dim() != 4:
        return SGLangAdapterDecision(False, "key/value cache must be [blocks, kv_heads, head_dim, block_size]", op)
    if key_cache.shape != value_cache.shape:
        return SGLangAdapterDecision(False, "key/value cache shapes must match", op)
    if key_cache.dtype != value_cache.dtype:
        return SGLangAdapterDecision(False, "key/value cache dtypes must match", op)
    if not key_cache.is_contiguous() or not value_cache.is_contiguous():
        return SGLangAdapterDecision(False, "key/value cache must be contiguous", op)
    if block_tables.dim() != 2:
        return SGLangAdapterDecision(False, "block_tables must be [batch, max_blocks]", op)
    if seq_lens.dim() != 1 or seq_lens.shape[0] != block_tables.shape[0]:
        return SGLangAdapterDecision(False, "seq_lens must be [batch]", op)
    if not _is_integer_tensor(block_tables) or not _is_integer_tensor(seq_lens):
        return SGLangAdapterDecision(False, "block_tables and seq_lens must be integer tensors", op)
    return SGLangAdapterDecision(True, "supported", op)


def try_kv_load_and_reshape_flash(
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    *,
    max_seq_len: int | None = None,
    out_k: torch.Tensor | None = None,
    out_v: torch.Tensor | None = None,
    require_env: bool = True,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    decision = check_kv_load_and_reshape_flash(
        key_cache,
        value_cache,
        block_tables,
        seq_lens,
        require_env=require_env,
    )
    if not decision.enabled:
        return None
    return kv_load_and_reshape_flash(
        key_cache,
        value_cache,
        block_tables,
        seq_lens,
        max_seq_len=max_seq_len,
        out_k=out_k,
        out_v=out_v,
    )


def resolve_sglang_attention_preload_replacement(
    *,
    mate_accepts_dense_kv: bool,
    parity_validated: bool,
    require_env: bool = True,
) -> SGLangAdapterDecision:
    op = "sglang_attention_preload_replace"
    if require_env and not is_sglang_attention_preload_route_enabled():
        return SGLangAdapterDecision(False, "MUSA_AITER_SGLANG_ATTENTION_PRELOAD is not enabled", op)
    if require_env and not is_sglang_attention_preload_replace_enabled():
        return SGLangAdapterDecision(False, "MUSA_AITER_SGLANG_ATTENTION_PRELOAD_REPLACE is not enabled", op)
    if not mate_accepts_dense_kv:
        return SGLangAdapterDecision(False, "MATE attention path does not accept dense preloaded KV tensors", op)
    if not parity_validated:
        return SGLangAdapterDecision(False, "attention preload replacement parity is not validated", op)
    return SGLangAdapterDecision(True, "supported", op)


def probe_sglang_attention_preload(
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    *,
    require_env: bool = True,
) -> SGLangAdapterDecision:
    op = "sglang_attention_preload_probe"
    if require_env and not is_sglang_attention_preload_probe_enabled():
        return SGLangAdapterDecision(False, "MUSA_AITER_SGLANG_ATTENTION_PRELOAD_PROBE is not enabled", op)
    decision = check_kv_load_and_reshape_flash(
        key_cache,
        value_cache,
        block_tables,
        seq_lens,
        require_env=require_env,
    )
    if decision.enabled:
        _log_route(
            "probe_supported route_type=attention_preload_probe op=%s key=(%s) block_tables=(%s) seq_lens=(%s)",
            op,
            _tensor_summary(key_cache),
            _tensor_summary(block_tables),
            _tensor_summary(seq_lens),
        )
        return SGLangAdapterDecision(True, "supported", op)
    _log_route("probe_rejected route_type=attention_preload_probe op=%s reason=%s", op, decision.reason)
    return SGLangAdapterDecision(False, decision.reason, op)


def try_sglang_attention_kv_preload_route(
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    *,
    max_seq_len: int | None = None,
    out_k: torch.Tensor | None = None,
    out_v: torch.Tensor | None = None,
    require_env: bool = True,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    if require_env and not is_sglang_attention_preload_route_enabled():
        _log_route("skip route_type=attention_preload reason=MUSA_AITER_SGLANG_ATTENTION_PRELOAD is not enabled")
        return None
    result = try_sglang_attention_kv_preload(
        key_cache,
        value_cache,
        block_tables,
        seq_lens,
        max_seq_len=max_seq_len,
        out_k=out_k,
        out_v=out_v,
        require_env=require_env,
    )
    if result is not None:
        _log_route("success route_type=attention_preload_route op=kv_load_and_reshape_flash batch=%s", int(seq_lens.shape[0]))
    return result


def try_sglang_attention_dense_kv_replacement(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    *,
    causal: bool = False,
    softmax_scale: float | None = None,
    max_seq_len: int | None = None,
    mate_accepts_dense_kv: bool = False,
    parity_validated: bool = False,
    require_env: bool = True,
) -> torch.Tensor | None:
    replacement = resolve_sglang_attention_preload_replacement(
        mate_accepts_dense_kv=mate_accepts_dense_kv,
        parity_validated=parity_validated,
        require_env=require_env,
    )
    if not replacement.enabled:
        _log_route("skip route_type=attention_dense_kv_replacement reason=%s", replacement.reason)
        return None
    dense = try_sglang_attention_kv_preload_route(
        key_cache,
        value_cache,
        block_tables,
        seq_lens,
        max_seq_len=max_seq_len,
        require_env=require_env,
    )
    if dense is None:
        return None
    dense_key, dense_value = dense
    query_was_decode = query.dim() == 3
    attention_query = query.unsqueeze(1) if query_was_decode else query
    try:
        import musa_aiter

        result = musa_aiter.flash_attn_func(
            attention_query,
            dense_key,
            dense_value,
            causal=causal,
            softmax_scale=softmax_scale,
        )
        if query_was_decode:
            result = result.squeeze(1)
    except Exception as exc:
        _log_route("fallback route_type=attention_dense_kv_replacement stage=dense_attention reason=%s", exc)
        return None
    _log_route("success route_type=attention_dense_kv_replacement batch=%s", int(seq_lens.shape[0]))
    return result


def _check_sglang_triton_indexed_preload(
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    kv_indptr: torch.Tensor,
    kv_indices: torch.Tensor,
    *,
    op: str,
    require_env: bool,
) -> SGLangAdapterDecision:
    env = _env_decision(op, require_env)
    if env is not None:
        return env
    tensors = (key_cache, value_cache, kv_indptr, kv_indices)
    if not all(isinstance(tensor, torch.Tensor) for tensor in tensors):
        return SGLangAdapterDecision(False, "all inputs must be torch.Tensor", op)
    if not all(_is_musa_tensor(tensor) for tensor in tensors):
        return SGLangAdapterDecision(False, "all inputs must be MUSA tensors", op)
    if key_cache.dim() != 3 or value_cache.dim() != 3:
        return SGLangAdapterDecision(False, "key/value cache must be [tokens, kv_heads, head_dim]", op)
    if key_cache.shape != value_cache.shape:
        return SGLangAdapterDecision(False, "key/value cache shapes must match", op)
    if key_cache.dtype != value_cache.dtype:
        return SGLangAdapterDecision(False, "key/value cache dtypes must match", op)
    if key_cache.dtype not in (torch.float32, torch.float16, torch.bfloat16):
        return SGLangAdapterDecision(False, "cache dtype must be float32, float16, or bfloat16", op)
    if kv_indptr.dim() != 1 or kv_indptr.numel() < 2:
        return SGLangAdapterDecision(False, "kv_indptr must be [batch + 1]", op)
    if kv_indices.dim() != 1:
        return SGLangAdapterDecision(False, "kv_indices must be 1-D", op)
    if not _is_integer_tensor(kv_indptr) or not _is_integer_tensor(kv_indices):
        return SGLangAdapterDecision(False, "kv_indptr and kv_indices must use integer dtype", op)
    return SGLangAdapterDecision(True, "supported", op)


def _triton_indexed_dense_load_pair(
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    kv_indptr: torch.Tensor,
    kv_indices: torch.Tensor,
    *,
    max_seq_len: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    dense_len = _resolve_triton_indexed_dense_len(kv_indptr, max_seq_len=max_seq_len)
    if dense_len is None:
        return None
    if not _is_triton_dense_replacement_within_perf_gate(dense_len):
        _profile_route(
            "skip route_type=triton_attention_dense_kv_replacement stage=perf_gate reason=max_seq_len_exceeds_limit max_seq_len=%s limit=%s",
            dense_len,
            sglang_attention_preload_max_seq_len(),
        )
        return None
    return kv_triton_indexed_to_dense(key_cache, value_cache, kv_indptr, kv_indices, max_seq_len=dense_len)


def _is_triton_dense_replacement_within_perf_gate(max_seq_len: int | None) -> bool:
    limit = sglang_attention_preload_max_seq_len()
    return limit is None or max_seq_len is None or int(max_seq_len) <= limit


def _is_triton_paged_sparse_replacement_within_perf_gate(max_seq_len: int | None) -> bool:
    limit = sglang_paged_sparse_attention_max_seq_len()
    return limit is None or max_seq_len is None or int(max_seq_len) <= limit


def _resolve_triton_indexed_dense_len(kv_indptr: torch.Tensor, *, max_seq_len: int | None = None) -> int | None:
    if max_seq_len is not None:
        return int(max_seq_len)
    if _is_musa_graph_capture_active():
        return None
    lens = kv_indptr[1:] - kv_indptr[:-1]
    return int(lens.max().item()) if int(lens.numel()) else 0


def _triton_indexed_dense_load(
    cache: torch.Tensor,
    kv_indptr: torch.Tensor,
    kv_indices: torch.Tensor,
    *,
    max_seq_len: int | None = None,
) -> torch.Tensor:
    batch = int(kv_indptr.numel() - 1)
    if max_seq_len is None:
        lens = kv_indptr[1:] - kv_indptr[:-1]
        dense_len = int(lens.max().item()) if int(lens.numel()) else 0
    else:
        dense_len = int(max_seq_len)
    out = cache.new_zeros((batch, dense_len, int(cache.shape[1]), int(cache.shape[2])))
    for batch_idx in range(batch):
        start = int(kv_indptr[batch_idx].item())
        end = int(kv_indptr[batch_idx + 1].item())
        slots = kv_indices[start:end].to(dtype=torch.long)
        copy_len = min(int(slots.numel()), dense_len)
        if copy_len <= 0:
            continue
        out[batch_idx, :copy_len].copy_(cache.index_select(0, slots[:copy_len]))
    return out


def try_sglang_triton_paged_sparse_attention_replacement(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    kv_indptr: torch.Tensor,
    kv_indices: torch.Tensor,
    *,
    causal: bool = False,
    softmax_scale: float | None = None,
    max_seq_len: int | None = None,
    mate_accepts_dense_kv: bool = False,
    parity_validated: bool = False,
    require_env: bool = True,
) -> torch.Tensor | None:
    if causal:
        _log_route("skip route_type=triton_paged_sparse_attention_replacement reason=causal_not_supported")
        return None
    replacement = resolve_sglang_attention_preload_replacement(
        mate_accepts_dense_kv=mate_accepts_dense_kv,
        parity_validated=parity_validated,
        require_env=require_env,
    )
    if not replacement.enabled:
        _log_route("skip route_type=triton_paged_sparse_attention_replacement reason=%s", replacement.reason)
        return None
    if require_env and not is_sglang_paged_sparse_attention_enabled():
        _log_route("skip route_type=triton_paged_sparse_attention_replacement reason=MUSA_AITER_SGLANG_PAGED_SPARSE_ATTENTION is not enabled")
        return None
    decision = _check_sglang_triton_indexed_preload(
        key_cache,
        value_cache,
        kv_indptr,
        kv_indices,
        op="sglang_triton_paged_sparse_attention_replacement",
        require_env=require_env,
    )
    if not decision.enabled:
        _log_route("fallback route_type=triton_paged_sparse_attention_replacement stage=contract reason=%s", decision.reason)
        return None
    query_was_decode = query.dim() == 3
    if not query_was_decode:
        _log_route("skip route_type=triton_paged_sparse_attention_replacement reason=query_must_be_decode_shape")
        return None
    if not _is_triton_paged_sparse_replacement_within_perf_gate(max_seq_len):
        _profile_route(
            "skip route_type=triton_paged_sparse_attention_replacement stage=perf_gate reason=max_seq_len_exceeds_limit max_seq_len=%s limit=%s",
            max_seq_len,
            sglang_paged_sparse_attention_max_seq_len(),
        )
        return None
    batch, num_q_heads, head_dim = map(int, query.shape)
    scratch_estimate = _estimate_tiled_paged_sparse_scratch_bytes(batch=batch, num_q_heads=num_q_heads, head_dim=head_dim, max_seq_len=max_seq_len)
    scratch_limit = sglang_paged_sparse_attention_scratch_max_bytes()
    if not _is_triton_paged_sparse_replacement_within_scratch_gate(batch=batch, num_q_heads=num_q_heads, head_dim=head_dim, max_seq_len=max_seq_len):
        _profile_route(
            "skip route_type=triton_paged_sparse_attention_replacement stage=scratch_gate reason=scratch_exceeds_limit max_seq_len=%s estimate_bytes=%s limit_bytes=%s compact_scratch=%s",
            max_seq_len,
            scratch_estimate,
            scratch_limit,
            os.environ.get("MUSA_AITER_PAGED_SPARSE_ATTENTION_COMPACT_SCRATCH", "1"),
        )
        return None
    _profile_route(
        "route_type=triton_paged_sparse_attention_replacement stage=native_input max_seq_len=%s tiled_env=%s tile_size=%s compact_scratch=%s scratch_estimate_bytes=%s scratch_limit_bytes=%s",
        max_seq_len,
        os.environ.get("MUSA_AITER_PAGED_SPARSE_ATTENTION_TILED", "0"),
        os.environ.get("MUSA_AITER_PAGED_SPARSE_ATTENTION_TILE_SIZE", "512"),
        os.environ.get("MUSA_AITER_PAGED_SPARSE_ATTENTION_COMPACT_SCRATCH", "1"),
        scratch_estimate,
        scratch_limit,
    )
    try:
        result = paged_sparse_decode_attention(
            query,
            key_cache,
            value_cache,
            kv_indptr,
            kv_indices,
            softmax_scale=softmax_scale,
            max_seq_len=max_seq_len,
        )
    except Exception as exc:
        _profile_route("fallback route_type=triton_paged_sparse_attention_replacement stage=native reason=%s", exc)
        return None
    _log_route("success route_type=triton_paged_sparse_attention_replacement batch=%s", int(kv_indptr.numel() - 1))
    return result


def try_sglang_triton_attention_dense_kv_replacement(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    kv_indptr: torch.Tensor,
    kv_indices: torch.Tensor,
    *,
    causal: bool = False,
    softmax_scale: float | None = None,
    max_seq_len: int | None = None,
    mate_accepts_dense_kv: bool = False,
    parity_validated: bool = False,
    require_env: bool = True,
) -> torch.Tensor | None:
    replacement = resolve_sglang_attention_preload_replacement(
        mate_accepts_dense_kv=mate_accepts_dense_kv,
        parity_validated=parity_validated,
        require_env=require_env,
    )
    if not replacement.enabled:
        _log_route("skip route_type=triton_attention_dense_kv_replacement reason=%s", replacement.reason)
        return None
    decision = _check_sglang_triton_indexed_preload(
        key_cache,
        value_cache,
        kv_indptr,
        kv_indices,
        op="sglang_triton_attention_dense_kv_replacement",
        require_env=require_env,
    )
    if not decision.enabled:
        _log_route("fallback route_type=triton_attention_dense_kv_replacement stage=contract reason=%s", decision.reason)
        return None
    capture_active = _is_musa_graph_capture_active()
    if capture_active and max_seq_len is None:
        _profile_route("skip route_type=triton_attention_dense_kv_replacement stage=graph_capture reason=max_seq_len_required_for_capture_safe_preload")
        return None
    query_was_decode = query.dim() == 3
    attention_query = query.unsqueeze(1) if query_was_decode else query
    try:
        dense_pair = _triton_indexed_dense_load_pair(key_cache, value_cache, kv_indptr, kv_indices, max_seq_len=max_seq_len)
        if dense_pair is None:
            return None
        dense_key, dense_value = dense_pair
        q_heads = int(attention_query.shape[2])
        kv_heads = int(dense_key.shape[2])
        if query_was_decode and not causal:
            result = dense_decode_attention(
                attention_query.squeeze(1),
                dense_key,
                dense_value,
                softmax_scale=softmax_scale,
            )
            _log_route("success route_type=triton_attention_dense_kv_replacement stage=native_dense_decode_attention batch=%s", int(kv_indptr.numel() - 1))
            return result
        import musa_aiter

        if capture_active:
            _profile_route("skip route_type=triton_attention_dense_kv_replacement stage=graph_capture reason=fallback_dense_attention_not_capture_safe")
            return None
        if q_heads != kv_heads and q_heads % kv_heads == 0:
            repeat = q_heads // kv_heads
            dense_key = dense_key.repeat_interleave(repeat, dim=2)
            dense_value = dense_value.repeat_interleave(repeat, dim=2)
        result = musa_aiter.flash_attn_func(
            attention_query,
            dense_key,
            dense_value,
            causal=causal,
            softmax_scale=softmax_scale,
        )
        if query_was_decode:
            result = result.squeeze(1)
    except Exception as exc:
        _profile_route("fallback route_type=triton_attention_dense_kv_replacement stage=dense_attention reason=%s", exc)
        return None
    _log_route("success route_type=triton_attention_dense_kv_replacement batch=%s", int(kv_indptr.numel() - 1))
    return result


def try_sglang_attention_kv_preload(
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    *,
    max_seq_len: int | None = None,
    out_k: torch.Tensor | None = None,
    out_v: torch.Tensor | None = None,
    require_env: bool = True,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    result = try_kv_load_and_reshape_flash(
        key_cache,
        value_cache,
        block_tables,
        seq_lens,
        max_seq_len=max_seq_len,
        out_k=out_k,
        out_v=out_v,
        require_env=require_env,
    )
    if result is not None:
        _log_route("success route_type=attention_preload op=kv_load_and_reshape_flash batch=%s", int(seq_lens.shape[0]))
    return result


def try_sglang_attention_fp8_kv_preload(
    key_cache_fp8: torch.Tensor,
    value_cache_fp8: torch.Tensor,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    k_scale: torch.Tensor | float,
    v_scale: torch.Tensor | float,
    *,
    max_seq_len: int | None = None,
    out_dtype: torch.dtype = torch.bfloat16,
    fp8_format: Literal["e4m3", "e5m2"] = "e4m3",
    out_k: torch.Tensor | None = None,
    out_v: torch.Tensor | None = None,
    require_env: bool = True,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    decision = check_sglang_attention_fp8_kv_preload_contract(
        key_cache_fp8,
        value_cache_fp8,
        block_tables,
        seq_lens,
        k_scale,
        v_scale,
        require_env=require_env,
    )
    if not decision.enabled:
        return None
    result = try_kv_load_and_dequantize_fp8_flash(
        key_cache_fp8,
        value_cache_fp8,
        block_tables,
        seq_lens,
        k_scale,
        v_scale,
        max_seq_len=max_seq_len,
        out_dtype=out_dtype,
        fp8_format=fp8_format,
        out_k=out_k,
        out_v=out_v,
        require_env=False,
    )
    if result is not None:
        _log_route("success route_type=attention_fp8_preload op=kv_load_and_dequantize_fp8_flash batch=%s fp8_format=%s", int(seq_lens.shape[0]), fp8_format)
    return result


def _valid_attention_fp8_scale_shape(scale: torch.Tensor | float, *, cache: torch.Tensor) -> bool:
    if not isinstance(scale, torch.Tensor):
        return True
    if scale.numel() == 1:
        return True
    if cache.dim() == 4:
        num_heads = int(cache.shape[1])
        return scale.dim() == 1 and int(scale.shape[0]) == num_heads
    if cache.dim() == 5:
        num_layers = int(cache.shape[0])
        num_heads = int(cache.shape[2])
        if scale.dim() == 1 and int(scale.shape[0]) == num_heads:
            return True
        return scale.dim() == 2 and tuple(scale.shape) == (num_layers, num_heads)
    return False


def check_sglang_attention_fp8_kv_preload_contract(
    key_cache_fp8: torch.Tensor,
    value_cache_fp8: torch.Tensor,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    k_scale: torch.Tensor | float,
    v_scale: torch.Tensor | float,
    *,
    require_env: bool = True,
) -> SGLangAdapterDecision:
    op = "sglang_attention_fp8_kv_preload"
    env = _env_decision(op, require_env)
    if env is not None:
        return env
    tensors = (key_cache_fp8, value_cache_fp8, block_tables, seq_lens)
    if not all(isinstance(tensor, torch.Tensor) for tensor in tensors):
        return SGLangAdapterDecision(False, "all inputs must be torch.Tensor", op)
    if key_cache_fp8.shape != value_cache_fp8.shape:
        return SGLangAdapterDecision(False, "key/value cache shapes must match", op)
    if key_cache_fp8.dim() not in (4, 5):
        return SGLangAdapterDecision(False, "FP8 cache must be single-layer or multi-layer paged layout", op)
    if key_cache_fp8.dtype is not torch.uint8 and str(key_cache_fp8.dtype) not in ("torch.float8_e4m3fn", "torch.float8_e5m2"):
        return SGLangAdapterDecision(False, "FP8 cache must be torch.uint8 or Torch FP8 dtype", op)
    for scale in (k_scale, v_scale):
        if isinstance(scale, torch.Tensor):
            if scale.dtype is not torch.float32:
                return SGLangAdapterDecision(False, "scale tensors must be float32", op)
            if not scale.is_contiguous():
                return SGLangAdapterDecision(False, "scale tensors must be contiguous", op)
            if not _valid_attention_fp8_scale_shape(scale, cache=key_cache_fp8):
                return SGLangAdapterDecision(False, "scale layout must be scalar, per-head, or per-layer-per-head", op)
    base = check_kv_load_and_dequantize_fp8_flash(
        key_cache_fp8,
        value_cache_fp8,
        block_tables,
        seq_lens,
        k_scale,
        v_scale,
        require_env=False,
    )
    if not base.enabled:
        return SGLangAdapterDecision(False, base.reason, op)
    return SGLangAdapterDecision(True, "supported", op)


def check_kv_load_and_dequantize_fp8_flash(
    key_cache_fp8: torch.Tensor,
    value_cache_fp8: torch.Tensor,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    k_scale: torch.Tensor | float,
    v_scale: torch.Tensor | float,
    *,
    require_env: bool = True,
) -> SGLangAdapterDecision:
    op = "kv_load_and_dequantize_fp8_flash"
    base = check_kv_load_and_reshape_flash(
        key_cache_fp8,
        value_cache_fp8,
        block_tables,
        seq_lens,
        require_env=require_env,
    )
    if not base.enabled:
        return SGLangAdapterDecision(False, base.reason, op)
    if key_cache_fp8.dtype is not torch.uint8 and str(key_cache_fp8.dtype) not in ("torch.float8_e4m3fn", "torch.float8_e5m2"):
        return SGLangAdapterDecision(False, "FP8 cache must be torch.uint8 or Torch FP8 dtype", op)
    for scale in (k_scale, v_scale):
        if isinstance(scale, torch.Tensor):
            if not _is_musa_tensor(scale):
                return SGLangAdapterDecision(False, "scale tensors must be on MUSA", op)
            if scale.dtype is not torch.float32:
                return SGLangAdapterDecision(False, "scale tensors must be float32", op)
            if not scale.is_contiguous():
                return SGLangAdapterDecision(False, "scale tensors must be contiguous", op)
    return SGLangAdapterDecision(True, "supported", op)


def try_kv_load_and_dequantize_fp8_flash(
    key_cache_fp8: torch.Tensor,
    value_cache_fp8: torch.Tensor,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    k_scale: torch.Tensor | float,
    v_scale: torch.Tensor | float,
    *,
    max_seq_len: int | None = None,
    out_dtype: torch.dtype = torch.bfloat16,
    fp8_format: Literal["e4m3", "e5m2"] = "e4m3",
    out_k: torch.Tensor | None = None,
    out_v: torch.Tensor | None = None,
    require_env: bool = True,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    decision = check_kv_load_and_dequantize_fp8_flash(
        key_cache_fp8,
        value_cache_fp8,
        block_tables,
        seq_lens,
        k_scale,
        v_scale,
        require_env=require_env,
    )
    if not decision.enabled:
        return None
    return kv_load_and_dequantize_fp8_flash(
        key_cache_fp8,
        value_cache_fp8,
        block_tables,
        seq_lens,
        k_scale,
        v_scale,
        max_seq_len=max_seq_len,
        out_dtype=out_dtype,
        fp8_format=fp8_format,
        out_k=out_k,
        out_v=out_v,
    )


def check_kv_copy_paged_blocks(
    src_key_cache: torch.Tensor,
    src_value_cache: torch.Tensor,
    dst_key_cache: torch.Tensor,
    dst_value_cache: torch.Tensor,
    block_mapping: torch.Tensor,
    *,
    fp8: bool = False,
    require_env: bool = True,
) -> SGLangAdapterDecision:
    op = "kv_copy_paged_blocks_fp8" if fp8 else "kv_copy_paged_blocks"
    env = _env_decision(op, require_env)
    if env is not None:
        return env
    tensors = (src_key_cache, src_value_cache, dst_key_cache, dst_value_cache, block_mapping)
    if not all(isinstance(tensor, torch.Tensor) for tensor in tensors):
        return SGLangAdapterDecision(False, "all inputs must be torch.Tensor", op)
    if not all(_is_musa_tensor(tensor) for tensor in tensors):
        return SGLangAdapterDecision(False, "all inputs must be MUSA tensors", op)
    if not _is_integer_tensor(block_mapping):
        return SGLangAdapterDecision(False, "block_mapping must use integer dtype", op)
    if block_mapping.dim() != 2 or block_mapping.shape[1] != 2:
        return SGLangAdapterDecision(False, "block_mapping must be [num_pairs, 2]", op)
    cache_tensors = (src_key_cache, src_value_cache, dst_key_cache, dst_value_cache)
    if not all(tensor.is_contiguous() for tensor in cache_tensors):
        return SGLangAdapterDecision(False, "cache tensors must be contiguous", op)
    if src_key_cache.shape != src_value_cache.shape or dst_key_cache.shape != dst_value_cache.shape:
        return SGLangAdapterDecision(False, "K/V cache shapes must match", op)
    if src_key_cache.dim() not in (4, 5) or dst_key_cache.dim() != src_key_cache.dim():
        return SGLangAdapterDecision(False, "cache tensors must be single-layer or multi-layer paged layout", op)
    src_inner_shape = src_key_cache.shape[1:] if src_key_cache.dim() == 4 else src_key_cache.shape[2:]
    dst_inner_shape = dst_key_cache.shape[1:] if dst_key_cache.dim() == 4 else dst_key_cache.shape[2:]
    if src_inner_shape != dst_inner_shape:
        return SGLangAdapterDecision(False, "source/destination cache inner shapes must match", op)
    if fp8:
        if src_key_cache.dtype is not torch.uint8 and str(src_key_cache.dtype) not in ("torch.float8_e4m3fn", "torch.float8_e5m2"):
            return SGLangAdapterDecision(False, "FP8 cache must be torch.uint8 or Torch FP8 dtype", op)
    elif src_key_cache.dtype not in (torch.float32, torch.float16, torch.bfloat16):
        return SGLangAdapterDecision(False, "cache dtype must be float32, float16, or bfloat16", op)
    if not (src_key_cache.dtype == src_value_cache.dtype == dst_key_cache.dtype == dst_value_cache.dtype):
        return SGLangAdapterDecision(False, "all cache dtypes must match", op)
    return SGLangAdapterDecision(True, "supported", op)


def try_kv_copy_paged_blocks(
    src_key_cache: torch.Tensor,
    src_value_cache: torch.Tensor,
    dst_key_cache: torch.Tensor,
    dst_value_cache: torch.Tensor,
    block_mapping: torch.Tensor,
    *,
    require_env: bool = True,
) -> int | None:
    decision = check_kv_copy_paged_blocks(
        src_key_cache,
        src_value_cache,
        dst_key_cache,
        dst_value_cache,
        block_mapping,
        require_env=require_env,
    )
    if not decision.enabled:
        return None
    return kv_copy_paged_blocks(src_key_cache, src_value_cache, dst_key_cache, dst_value_cache, block_mapping)


def try_kv_copy_paged_blocks_fp8(
    src_key_cache_fp8: torch.Tensor,
    src_value_cache_fp8: torch.Tensor,
    dst_key_cache_fp8: torch.Tensor,
    dst_value_cache_fp8: torch.Tensor,
    block_mapping: torch.Tensor,
    *,
    require_env: bool = True,
) -> int | None:
    decision = check_kv_copy_paged_blocks(
        src_key_cache_fp8,
        src_value_cache_fp8,
        dst_key_cache_fp8,
        dst_value_cache_fp8,
        block_mapping,
        fp8=True,
        require_env=require_env,
    )
    if not decision.enabled:
        return None
    return kv_copy_paged_blocks_fp8(src_key_cache_fp8, src_value_cache_fp8, dst_key_cache_fp8, dst_value_cache_fp8, block_mapping)


__all__ = [
    "HiCacheIOBackend",
    "SGLangAdapterDecision",
    "SGLangHiCacheBackendDecision",
    "check_kv_copy_paged_blocks",
    "check_kv_load_and_dequantize_fp8_flash",
    "check_kv_load_and_reshape_flash",
    "check_sglang_attention_fp8_kv_preload_contract",
    "clear_sglang_cpu_host_staging_cache",
    "is_sglang_cpu_host_staging_cache_enabled",
    "is_sglang_kv_route_logging_enabled",
    "is_sglang_kv_route_profile_enabled",
    "is_sglang_attention_preload_probe_enabled",
    "is_sglang_attention_preload_route_enabled",
    "is_sglang_attention_preload_replace_enabled",
    "is_sglang_musa_aiter_adapter_enabled",
    "is_sglang_native_cpu_host_kv_transfer_enabled",
    "probe_sglang_attention_preload",
    "resolve_hicache_io_backend_for_musa",
    "resolve_sglang_attention_preload_replacement",
    "sglang_integration_import_status",
    "try_kv_copy_paged_blocks",
    "try_kv_copy_paged_blocks_fp8",
    "try_kv_load_and_dequantize_fp8_flash",
    "try_kv_load_and_reshape_flash",
    "try_sglang_attention_fp8_kv_preload",
    "try_sglang_attention_dense_kv_replacement",
    "try_sglang_attention_kv_preload",
    "try_sglang_attention_kv_preload_route",
    "try_sglang_hicache_direct_transfer",
    "try_sglang_hicache_layer_first_cpu_host_transfer",
    "try_sglang_hicache_layer_first_cpu_host_transfer_multi_layer",
    "try_sglang_hicache_layer_first_transfer",
]
