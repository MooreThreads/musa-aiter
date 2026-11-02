from __future__ import annotations

import argparse
import shutil
from datetime import datetime
from pathlib import Path

IMPORT_NEEDLE = "import logging\n"
IMPORT_REPLACEMENT = "import logging\nimport os\n"

PATCH_MARKER = "MUSA_AITER_SGLANG_HICACHE_ROUTE_PATCH_V4"
ATTENTION_PROBE_PATCH_MARKER = "MUSA_AITER_SGLANG_ATTENTION_PRELOAD_PROBE_PATCH_V1"
ATTENTION_BACKEND_PARITY_PATCH_MARKER = "MUSA_AITER_SGLANG_ATTENTION_BACKEND_PARITY_PATCH_V1"
TRITON_BACKEND_PARITY_PATCH_MARKER = "MUSA_AITER_SGLANG_TRITON_DECODE_PARITY_PATCH_V10"
OLD_PROBE_MARKER = "MUSA_AITER_SGLANG_HICACHE_PATCH_V1"
OLD_ROUTE_MARKERS = ("MUSA_AITER_SGLANG_HICACHE_ROUTE_PATCH_V2", "MUSA_AITER_SGLANG_HICACHE_ROUTE_PATCH_V3")
OLD_TRITON_BACKEND_PARITY_MARKERS = (
    "MUSA_AITER_SGLANG_TRITON_DECODE_PARITY_PATCH_V1",
    "MUSA_AITER_SGLANG_TRITON_DECODE_PARITY_PATCH_V2",
    "MUSA_AITER_SGLANG_TRITON_DECODE_PARITY_PATCH_V3",
    "MUSA_AITER_SGLANG_TRITON_DECODE_PARITY_PATCH_V4",
    "MUSA_AITER_SGLANG_TRITON_DECODE_PARITY_PATCH_V5",
    "MUSA_AITER_SGLANG_TRITON_DECODE_PARITY_PATCH_V6",
    "MUSA_AITER_SGLANG_TRITON_DECODE_PARITY_PATCH_V7",
    "MUSA_AITER_SGLANG_TRITON_DECODE_PARITY_PATCH_V8",
    "MUSA_AITER_SGLANG_TRITON_DECODE_PARITY_PATCH_V9",
)
HELPER_NEEDLE = "logger = logging.getLogger(__name__)\n\n\n"
HELPER_BLOCK = f'''logger = logging.getLogger(__name__)


{PATCH_MARKER} = True


def _musa_aiter_hicache_kernel_route(
    host_pool,
    device_pool,
    host_indices,
    device_indices,
    *,
    layer_id,
    direction: str,
) -> bool:
    if os.environ.get("MUSA_AITER_SGLANG_ADAPTER", "0") != "1":
        return False
    if getattr(host_pool, "layout", None) != "layer_first":
        return False
    if layer_id is None:
        return False
    try:
        from sglang.srt.utils import is_musa
        from musa_aiter.integration import (
            is_sglang_kv_route_logging_enabled,
            resolve_hicache_io_backend_for_musa,
            sglang_integration_import_status,
            try_sglang_hicache_layer_first_cpu_host_transfer,
            try_sglang_hicache_layer_first_cpu_host_transfer_multi_layer,
            try_sglang_hicache_layer_first_transfer,
        )

        status = sglang_integration_import_status()
        decision = resolve_hicache_io_backend_for_musa(
            "kernel",
            is_musa=is_musa(),
            adapter_available=status.get("musa_aiter", False),
            layout_supported=True,
        )
        if not decision.use_musa_aiter_adapter:
            return False
        layer_offset = layer_id - getattr(host_pool, "start_layer", 0)
        host_key_cache = host_pool.k_buffer[layer_offset]
        host_value_cache = host_pool.v_buffer[layer_offset]
        device_key_cache = device_pool.k_buffer[layer_offset]
        device_value_cache = device_pool.v_buffer[layer_offset]
        page_size = getattr(host_pool, "page_size", 1)
        route_direction = "host_to_device" if direction == "load_to_device" else "device_to_host"
        host_device_type = str(getattr(host_key_cache, "device", "")).split(":")[0]
        device_device_type = str(getattr(device_key_cache, "device", "")).split(":")[0]
        if host_device_type == "cpu" and device_device_type == "musa":
            routed = try_sglang_hicache_layer_first_cpu_host_transfer(
                host_key_cache,
                host_value_cache,
                device_key_cache,
                device_value_cache,
                host_indices,
                device_indices,
                page_size=page_size,
                direction=route_direction,
            )
        else:
            routed = try_sglang_hicache_layer_first_transfer(
                host_key_cache,
                host_value_cache,
                device_key_cache,
                device_value_cache,
                host_indices,
                device_indices,
                page_size=page_size,
                direction=route_direction,
            )
            if not routed:
                routed = try_sglang_hicache_layer_first_cpu_host_transfer(
                    host_key_cache,
                    host_value_cache,
                    device_key_cache,
                    device_value_cache,
                    host_indices,
                    device_indices,
                    page_size=page_size,
                    direction=route_direction,
                )
        if routed and is_sglang_kv_route_logging_enabled():
            logger.info(
                "musa_aiter routed HiCache KV transfer: direction=%s layer_id=%s layout=%s tokens=%s",
                direction,
                layer_id,
                getattr(host_pool, "layout", None),
                int(host_indices.numel()),
            )
        return routed
    except Exception as exc:
        if not getattr(_musa_aiter_hicache_kernel_route, "_warned", False):
            logger.warning("musa_aiter HiCache route disabled: %s", exc)
            setattr(_musa_aiter_hicache_kernel_route, "_warned", True)
        return False


def _musa_aiter_hicache_kernel_route_all_layers(
    host_pool,
    device_pool,
    host_indices,
    device_indices,
    *,
    direction: str,
) -> bool:
    if os.environ.get("MUSA_AITER_SGLANG_ADAPTER", "0") != "1":
        return False
    if getattr(host_pool, "layout", None) != "layer_first":
        return False
    try:
        from sglang.srt.utils import is_musa
        from musa_aiter.integration import (
            is_sglang_kv_route_logging_enabled,
            resolve_hicache_io_backend_for_musa,
            sglang_integration_import_status,
            try_sglang_hicache_layer_first_cpu_host_transfer_multi_layer,
        )

        status = sglang_integration_import_status()
        decision = resolve_hicache_io_backend_for_musa(
            "kernel",
            is_musa=is_musa(),
            adapter_available=status.get("musa_aiter", False),
            layout_supported=True,
        )
        if not decision.use_musa_aiter_adapter:
            return False
        host_key_cache = host_pool.k_buffer
        host_value_cache = host_pool.v_buffer
        device_key_cache = device_pool.k_buffer[getattr(host_pool, "start_layer", 0): getattr(host_pool, "start_layer", 0) + getattr(host_pool, "layer_num", 0)]
        device_value_cache = device_pool.v_buffer[getattr(host_pool, "start_layer", 0): getattr(host_pool, "start_layer", 0) + getattr(host_pool, "layer_num", 0)]
        host_device_type = str(getattr(host_key_cache, "device", "")).split(":")[0]
        device_device_type = str(getattr(device_key_cache, "device", "")).split(":")[0]
        if host_device_type != "cpu" or device_device_type != "musa":
            return False
        routed = try_sglang_hicache_layer_first_cpu_host_transfer_multi_layer(
            host_key_cache,
            host_value_cache,
            device_key_cache,
            device_value_cache,
            host_indices,
            device_indices,
            page_size=getattr(host_pool, "page_size", 1),
            direction="host_to_device" if direction == "load_to_device" else "device_to_host",
        )
        if routed and is_sglang_kv_route_logging_enabled():
            logger.info(
                "musa_aiter routed all-layer HiCache KV transfer: direction=%s layout=%s layers=%s tokens=%s",
                direction,
                getattr(host_pool, "layout", None),
                getattr(host_pool, "layer_num", None),
                int(host_indices.numel()),
            )
        return routed
    except Exception as exc:
        if not getattr(_musa_aiter_hicache_kernel_route_all_layers, "_warned", False):
            logger.warning("musa_aiter all-layer HiCache route disabled: %s", exc)
            setattr(_musa_aiter_hicache_kernel_route_all_layers, "_warned", True)
        return False


'''


def _old_backup_mha_replacement_v2() -> str:
    return '''        if io_backend == "kernel":
            if self.layout == "layer_first":
                routed_layers = 0
                for _musa_aiter_layer_id in range(self.start_layer, self.start_layer + self.layer_num):
                    if _musa_aiter_hicache_kernel_route(
                        self,
                        device_pool,
                        host_indices,
                        device_indices,
                        layer_id=_musa_aiter_layer_id,
                        direction="backup_from_device",
                    ):
                        routed_layers += 1
                    else:
                        break
                if routed_layers == self.layer_num:
                    return
                if self.can_use_jit:
                    jit_transfer_hicache_all_layer(
'''


def _old_backup_mla_replacement_v2() -> str:
    return '''        if io_backend == "kernel":
            if self.layout == "layer_first":
                routed_layers = 0
                for _musa_aiter_layer_id in range(self.start_layer, self.start_layer + self.layer_num):
                    if _musa_aiter_hicache_kernel_route(
                        self,
                        device_pool,
                        host_indices,
                        device_indices,
                        layer_id=_musa_aiter_layer_id,
                        direction="backup_from_device",
                    ):
                        routed_layers += 1
                    else:
                        break
                if routed_layers == self.layer_num:
                    return
                transfer_kv_all_layer_mla(
'''


LOAD_MHA_NEEDLE = '''        if io_backend == "kernel":
            if self.layout == "layer_first":
                if self.can_use_jit:
                    jit_transfer_hicache_one_layer(
'''
LOAD_MHA_REPLACEMENT = '''        if io_backend == "kernel":
            if _musa_aiter_hicache_kernel_route(
                self,
                device_pool,
                host_indices,
                device_indices,
                layer_id=layer_id,
                direction="load_to_device",
            ):
                return
            if self.layout == "layer_first":
                if self.can_use_jit:
                    jit_transfer_hicache_one_layer(
'''

LOAD_MLA_NEEDLE = '''        if io_backend == "kernel":
            if self.layout == "layer_first":
                transfer_kv_per_layer_mla(
'''
LOAD_MLA_REPLACEMENT = '''        if io_backend == "kernel":
            if _musa_aiter_hicache_kernel_route(
                self,
                device_pool,
                host_indices,
                device_indices,
                layer_id=layer_id,
                direction="load_to_device",
            ):
                return
            if self.layout == "layer_first":
                transfer_kv_per_layer_mla(
'''

BACKUP_MHA_NEEDLE = '''        if io_backend == "kernel":
            if self.layout == "layer_first":
                if self.can_use_jit:
                    jit_transfer_hicache_all_layer(
'''
BACKUP_MHA_REPLACEMENT = '''        if io_backend == "kernel":
            if self.layout == "layer_first":
                if _musa_aiter_hicache_kernel_route_all_layers(
                    self,
                    device_pool,
                    host_indices,
                    device_indices,
                    direction="backup_from_device",
                ):
                    return
                routed_layers = 0
                for _musa_aiter_layer_id in range(self.start_layer, self.start_layer + self.layer_num):
                    if _musa_aiter_hicache_kernel_route(
                        self,
                        device_pool,
                        host_indices,
                        device_indices,
                        layer_id=_musa_aiter_layer_id,
                        direction="backup_from_device",
                    ):
                        routed_layers += 1
                    else:
                        break
                if routed_layers == self.layer_num:
                    return
                if self.can_use_jit:
                    jit_transfer_hicache_all_layer(
'''

BACKUP_MLA_NEEDLE = '''        if io_backend == "kernel":
            if self.layout == "layer_first":
                transfer_kv_all_layer_mla(
'''
BACKUP_MLA_REPLACEMENT = '''        if io_backend == "kernel":
            if self.layout == "layer_first":
                if _musa_aiter_hicache_kernel_route_all_layers(
                    self,
                    device_pool,
                    host_indices,
                    device_indices,
                    direction="backup_from_device",
                ):
                    return
                routed_layers = 0
                for _musa_aiter_layer_id in range(self.start_layer, self.start_layer + self.layer_num):
                    if _musa_aiter_hicache_kernel_route(
                        self,
                        device_pool,
                        host_indices,
                        device_indices,
                        layer_id=_musa_aiter_layer_id,
                        direction="backup_from_device",
                    ):
                        routed_layers += 1
                    else:
                        break
                if routed_layers == self.layer_num:
                    return
                transfer_kv_all_layer_mla(
'''


ATTENTION_IMPORT_NEEDLE = "import threading\n"
ATTENTION_IMPORT_REPLACEMENT = "import logging\nimport os\nimport threading\n"
ATTENTION_HELPER_NEEDLE = "_flash_attention_context = threading.local()\n\n\n"
ATTENTION_HELPER_BLOCK = f'''_flash_attention_context = threading.local()

logger = logging.getLogger(__name__)


{ATTENTION_PROBE_PATCH_MARKER} = True


def _musa_aiter_attention_preload_probe(
    k_cache,
    v_cache,
    page_table,
    cache_seqlens,
) -> None:
    route_enabled = os.environ.get("MUSA_AITER_SGLANG_ATTENTION_PRELOAD", "0") == "1"
    probe_enabled = os.environ.get("MUSA_AITER_SGLANG_ATTENTION_PRELOAD_PROBE", "0") == "1"
    if not route_enabled and not probe_enabled:
        return
    try:
        if page_table is None or cache_seqlens is None:
            return
        logger = globals().get("logger")
        if route_enabled:
            from musa_aiter.integration import (
                resolve_sglang_attention_preload_replacement,
                try_sglang_attention_kv_preload_route,
            )

            routed = try_sglang_attention_kv_preload_route(
                k_cache,
                v_cache,
                page_table,
                cache_seqlens,
                require_env=True,
            )
            replacement = resolve_sglang_attention_preload_replacement(
                mate_accepts_dense_kv=False,
                parity_validated=False,
                require_env=True,
            )
            if logger is not None:
                logger.info(
                    "musa_aiter attention preload route candidate: routed=%s replace_enabled=%s replace_reason=%s k_cache_shape=%s page_table_shape=%s cache_seqlens_shape=%s",
                    routed is not None,
                    replacement.enabled,
                    replacement.reason,
                    tuple(getattr(k_cache, "shape", ())),
                    tuple(getattr(page_table, "shape", ())),
                    tuple(getattr(cache_seqlens, "shape", ())),
                )
            return
        from musa_aiter.integration import probe_sglang_attention_preload

        decision = probe_sglang_attention_preload(
            k_cache,
            v_cache,
            page_table,
            cache_seqlens,
            require_env=True,
        )
        if logger is not None:
            logger.info(
                "musa_aiter attention preload probe: enabled=%s reason=%s k_cache_shape=%s page_table_shape=%s cache_seqlens_shape=%s",
                decision.enabled,
                decision.reason,
                tuple(getattr(k_cache, "shape", ())),
                tuple(getattr(page_table, "shape", ())),
                tuple(getattr(cache_seqlens, "shape", ())),
            )
    except Exception as exc:
        logger = globals().get("logger")
        if logger is not None and not getattr(_musa_aiter_attention_preload_probe, "_warned", False):
            logger.warning("musa_aiter attention preload probe disabled: %s", exc)
            setattr(_musa_aiter_attention_preload_probe, "_warned", True)


'''
ATTENTION_CALL_NEEDLE = '''    # Get context and compute scheduler_metadata if not provided
    if scheduler_metadata is None:
'''
ATTENTION_CALL_REPLACEMENT = '''    _musa_aiter_attention_preload_probe(k_cache, v_cache, page_table, cache_seqlens)

    # Get context and compute scheduler_metadata if not provided
    if scheduler_metadata is None:
'''


def _strip_attention_probe_patch(text: str) -> str:
    if ATTENTION_PROBE_PATCH_MARKER not in text:
        return text
    text = text.replace(ATTENTION_CALL_REPLACEMENT, ATTENTION_CALL_NEEDLE)
    start = text.index(f"{ATTENTION_PROBE_PATCH_MARKER} = True")
    end_marker = "\n\n\n@dataclass"
    end = text.index(end_marker, start)
    text = text[:start] + text[end + 3 :]
    text = text.replace(ATTENTION_IMPORT_REPLACEMENT, ATTENTION_IMPORT_NEEDLE, 1)
    return text


def patch_attention_probe_file(path: Path, *, backup: bool) -> bool:
    text = path.read_text()
    if ATTENTION_PROBE_PATCH_MARKER in text:
        print(f"already patched: {path}")
        return False
    original_text = text
    text = _strip_attention_probe_patch(text)
    if ATTENTION_IMPORT_REPLACEMENT in text:
        pass
    elif ATTENTION_IMPORT_NEEDLE in text:
        text = text.replace(ATTENTION_IMPORT_NEEDLE, ATTENTION_IMPORT_REPLACEMENT, 1)
    else:
        raise RuntimeError("attention import insertion point not found")
    if ATTENTION_HELPER_NEEDLE not in text:
        raise RuntimeError("attention helper insertion point not found")
    if text.count(ATTENTION_CALL_NEEDLE) != 1:
        raise RuntimeError(f"expected 1 attention probe hook point, found {text.count(ATTENTION_CALL_NEEDLE)}")
    if backup:
        stamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
        backup_path = path.with_suffix(path.suffix + f".musa_aiter_backup_{stamp}")
        backup_path.write_text(original_text)
        print(f"backup: {backup_path}")
    text = text.replace(ATTENTION_HELPER_NEEDLE, ATTENTION_HELPER_BLOCK, 1)
    text = text.replace(ATTENTION_CALL_NEEDLE, ATTENTION_CALL_REPLACEMENT, 1)
    path.write_text(text)
    print(f"patched: {path}")
    return True


ATTENTION_BACKEND_HELPER_NEEDLE = "logger = logging.getLogger(__name__)\n\n\n"
ATTENTION_BACKEND_HELPER_BLOCK = f'''logger = logging.getLogger(__name__)


{ATTENTION_BACKEND_PARITY_PATCH_MARKER} = True


def _musa_aiter_attention_backend_parity_probe(
    original_output,
    q,
    k_cache,
    v_cache,
    page_table,
    cache_seqlens,
    layer,
    *,
    causal: bool,
    softmax_scale,
):
    if os.environ.get("MUSA_AITER_SGLANG_ATTENTION_PRELOAD_PARITY", "0") != "1":
        return original_output
    logger = globals().get("logger")
    try:
        if logger is not None:
            logger.info(
                "musa_aiter attention backend parity reached: original_shape=%s q_shape=%s k_cache_shape=%s page_table_shape=%s cache_seqlens_shape=%s",
                tuple(getattr(original_output, "shape", ())),
                tuple(getattr(q, "shape", ())),
                tuple(getattr(k_cache, "shape", ())),
                tuple(getattr(page_table, "shape", ())),
                tuple(getattr(cache_seqlens, "shape", ())),
            )
        if page_table is None or cache_seqlens is None:
            if logger is not None:
                logger.info("musa_aiter attention backend parity unsupported: missing page_table/cache_seqlens")
            return original_output
        from musa_aiter.integration import try_sglang_attention_dense_kv_replacement

        dense_output = try_sglang_attention_dense_kv_replacement(
            q,
            k_cache,
            v_cache,
            page_table,
            cache_seqlens,
            causal=causal,
            softmax_scale=softmax_scale,
            mate_accepts_dense_kv=True,
            parity_validated=True,
            require_env=True,
        )
        if dense_output is None:
            if logger is not None:
                logger.info("musa_aiter attention backend parity unsupported: dense_output=None")
            return original_output
        if logger is not None:
            original_dense = original_output.reshape_as(dense_output)
            diff = (dense_output.float() - original_dense.float()).abs()
            logger.info(
                "musa_aiter attention backend parity: dense_shape=%s original_shape=%s max_abs=%s mean_abs=%s",
                tuple(getattr(dense_output, "shape", ())),
                tuple(getattr(original_output, "shape", ())),
                float(diff.max().item()) if diff.numel() else 0.0,
                float(diff.mean().item()) if diff.numel() else 0.0,
            )
        return original_output
    except Exception as exc:
        logger = globals().get("logger")
        if logger is not None and not getattr(_musa_aiter_attention_backend_parity_probe, "_warned", False):
            logger.warning("musa_aiter attention backend parity disabled: %s", exc)
            setattr(_musa_aiter_attention_backend_parity_probe, "_warned", True)
        return original_output


'''
ATTENTION_BACKEND_RETURN_NEEDLE = '''        return o.view(-1, layer.tp_q_head_num * layer.v_head_dim)
'''
ATTENTION_BACKEND_RETURN_REPLACEMENT = '''        o = _musa_aiter_attention_backend_parity_probe(
            o,
            q_reshaped if "q_reshaped" in locals() else q_rope,
            key_cache if "key_cache" in locals() else k_rope_cache,
            value_cache if "value_cache" in locals() else c_kv_cache,
            page_table if "page_table" in locals() else metadata.page_table,
            cache_seqlens if "cache_seqlens" in locals() else metadata.cache_seqlens_int32,
            layer,
            causal=causal,
            softmax_scale=layer.scaling,
        )
        return o.view(-1, layer.tp_q_head_num * layer.v_head_dim)
'''


def patch_attention_backend_parity_file(path: Path, *, backup: bool) -> bool:
    text = path.read_text()
    if ATTENTION_BACKEND_PARITY_PATCH_MARKER in text:
        print(f"already patched: {path}")
        return False
    original_text = text
    if "import os\n" not in text:
        if "import logging\n" in text:
            text = text.replace("import logging\n", "import logging\nimport os\n", 1)
        elif "from typing import" in text:
            line_end = text.index("\n", text.index("from typing import")) + 1
            text = text[:line_end] + "\nimport logging\nimport os\n" + text[line_end:]
        elif "from __future__ import annotations\n" in text:
            text = text.replace("from __future__ import annotations\n", "from __future__ import annotations\n\nimport logging\nimport os\n", 1)
        else:
            raise RuntimeError("backend import insertion point not found")
    if ATTENTION_BACKEND_HELPER_NEEDLE in text:
        text = text.replace(ATTENTION_BACKEND_HELPER_NEEDLE, ATTENTION_BACKEND_HELPER_BLOCK, 1)
    elif "\n\n@dataclass\n" in text:
        text = text.replace("\n\n@dataclass\n", "\n\n" + ATTENTION_BACKEND_HELPER_BLOCK + "@dataclass\n", 1)
    else:
        raise RuntimeError("backend helper insertion point not found")
    hook_count = text.count(ATTENTION_BACKEND_RETURN_NEEDLE)
    if hook_count < 1:
        raise RuntimeError(f"expected at least 1 backend parity return hook point, found {hook_count}")
    if backup:
        stamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
        backup_path = path.with_suffix(path.suffix + f".musa_aiter_backup_{stamp}")
        backup_path.write_text(original_text)
        print(f"backup: {backup_path}")
    text = text.replace(ATTENTION_BACKEND_RETURN_NEEDLE, ATTENTION_BACKEND_RETURN_REPLACEMENT)
    path.write_text(text)
    print(f"patched: {path}")
    return True


TRITON_HELPER_BLOCK = f'''logger = logging.getLogger(__name__)


{TRITON_BACKEND_PARITY_PATCH_MARKER} = True


def _musa_aiter_triton_decode_parity_probe(
    original_output,
    q,
    key_cache,
    value_cache,
    kv_indptr,
    kv_indices,
    layer,
    max_seq_len=None,
):
    if os.environ.get("MUSA_AITER_SGLANG_ATTENTION_PRELOAD_PARITY", "0") != "1":
        return original_output
    logger = globals().get("logger")
    import time
    debug_t0 = time.perf_counter()
    debug_capture_at_entry = False
    try:
        if hasattr(torch, "musa"):
            capture_status = getattr(torch.musa, "is_current_stream_capturing", None)
            if callable(capture_status):
                debug_capture_at_entry = bool(capture_status())
    except Exception:
        debug_capture_at_entry = False
    try:
        if logger is not None:
            logger.info(
                "musa_aiter triton decode parity reached: original_shape=%s q_shape=%s key_cache_shape=%s kv_indptr_shape=%s kv_indices_shape=%s",
                tuple(getattr(original_output, "shape", ())),
                tuple(getattr(q, "shape", ())),
                tuple(getattr(key_cache, "shape", ())),
                tuple(getattr(kv_indptr, "shape", ())),
                tuple(getattr(kv_indices, "shape", ())),
            )
        if os.environ.get("MUSA_AITER_SGLANG_ATTENTION_PRELOAD_OUTPUT_REPLACE", "0") == "1" and os.environ.get("MUSA_AITER_SGLANG_ALLOW_AFTER_ORIGINAL_OUTPUT_REPLACE", "0") != "1":
            if logger is not None:
                logger.info(
                    "musa_aiter triton decode output replacement disabled: reason=after_original_double_compute_guard allow_env=MUSA_AITER_SGLANG_ALLOW_AFTER_ORIGINAL_OUTPUT_REPLACE"
                )
            return original_output
        from musa_aiter.integration import (
            try_sglang_triton_attention_dense_kv_replacement,
            try_sglang_triton_paged_sparse_attention_replacement,
        )

        effective_max_seq_len = max_seq_len
        if not debug_capture_at_entry:
            try:
                if kv_indptr is not None and int(kv_indptr.numel()) > 1:
                    seq_lens = kv_indptr[1:] - kv_indptr[:-1]
                    exact_max_seq_len = int(seq_lens.max().item()) if int(seq_lens.numel()) > 0 else 1
                    if effective_max_seq_len is None or exact_max_seq_len < int(effective_max_seq_len):
                        effective_max_seq_len = max(1, exact_max_seq_len)
            except Exception:
                pass
        if effective_max_seq_len is None:
            effective_batch = int(q.shape[0]) if len(q.shape) > 0 else 1
            effective_batch = max(1, effective_batch)
            effective_max_seq_len = (int(kv_indices.numel()) + effective_batch - 1) // effective_batch
        if logger is not None:
            effective_batch = int(q.shape[0]) if len(q.shape) > 0 else 1
            num_q_heads = int(q.shape[1]) if len(q.shape) > 1 else 1
            head_dim = int(q.shape[2]) if len(q.shape) > 2 else 1
            tile_size = max(1, int(os.environ.get("MUSA_AITER_PAGED_SPARSE_ATTENTION_TILE_SIZE", "512") or "512"))
            num_tiles = max(1, (int(effective_max_seq_len) + tile_size - 1) // tile_size)
            partial_out_bytes = head_dim * (2 if os.environ.get("MUSA_AITER_PAGED_SPARSE_ATTENTION_COMPACT_SCRATCH", "1") != "0" else 4)
            scratch_estimate = effective_batch * num_q_heads * num_tiles * (2 * 4 + partial_out_bytes)
            logger.info(
                "musa_aiter triton decode paged_sparse native_input: max_seq_len=%s tiled_env=%s tile_size=%s compact_scratch=%s scratch_estimate_bytes=%s scratch_limit_bytes=%s",
                effective_max_seq_len,
                os.environ.get("MUSA_AITER_PAGED_SPARSE_ATTENTION_TILED", "0"),
                tile_size,
                os.environ.get("MUSA_AITER_PAGED_SPARSE_ATTENTION_COMPACT_SCRATCH", "1"),
                scratch_estimate,
                os.environ.get("MUSA_AITER_SGLANG_PAGED_SPARSE_ATTENTION_SCRATCH_MAX_BYTES", str(512 * 1024 * 1024)),
            )

        paged_sparse_output = try_sglang_triton_paged_sparse_attention_replacement(
            q,
            key_cache,
            value_cache,
            kv_indptr,
            kv_indices,
            causal=False,
            softmax_scale=layer.scaling,
            max_seq_len=effective_max_seq_len,
            mate_accepts_dense_kv=True,
            parity_validated=True,
            require_env=True,
        )
        if logger is not None and os.environ.get("MUSA_AITER_SGLANG_DEBUG_TIMING", "0") == "1":
            if not debug_capture_at_entry:
                try:
                    if hasattr(torch, "musa"):
                        torch.musa.synchronize()
                except Exception:
                    pass
            logger.info(
                "musa_aiter triton decode timing: stage=after_paged_sparse elapsed_ms=%.3f capture=%s batch=%s q_heads=%s max_seq_len=%s kv_tokens=%s replacement_none=%s",
                (time.perf_counter() - debug_t0) * 1000.0,
                debug_capture_at_entry,
                int(q.shape[0]) if len(q.shape) > 0 else -1,
                int(q.shape[1]) if len(q.shape) > 1 else -1,
                effective_max_seq_len,
                int(kv_indices.numel()),
                paged_sparse_output is None,
            )
        replacement_output = paged_sparse_output
        replacement_route = "paged_sparse"
        if replacement_output is None:
            dense_output = try_sglang_triton_attention_dense_kv_replacement(
                q,
                key_cache,
                value_cache,
                kv_indptr,
                kv_indices,
                causal=False,
                softmax_scale=layer.scaling,
                max_seq_len=max_seq_len,
                mate_accepts_dense_kv=True,
                parity_validated=True,
                require_env=True,
            )
            replacement_output = dense_output
            replacement_route = "dense"
        if replacement_output is None:
            if logger is not None:
                logger.info("musa_aiter triton decode parity unsupported: replacement_output=None")
            return original_output
        capture_active = False
        try:
            if hasattr(torch, "musa"):
                capture_status = getattr(torch.musa, "is_current_stream_capturing", None)
                if callable(capture_status):
                    capture_active = bool(capture_status())
        except Exception:
            capture_active = False
        if logger is not None and not capture_active:
            original_dense = original_output.reshape_as(replacement_output)
            diff = (replacement_output.float() - original_dense.float()).abs()
            logger.info(
                "musa_aiter triton decode parity: route=%s replacement_shape=%s original_shape=%s max_abs=%s mean_abs=%s",
                replacement_route,
                tuple(getattr(replacement_output, "shape", ())),
                tuple(getattr(original_output, "shape", ())),
                float(diff.max().item()) if diff.numel() else 0.0,
                float(diff.mean().item()) if diff.numel() else 0.0,
            )
        if os.environ.get("MUSA_AITER_SGLANG_ATTENTION_PRELOAD_OUTPUT_REPLACE", "0") == "1":
            if logger is not None:
                logger.info("musa_aiter triton decode output replaced: route=%s replacement_shape=%s original_shape=%s", replacement_route, tuple(getattr(replacement_output, "shape", ())), tuple(getattr(original_output, "shape", ())))
            return replacement_output.reshape_as(original_output)
        return original_output
    except Exception as exc:
        logger = globals().get("logger")
        if logger is not None and not getattr(_musa_aiter_triton_decode_parity_probe, "_warned", False):
            logger.warning("musa_aiter triton decode parity disabled: %s", exc)
            setattr(_musa_aiter_triton_decode_parity_probe, "_warned", True)
        return original_output


'''
TRITON_RETURN_NEEDLE = '''        return o
'''
TRITON_RETURN_REPLACEMENT = '''        if os.environ.get("MUSA_AITER_SGLANG_ATTENTION_PRELOAD_PARITY", "0") == "1":
            o = _musa_aiter_triton_decode_parity_probe(
                o,
                q.view(-1, layer.tp_q_head_num, layer.qk_head_dim),
                forward_batch.token_to_kv_pool.get_key_buffer(layer.layer_id),
                forward_batch.token_to_kv_pool.get_value_buffer(layer.layer_id),
                kv_indptr,
                kv_indices,
                layer,
                max_seq_len=getattr(self.forward_metadata, "max_seq_len", None),
            )
        return o
'''


def _strip_old_triton_parity_patch(text: str) -> str:
    if not any(marker in text for marker in OLD_TRITON_BACKEND_PARITY_MARKERS):
        return text
    start_candidates = [text.find(f"{marker} = True") for marker in OLD_TRITON_BACKEND_PARITY_MARKERS]
    start_candidates = [index for index in start_candidates if index >= 0]
    start = min(start_candidates)
    helper_start = text.rfind("logger = logging.getLogger(__name__)", 0, start)
    if helper_start < 0:
        helper_start = start
    end_marker = "\n\nclass Triton"
    end = text.index(end_marker, start)
    text = text[:helper_start] + text[end + 2 :]
    old_call_start = text.find("        o = _musa_aiter_triton_decode_parity_probe(\n")
    if old_call_start >= 0:
        old_call_end = text.find("        return o\n", old_call_start)
        if old_call_end >= 0:
            text = text[:old_call_start] + text[old_call_end:]
    return text


def patch_triton_backend_parity_file(path: Path, *, backup: bool) -> bool:
    text = path.read_text()
    old_seq_len_fallback = """        effective_max_seq_len = max_seq_len
        if effective_max_seq_len is None:
            effective_max_seq_len = int(kv_indices.numel())
"""
    new_seq_len_fallback = """        effective_max_seq_len = max_seq_len
        if effective_max_seq_len is None:
            effective_batch = int(q.shape[0]) if len(q.shape) > 0 else 1
            effective_batch = max(1, effective_batch)
            effective_max_seq_len = (int(kv_indices.numel()) + effective_batch - 1) // effective_batch
"""
    if TRITON_BACKEND_PARITY_PATCH_MARKER in text:
        if old_seq_len_fallback in text:
            if backup:
                stamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
                backup_path = path.with_suffix(path.suffix + f".musa_aiter_backup_{stamp}")
                backup_path.write_text(text)
                print(f"backup: {backup_path}")
            path.write_text(text.replace(old_seq_len_fallback, new_seq_len_fallback, 1))
            print(f"updated: {path}")
            return True
        print(f"already patched: {path}")
        return False
    original_text = text
    text = _strip_old_triton_parity_patch(text)
    if "import os\n" not in text:
        if "from __future__ import annotations\n" in text:
            text = text.replace("from __future__ import annotations\n", "from __future__ import annotations\n\nimport logging\nimport os\n", 1)
        elif "import torch\n" in text:
            text = text.replace("import torch\n", "import logging\nimport os\n\nimport torch\n", 1)
        else:
            raise RuntimeError("triton import insertion point not found")
    elif "import logging\n" not in text:
        text = text.replace("import os\n", "import logging\nimport os\n", 1)
    if "\n\nclass Triton" in text:
        text = text.replace("\n\nclass Triton", "\n\n" + TRITON_HELPER_BLOCK + "class Triton", 1)
    else:
        raise RuntimeError("triton helper insertion point not found")
    hook_count = text.count(TRITON_RETURN_NEEDLE)
    if hook_count < 1:
        raise RuntimeError(f"expected at least 1 triton return hook point, found {hook_count}")
    if backup:
        stamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
        backup_path = path.with_suffix(path.suffix + f".musa_aiter_backup_{stamp}")
        backup_path.write_text(original_text)
        print(f"backup: {backup_path}")
    hook_index = text.rfind(TRITON_RETURN_NEEDLE)
    text = text[:hook_index] + TRITON_RETURN_REPLACEMENT + text[hook_index + len(TRITON_RETURN_NEEDLE) :]
    path.write_text(text)
    print(f"patched: {path}")
    return True



def _strip_v1_probe(text: str) -> str:
    if OLD_PROBE_MARKER not in text:
        return text
    start = text.index(f"{OLD_PROBE_MARKER} = True")
    end_marker = "\n\n\nclass HostKVCache"
    end = text.index(end_marker, start)
    text = text[:start] + text[end + 3 :]
    text = text.replace("import os\n", "", 1)
    old_call = '''            _musa_aiter_hicache_kernel_probe(
                self,
                device_pool,
                host_indices,
                device_indices,
                layer_id=locals().get("layer_id"),
                direction="load_to_device" if "layer_id" in locals() else "backup_from_device",
            )
'''
    text = text.replace(old_call, "")
    return text


def _strip_old_route_patch(text: str) -> str:
    if not any(marker in text for marker in OLD_ROUTE_MARKERS):
        return text
    start_candidates = [text.find(f"{marker} = True") for marker in OLD_ROUTE_MARKERS]
    start_candidates = [index for index in start_candidates if index >= 0]
    start = min(start_candidates)
    end_marker = "\ndef synchronized("
    end = text.index(end_marker, start)
    text = text[:start] + text[end + 1 :]
    text = text.replace("import os\n", "", 1)
    for replacement, needle in (
        (LOAD_MHA_REPLACEMENT, LOAD_MHA_NEEDLE),
        (LOAD_MLA_REPLACEMENT, LOAD_MLA_NEEDLE),
        (BACKUP_MHA_REPLACEMENT, BACKUP_MHA_NEEDLE),
        (BACKUP_MLA_REPLACEMENT, BACKUP_MLA_NEEDLE),
        (_old_backup_mha_replacement_v2(), BACKUP_MHA_NEEDLE),
        (_old_backup_mla_replacement_v2(), BACKUP_MLA_NEEDLE),
    ):
        text = text.replace(replacement, needle)
    return text


def patch_file(path: Path, *, backup: bool, dry_run: bool = False) -> bool:
    text = path.read_text()
    if PATCH_MARKER in text:
        print(f"already patched: {path}")
        return False
    original_text = text
    text = _strip_old_route_patch(_strip_v1_probe(text))
    if IMPORT_REPLACEMENT in text:
        pass
    elif IMPORT_NEEDLE in text:
        text = text.replace(IMPORT_NEEDLE, IMPORT_REPLACEMENT, 1)
    else:
        raise RuntimeError("import insertion point not found")
    if HELPER_NEEDLE not in text:
        raise RuntimeError("helper insertion point not found")
    hook_counts = {
        "MHA load": text.count(LOAD_MHA_NEEDLE),
        "MLA load": text.count(LOAD_MLA_NEEDLE),
        "MHA all-layer backup": text.count(BACKUP_MHA_NEEDLE),
        "MLA all-layer backup": text.count(BACKUP_MLA_NEEDLE),
    }
    # Multi-injection (REQ-5): DSV4-ized memory_pool_host.py has up to 9
    # load_to_device_per_layer variants. Require >=1 per family, inject at
    # every match, and report counts. Missing families fail-soft with a
    # manual-integration note.
    missing = [name for name, count in hook_counts.items() if count < 1]
    for name, count in hook_counts.items():
        print(f"hook point: {name}: {count} match(es)")
    if missing:
        raise RuntimeError(
            f"no hook point found for: {', '.join(missing)}; "
            "sglang memory_pool_host.py may have drifted — see manual integration notes in "
            "docs/DSV4_NATIVE_KV_TRANSFER_DESIGN_TDD.md section 3.5"
        )
    if dry_run:
        print(f"dry-run: would patch {path} ({sum(hook_counts.values())} injection point(s))")
        return False
    if backup:
        stamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
        backup_path = path.with_suffix(path.suffix + f".musa_aiter_backup_{stamp}")
        backup_path.write_text(original_text)
        print(f"backup: {backup_path}")
    text = text.replace(HELPER_NEEDLE, HELPER_BLOCK, 1)
    text = text.replace(BACKUP_MHA_NEEDLE, BACKUP_MHA_REPLACEMENT)
    text = text.replace(BACKUP_MLA_NEEDLE, BACKUP_MLA_REPLACEMENT)
    text = text.replace(LOAD_MHA_NEEDLE, LOAD_MHA_REPLACEMENT)
    text = text.replace(LOAD_MLA_NEEDLE, LOAD_MLA_REPLACEMENT)
    path.write_text(text)
    print(f"patched: {path} ({sum(hook_counts.values())} injection point(s))")
    return True


def restore_latest(path: Path) -> bool:
    backups = sorted(path.parent.glob(path.name + ".musa_aiter_backup_*"))
    if not backups:
        print(f"no backup found for {path}")
        return False
    latest = backups[-1]
    shutil.copy2(latest, path)
    print(f"restored: {path} from {latest}")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="Patch SGLang vendor checkout for musa_aiter HiCache routing and attention preload probing.")
    parser.add_argument("--sglang-root", default="/sgl-workspace/sglang/python")
    parser.add_argument("--file", default="sglang/srt/mem_cache/memory_pool_host.py")
    parser.add_argument("--attention-probe", action="store_true", help="Patch the MUSA flash attention wrapper with disabled-by-default preload probe/route hooks.")
    parser.add_argument("--attention-backend-parity", action="store_true", help="Patch the SGLang FA backend with disabled-by-default musa_aiter dense-attention parity probing.")
    parser.add_argument("--triton-backend-parity", action="store_true", help="Patch the SGLang Triton decode backend with disabled-by-default parity reachability diagnostics.")
    parser.add_argument("--attention-file", default="sglang/srt/hardware_backend/musa/attention/flash_attention.py")
    parser.add_argument("--attention-backend-file", default="sglang/srt/layers/attention/flashattention_backend.py")
    parser.add_argument("--triton-backend-file", default="sglang/srt/layers/attention/triton_backend.py")
    parser.add_argument("--no-backup", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Report hook-point counts and would-be injections without writing.")
    parser.add_argument("--restore", action="store_true")
    args = parser.parse_args()

    attention_modes = sum(bool(value) for value in (args.attention_probe, args.attention_backend_parity, args.triton_backend_parity))
    if attention_modes > 1:
        raise RuntimeError("choose only one attention patch mode")
    if args.triton_backend_parity:
        target = Path(args.sglang_root) / args.triton_backend_file
    elif args.attention_backend_parity:
        target = Path(args.sglang_root) / args.attention_backend_file
    elif args.attention_probe:
        target = Path(args.sglang_root) / args.attention_file
    else:
        target = Path(args.sglang_root) / args.file
    if not target.exists():
        raise FileNotFoundError(target)
    if args.restore:
        restore_latest(target)
    elif args.triton_backend_parity:
        patch_triton_backend_parity_file(target, backup=not args.no_backup)
    elif args.attention_backend_parity:
        patch_attention_backend_parity_file(target, backup=not args.no_backup)
    elif args.attention_probe:
        patch_attention_probe_file(target, backup=not args.no_backup)
    else:
        patch_file(target, backup=not args.no_backup, dry_run=args.dry_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
