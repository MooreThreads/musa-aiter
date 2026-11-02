from pathlib import Path

from scripts.patch_sglang_musa_aiter_hicache import (
    ATTENTION_BACKEND_PARITY_PATCH_MARKER,
    TRITON_BACKEND_PARITY_PATCH_MARKER,
    patch_attention_backend_parity_file,
    patch_triton_backend_parity_file,
)


BACKEND_TEXT = '''import logging

logger = logging.getLogger(__name__)


class FlashAttentionBackend:
    def forward(self, q, layer, forward_batch, metadata, result):
        if False:
            o = result
        else:
            o = result

        return o.view(-1, layer.tp_q_head_num * layer.v_head_dim)
'''


TRITON_TEXT = '''from __future__ import annotations

import torch


class TritonAttnBackend:
    def forward_decode(self, q, k, v, layer, forward_batch, save_kv_cache=True):
        q = q.reshape(-1, layer.tp_q_head_num * layer.qk_head_dim)
        o = torch.empty_like(q)
        kv_indptr = self.forward_metadata.kv_indptr
        kv_indices = self.forward_metadata.kv_indices
        self.decode_attention_fwd(
            q.view(-1, layer.tp_q_head_num, layer.qk_head_dim),
            forward_batch.token_to_kv_pool.get_key_buffer(layer.layer_id),
            forward_batch.token_to_kv_pool.get_value_buffer(layer.layer_id),
            o.view(-1, layer.tp_q_head_num, layer.v_head_dim),
            kv_indptr,
            kv_indices,
        )
        return o
'''


def test_patch_attention_backend_parity_file_inserts_disabled_probe(tmp_path: Path):
    path = tmp_path / "flashattention_backend.py"
    path.write_text(BACKEND_TEXT)

    patched = patch_attention_backend_parity_file(path, backup=False)

    text = path.read_text()
    assert patched is True
    assert ATTENTION_BACKEND_PARITY_PATCH_MARKER in text
    assert "def _musa_aiter_attention_backend_parity_probe" in text
    assert "MUSA_AITER_SGLANG_ATTENTION_PRELOAD_PARITY" in text
    assert "musa_aiter attention backend parity reached" in text
    assert "musa_aiter attention backend parity unsupported" in text
    assert "_musa_aiter_attention_backend_parity_probe(" in text
    assert "return o.view(-1, layer.tp_q_head_num * layer.v_head_dim)" in text


def test_patch_attention_backend_parity_file_is_idempotent(tmp_path: Path):
    path = tmp_path / "flashattention_backend.py"
    path.write_text(BACKEND_TEXT)

    assert patch_attention_backend_parity_file(path, backup=False) is True
    once = path.read_text()
    assert patch_attention_backend_parity_file(path, backup=False) is False
    assert path.read_text() == once


def test_patch_attention_backend_parity_file_patches_final_return_when_multiple(tmp_path: Path):
    path = tmp_path / "flashattention_backend.py"
    path.write_text(BACKEND_TEXT + "\n" + BACKEND_TEXT)

    assert patch_attention_backend_parity_file(path, backup=False) is True

    text = path.read_text()
    assert text.count("_musa_aiter_attention_backend_parity_probe(") == 3
    assert text.count("return o.view(-1, layer.tp_q_head_num * layer.v_head_dim)") == 2


def test_patch_triton_backend_parity_file_inserts_disabled_probe(tmp_path: Path):
    path = tmp_path / "triton_backend.py"
    path.write_text(TRITON_TEXT)

    patched = patch_triton_backend_parity_file(path, backup=False)

    text = path.read_text()
    assert patched is True
    assert TRITON_BACKEND_PARITY_PATCH_MARKER in text
    assert "def _musa_aiter_triton_decode_parity_probe" in text
    assert "MUSA_AITER_SGLANG_ATTENTION_PRELOAD_PARITY" in text
    assert "musa_aiter triton decode parity reached" in text
    assert "musa_aiter triton decode parity:" in text
    assert "try_sglang_triton_paged_sparse_attention_replacement" in text
    assert "try_sglang_triton_attention_dense_kv_replacement" in text
    assert "replacement_route = \"paged_sparse\"" in text
    assert "replacement_route = \"dense\"" in text
    assert "is_current_stream_capturing" in text
    assert "MUSA_AITER_SGLANG_ATTENTION_PRELOAD_OUTPUT_REPLACE" in text
    assert "musa_aiter triton decode output replaced" in text
    assert "effective_max_seq_len = max_seq_len" in text
    assert "effective_batch = int(q.shape[0])" in text
    assert "effective_max_seq_len = (int(kv_indices.numel()) + effective_batch - 1) // effective_batch" in text
    assert "musa_aiter triton decode paged_sparse native_input" in text
    assert "compact_scratch=%s" in text
    assert "scratch_estimate_bytes=%s" in text
    assert "MUSA_AITER_SGLANG_DEBUG_TIMING" in text
    assert "musa_aiter triton decode timing: stage=after_paged_sparse" in text
    assert "exact_max_seq_len = int(seq_lens.max().item())" in text
    assert "exact_max_seq_len < int(effective_max_seq_len)" in text
    assert "MUSA_AITER_SGLANG_ALLOW_AFTER_ORIGINAL_OUTPUT_REPLACE" in text
    assert "reason=after_original_double_compute_guard" in text
    assert 'if os.environ.get("MUSA_AITER_SGLANG_ATTENTION_PRELOAD_PARITY", "0") == "1"' in text
    assert "MUSA_AITER_PAGED_SPARSE_ATTENTION_COMPACT_SCRATCH" in text
    assert "max_seq_len=effective_max_seq_len" in text
    assert "max_seq_len=getattr(self.forward_metadata, \"max_seq_len\", None)" in text
    assert "q.view(-1, layer.tp_q_head_num, layer.qk_head_dim)" in text
    assert "_musa_aiter_triton_decode_parity_probe(" in text
    assert "return o" in text


def test_patch_triton_backend_parity_file_upgrades_v1_patch(tmp_path: Path):
    path = tmp_path / "triton_backend.py"
    v1_text = TRITON_TEXT.replace(
        "class TritonAttnBackend",
        '''logger = logging.getLogger(__name__)


MUSA_AITER_SGLANG_TRITON_DECODE_PARITY_PATCH_V1 = True


def _musa_aiter_triton_decode_parity_probe(original_output, q, key_cache, value_cache, kv_indptr, kv_indices, layer):
    return original_output


class TritonAttnBackend''',
    ).replace(
        "        return o\n",
        '''        o = _musa_aiter_triton_decode_parity_probe(
            o,
            q.view(-1, layer.tp_q_head_num, layer.qk_head_dim),
            forward_batch.token_to_kv_pool.get_key_buffer(layer.layer_id),
            forward_batch.token_to_kv_pool.get_value_buffer(layer.layer_id),
            kv_indptr,
            kv_indices,
            layer,
        )
        return o
''',
    )
    path.write_text(v1_text)

    assert patch_triton_backend_parity_file(path, backup=False) is True

    text = path.read_text()
    assert "MUSA_AITER_SGLANG_TRITON_DECODE_PARITY_PATCH_V1 = True" not in text
    assert TRITON_BACKEND_PARITY_PATCH_MARKER in text
    assert "MUSA_AITER_SGLANG_ATTENTION_PRELOAD_OUTPUT_REPLACE" in text
    assert text.count("def _musa_aiter_triton_decode_parity_probe") == 1
    assert text.count("_musa_aiter_triton_decode_parity_probe(") == 2


def test_patch_triton_backend_parity_file_is_idempotent(tmp_path: Path):
    path = tmp_path / "triton_backend.py"
    path.write_text(TRITON_TEXT)

    assert patch_triton_backend_parity_file(path, backup=False) is True
    once = path.read_text()
    assert patch_triton_backend_parity_file(path, backup=False) is False
    assert path.read_text() == once


def test_patch_triton_backend_parity_file_upgrades_v2_patch(tmp_path: Path):
    path = tmp_path / "triton_backend.py"
    path.write_text(TRITON_TEXT)
    assert patch_triton_backend_parity_file(path, backup=False) is True
    v2_text = path.read_text().replace(
        "MUSA_AITER_SGLANG_TRITON_DECODE_PARITY_PATCH_V10",
        "MUSA_AITER_SGLANG_TRITON_DECODE_PARITY_PATCH_V2",
    ).replace(
        "        from musa_aiter.integration import (\n            try_sglang_triton_attention_dense_kv_replacement,\n            try_sglang_triton_paged_sparse_attention_replacement,\n        )\n\n        paged_sparse_output = try_sglang_triton_paged_sparse_attention_replacement(\n            q,\n            key_cache,\n            value_cache,\n            kv_indptr,\n            kv_indices,\n            causal=False,\n            softmax_scale=layer.scaling,\n            mate_accepts_dense_kv=True,\n            parity_validated=True,\n            require_env=True,\n        )\n        replacement_output = paged_sparse_output\n        replacement_route = \"paged_sparse\"\n        if replacement_output is None:\n            dense_output = try_sglang_triton_attention_dense_kv_replacement(\n                q,\n                key_cache,\n                value_cache,\n                kv_indptr,\n                kv_indices,\n                causal=False,\n                softmax_scale=layer.scaling,\n                max_seq_len=max_seq_len,\n                mate_accepts_dense_kv=True,\n                parity_validated=True,\n                require_env=True,\n            )\n            replacement_output = dense_output\n            replacement_route = \"dense\"\n        if replacement_output is None:\n            if logger is not None:\n                logger.info(\"musa_aiter triton decode parity unsupported: replacement_output=None\")\n            return original_output\n        if logger is not None:\n            original_dense = original_output.reshape_as(replacement_output)\n            diff = (replacement_output.float() - original_dense.float()).abs()\n            logger.info(\n                \"musa_aiter triton decode parity: route=%s replacement_shape=%s original_shape=%s max_abs=%s mean_abs=%s\",\n                replacement_route,\n                tuple(getattr(replacement_output, \"shape\", ())),\n                tuple(getattr(original_output, \"shape\", ())),\n                float(diff.max().item()) if diff.numel() else 0.0,\n                float(diff.mean().item()) if diff.numel() else 0.0,\n            )\n        if os.environ.get(\"MUSA_AITER_SGLANG_ATTENTION_PRELOAD_OUTPUT_REPLACE\", \"0\") == \"1\":\n            if logger is not None:\n                logger.info(\"musa_aiter triton decode output replaced: route=%s replacement_shape=%s original_shape=%s\", replacement_route, tuple(getattr(replacement_output, \"shape\", ())), tuple(getattr(original_output, \"shape\", ())))\n            return replacement_output.reshape_as(original_output)\n        return original_output",
        "        from musa_aiter.integration import try_sglang_triton_attention_dense_kv_replacement\n\n        dense_output = try_sglang_triton_attention_dense_kv_replacement(\n            q,\n            key_cache,\n            value_cache,\n            kv_indptr,\n            kv_indices,\n            causal=False,\n            softmax_scale=layer.scaling,\n            max_seq_len=max_seq_len,\n            mate_accepts_dense_kv=True,\n            parity_validated=True,\n            require_env=True,\n        )\n        if dense_output is None:\n            if logger is not None:\n                logger.info(\"musa_aiter triton decode parity unsupported: dense_output=None\")\n            return original_output\n        if logger is not None:\n            original_dense = original_output.reshape_as(dense_output)\n            diff = (dense_output.float() - original_dense.float()).abs()\n            logger.info(\n                \"musa_aiter triton decode parity: dense_shape=%s original_shape=%s max_abs=%s mean_abs=%s\",\n                tuple(getattr(dense_output, \"shape\", ())),\n                tuple(getattr(original_output, \"shape\", ())),\n                float(diff.max().item()) if diff.numel() else 0.0,\n                float(diff.mean().item()) if diff.numel() else 0.0,\n            )\n        if os.environ.get(\"MUSA_AITER_SGLANG_ATTENTION_PRELOAD_OUTPUT_REPLACE\", \"0\") == \"1\":\n            if logger is not None:\n                logger.info(\"musa_aiter triton decode output replaced: dense_shape=%s original_shape=%s\", tuple(getattr(dense_output, \"shape\", ())), tuple(getattr(original_output, \"shape\", ())))\n            return dense_output.reshape_as(original_output)\n        return original_output",
    )
    path.write_text(v2_text)

    assert patch_triton_backend_parity_file(path, backup=False) is True

    text = path.read_text()
    assert "MUSA_AITER_SGLANG_TRITON_DECODE_PARITY_PATCH_V2" not in text
    assert TRITON_BACKEND_PARITY_PATCH_MARKER in text
    assert "try_sglang_triton_paged_sparse_attention_replacement" in text
    assert text.count("def _musa_aiter_triton_decode_parity_probe") == 1
    assert text.count("_musa_aiter_triton_decode_parity_probe(") == 2
