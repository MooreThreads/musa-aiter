import pytest
import torch

from musa_aiter.ops.paged_sparse_attention import (
    clear_paged_sparse_tiled_scratch_cache,
    is_native_paged_sparse_decode_attention_available,
    paged_sparse_decode_attention,
)


def _require_musa_device() -> torch.device:
    if not hasattr(torch, "musa") or not torch.musa.is_available():
        pytest.skip("TorchMUSA is not available")
    return torch.device("musa:0")


def _reference_paged_sparse_attention(q, k, v, kv_indptr, kv_indices, *, softmax_scale=None):
    batch, q_heads, head_dim = map(int, q.shape)
    kv_heads = int(k.shape[1])
    scale = float(softmax_scale) if softmax_scale is not None else 1.0 / (head_dim**0.5)
    out = torch.empty_like(q)
    for batch_idx in range(batch):
        start = int(kv_indptr[batch_idx].item())
        end = int(kv_indptr[batch_idx + 1].item())
        slots = kv_indices[start:end].to(dtype=torch.long)
        key = k.index_select(0, slots)
        value = v.index_select(0, slots)
        for q_head in range(q_heads):
            kv_head = q_head // (q_heads // kv_heads)
            scores = torch.einsum("d,sd->s", q[batch_idx, q_head].float(), key[:, kv_head].float()) * scale
            probs = torch.softmax(scores, dim=-1)
            out[batch_idx, q_head] = torch.einsum("s,sd->d", probs, value[:, kv_head].float()).to(q.dtype)
    return out


def test_paged_sparse_decode_attention_rejects_bad_rank():
    q = torch.zeros((1, 2, 4, 8), dtype=torch.float32)
    k = torch.zeros((4, 2, 8), dtype=torch.float32)
    v = torch.zeros_like(k)
    kv_indptr = torch.tensor([0, 2], dtype=torch.long)
    kv_indices = torch.tensor([0, 1], dtype=torch.long)

    with pytest.raises(ValueError, match="q must be"):
        paged_sparse_decode_attention(q, k, v, kv_indptr, kv_indices)


def test_paged_sparse_decode_attention_rejects_q_heads_not_divisible_by_kv_heads():
    q = torch.zeros((1, 3, 8), dtype=torch.float32)
    k = torch.zeros((4, 2, 8), dtype=torch.float32)
    v = torch.zeros_like(k)
    kv_indptr = torch.tensor([0, 2], dtype=torch.long)
    kv_indices = torch.tensor([0, 1], dtype=torch.long)

    with pytest.raises(ValueError, match="divisible"):
        paged_sparse_decode_attention(q, k, v, kv_indptr, kv_indices)


def test_paged_sparse_decode_attention_rejects_non_integer_indices():
    q = torch.zeros((1, 2, 8), dtype=torch.float32)
    k = torch.zeros((4, 2, 8), dtype=torch.float32)
    v = torch.zeros_like(k)
    kv_indptr = torch.tensor([0.0, 2.0], dtype=torch.float32)
    kv_indices = torch.tensor([0, 1], dtype=torch.long)

    with pytest.raises(ValueError, match="integer"):
        paged_sparse_decode_attention(q, k, v, kv_indptr, kv_indices)


def test_paged_sparse_decode_attention_musa_float_matches_reference():
    device = _require_musa_device()
    assert is_native_paged_sparse_decode_attention_available() is True
    dtype = torch.float32
    q_cpu = torch.randn((2, 4, 8), dtype=dtype)
    k_cpu = torch.randn((8, 2, 8), dtype=dtype)
    v_cpu = torch.randn((8, 2, 8), dtype=dtype)
    kv_indptr_cpu = torch.tensor([0, 3, 7], dtype=torch.long)
    kv_indices_cpu = torch.tensor([0, 2, 4, 1, 3, 5, 7], dtype=torch.long)
    expected = _reference_paged_sparse_attention(q_cpu, k_cpu, v_cpu, kv_indptr_cpu, kv_indices_cpu)

    actual = paged_sparse_decode_attention(
        q_cpu.to(device),
        k_cpu.to(device),
        v_cpu.to(device),
        kv_indptr_cpu.to(device),
        kv_indices_cpu.to(device),
    )

    torch.testing.assert_close(actual.cpu(), expected, rtol=1e-4, atol=1e-4)


def test_paged_sparse_decode_attention_musa_bfloat16_gqa_matches_reference():
    device = _require_musa_device()
    assert is_native_paged_sparse_decode_attention_available() is True
    dtype = torch.bfloat16
    q_cpu = torch.randn((2, 4, 8), dtype=dtype)
    k_cpu = torch.randn((8, 2, 8), dtype=dtype)
    v_cpu = torch.randn((8, 2, 8), dtype=dtype)
    kv_indptr_cpu = torch.tensor([0, 3, 7], dtype=torch.long)
    kv_indices_cpu = torch.tensor([0, 2, 4, 1, 3, 5, 7], dtype=torch.long)
    expected = _reference_paged_sparse_attention(q_cpu, k_cpu, v_cpu, kv_indptr_cpu, kv_indices_cpu)

    actual = paged_sparse_decode_attention(
        q_cpu.to(device),
        k_cpu.to(device),
        v_cpu.to(device),
        kv_indptr_cpu.to(device),
        kv_indices_cpu.to(device),
    )

    torch.testing.assert_close(actual.cpu(), expected, rtol=5e-2, atol=5e-2)


def test_paged_sparse_decode_attention_musa_bfloat16_real_head_dim_smoke():
    device = _require_musa_device()
    assert is_native_paged_sparse_decode_attention_available() is True
    dtype = torch.bfloat16
    q_cpu = torch.randn((1, 4, 128), dtype=dtype)
    k_cpu = torch.randn((128, 2, 128), dtype=dtype)
    v_cpu = torch.randn((128, 2, 128), dtype=dtype)
    kv_indptr_cpu = torch.tensor([0, 128], dtype=torch.long)
    kv_indices_cpu = torch.arange(128, dtype=torch.long)
    expected = _reference_paged_sparse_attention(q_cpu, k_cpu, v_cpu, kv_indptr_cpu, kv_indices_cpu)

    actual = paged_sparse_decode_attention(
        q_cpu.to(device),
        k_cpu.to(device),
        v_cpu.to(device),
        kv_indptr_cpu.to(device),
        kv_indices_cpu.to(device),
    )

    torch.testing.assert_close(actual.cpu(), expected, rtol=6e-2, atol=6e-2)


def test_paged_sparse_decode_attention_musa_bfloat16_tiled_matches_block(monkeypatch):
    device = _require_musa_device()
    assert is_native_paged_sparse_decode_attention_available() is True
    dtype = torch.bfloat16
    q = torch.randn((2, 4, 128), device=device, dtype=dtype)
    k = torch.randn((512, 2, 128), device=device, dtype=dtype)
    v = torch.randn_like(k)
    kv_indptr = torch.tensor([0, 256, 512], device=device, dtype=torch.long)
    kv_indices = torch.arange(512, device=device, dtype=torch.long)

    monkeypatch.delenv("MUSA_AITER_PAGED_SPARSE_ATTENTION_TILED", raising=False)
    block = paged_sparse_decode_attention(q, k, v, kv_indptr, kv_indices)
    monkeypatch.setenv("MUSA_AITER_PAGED_SPARSE_ATTENTION_TILED", "1")
    monkeypatch.setenv("MUSA_AITER_PAGED_SPARSE_ATTENTION_TILE_SIZE", "128")
    tiled = paged_sparse_decode_attention(q, k, v, kv_indptr, kv_indices, max_seq_len=256, tile_size=128)

    torch.testing.assert_close(tiled.cpu(), block.cpu(), rtol=6e-2, atol=6e-2)


def test_paged_sparse_decode_attention_tiled_explicit_metadata_avoids_item(monkeypatch):
    device = _require_musa_device()
    assert is_native_paged_sparse_decode_attention_available() is True
    clear_paged_sparse_tiled_scratch_cache()
    q = torch.randn((1, 4, 128), device=device, dtype=torch.bfloat16)
    k = torch.randn((256, 2, 128), device=device, dtype=torch.bfloat16)
    v = torch.randn_like(k)
    kv_indptr = torch.tensor([0, 256], device=device, dtype=torch.long)
    kv_indices = torch.arange(256, device=device, dtype=torch.long)

    monkeypatch.setenv("MUSA_AITER_PAGED_SPARSE_ATTENTION_TILED", "1")
    monkeypatch.setattr(torch.Tensor, "item", lambda self: (_ for _ in ()).throw(RuntimeError("item disabled")))

    actual = paged_sparse_decode_attention(q, k, v, kv_indptr, kv_indices, max_seq_len=256, tile_size=128)

    assert actual.shape == q.shape
