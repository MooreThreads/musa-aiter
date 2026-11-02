import math

import pytest
import torch

import musa_aiter


def manual_paged_legacy_noncausal(
    query, key_cache, value_cache, block_tables, seq_lens, block_size, num_kv_heads
):
    """3D-query reference (full softmax over cached prefix — legacy default)."""

    out = torch.empty_like(query)
    scale = 1 / math.sqrt(query.shape[-1])
    repeat = query.shape[1] // num_kv_heads
    for b in range(query.shape[0]):
        seq_len = int(seq_lens[b])
        blocks = (seq_len + block_size - 1) // block_size
        ks, vs = [], []
        for j in range(blocks):
            bid = int(block_tables[b, j])
            ks.append(key_cache[bid].permute(2, 0, 1))
            vs.append(value_cache[bid].permute(2, 0, 1))
        k = torch.cat(ks, 0)[:seq_len]
        v = torch.cat(vs, 0)[:seq_len]
        if repeat != 1:
            k = k.repeat_interleave(repeat, 1)
            v = v.repeat_interleave(repeat, 1)
        scores = torch.einsum("hd,shd->hs", query[b].float(), k.float()) * scale
        probs = torch.softmax(scores, -1)
        out[b] = torch.einsum("hs,shd->hd", probs, v.float()).to(query.dtype)
    return out


def manual_paged_causal_multiquery(
    query_4d,
    key_cache,
    value_cache,
    block_tables,
    seq_lens,
    block_size,
    num_kv_heads,
    scale: float | None,
):
    """Matches :func:`musa_aiter.ops.paged_attn.paged_attention_v1` causal rule."""

    scale = scale if scale is not None else 1 / math.sqrt(query_4d.shape[-1])
    out = torch.empty_like(query_4d)
    repeat = query_4d.shape[2] // num_kv_heads
    _, Q, _, _ = query_4d.shape
    for b in range(query_4d.shape[0]):
        seq_len = int(seq_lens[b])
        blocks = (seq_len + block_size - 1) // block_size
        ks, vs = [], []
        for j in range(blocks):
            bid = int(block_tables[b, j])
            ks.append(key_cache[bid].permute(2, 0, 1))
            vs.append(value_cache[bid].permute(2, 0, 1))
        k = torch.cat(ks, 0)[:seq_len]
        v = torch.cat(vs, 0)[:seq_len]
        if repeat != 1:
            k = k.repeat_interleave(repeat, 1)
            v = v.repeat_interleave(repeat, 1)
        for qi in range(Q):
            kv_lim = seq_len - Q + qi + 1
            k_sub = k[:kv_lim]
            v_sub = v[:kv_lim]
            qh = query_4d[b, qi].float()
            scores = torch.einsum("hd,shd->hs", qh, k_sub.float()) * scale
            probs = torch.softmax(scores, -1)
            out[b, qi] = torch.einsum(
                "hs,shd->hd", probs, v_sub.float()
            ).to(query_4d.dtype)
    return out


def test_paged_attention_reference():
    num_seqs = 2
    qh = 4
    kvh = 2
    d = 8
    block = 4
    num_blocks = 4
    query = torch.randn(num_seqs, qh, d)
    key_cache = torch.randn(num_blocks, kvh, d, block)
    value_cache = torch.randn(num_blocks, kvh, d, block)
    block_tables = torch.tensor([[0, 1], [2, 3]], dtype=torch.int32)
    seq_lens = torch.tensor([5, 7], dtype=torch.int32)
    out = musa_aiter.paged_attention(
        query,
        key_cache,
        value_cache,
        block_tables,
        seq_lens,
        num_kv_heads=kvh,
        block_size=block,
    )
    ref = manual_paged_legacy_noncausal(
        query,
        key_cache,
        value_cache,
        block_tables,
        seq_lens,
        block,
        kvh,
    )
    torch.testing.assert_close(out, ref, rtol=1e-5, atol=1e-5)


def test_paged_attention_prefill_four_d_causal():
    num_seqs = 1
    Q = 3
    qh = 2
    kvh = 2
    d = 4
    block = 8
    num_blocks = 1
    seq_len = 3
    query = torch.randn(num_seqs, Q, qh, d)
    key_cache = torch.randn(num_blocks, kvh, d, block)
    value_cache = torch.randn(num_blocks, kvh, d, block)
    block_tables = torch.tensor([[0]], dtype=torch.int32)
    seq_lens = torch.tensor([seq_len], dtype=torch.int32)
    scale = 0.5
    ref = manual_paged_causal_multiquery(
        query,
        key_cache,
        value_cache,
        block_tables,
        seq_lens,
        block,
        kvh,
        scale,
    )
    out = musa_aiter.paged_attention(
        query,
        key_cache,
        value_cache,
        block_tables,
        seq_lens,
        num_kv_heads=kvh,
        block_size=block,
        scale=scale,
        causal=True,
    )
    torch.testing.assert_close(out, ref, rtol=1e-5, atol=1e-5)


def test_paged_attention_prefill_four_d_via_attention_type():
    """Decoder role implies causal masking when ``Q > 1``."""

    qh = kvh = 2
    d = 4
    block = 8
    Q = seq_len = 4
    query = torch.randn(1, Q, qh, d)
    key_cache = torch.randn(1, kvh, d, block)
    value_cache = torch.randn(1, kvh, d, block)
    block_tables = torch.tensor([[0]], dtype=torch.int32)
    seq_lens = torch.tensor([seq_len], dtype=torch.int32)
    ref = musa_aiter.paged_attention(
        query,
        key_cache,
        value_cache,
        block_tables,
        seq_lens,
        num_kv_heads=kvh,
        block_size=block,
        causal=True,
    )
    out = musa_aiter.paged_attention(
        query,
        key_cache,
        value_cache,
        block_tables,
        seq_lens,
        num_kv_heads=kvh,
        block_size=block,
        attention_type=musa_aiter.AttentionType.DECODER,
    )
    torch.testing.assert_close(out, ref, rtol=1e-5, atol=1e-5)


def test_paged_attention_causal_raises_when_seq_shorter_than_q():
    query = torch.randn(1, 4, 2, 4)
    key_cache = torch.randn(1, 2, 4, 8)
    value_cache = torch.randn(1, 2, 4, 8)
    block_tables = torch.tensor([[0]], dtype=torch.int32)
    seq_lens = torch.tensor([3], dtype=torch.int32)
    with pytest.raises(ValueError, match="seq_lens"):
        musa_aiter.paged_attention(
            query,
            key_cache,
            value_cache,
            block_tables,
            seq_lens,
            num_kv_heads=2,
            block_size=8,
            causal=True,
        )


def test_paged_attention_out_api():
    query = torch.randn(1, 2, 4)
    key_cache = torch.randn(1, 2, 4, 2)
    value_cache = torch.randn(1, 2, 4, 2)
    block_tables = torch.tensor([[0]], dtype=torch.int32)
    seq_lens = torch.tensor([2], dtype=torch.int32)
    out = torch.empty_like(query)
    ret = musa_aiter.paged_attention_v1(
        out,
        query,
        key_cache,
        value_cache,
        num_kv_heads=2,
        scale=0.5,
        block_tables=block_tables,
        seq_lens=seq_lens,
        block_size=2,
    )
    assert ret is out

