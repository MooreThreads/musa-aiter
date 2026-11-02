import math

import pytest
import torch

import musa_aiter


def manual_attention(q, k, v, causal=False):
    scale = 1.0 / math.sqrt(q.shape[-1])
    qh = q.transpose(1, 2).float()
    kh = k.transpose(1, 2).float()
    vh = v.transpose(1, 2).float()
    scores = torch.matmul(qh, kh.transpose(-2, -1)) * scale
    if causal:
        s = q.shape[1]
        mask = torch.ones((s, s), dtype=torch.bool).tril()
        scores = scores.masked_fill(~mask, float("-inf"))
    probs = torch.softmax(scores, dim=-1)
    return torch.matmul(probs, vh).transpose(1, 2).to(q.dtype)


def test_flash_attn_func_matches_manual_noncausal():
    q = torch.randn(2, 5, 4, 8)
    k = torch.randn(2, 5, 4, 8)
    v = torch.randn(2, 5, 4, 8)
    out = musa_aiter.flash_attn_func(q, k, v, causal=False)
    ref = manual_attention(q, k, v, causal=False)
    torch.testing.assert_close(out, ref, rtol=1e-5, atol=1e-5)


def test_flash_attn_func_matches_manual_causal():
    q = torch.randn(1, 4, 2, 8)
    k = torch.randn(1, 4, 2, 8)
    v = torch.randn(1, 4, 2, 8)
    out = musa_aiter.flash_attn_func(q, k, v, causal=True)
    ref = manual_attention(q, k, v, causal=True)
    torch.testing.assert_close(out, ref, rtol=1e-5, atol=1e-5)


def test_flash_attn_func_gqa_shape():
    q = torch.randn(1, 3, 4, 8)
    k = torch.randn(1, 3, 2, 8)
    v = torch.randn(1, 3, 2, 8)
    out = musa_aiter.flash_attn_func(q, k, v)
    assert out.shape == q.shape


def test_flash_attn_attention_type_decoder_defaults_causal():
    q = torch.randn(1, 4, 2, 8)
    k = torch.randn(1, 4, 2, 8)
    v = torch.randn(1, 4, 2, 8)
    out_explicit = musa_aiter.flash_attn_func(q, k, v, causal=True)
    out_typed = musa_aiter.flash_attn_func(
        q,
        k,
        v,
        attention_type=musa_aiter.AttentionType.DECODER,
    )
    torch.testing.assert_close(out_typed, out_explicit, rtol=1e-5, atol=1e-5)


def test_flash_attn_attention_type_encoder_only_non_causal():
    q = torch.randn(1, 4, 2, 8)
    k = torch.randn(1, 4, 2, 8)
    v = torch.randn(1, 4, 2, 8)
    out_explicit = musa_aiter.flash_attn_func(q, k, v, causal=False)
    out_typed = musa_aiter.flash_attn_func(
        q,
        k,
        v,
        attention_type=musa_aiter.AttentionType.ENCODER_ONLY,
    )
    torch.testing.assert_close(out_typed, out_explicit, rtol=1e-5, atol=1e-5)


def test_flash_attn_causal_kw_overrides_attention_type():
    q = torch.randn(1, 4, 2, 8)
    k = torch.randn(1, 4, 2, 8)
    v = torch.randn(1, 4, 2, 8)
    out_noncausal = musa_aiter.flash_attn_func(
        q,
        k,
        v,
        attention_type=musa_aiter.AttentionType.DECODER,
        causal=False,
    )
    ref = manual_attention(q, k, v, causal=False)
    torch.testing.assert_close(out_noncausal, ref, rtol=1e-5, atol=1e-5)


def test_flash_attn_encoder_decoder_cross_sequence_length():
    q = torch.randn(1, 2, 4, 8)
    k = torch.randn(1, 5, 4, 8)
    v = torch.randn(1, 5, 4, 8)
    out = musa_aiter.flash_attn_func(
        q,
        k,
        v,
        attention_type=musa_aiter.AttentionType.ENCODER_DECODER,
    )
    assert out.shape == q.shape


def test_flash_attn_unknown_attention_type_raises():
    q = torch.randn(1, 2, 2, 8)
    k = torch.randn(1, 2, 2, 8)
    v = torch.randn(1, 2, 2, 8)
    with pytest.raises(ValueError, match="unknown attention_type"):
        musa_aiter.flash_attn_func(q, k, v, attention_type="vlm_cross")
