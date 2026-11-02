"""Attention type coercion (vLLM / SGLang alignment)."""

import pytest

from musa_aiter.attention_type import AttentionType, coerce_attention_type, default_causal_for_attention_type


def test_coerce_none():
    assert coerce_attention_type(None) is None


def test_coerce_enum_roundtrip():
    assert coerce_attention_type(AttentionType.DECODER) is AttentionType.DECODER


def test_coerce_string_case_insensitive():
    assert coerce_attention_type("ENCODER_ONLY") is AttentionType.ENCODER_ONLY
    assert coerce_attention_type("decoder_bidirectional") is AttentionType.DECODER_BIDIRECTIONAL


def test_coerce_invalid():
    with pytest.raises(ValueError, match="unknown attention_type"):
        coerce_attention_type("not_an_attention_kind")


@pytest.mark.parametrize(
    ("member", "causal"),
    [
        (AttentionType.DECODER, True),
        (AttentionType.ENCODER, False),
        (AttentionType.ENCODER_ONLY, False),
        (AttentionType.ENCODER_DECODER, False),
        (AttentionType.DECODER_BIDIRECTIONAL, False),
    ],
)
def test_default_causal_mapping(member: AttentionType, causal: bool) -> None:
    assert default_causal_for_attention_type(member) is causal
