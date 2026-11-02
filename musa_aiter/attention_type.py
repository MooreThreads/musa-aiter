"""Attention role labels for framework integration (vLLM / SGLang alignment).

`string`-subclass `Enum` values match upstream spellings so callsites and
``torch.compile`` string annotations can interoperate:

- **vLLM v1** — ``vllm.v1.attention.backend.AttentionType``:
  ``decoder``, ``encoder``, ``encoder_only``, ``encoder_decoder``.
- **SGLang** — ``sglang.srt.layers.radix_attention.AttentionType`` adds
  ``decoder_bidirectional`` (vLLM has no direct equivalent; use together with
  vLLM's separate non-causal decoder flags in full frameworks).

Use with :func:`musa_aiter.flash_attn_func`, :func:`musa_aiter.paged_attention`, and
:func:`musa_aiter.paged_attention_v1` via the ``attention_type`` keyword.
"""

from __future__ import annotations

from enum import Enum


class AttentionType(str, Enum):
    """Attention *role* for dispatch metadata and default ``causal`` policy (dense + paged)."""

    DECODER = "decoder"
    """Decoder self-attention; default ``causal=True`` for dense and multi-query paged paths."""

    ENCODER = "encoder"
    """Encoder self-attention in encoder–decoder stacks (vLLM); default non-causal."""

    ENCODER_ONLY = "encoder_only"
    """Encoder-only model self-attention (vLLM + SGLang); default non-causal."""

    ENCODER_DECODER = "encoder_decoder"
    """Cross-attention: decoder queries vs encoder K/V (vLLM); default non-causal."""

    DECODER_BIDIRECTIONAL = "decoder_bidirectional"
    """Bidirectional attention inside a decoder stack (e.g. multimodal prefix, SGLang)."""


def coerce_attention_type(value: AttentionType | str | None) -> AttentionType | None:
    if value is None:
        return None
    if isinstance(value, AttentionType):
        return value
    s = str(value).strip().lower()
    for m in AttentionType:
        if m.value == s:
            return m
    raise ValueError(f"unknown attention_type {value!r}; expected one of {[e.value for e in AttentionType]}")


def default_causal_for_attention_type(attention_type: AttentionType) -> bool:
    """Return the usual causal mask flag for *attention_type* (dense reference path)."""

    if attention_type is AttentionType.DECODER:
        return True
    return False
