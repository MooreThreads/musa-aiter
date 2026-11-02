"""Shared helpers for `verify_*_musa_aiter.py` example scripts.

Keeps operator checks identical across framework verifiers (vLLM, SGLang, …).
"""

from __future__ import annotations

import sys

import musa_aiter
import torch


def assert_device_available(device: str) -> None:
    if device == "cpu":
        return
    d = device.lower()
    if d.startswith("musa"):
        if not hasattr(torch, "musa"):
            print(
                "TorchMUSA is required for device starting with 'musa:' "
                "(install torch_musa / vendor stack).",
                file=sys.stderr,
            )
            raise SystemExit(1)
        if not torch.musa.is_available():
            print("torch.musa.is_available() is False.", file=sys.stderr)
            raise SystemExit(1)
        return
    if d.startswith("cuda"):
        if not torch.cuda.is_available():
            print("torch.cuda.is_available() is False for cuda device.", file=sys.stderr)
            raise SystemExit(1)
        return
    print(
        f"Unsupported device {device!r}; use cpu, musa:N, or cuda:N.",
        file=sys.stderr,
    )
    raise SystemExit(1)


def run_musa_aiter_ops(device: str) -> None:
    """Attention + paged decode + fused MoE smoke (same coverage as vLLM verifier)."""

    d = device
    q = torch.randn(1, 8, 4, 16, device=d)
    k = torch.randn(1, 8, 4, 16, device=d)
    v = torch.randn(1, 8, 4, 16, device=d)
    pre = musa_aiter.flash_attn_func(q, k, v, causal=True)
    if pre.shape != q.shape:
        raise AssertionError((pre.shape, q.shape))

    query = torch.randn(2, 4, 16, device=d)
    key_cache = torch.randn(4, 2, 16, 4, device=d)
    value_cache = torch.randn(4, 2, 16, 4, device=d)
    block_tables = torch.tensor([[0, 1], [2, 3]], dtype=torch.int32, device=d)
    seq_lens = torch.tensor([5, 7], dtype=torch.int32, device=d)
    dec = musa_aiter.paged_attention(
        query,
        key_cache,
        value_cache,
        block_tables,
        seq_lens,
        num_kv_heads=2,
        block_size=4,
    )
    if dec.shape != query.shape:
        raise AssertionError((dec.shape, query.shape))

    tokens, hidden, inter, experts = 8, 16, 32, 4
    x = torch.randn(tokens, hidden, device=d)
    w1 = torch.randn(experts, inter * 2, hidden, device=d)
    w2 = torch.randn(experts, hidden, inter, device=d)
    logits = torch.randn(tokens, experts, device=d)
    weights, ids = musa_aiter.topk_softmax(logits, 2)
    moe_out = musa_aiter.fused_moe(x, w1, w2, weights, ids)
    if moe_out.shape != x.shape:
        raise AssertionError((moe_out.shape, x.shape))
