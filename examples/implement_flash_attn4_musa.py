#!/usr/bin/env python3
"""Skeleton: FA4-aligned dense attention façade for MUSA bring-up.

Upstream FlashAttention‑4 targets optimised **CUDA** stacks; on **MUSA** you implement
(or vendor-wrap) kernels under **mcc** / MuDNN / MUBIN. This script stays **standalone**:
it does **not** add ``flash_attn4_func`` to the public ``musa_aiter`` package until you
follow the repository's registry, API policy, and tests.

Demonstrates:
  * Same tensor contract as ``musa_aiter.flash_attn_func`` ([B,S,H,D]).
  * Optional env ``MUSA_AITER_NATIVE_FA4`` hook (currently always falls back).
  * ``musa_aiter trace`` for dispatch visibility.

Usage::

    PYTHONPATH=. python examples/implement_flash_attn4_musa.py
    PYTHONPATH=. python -m musa_aiter.cli.main trace -- python examples/implement_flash_attn4_musa.py
"""

from __future__ import annotations

import argparse
import os

import torch

import musa_aiter
from musa_aiter.attention_type import AttentionType


def _try_native_fa4_placeholder(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    causal: bool,
) -> torch.Tensor | None:
    """Replace with ctypes/pybind that loads your FA4-aligned MUSA kernel.

    Typical steps:
      1. ``mcc``-compile ``.mu`` / link vendor ``.so``
      2. ``ctypes.CDLL`` or pybind extension
      3. Validate parity vs ``flash_attn_func`` on ``op_tests``

    Returning ``None`` keeps the Torch reference path.
    """
    if os.environ.get("MUSA_AITER_NATIVE_FA4", "0") != "1":
        return None
    # Stub: flip to dlopen once a symbol exists, e.g. musa_aiter_fa4_forward
    print(
        "[fa4_skeleton] MUSA_AITER_NATIVE_FA4=1 but no native FA4 "
        ".so wired — using reference fallback.",
        flush=True,
    )
    return None


def flash_attn4_func_skeleton(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    causal: bool = True,
    attention_type: AttentionType | None = AttentionType.DECODER,
) -> torch.Tensor:
    """FA4 *style* façade: native slot + reference parity via ``flash_attn_func``."""

    native = _try_native_fa4_placeholder(q, k, v, causal=causal)
    if native is not None:
        return native
    # Reference path + framework-aligned attention labeling
    return musa_aiter.flash_attn_func(
        q,
        k,
        v,
        causal=causal,
        attention_type=attention_type,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu", help="cpu | musa[:N]")
    args = parser.parse_args()
    dev = args.device

    torch.manual_seed(0)
    q = torch.randn(1, 8, 4, 32, device=dev)
    k = torch.randn(1, 8, 4, 32, device=dev)
    v = torch.randn(1, 8, 4, 32, device=dev)

    y = flash_attn4_func_skeleton(q, k, v, causal=True)
    ref = musa_aiter.flash_attn_func(
        q, k, v, causal=True, attention_type=AttentionType.DECODER
    )
    torch.testing.assert_close(y, ref)
    print("flash_attn4_func_skeleton:", tuple(y.shape))
    print("parity_with_flash_attn_func: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
