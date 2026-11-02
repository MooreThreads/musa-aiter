#!/usr/bin/env python3
"""Small K1-K6 KV transfer matrix benchmark.

The benchmark intentionally reports the selected implementation family and
does not claim native performance unless the requested device and native
capability are both available.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from typing import Any

import torch

import musa_aiter


def _sync(device: torch.device) -> None:
    if device.type == "musa":
        torch.musa.synchronize()


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.device.startswith("musa"):
        if not hasattr(torch, "musa") or not torch.musa.is_available():
            raise RuntimeError("requested MUSA benchmark but TorchMUSA is unavailable")
        device = torch.device(args.device)
    else:
        device = torch.device("cpu")

    dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16
    source = torch.randn(
        args.layers,
        args.pages,
        args.item_elements,
        dtype=dtype,
        device=device,
    )
    destination = torch.zeros(
        args.pages,
        args.layers,
        args.item_elements,
        dtype=dtype,
        device=device,
    )
    value_source = torch.randn_like(source)
    value_destination = torch.zeros_like(destination)
    source_indices = torch.arange(args.tokens, dtype=torch.int64, device=device) % args.pages
    destination_indices = torch.arange(args.tokens, dtype=torch.int64, device=device) % args.pages

    for _ in range(args.warmup):
        musa_aiter.hicache_transfer_kv_all_layer_lf_pf(
            source,
            destination,
            value_source,
            value_destination,
            source_indices,
            destination_indices,
        )
    _sync(device)

    start = time.perf_counter()
    for _ in range(args.iters):
        musa_aiter.hicache_transfer_kv_all_layer_lf_pf(
            source,
            destination,
            value_source,
            value_destination,
            source_indices,
            destination_indices,
        )
    _sync(device)
    elapsed = (time.perf_counter() - start) / args.iters

    item_bytes = args.item_elements * source.element_size()
    transferred_bytes = args.tokens * args.layers * item_bytes * 2
    native = (
        device.type == "musa"
        and os.environ.get("MUSA_AITER_NATIVE_HICACHE_TRANSFER") == "1"
        and musa_aiter.is_native_hicache_transfer_available()
    )
    return {
        "op": "hicache_transfer_kv_all_layer_lf_pf",
        "device": str(device),
        "dtype": args.dtype,
        "layers": args.layers,
        "pages": args.pages,
        "tokens": args.tokens,
        "item_elements": args.item_elements,
        "backend": "hicache-musa-native" if native else "torch-reference",
        "latency_us": elapsed * 1e6,
        "bytes": transferred_bytes,
        "bandwidth_gbps": transferred_bytes / elapsed / 1e9,
        "native_capability": bool(
            device.type == "musa" and musa_aiter.is_native_hicache_transfer_available()
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", choices=("float16", "bfloat16"), default="float16")
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--pages", type=int, default=256)
    parser.add_argument("--tokens", type=int, default=128)
    parser.add_argument("--item-elements", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    result = run(args)
    print(json.dumps(result, sort_keys=True) if args.json else result)


if __name__ == "__main__":
    main()
