#!/usr/bin/env python3
"""Micro-benchmarks for smoke / teaching operators (JIT optional via ``--prefer-native``).

See ``benchmarks/README.md`` for suite layout.
"""

from __future__ import annotations

import argparse
import json

import torch

import musa_aiter
from musa_aiter.benchmarking import benchmark_callable


def main() -> int:
    choices = ["add", "multiply", "reduce_sum", "softmax_last_dim", "rmsnorm"]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--op", choices=choices, default="add")
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--prefer-native",
        action="store_true",
        help="use native ctypes paths where implemented (needs MUSA + MUSA_AITER_NATIVE_SMOKE=1 for device)",
    )
    parser.add_argument("--json", action="store_true")
    parser.add_argument(
        "--n",
        type=int,
        default=None,
        help="element count for unary/binary elementwise-ish ops (default op-specific)",
    )
    parser.add_argument(
        "--rows",
        type=int,
        default=64,
        help="logical rows for reduce_sum / softmax / rmsnorm",
    )
    parser.add_argument(
        "--cols",
        type=int,
        default=128,
        help="logical last-dim size for softmax / rmsnorm (≤ SOFTMAX_LD_MAX_DIM / RMSNORM_NATIVE_MAX_DIM)",
    )
    args = parser.parse_args()

    if args.device == "musa":
        try:
            import torch_musa  # noqa: F401
        except Exception as exc:  # pragma: no cover
            raise SystemExit(f"torch_musa not available: {exc}")

    prefer = bool(args.prefer_native)

    def make_fn():
        n = args.n if args.n is not None else 1_048_576
        if args.op == "add":
            a = torch.randn(n, device=args.device)
            b = torch.randn_like(a)

            def _run():
                return musa_aiter.add(a, b, prefer_native=prefer)

            return _run, str(a.dtype)
        if args.op == "multiply":
            a = torch.randn(n, device=args.device)
            b = torch.randn_like(a)

            def _run():
                return musa_aiter.multiply(a, b, prefer_native=prefer)

            return _run, str(a.dtype)
        if args.op == "reduce_sum":
            r, c = max(1, args.rows), max(1, args.cols)
            x = torch.randn(r, c, device=args.device)

            def _run():
                return musa_aiter.reduce_sum(x, prefer_native=prefer)

            return _run, str(x.dtype)
        if args.op == "softmax_last_dim":
            r, c = max(1, args.rows), max(1, args.cols)
            x = torch.randn(r, c, device=args.device)

            def _run():
                return musa_aiter.softmax_last_dim(x, prefer_native=prefer)

            return _run, str(x.dtype)
        if args.op == "rmsnorm":
            r, c = max(1, args.rows), max(1, args.cols)
            x = torch.randn(r, c, device=args.device)
            w = torch.randn(c, device=args.device)

            def _run():
                return musa_aiter.rmsnorm(x, w, prefer_native=prefer)

            return _run, str(x.dtype)
        raise AssertionError(args.op)

    fn, dtype = make_fn()

    if args.iters <= 0:
        raise SystemExit("--iters must be positive")

    result = benchmark_callable(
        args.op,
        fn,
        iterations=args.iters,
        warmup=args.warmup,
        device=args.device,
        dtype=dtype,
    )
    if args.json:
        print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
        return 0

    suffix = "(prefer_native)" if prefer else "(reference)"
    print(
        f"op={args.op} device={args.device} {suffix} iters={result.iterations} "
        f"avg_ms={result.avg_ms:.3f} min_ms={result.min_ms:.3f} max_ms={result.max_ms:.3f}",
    )
    out = fn()
    if isinstance(out, torch.Tensor):
        print(f"tiny_check={float(out.flatten()[0]):.8f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
