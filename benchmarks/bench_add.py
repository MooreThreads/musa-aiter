"""Micro-benchmark for ``musa_aiter.add``. For the unified smoke suite see ``benchmarks/bench_smoke.py`` and ``benchmarks/README.md``."""

from __future__ import annotations

import argparse
import time

import torch

import musa_aiter


def _sync(device: str) -> None:
    if device == "cpu":
        return
    d = str(device).lower()
    if d.startswith("musa") and hasattr(torch, "musa"):
        torch.musa.synchronize()
    elif d.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()
    elif hasattr(torch, "musa"):
        torch.musa.synchronize()
    elif torch.cuda.is_available():
        torch.cuda.synchronize()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=1_000_000)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    device = args.device
    if device == "musa":
        try:
            import torch_musa  # noqa: F401
        except Exception as exc:
            raise SystemExit(f"torch_musa not available: {exc}")

    a = torch.randn(args.n, device=device)
    b = torch.randn(args.n, device=device)

    for _ in range(5):
        musa_aiter.add(a, b, prefer_native=False)

    _sync(device)
    start = time.perf_counter()
    for _ in range(args.iters):
        out = musa_aiter.add(a, b, prefer_native=False)
    _sync(device)
    elapsed = time.perf_counter() - start
    print(f"n={args.n} iters={args.iters} device={device} avg_ms={elapsed / args.iters * 1000:.3f}")
    print(float(out[0]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
