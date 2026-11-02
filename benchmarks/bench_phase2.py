from __future__ import annotations

import argparse
import json

import torch

import musa_aiter
from musa_aiter.benchmarking import benchmark_callable


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--op", choices=["gemm", "grouped_gemm", "fused_moe"], default="gemm")
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    device = args.device

    if args.op == "gemm":
        a = torch.randn(256, 256, device=device)
        b = torch.randn(256, 256, device=device)
        fn = lambda: musa_aiter.gemm(a, b)
        dtype = str(a.dtype)
    elif args.op == "grouped_gemm":
        a = torch.randn(512, 128, device=device)
        b = torch.randn(4, 128, 256, device=device)
        idx = torch.randint(0, 4, (512,), device=device)
        fn = lambda: musa_aiter.grouped_gemm(a, b, idx)
        dtype = str(a.dtype)
    else:
        tokens, hidden, inter, experts = 128, 64, 128, 4
        x = torch.randn(tokens, hidden, device=device)
        w1 = torch.randn(experts, inter * 2, hidden, device=device)
        w2 = torch.randn(experts, hidden, inter, device=device)
        tw = torch.randn(tokens, 2, device=device)
        ti = torch.randint(0, experts, (tokens, 2), device=device)
        fn = lambda: musa_aiter.fused_moe(x, w1, w2, tw, ti)
        dtype = str(x.dtype)

    result = benchmark_callable(args.op, fn, iterations=args.iters, device=device, dtype=dtype)
    if args.json:
        print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
    else:
        print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
