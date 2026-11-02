from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict, dataclass

import torch

from musa_aiter.ops.paged_sparse_attention import paged_sparse_decode_attention


@dataclass
class BenchCase:
    mode: str
    batch: int
    q_heads: int
    kv_heads: int
    head_dim: int
    seq_len: int
    dtype: str
    warmup: int
    iters: int
    mean_ms: float
    p50_ms: float
    p95_ms: float


def _require_musa_device() -> torch.device:
    if not hasattr(torch, "musa") or not torch.musa.is_available():
        raise RuntimeError("TorchMUSA is not available")
    return torch.device("musa:0")


def _sync() -> None:
    if hasattr(torch, "musa"):
        torch.musa.synchronize()


def _dtype(name: str) -> torch.dtype:
    if name == "float32":
        return torch.float32
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    raise ValueError(f"unsupported dtype: {name}")


def run_case(*, mode: str, batch: int, q_heads: int, kv_heads: int, head_dim: int, seq_len: int, dtype_name: str, warmup: int, iters: int, tile_size: int) -> BenchCase:
    device = _require_musa_device()
    if mode == "tiled":
        os.environ["MUSA_AITER_PAGED_SPARSE_ATTENTION_TILED"] = "1"
        os.environ["MUSA_AITER_PAGED_SPARSE_ATTENTION_TILE_SIZE"] = str(tile_size)
    else:
        os.environ.pop("MUSA_AITER_PAGED_SPARSE_ATTENTION_TILED", None)
    dtype = _dtype(dtype_name)
    total_tokens = batch * seq_len
    q = torch.randn((batch, q_heads, head_dim), device=device, dtype=dtype)
    k = torch.randn((total_tokens, kv_heads, head_dim), device=device, dtype=dtype)
    v = torch.randn_like(k)
    kv_indptr = torch.arange(0, total_tokens + 1, seq_len, device=device, dtype=torch.long)
    kv_indices = torch.arange(total_tokens, device=device, dtype=torch.long)

    for _ in range(warmup):
        paged_sparse_decode_attention(q, k, v, kv_indptr, kv_indices, max_seq_len=seq_len, tile_size=tile_size)
    _sync()

    times = []
    for _ in range(iters):
        start = time.perf_counter()
        paged_sparse_decode_attention(q, k, v, kv_indptr, kv_indices, max_seq_len=seq_len, tile_size=tile_size)
        _sync()
        times.append((time.perf_counter() - start) * 1000.0)

    ordered = sorted(times)
    return BenchCase(
        mode=mode,
        batch=batch,
        q_heads=q_heads,
        kv_heads=kv_heads,
        head_dim=head_dim,
        seq_len=seq_len,
        dtype=dtype_name,
        warmup=warmup,
        iters=iters,
        mean_ms=sum(times) / len(times),
        p50_ms=ordered[len(ordered) // 2],
        p95_ms=ordered[max(0, int(len(ordered) * 0.95) - 1)],
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dtype", default="bfloat16", choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--batch", default="1,4,8")
    parser.add_argument("--seq-len", default="128,1024,4096")
    parser.add_argument("--q-heads", type=int, default=64)
    parser.add_argument("--kv-heads", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--mode", default="block", choices=["block", "tiled", "both"])
    parser.add_argument("--tile-size", type=int, default=512)
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    rows = []
    modes = ["block", "tiled"] if args.mode == "both" else [args.mode]
    for mode in modes:
        for batch in [int(x) for x in args.batch.split(",") if x]:
            for seq_len in [int(x) for x in args.seq_len.split(",") if x]:
                row = run_case(
                    mode=mode,
                    batch=batch,
                    q_heads=args.q_heads,
                    kv_heads=args.kv_heads,
                    head_dim=args.head_dim,
                    seq_len=seq_len,
                    dtype_name=args.dtype,
                    warmup=args.warmup,
                    iters=args.iters,
                    tile_size=args.tile_size,
                )
                payload = asdict(row)
                print(json.dumps(payload, ensure_ascii=False))
                rows.append(payload)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(rows, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
