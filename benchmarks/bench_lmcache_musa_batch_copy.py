from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from collections.abc import Callable
from typing import Any

import torch

import musa_aiter
from musa_aiter.ops.lmcache_musa_batch_copy_ext import load_lmcache_musa_batch_copy_ext
from musa_aiter.ops.lmcache_musa_batch_copy_plan import build_batch_copy_plan


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, int((len(ordered) - 1) * fraction))
    return ordered[index]


def _measure(
    fn: Callable[[], None], *, warmup: int, iterations: int
) -> list[dict[str, float]]:
    for _ in range(warmup):
        fn()
    torch.musa.synchronize()

    samples: list[dict[str, float]] = []
    for _ in range(iterations):
        start = time.perf_counter()
        fn()
        enqueue_end = time.perf_counter()
        torch.musa.synchronize()
        end = time.perf_counter()
        samples.append(
            {
                "enqueue_ms": (enqueue_end - start) * 1_000,
                "wait_ms": (end - enqueue_end) * 1_000,
                "total_ms": (end - start) * 1_000,
            }
        )
    return samples


def _summary(samples: list[dict[str, float]], metric: str) -> dict[str, float]:
    values = [sample[metric] for sample in samples]
    return {
        f"avg_{metric}": statistics.fmean(values),
        f"p50_{metric}": _percentile(values, 0.50),
        f"p95_{metric}": _percentile(values, 0.95),
    }


def _result(
    name: str, samples: list[dict[str, float]], total_bytes: int
) -> dict[str, float | str]:
    result: dict[str, float | str] = {"backend": name}
    for metric in ("enqueue_ms", "wait_ms", "total_ms"):
        result.update(_summary(samples, metric))
    average_total_ms = float(result["avg_total_ms"])
    result["effective_gbps"] = total_bytes / (average_total_ms / 1_000) / 1_000_000_000
    # Keep avg_ms as a compact compatibility field for existing report parsers.
    result["avg_ms"] = average_total_ms
    return result


def _run_direction(
    direction: str,
    *,
    extension: Any,
    entry_count: int,
    bytes_per_entry: int,
    warmup: int,
    iterations: int,
    pin_chunk_size: int | None,
    max_batch_entries: int,
    prefer_overlap_with_compute: bool,
) -> dict[str, object]:
    total_bytes = entry_count * bytes_per_entry
    host = torch.empty(total_bytes, dtype=torch.uint8, pin_memory=True).random_()
    device = torch.empty(total_bytes, dtype=torch.uint8, device="musa")
    offsets = [index * bytes_per_entry for index in range(entry_count)]
    sizes = [bytes_per_entry] * entry_count

    if direction == "h2d":
        destinations, sources = [device] * entry_count, [host] * entry_count
    else:
        device.copy_(host, non_blocking=True)
        torch.musa.synchronize()
        host.zero_()
        destinations, sources = [host] * entry_count, [device] * entry_count

    native_plan = build_batch_copy_plan(
        direction=direction,
        sizes=sizes,
        source_capacities=[total_bytes] * entry_count,
        destination_capacities=[total_bytes] * entry_count,
        source_offsets=offsets,
        destination_offsets=offsets,
        host_registration_offsets=offsets,
        pin_chunk_size=pin_chunk_size,
        destination_keys=[id(destinations[0])] * entry_count,
    )
    native_destinations = [destinations[entry.input_index] for entry in native_plan.entries]
    native_sources = [sources[entry.input_index] for entry in native_plan.entries]
    native_destination_offsets = [entry.destination_offset for entry in native_plan.entries]
    native_source_offsets = [entry.source_offset for entry in native_plan.entries]
    native_sizes = [entry.size for entry in native_plan.entries]

    def native_batch_copy() -> None:
        for start in range(0, len(native_plan.entries), max_batch_entries):
            end = min(start + max_batch_entries, len(native_plan.entries))
            submitted = extension.execute_batch_copy(
                native_destinations[start:end],
                native_sources[start:end],
                native_destination_offsets[start:end],
                native_source_offsets[start:end],
                native_sizes[start:end],
                prefer_overlap_with_compute,
            )
            if not submitted:
                raise RuntimeError("MUSA batch-copy backend became unavailable")

    def facade_batch_copy() -> None:
        submitted = musa_aiter.lmcache_musa_memcpy_batch_async(
            destinations,
            sources,
            sizes,
            destination_offsets=offsets,
            source_offsets=offsets,
            pin_chunk_size=pin_chunk_size,
            prefer_overlap_with_compute=prefer_overlap_with_compute,
            max_batch_entries=max_batch_entries,
        )
        if not submitted:
            raise RuntimeError("MUSA batch-copy facade became unavailable")

    def native_copy_loop() -> None:
        if not extension.execute_copy_loop(
            native_destinations,
            native_sources,
            native_destination_offsets,
            native_source_offsets,
            native_sizes,
        ):
            raise RuntimeError("MUSA memcpyAsync loop became unavailable")

    def check_result() -> None:
        actual = device.cpu() if direction == "h2d" else host
        expected = host if direction == "h2d" else device.cpu()
        if not torch.equal(actual, expected):
            raise RuntimeError(f"{direction} correctness check failed")

    for method in (native_batch_copy, facade_batch_copy, native_copy_loop):
        method()
        torch.musa.synchronize()
        check_result()

    measurements = [
        _result(
            "musaMemcpyBatchAsync_native",
            _measure(native_batch_copy, warmup=warmup, iterations=iterations),
            total_bytes,
        ),
        _result(
            "musaMemcpyAsync_loop_native",
            _measure(native_copy_loop, warmup=warmup, iterations=iterations),
            total_bytes,
        ),
        _result(
            "lmcache_batch_facade",
            _measure(facade_batch_copy, warmup=warmup, iterations=iterations),
            total_bytes,
        ),
    ]
    return {
        "direction": direction,
        "entry_count": entry_count,
        "bytes_per_entry": bytes_per_entry,
        "total_bytes": total_bytes,
        "submitted_entry_count": len(native_plan.entries),
        "pin_chunk_size": pin_chunk_size,
        "max_batch_entries": max_batch_entries,
        "prefer_overlap_with_compute": prefer_overlap_with_compute,
        "warmup": warmup,
        "iterations": iterations,
        "measurements": measurements,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Benchmark the optional LMCache MUSA batch-copy backend"
    )
    parser.add_argument("--direction", choices=("h2d", "d2h", "both"), default="both")
    parser.add_argument("--entries", type=int, default=128)
    parser.add_argument("--bytes-per-entry", type=int, default=256 * 1024)
    parser.add_argument("--pin-chunk-size", type=int, default=0)
    parser.add_argument("--max-batch-entries", type=int, default=1024)
    parser.add_argument("--prefer-overlap-with-compute", action="store_true")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    args = parser.parse_args()

    if args.entries <= 0 or args.bytes_per_entry <= 0 or args.warmup < 0 or args.iters <= 0:
        parser.error(
            "entries, bytes-per-entry, and iters must be positive; warmup must be non-negative"
        )
    if args.pin_chunk_size < 0 or args.max_batch_entries <= 0:
        parser.error("pin-chunk-size must be non-negative and max-batch-entries must be positive")
    if not hasattr(torch, "musa") or not torch.musa.is_available():
        parser.error("TorchMUSA with a visible MUSA device is required")
    if os.environ.get("MUSA_AITER_LMCACHE_BATCH_COPY", "0") != "1" and os.environ.get(
        "LMCACHE_MUSA_DIRECT_COPY", "0"
    ) != "1":
        parser.error("set MUSA_AITER_LMCACHE_BATCH_COPY=1")
    if not musa_aiter.is_native_lmcache_musa_batch_copy_available():
        parser.error("MUSA SDK 5.1+ batch-copy extension is unavailable")
    extension = load_lmcache_musa_batch_copy_ext()
    if extension is None or not hasattr(extension, "execute_copy_loop"):
        parser.error("native musaMemcpyAsync loop baseline is unavailable")

    directions = ("h2d", "d2h") if args.direction == "both" else (args.direction,)
    results = [
        _run_direction(
            direction,
            extension=extension,
            entry_count=args.entries,
            bytes_per_entry=args.bytes_per_entry,
            warmup=args.warmup,
            iterations=args.iters,
            pin_chunk_size=args.pin_chunk_size or None,
            max_batch_entries=args.max_batch_entries,
            prefer_overlap_with_compute=args.prefer_overlap_with_compute,
        )
        for direction in directions
    ]
    print(
        json.dumps(
            {
                "runtime_versions": musa_aiter.native_lmcache_musa_batch_copy_runtime_versions(),
                "results": results,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
