from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable

import torch


def _sync_device(device: str) -> None:
    """Synchronize the accelerator implied by ``device`` (MUSA preferred over CUDA when named)."""
    if device == "cpu":
        return
    d = device.lower()
    if d.startswith("musa") and hasattr(torch, "musa"):
        torch.musa.synchronize()
    elif d.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()
    elif hasattr(torch, "musa"):
        torch.musa.synchronize()
    elif torch.cuda.is_available():
        torch.cuda.synchronize()


@dataclass(frozen=True)
class BenchmarkResult:
    name: str
    device: str
    dtype: str
    iterations: int
    avg_ms: float
    min_ms: float
    max_ms: float

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "device": self.device,
            "dtype": self.dtype,
            "iterations": self.iterations,
            "avg_ms": self.avg_ms,
            "min_ms": self.min_ms,
            "max_ms": self.max_ms,
        }


def benchmark_callable(
    name: str,
    fn: Callable[[], object],
    *,
    iterations: int = 20,
    warmup: int = 5,
    device: str = "cpu",
    dtype: str = "unknown",
) -> BenchmarkResult:
    for _ in range(warmup):
        fn()
    _sync_device(device)

    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        fn()
        _sync_device(device)
        times.append((time.perf_counter() - start) * 1000)

    return BenchmarkResult(
        name=name,
        device=device,
        dtype=dtype,
        iterations=iterations,
        avg_ms=sum(times) / len(times),
        min_ms=min(times),
        max_ms=max(times),
    )
