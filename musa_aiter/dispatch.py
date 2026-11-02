from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class DispatchRecord:
    op: str
    backend: str
    device: str
    dtype: str
    reason: str


def trace_enabled() -> bool:
    return os.environ.get("MUSA_AITER_TRACE_DISPATCH", "0") in {"1", "true", "TRUE", "yes", "YES"}


def emit(record: DispatchRecord) -> None:
    if not trace_enabled():
        return
    print(
        "[MUSA_AITER DISPATCH] "
        f"op={record.op} backend={record.backend} device={record.device} "
        f"dtype={record.dtype} reason={record.reason}",
        file=sys.stderr,
    )


def tensor_device_dtype(x: Any) -> tuple[str, str]:
    device = getattr(x, "device", "unknown")
    dtype = getattr(x, "dtype", "unknown")
    return str(device), str(dtype)
