from __future__ import annotations

import os
from typing import Protocol, Sequence, cast

import torch

from .lmcache_musa_batch_copy_ext import (
    lmcache_musa_batch_copy_requested,
    load_lmcache_musa_batch_copy_ext,
)
from .lmcache_musa_batch_copy_plan import build_batch_copy_plan as _build_batch_copy_plan
from .lmcache_musa_batch_copy_plan import chunk_batch_copy_plan as _chunk_batch_copy_plan
from .kv_backend import KVBackendCapabilities, KVBackendDecision, select_kv_backend
from .kv_transfer_plan import KVTransferPlan

_ABI_VERSION = 1
_DEFAULT_MAX_BATCH_ENTRIES = 1024


class _BatchCopyExtension(Protocol):
    def abi_version(self) -> int: ...

    def runtime_version(self) -> int: ...

    def driver_version(self) -> int: ...

    def batch_memcpy_supported(self) -> bool: ...

    def execute_batch_copy(
        self,
        destinations: Sequence[torch.Tensor],
        sources: Sequence[torch.Tensor],
        destination_offsets: Sequence[int],
        source_offsets: Sequence[int],
        sizes: Sequence[int],
        prefer_overlap_with_compute: bool,
    ) -> bool: ...


def native_lmcache_musa_batch_copy_abi_version() -> int:
    return _ABI_VERSION


def native_lmcache_musa_batch_copy_runtime_versions() -> tuple[int, int] | None:
    """Return (runtime, driver) versions from the optional native module."""
    if not lmcache_musa_batch_copy_requested():
        return None
    extension = load_lmcache_musa_batch_copy_ext()
    if extension is None:
        return None
    candidate = cast(_BatchCopyExtension, extension)
    try:
        if candidate.abi_version() != _ABI_VERSION:
            return None
        return candidate.runtime_version(), candidate.driver_version()
    except Exception:
        return None


def is_native_lmcache_musa_batch_copy_available() -> bool:
    if not lmcache_musa_batch_copy_requested():
        return False
    return _compatible_extension(load_lmcache_musa_batch_copy_ext()) is not None


def select_lmcache_batch_copy_backend(
    plan: KVTransferPlan,
    *,
    native_available: bool | None = None,
    require_native: bool = False,
) -> KVBackendDecision:
    """Classify an LMCache plan before submitting byte copies.

    The planner remains responsible for producing affine byte ranges. This
    helper only connects that plan to the common backend selector; it does not
    submit work or force-load the optional native extension when the caller
    already knows its capability.
    """
    if native_available is None:
        native_available = is_native_lmcache_musa_batch_copy_available()
    return select_kv_backend(
        plan,
        capabilities=KVBackendCapabilities(
            direct_batch=bool(native_available),
            reference=True,
        ),
        require_native=require_native,
    )


def _compatible_extension(extension: object | None) -> _BatchCopyExtension | None:
    if extension is None:
        return None
    candidate = cast(_BatchCopyExtension, extension)
    try:
        if candidate.abi_version() != _ABI_VERSION or not candidate.batch_memcpy_supported():
            return None
    except Exception:
        return None
    return candidate


def _tensor_device_kind(tensor: torch.Tensor) -> str:
    device_type = tensor.device.type.lower()
    if device_type == "cpu":
        return "cpu"
    if device_type in {"musa", "privateuseone"}:
        return "musa"
    return "unsupported"


def _tensor_capacity_bytes(tensor: torch.Tensor) -> int:
    return tensor.numel() * tensor.element_size()


def _musa_device_key(tensor: torch.Tensor) -> tuple[str, int | None]:
    return (tensor.device.type, tensor.device.index)


def lmcache_musa_memcpy_batch_async(
    destinations: Sequence[torch.Tensor],
    sources: Sequence[torch.Tensor],
    sizes: Sequence[int],
    *,
    destination_offsets: Sequence[int] | None = None,
    source_offsets: Sequence[int] | None = None,
    host_registration_offsets: Sequence[int] | None = None,
    pin_chunk_size: int | None = None,
    require_pinned_host: bool = True,
    prefer_overlap_with_compute: bool = False,
    max_batch_entries: int | None = None,
) -> bool:
    """Submit an opt-in host/device byte-copy batch on the current MUSA stream.

    ``True`` means the batch was submitted (or was empty). ``False`` means the
    optional backend was disabled or unavailable, allowing the caller to use
    its existing transfer path. ``max_batch_entries`` is a conservative
    per-submission cap; oversized plans are submitted as ordered chunks on the
    same current stream. LMCache callers should use the internal wrapper so
    host registration boundaries cannot be omitted.
    """
    destinations = tuple(destinations)
    sources = tuple(sources)
    sizes = tuple(sizes)
    count = len(sizes)
    if not destinations and not sources and not sizes:
        return True
    if len(destinations) != count or len(sources) != count:
        raise ValueError("destinations, sources, and sizes must have the same length")

    directions: list[str] = []
    musa_devices: set[tuple[str, int | None]] = set()
    for index, (destination, source) in enumerate(zip(destinations, sources)):
        if not isinstance(destination, torch.Tensor) or not isinstance(source, torch.Tensor):
            raise TypeError("destinations and sources must contain torch.Tensor values")
        if not destination.is_contiguous() or not source.is_contiguous():
            raise ValueError("batch-copy tensors must be contiguous")
        destination_kind = _tensor_device_kind(destination)
        source_kind = _tensor_device_kind(source)
        if (source_kind, destination_kind) == ("cpu", "musa"):
            direction, host_tensor, musa_tensor = "h2d", source, destination
        elif (source_kind, destination_kind) == ("musa", "cpu"):
            direction, host_tensor, musa_tensor = "d2h", destination, source
        else:
            return False
        if require_pinned_host and not host_tensor.is_pinned():
            raise ValueError(f"host tensor at index {index} must use pinned memory")
        directions.append(direction)
        musa_devices.add(_musa_device_key(musa_tensor))

    if len(set(directions)) != 1:
        raise ValueError("one batch cannot mix h2d and d2h copies")
    if len(musa_devices) != 1:
        raise ValueError("all batch entries must use the same MUSA device")

    plan = _build_batch_copy_plan(
        direction=directions[0],
        sizes=sizes,
        source_capacities=[_tensor_capacity_bytes(tensor) for tensor in sources],
        destination_capacities=[_tensor_capacity_bytes(tensor) for tensor in destinations],
        source_offsets=source_offsets,
        destination_offsets=destination_offsets,
        host_registration_offsets=host_registration_offsets,
        pin_chunk_size=pin_chunk_size,
        destination_keys=[id(tensor) for tensor in destinations],
    )

    if not lmcache_musa_batch_copy_requested():
        return False
    extension = _compatible_extension(load_lmcache_musa_batch_copy_ext())
    if extension is None:
        return False

    batch_limit = _resolve_max_batch_entries(max_batch_entries)
    chunks = _chunk_batch_copy_plan(plan, batch_limit)
    for chunk_index, entries in enumerate(chunks):
        expanded_destinations = [destinations[entry.input_index] for entry in entries]
        expanded_sources = [sources[entry.input_index] for entry in entries]
        submitted = extension.execute_batch_copy(
            expanded_destinations,
            expanded_sources,
            [entry.destination_offset for entry in entries],
            [entry.source_offset for entry in entries],
            [entry.size for entry in entries],
            prefer_overlap_with_compute,
        )
        if not submitted:
            if chunk_index:
                raise RuntimeError(
                    "MUSA batch-copy chunk failed after an earlier chunk was submitted"
                )
            return False
    return True


def lmcache_musa_memcpy_batch_async_internal(
    destinations: Sequence[torch.Tensor],
    sources: Sequence[torch.Tensor],
    sizes: Sequence[int],
    *,
    host_registration_offsets: Sequence[int],
    pin_chunk_size: int,
    destination_offsets: Sequence[int] | None = None,
    source_offsets: Sequence[int] | None = None,
    require_pinned_host: bool = True,
    prefer_overlap_with_compute: bool = False,
    max_batch_entries: int | None = None,
) -> bool:
    """LMCache-only facade requiring an explicit host registration boundary."""
    if host_registration_offsets is None or pin_chunk_size is None:
        raise ValueError("LMCache internal API requires a host registration boundary")
    return lmcache_musa_memcpy_batch_async(
        destinations,
        sources,
        sizes,
        destination_offsets=destination_offsets,
        source_offsets=source_offsets,
        host_registration_offsets=host_registration_offsets,
        pin_chunk_size=pin_chunk_size,
        require_pinned_host=require_pinned_host,
        prefer_overlap_with_compute=prefer_overlap_with_compute,
        max_batch_entries=max_batch_entries,
    )


def _resolve_max_batch_entries(value: int | None) -> int:
    if value is not None:
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError("max_batch_entries must be a positive integer")
        return value

    configured = os.environ.get(
        "MUSA_AITER_LMCACHE_BATCH_MAX_ENTRIES",
        str(_DEFAULT_MAX_BATCH_ENTRIES),
    )
    try:
        configured_value = int(configured)
    except ValueError as exc:
        raise ValueError("MUSA_AITER_LMCACHE_BATCH_MAX_ENTRIES must be an integer") from exc
    if configured_value <= 0:
        raise ValueError("MUSA_AITER_LMCACHE_BATCH_MAX_ENTRIES must be positive")
    return configured_value
