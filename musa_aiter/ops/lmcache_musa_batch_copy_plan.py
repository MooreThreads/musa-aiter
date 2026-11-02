from __future__ import annotations

from dataclasses import dataclass
from typing import Hashable, Sequence


@dataclass(frozen=True)
class BatchCopyEntry:
    input_index: int
    source_offset: int
    destination_offset: int
    size: int
    host_registration_offset: int


@dataclass(frozen=True)
class BatchCopyPlan:
    direction: str
    entries: tuple[BatchCopyEntry, ...]
    total_bytes: int
    input_entry_count: int


def _int_values(name: str, values: Sequence[int], expected_length: int) -> tuple[int, ...]:
    if len(values) != expected_length:
        raise ValueError(f"{name} must have the same length as sizes")
    converted: list[int] = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{name} must contain integers")
        converted.append(value)
    return tuple(converted)


def _optional_int_values(
    name: str,
    values: Sequence[int] | None,
    expected_length: int,
    default: Sequence[int],
) -> tuple[int, ...]:
    return _int_values(name, default if values is None else values, expected_length)


def _keys(
    name: str,
    values: Sequence[Hashable] | None,
    expected_length: int,
) -> tuple[Hashable, ...]:
    result: tuple[Hashable, ...] = tuple(range(expected_length)) if values is None else tuple(values)
    if len(result) != expected_length:
        raise ValueError(f"{name} must have the same length as sizes")
    for value in result:
        try:
            hash(value)
        except TypeError as exc:
            raise ValueError(f"{name} must contain hashable values") from exc
    return result


def _reject_overlapping_destinations(
    destination_offsets: Sequence[int],
    sizes: Sequence[int],
    destination_keys: Sequence[Hashable],
) -> None:
    ranges_by_destination: dict[Hashable, list[tuple[int, int]]] = {}
    for offset, size, key in zip(destination_offsets, sizes, destination_keys):
        start, end = offset, offset + size
        ranges_by_destination.setdefault(key, []).append((start, end))

    for ranges in ranges_by_destination.values():
        ranges.sort(key=lambda interval: interval[0])
        for previous, current in zip(ranges, ranges[1:]):
            if current[0] < previous[1]:
                raise ValueError("overlapping destination ranges are invalid in an unordered batch")


def build_batch_copy_plan(
    *,
    direction: str,
    sizes: Sequence[int],
    source_capacities: Sequence[int],
    destination_capacities: Sequence[int],
    source_offsets: Sequence[int] | None = None,
    destination_offsets: Sequence[int] | None = None,
    host_registration_offsets: Sequence[int] | None = None,
    pin_chunk_size: int | None = None,
    destination_keys: Sequence[Hashable] | None = None,
) -> BatchCopyPlan:
    """Validate and split one unordered batch without depending on Torch."""
    if direction not in {"h2d", "d2h"}:
        raise ValueError("direction must be 'h2d' or 'd2h'")
    if pin_chunk_size is not None and (
        isinstance(pin_chunk_size, bool) or not isinstance(pin_chunk_size, int) or pin_chunk_size <= 0
    ):
        raise ValueError("pin_chunk_size must be a positive integer")

    count = len(sizes)
    normalized_sizes = _int_values("sizes", sizes, count)
    normalized_source_capacities = _int_values("source_capacities", source_capacities, count)
    normalized_destination_capacities = _int_values(
        "destination_capacities", destination_capacities, count
    )
    normalized_source_offsets = _optional_int_values("source_offsets", source_offsets, count, [0] * count)
    normalized_destination_offsets = _optional_int_values(
        "destination_offsets", destination_offsets, count, [0] * count
    )
    default_host_offsets = normalized_source_offsets if direction == "h2d" else normalized_destination_offsets
    normalized_host_offsets = _optional_int_values(
        "host_registration_offsets", host_registration_offsets, count, default_host_offsets
    )
    normalized_destination_keys = _keys("destination_keys", destination_keys, count)

    for index, size in enumerate(normalized_sizes):
        if size <= 0:
            raise ValueError(f"sizes[{index}] must be positive")
        source_offset = normalized_source_offsets[index]
        destination_offset = normalized_destination_offsets[index]
        host_offset = normalized_host_offsets[index]
        if source_offset < 0 or destination_offset < 0 or host_offset < 0:
            raise ValueError("copy offsets must be non-negative")
        if (
            normalized_destination_capacities[index] < 0
            or destination_offset + size > normalized_destination_capacities[index]
        ):
            raise ValueError(f"destination range at index {index} exceeds tensor capacity")
        if normalized_source_capacities[index] < 0 or source_offset + size > normalized_source_capacities[index]:
            raise ValueError(f"source range at index {index} exceeds tensor capacity")

    _reject_overlapping_destinations(
        normalized_destination_offsets,
        normalized_sizes,
        normalized_destination_keys,
    )

    entries: list[BatchCopyEntry] = []
    for index, size in enumerate(normalized_sizes):
        remaining = size
        source_offset = normalized_source_offsets[index]
        destination_offset = normalized_destination_offsets[index]
        host_offset = normalized_host_offsets[index]
        while remaining:
            chunk_size = remaining
            if pin_chunk_size is not None:
                until_boundary = pin_chunk_size - (host_offset % pin_chunk_size)
                chunk_size = min(chunk_size, until_boundary)
            entries.append(
                BatchCopyEntry(
                    input_index=index,
                    source_offset=source_offset,
                    destination_offset=destination_offset,
                    size=chunk_size,
                    host_registration_offset=host_offset,
                )
            )
            remaining -= chunk_size
            source_offset += chunk_size
            destination_offset += chunk_size
            host_offset += chunk_size

    return BatchCopyPlan(
        direction=direction,
        entries=tuple(entries),
        total_bytes=sum(normalized_sizes),
        input_entry_count=count,
    )


def chunk_batch_copy_plan(
    plan: BatchCopyPlan,
    max_batch_entries: int,
) -> tuple[tuple[BatchCopyEntry, ...], ...]:
    """Split an already validated plan into stream-ordered submissions."""
    if isinstance(max_batch_entries, bool) or not isinstance(max_batch_entries, int) or max_batch_entries <= 0:
        raise ValueError("max_batch_entries must be a positive integer")
    return tuple(
        tuple(plan.entries[start : start + max_batch_entries])
        for start in range(0, len(plan.entries), max_batch_entries)
    )
