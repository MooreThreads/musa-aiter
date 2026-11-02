from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from .kv_layout import KVLayoutKind, KVLayoutSpec, KVScaleKind

TransferDirection = Literal["h2d", "d2h", "d2d"]


@dataclass(frozen=True)
class KVTransferEntry:
    src_offset_bytes: int
    dst_offset_bytes: int
    nbytes: int
    layer: int = 0
    block: int = 0
    kv_plane: int = 0


@dataclass(frozen=True)
class KVTransferPlan:
    source: KVLayoutSpec
    destination: KVLayoutSpec
    direction: TransferDirection
    num_tokens: int
    skip_prefix_n_tokens: int
    entries: tuple[KVTransferEntry, ...]
    stream_id: int | None
    direct_eligible: bool
    direct_reject_reason: str | None


def _ranges_overlap(left: tuple[int, int], right: tuple[int, int]) -> bool:
    return left[0] < right[1] and right[0] < left[1]


def validate_transfer_plan(plan: KVTransferPlan) -> KVTransferPlan:
    if plan.direction not in {"h2d", "d2h", "d2d"}:
        raise ValueError(f"unsupported transfer direction={plan.direction!r}")
    if plan.num_tokens < 0:
        raise ValueError("num_tokens must be non-negative")
    if not 0 <= plan.skip_prefix_n_tokens <= plan.num_tokens:
        raise ValueError("skip_prefix_n_tokens must be within num_tokens")
    if plan.stream_id is not None and plan.stream_id < 0:
        raise ValueError("stream_id must be non-negative")

    destination_ranges: list[tuple[int, int]] = []
    for entry in plan.entries:
        if entry.src_offset_bytes < 0 or entry.dst_offset_bytes < 0:
            raise ValueError("transfer offsets must be non-negative")
        if entry.nbytes <= 0:
            raise ValueError("transfer nbytes must be positive")
        if entry.layer < 0 or entry.block < 0 or entry.kv_plane < 0:
            raise ValueError("transfer indices must be non-negative")
        destination_ranges.append(
            (entry.dst_offset_bytes, entry.dst_offset_bytes + entry.nbytes)
        )

    destination_ranges.sort()
    for previous, current in zip(destination_ranges, destination_ranges[1:]):
        if _ranges_overlap(previous, current):
            raise ValueError("destination transfer ranges overlap")
    return plan


def assess_direct_eligibility(
    source: KVLayoutSpec, destination: KVLayoutSpec
) -> tuple[bool, str | None]:
    if source.requires_layout_transform or destination.requires_layout_transform:
        return False, "layout transform is required"
    if not source.is_contiguous or not destination.is_contiguous:
        return False, "layout is not contiguous"
    if source.dtype_name != destination.dtype_name:
        return False, "dtype transform is required"
    if source.organization != destination.organization:
        return False, "K/V organization transform is required"
    if source.scale_kind is not KVScaleKind.NONE or destination.scale_kind is not KVScaleKind.NONE:
        return False, "scale transform is required"
    if source.kind in {
        KVLayoutKind.PAGE_FIRST,
        KVLayoutKind.PAGE_FIRST_DIRECT,
        KVLayoutKind.PAGE_HEAD,
    } or destination.kind in {
        KVLayoutKind.PAGE_FIRST,
        KVLayoutKind.PAGE_FIRST_DIRECT,
        KVLayoutKind.PAGE_HEAD,
    }:
        return False, "page layout transform is required"
    return True, None
