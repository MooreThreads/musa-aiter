from __future__ import annotations

import pytest

from musa_aiter.ops.kv_layout import KVLayoutSpec
from musa_aiter.ops.kv_transfer_plan import (
    KVTransferEntry,
    KVTransferPlan,
    assess_direct_eligibility,
    validate_transfer_plan,
)


def _paged() -> KVLayoutSpec:
    return KVLayoutSpec.standard_paged(
        num_layers=1,
        num_blocks=4,
        block_size=16,
        num_heads=2,
        head_size=8,
        dtype_name="float16",
        row_bytes=32,
        block_stride_bytes=512,
    )


def _linear() -> KVLayoutSpec:
    return KVLayoutSpec.linear(
        num_layers=1,
        num_tokens=32,
        num_heads=2,
        head_size=8,
        dtype_name="float16",
    )


def _plan(*entries: KVTransferEntry, direct_eligible: bool = False, reason: str | None = None):
    return KVTransferPlan(
        source=_paged(),
        destination=_linear(),
        direction="d2d",
        num_tokens=32,
        skip_prefix_n_tokens=0,
        entries=tuple(entries),
        stream_id=None,
        direct_eligible=direct_eligible,
        direct_reject_reason=reason,
    )


def test_validate_transfer_plan_accepts_non_overlapping_entries():
    plan = _plan(
        KVTransferEntry(0, 0, 16, layer=0, block=0),
        KVTransferEntry(32, 16, 16, layer=0, block=1),
    )

    assert validate_transfer_plan(plan) is plan


def test_plan_rejects_overlapping_destinations():
    plan = _plan(
        KVTransferEntry(0, 0, 16),
        KVTransferEntry(32, 8, 16),
    )

    with pytest.raises(ValueError, match="destination.*overlap"):
        validate_transfer_plan(plan)


def test_plan_rejects_prefix_larger_than_token_count():
    plan = KVTransferPlan(
        source=_paged(),
        destination=_linear(),
        direction="d2d",
        num_tokens=4,
        skip_prefix_n_tokens=5,
        entries=(),
        stream_id=None,
        direct_eligible=False,
        direct_reject_reason="empty",
    )

    with pytest.raises(ValueError, match="skip_prefix_n_tokens"):
        validate_transfer_plan(plan)


def test_page_first_plan_is_not_direct_eligible():
    source = KVLayoutSpec.page_first(
        num_layers=1,
        num_blocks=4,
        block_size=16,
        num_heads=2,
        head_size=8,
        dtype_name="float16",
        row_bytes=32,
        block_stride_bytes=512,
    )
    destination = _linear()
    eligible, reason = assess_direct_eligibility(source, destination)

    assert eligible is False
    assert "layout transform" in reason


def test_same_geometry_can_use_direct_batch_copy():
    source = _linear()
    destination = KVLayoutSpec.linear(
        num_layers=1,
        num_tokens=32,
        num_heads=2,
        head_size=8,
        dtype_name="float16",
    )

    eligible, reason = assess_direct_eligibility(source, destination)

    assert eligible is True
    assert reason is None


def test_direct_plan_requires_reason_when_rejected():
    plan = _plan(direct_eligible=False)

    assert validate_transfer_plan(plan) is plan
