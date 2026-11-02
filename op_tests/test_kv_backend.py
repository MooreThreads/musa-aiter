from __future__ import annotations

import pytest

from musa_aiter.ops.kv_backend import (
    KVBackend,
    KVBackendCapabilities,
    KVBackendDecision,
    select_kv_backend,
)
from musa_aiter.ops.kv_layout import KVLayoutSpec
from musa_aiter.ops.kv_transfer_plan import KVTransferEntry, KVTransferPlan


def _plan(*, direct_eligible: bool, reason: str | None = None) -> KVTransferPlan:
    linear = KVLayoutSpec.linear(
        num_layers=1,
        num_tokens=4,
        num_heads=2,
        head_size=8,
        dtype_name="float16",
    )
    return KVTransferPlan(
        source=linear,
        destination=linear,
        direction="d2d",
        num_tokens=4,
        skip_prefix_n_tokens=0,
        entries=(KVTransferEntry(0, 0, 16),),
        stream_id=None,
        direct_eligible=direct_eligible,
        direct_reject_reason=reason,
    )


def test_specialized_native_has_priority_over_direct():
    decision = select_kv_backend(
        _plan(direct_eligible=True),
        capabilities=KVBackendCapabilities(
            specialized_native=True,
            fused_native=True,
            direct_batch=True,
            reference=True,
        ),
    )

    assert decision == KVBackendDecision(
        backend=KVBackend.SPECIALIZED_NATIVE,
        accepted=True,
        reason="exact specialized native capability is available",
        requires_sync=False,
    )


def test_direct_is_selected_only_for_direct_eligible_plan():
    decision = select_kv_backend(
        _plan(direct_eligible=True),
        capabilities=KVBackendCapabilities(direct_batch=True),
    )

    assert decision.backend is KVBackend.DIRECT_BATCH
    assert decision.accepted is True
    assert decision.requires_sync is False


def test_non_direct_plan_falls_back_with_plan_reason():
    decision = select_kv_backend(
        _plan(direct_eligible=False, reason="layout transform is required"),
        capabilities=KVBackendCapabilities(direct_batch=True, reference=True),
    )

    assert decision.backend is KVBackend.REFERENCE
    assert decision.accepted is True
    assert "layout transform is required" in decision.reason


def test_require_native_rejects_when_no_native_backend_is_available():
    decision = select_kv_backend(
        _plan(direct_eligible=True),
        capabilities=KVBackendCapabilities(reference=True),
        require_native=True,
    )

    assert decision.backend is KVBackend.REJECTED
    assert decision.accepted is False
    assert "native backend required" in decision.reason


def test_invalid_plan_is_rejected_before_backend_selection():
    invalid = _plan(direct_eligible=True)
    invalid = KVTransferPlan(
        source=invalid.source,
        destination=invalid.destination,
        direction="d2d",
        num_tokens=4,
        skip_prefix_n_tokens=5,
        entries=invalid.entries,
        stream_id=None,
        direct_eligible=True,
        direct_reject_reason=None,
    )

    with pytest.raises(ValueError, match="skip_prefix_n_tokens"):
        select_kv_backend(invalid)


def test_lmcache_batch_copy_wrapper_uses_plan_selector():
    from musa_aiter.ops.lmcache_musa_batch_copy import (
        select_lmcache_batch_copy_backend,
    )

    decision = select_lmcache_batch_copy_backend(
        _plan(direct_eligible=True),
        native_available=True,
    )

    assert decision.backend is KVBackend.DIRECT_BATCH
    assert decision.accepted is True


def test_public_package_exports_backend_selector():
    import musa_aiter

    assert musa_aiter.KVBackend is KVBackend
    assert musa_aiter.select_kv_backend is select_kv_backend
