from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .kv_transfer_plan import KVTransferPlan, validate_transfer_plan


class KVBackend(str, Enum):
    SPECIALIZED_NATIVE = "specialized_native"
    FUSED_NATIVE = "fused_native"
    DIRECT_BATCH = "direct_batch"
    REFERENCE = "reference"
    REJECTED = "rejected"


@dataclass(frozen=True)
class KVBackendCapabilities:
    specialized_native: bool = False
    fused_native: bool = False
    direct_batch: bool = False
    reference: bool = True


@dataclass(frozen=True)
class KVBackendDecision:
    backend: KVBackend
    accepted: bool
    reason: str
    requires_sync: bool


def select_kv_backend(
    plan: KVTransferPlan,
    *,
    capabilities: KVBackendCapabilities | None = None,
    require_native: bool = False,
) -> KVBackendDecision:
    validate_transfer_plan(plan)
    capabilities = capabilities or KVBackendCapabilities()

    if capabilities.specialized_native:
        return KVBackendDecision(
            backend=KVBackend.SPECIALIZED_NATIVE,
            accepted=True,
            reason="exact specialized native capability is available",
            requires_sync=False,
        )
    if capabilities.fused_native:
        return KVBackendDecision(
            backend=KVBackend.FUSED_NATIVE,
            accepted=True,
            reason="layout transform fused native capability is available",
            requires_sync=False,
        )
    if plan.direct_eligible and capabilities.direct_batch:
        return KVBackendDecision(
            backend=KVBackend.DIRECT_BATCH,
            accepted=True,
            reason="affine contiguous transfer is eligible for direct batch copy",
            requires_sync=False,
        )

    plan_reason = plan.direct_reject_reason or "no native backend capability is available"
    if require_native:
        return KVBackendDecision(
            backend=KVBackend.REJECTED,
            accepted=False,
            reason=f"native backend required; {plan_reason}",
            requires_sync=False,
        )
    if capabilities.reference:
        return KVBackendDecision(
            backend=KVBackend.REFERENCE,
            accepted=True,
            reason=f"reference fallback: {plan_reason}",
            requires_sync=False,
        )
    return KVBackendDecision(
        backend=KVBackend.REJECTED,
        accepted=False,
        reason=f"no acceptable backend: {plan_reason}",
        requires_sync=False,
    )
