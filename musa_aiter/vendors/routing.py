"""Resolve dispatch labels when a vendor library is present (Phase P7).

``MUSA_AITER_VENDOR_BACKEND`` selects a catalog id (for example ``mudnn``) or
``auto`` to accept any target listed on the operator capability. When
``MUSA_AITER_VENDOR_LIB`` loads successfully, the primary dispatch line switches
to ``torch-reference+vendors``; math still runs through PyTorch until vendor
symbols are bound per operator.
"""

from __future__ import annotations

import os

from musa_aiter.registry import get
from musa_aiter.vendors.loader import get_vendor_library


def resolve_dispatch_backend(op: str, default_label: str = "torch-reference") -> tuple[str, str]:
    """Return ``(emit_backend_label, reason_suffix)``.

    *reason_suffix* is empty or starts with ``"; "`` for trace-friendly metadata.
    """

    req = os.environ.get("MUSA_AITER_VENDOR_BACKEND", "").strip().lower()
    if not req:
        return default_label, ""
    try:
        cap = get(op)
    except KeyError:
        return default_label, "; vendor-unknown-op"
    targets = tuple(v.lower() for v in cap.vendor_backends)
    if not targets:
        return default_label, "; vendor-not-applicable"
    if req != "auto" and req not in targets:
        return default_label, "; vendor-request-not-in-matrix"
    if get_vendor_library() is None:
        return default_label, "; vendor-lib-missing"
    return f"{default_label}+vendors", "; vendor-lib-routing-active"
