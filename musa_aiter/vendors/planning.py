"""Optional stderr notes when ``MUSA_AITER_TRACE_DISPATCH=1`` (vendor roadmap visibility)."""

from __future__ import annotations

import os
import sys

from musa_aiter.dispatch import trace_enabled
from musa_aiter.registry import get
from musa_aiter.vendors.loader import vendor_library_loaded


def emit_vendor_planning_note(op: str) -> None:
    """Emit a single-line note listing planned ``vendor_backends`` for *op* when tracing is on.

    When ``MUSA_AITER_VENDOR_LIB`` loads, stderr notes ``vendor_dlopen=ok``; tensor
    math may still use the reference path until operator-specific vendor symbols
    are wired.
    """
    if not trace_enabled():
        return
    try:
        cap = get(op)
    except KeyError:
        return
    if not cap.vendor_backends:
        return
    override = os.environ.get("MUSA_AITER_VENDOR_BACKEND", "").strip()
    msg = f"[MUSA_AITER VENDOR] op={op} vendor_targets={','.join(cap.vendor_backends)}"
    if vendor_library_loaded():
        msg += " vendor_dlopen=ok"
    if override:
        msg += f" requested={override!r}"
        if not vendor_library_loaded():
            msg += " (vendor lib missing or unloadable; reference path)"
    print(msg, file=sys.stderr)
