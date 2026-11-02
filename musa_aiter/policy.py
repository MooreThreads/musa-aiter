from __future__ import annotations

import os
from enum import Enum


class FallbackPolicy(str, Enum):
    ALLOW = "allow"
    WARN = "warn"
    REQUIRE_NATIVE = "require-native"


def current_policy() -> FallbackPolicy:
    raw = os.environ.get("MUSA_AITER_FALLBACK_POLICY", "allow").strip().lower()
    aliases = {
        "1": "require-native",
        "0": "allow",
        "require_native": "require-native",
        "native": "require-native",
    }
    raw = aliases.get(raw, raw)
    try:
        return FallbackPolicy(raw)
    except ValueError:
        return FallbackPolicy.ALLOW


def ensure_reference_allowed(op: str, native_available: bool = False) -> None:
    policy = current_policy()
    if native_available:
        return
    if policy == FallbackPolicy.REQUIRE_NATIVE:
        raise RuntimeError(
            f"{op}: native backend required by MUSA_AITER_FALLBACK_POLICY=require-native, "
            "but no native backend is available in this environment."
        )
    if policy == FallbackPolicy.WARN:
        import warnings

        warnings.warn(
            f"{op}: using reference fallback because native backend is unavailable.",
            RuntimeWarning,
            stacklevel=2,
        )
