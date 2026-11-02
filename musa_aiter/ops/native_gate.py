"""Canonical opt-in policy for native MUSA operator dispatch."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class NativeGateDecision:
    operation: str
    requested: bool
    available: bool
    mode: str
    reason: str


def native_requested(*env_names: str) -> bool:
    """Return true when any supplied native gate alias is explicitly enabled."""

    return any(os.environ.get(name, "0") == "1" for name in env_names)


def decide_native_gate(
    operation: str,
    env_names: Iterable[str],
    *,
    available: bool,
    native_required: bool = False,
) -> NativeGateDecision:
    requested = native_requested(*tuple(env_names))
    if not requested:
        return NativeGateDecision(
            operation,
            requested=False,
            available=available,
            mode="reference",
            reason="native_gate_off",
        )
    if available:
        return NativeGateDecision(
            operation,
            requested=True,
            available=True,
            mode="native",
            reason="native_available",
        )
    return NativeGateDecision(
        operation,
        requested=True,
        available=False,
        mode="error" if native_required else "fallback",
        reason="native_unavailable",
    )


__all__ = ["NativeGateDecision", "decide_native_gate", "native_requested"]
