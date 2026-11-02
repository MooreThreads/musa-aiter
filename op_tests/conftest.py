"""Shared pytest configuration for ``op_tests/``.

Keeps RNG deterministic across suites so flaky tolerances stabilize.
TorchMUSA/MUSA CUDA-style seeds applied when backends exist.
"""

from __future__ import annotations

from collections.abc import Generator

import pytest
import torch


@pytest.fixture(autouse=True)
def _torch_deterministic_seed() -> Generator[None, None, None]:
    torch.manual_seed(42)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    musa_mod = getattr(torch, "musa", None)
    if musa_mod is not None:
        try:
            if musa_mod.is_available():
                ms = getattr(musa_mod, "manual_seed_all", None)
                if callable(ms):
                    ms(42)
        except RuntimeError:
            pass
    yield


@pytest.fixture(scope="session")
def musa_available() -> bool:
    """Read-only probe for parametrization (prefer ``pytest.mark.musa`` skips)."""

    if not hasattr(torch, "musa"):
        return False
    try:
        return bool(torch.musa.is_available())
    except Exception:
        return False
