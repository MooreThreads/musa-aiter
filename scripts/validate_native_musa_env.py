#!/usr/bin/env python3
"""Validate the environment for native MUSA execution (informational exit codes).

Exit 0 always: prints an environment summary suitable for local diagnostics.
Use `--strict` to exit 1 if MUSA is expected but not available (e.g. in a lab job).

Example::

    python scripts/validate_native_musa_env.py
    python scripts/validate_native_musa_env.py --strict
"""

from __future__ import annotations

import argparse
import shutil


def _ok(msg: str) -> str:
    return f"OK   {msg}"


def _fail(msg: str) -> str:
    return f"FAIL {msg}"


def _warn(msg: str) -> str:
    return f"WARN {msg}"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--strict",
        action="store_true",
        help="Exit 1 if torch_musa missing or no MUSA device.",
    )
    args = p.parse_args()

    lines: list[str] = []
    hard_fail = False

    for cmd, label in (("mcc", "mcc compiler"), ("mthreads-gmi", "mthreads-gmi")):
        path = shutil.which(cmd)
        if path:
            lines.append(_ok(f"{label}: {path}"))
        else:
            lines.append(_warn(f"{label} not on PATH"))
            if args.strict:
                hard_fail = True

    try:
        import torch
    except ImportError:
        lines.append(_fail("torch not importable"))
        print("\n".join(lines))
        return 1

    lines.append(_ok(f"torch {torch.__version__}"))

    try:
        import torch_musa  # noqa: F401
    except ImportError:
        lines.append(_fail("torch_musa import failed"))
        print("\n".join(lines))
        return 1 if args.strict else 0

    if not hasattr(torch, "musa"):
        lines.append(_fail("torch.musa missing"))
        print("\n".join(lines))
        return 1 if args.strict else 0

    musa_ok = torch.musa.is_available()
    if musa_ok:
        lines.append(_ok("torch.musa.is_available() is True"))
        try:
            d = torch.musa.device_count()
            lines.append(_ok(f"torch.musa.device_count() = {d}"))
        except Exception as e:
            lines.append(_warn(f"device_count: {e}"))
    else:
        lines.append(_fail("torch.musa.is_available() is False"))
        hard_fail = True

    print("\n".join(lines))
    if args.strict and hard_fail:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
