"""Optional Phase P7 shared-library probe (``MUSA_AITER_VENDOR_LIB``).

Loads at most once per process. Dlopen succeeds only when the path points to a
valid shared object compatible with this host; ABI symbols are validated by
individual operator integrations as vendor backends arrive.
"""

from __future__ import annotations

import os
from ctypes import CDLL
from pathlib import Path

_attempted = False
_handle: CDLL | None = None


def reset_vendor_library_cache() -> None:
    """Testing helper: forget load state so env changes apply."""
    global _attempted, _handle
    _attempted = False
    _handle = None


def get_vendor_library() -> CDLL | None:
    """Return a cached ``ctypes.CDLL`` handle or ``None`` if unset / load fails."""
    global _attempted, _handle
    if _attempted:
        return _handle
    _attempted = True
    raw = os.environ.get("MUSA_AITER_VENDOR_LIB", "").strip()
    if not raw:
        _handle = None
        return None
    p = Path(raw)
    if not p.is_file():
        _handle = None
        return None
    try:
        _handle = CDLL(str(p))
    except OSError:
        _handle = None
    return _handle


def vendor_library_loaded() -> bool:
    """True when ``MUSA_AITER_VENDOR_LIB`` resolved to a loadable shared library."""
    return get_vendor_library() is not None
