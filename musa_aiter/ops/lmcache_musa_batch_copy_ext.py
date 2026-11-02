from __future__ import annotations

import os
from typing import Any

from musa_aiter._paths import project_data_path

_EXT_SOURCE = project_data_path("csrc", "kvcache", "lmcache_musa_batch_copy_ext.cpp")
_EXT_MODULE: Any | None = None
_EXT_ATTEMPTED = False
_EXT_ERROR: Exception | None = None


def lmcache_musa_batch_copy_requested() -> bool:
    return os.environ.get("MUSA_AITER_LMCACHE_BATCH_COPY", "0") == "1" or os.environ.get(
        "LMCACHE_MUSA_DIRECT_COPY", "0"
    ) == "1"


def load_lmcache_musa_batch_copy_ext() -> Any | None:
    """Build and load the optional MUSA SDK batch-copy extension once."""
    global _EXT_MODULE, _EXT_ATTEMPTED, _EXT_ERROR
    if not lmcache_musa_batch_copy_requested():
        return None
    if _EXT_MODULE is not None:
        return _EXT_MODULE
    if _EXT_ATTEMPTED:
        return None

    _EXT_ATTEMPTED = True
    try:
        from torch_musa.utils.musa_extension import load

        _EXT_MODULE = load(
            name="musa_aiter_lmcache_musa_batch_copy_ext",
            sources=[str(_EXT_SOURCE)],
            extra_cflags=["-O3"],
            verbose=os.environ.get("MUSA_AITER_VERBOSE_EXT_BUILD", "0") == "1",
        )
        return _EXT_MODULE
    except Exception as exc:
        _EXT_ERROR = exc
        return None


def lmcache_musa_batch_copy_ext_error() -> Exception | None:
    return _EXT_ERROR
