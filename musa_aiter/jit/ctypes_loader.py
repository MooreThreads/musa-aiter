from __future__ import annotations

import ctypes
from dataclasses import dataclass
from pathlib import Path


@dataclass
class LoadedLibrary:
    path: Path
    handle: ctypes.CDLL


def load_library(path: str | Path) -> LoadedLibrary:
    resolved = Path(path).expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(resolved)
    return LoadedLibrary(path=resolved, handle=ctypes.CDLL(str(resolved)))


def has_symbol(lib: LoadedLibrary, symbol: str) -> bool:
    try:
        getattr(lib.handle, symbol)
        return True
    except AttributeError:
        return False
