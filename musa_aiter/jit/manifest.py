from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

from musa_aiter.version import __version__


def cache_root() -> Path:
    return Path(os.environ.get("MUSA_AITER_CACHE_DIR", "~/.cache/musa_aiter")).expanduser()


def mcc_path() -> Optional[str]:
    return shutil.which("mcc")


def mcc_version() -> str:
    path = mcc_path()
    if not path:
        return "unavailable"
    try:
        out = subprocess.check_output([path, "--version"], text=True, stderr=subprocess.STDOUT, timeout=10)
        return out.strip().splitlines()[0] if out.strip() else "unknown"
    except Exception as exc:
        return f"error: {exc!r}"


@dataclass(frozen=True)
class BuildManifest:
    project: str
    version: str
    op: str
    backend: str
    arch: str
    compiler: str
    compiler_version: str
    source_hash: str
    build_flags: list[str]

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, sort_keys=True)


def hash_sources(paths: list[Path]) -> str:
    h = hashlib.sha256()
    for path in sorted(paths):
        h.update(str(path).encode())
        if path.exists():
            h.update(path.read_bytes())
        else:
            h.update(b"<missing>")
    return h.hexdigest()[:16]


def detect_arch() -> str:
    return os.environ.get("MUSA_AITER_ARCHS", "native")


def make_manifest(op: str, backend: str, sources: list[Path], flags: list[str] | None = None) -> BuildManifest:
    return BuildManifest(
        project="musa-aiter",
        version=__version__,
        op=op,
        backend=backend,
        arch=detect_arch(),
        compiler=mcc_path() or "mcc-not-found",
        compiler_version=mcc_version(),
        source_hash=hash_sources(sources),
        build_flags=flags or [],
    )
