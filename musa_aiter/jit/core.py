from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from .manifest import cache_root, make_manifest


@dataclass
class JitResult:
    ok: bool
    library_path: Path | None
    build_dir: Path
    message: str


class JitUnavailable(RuntimeError):
    pass


def _compiler() -> str:
    compiler = shutil.which("mcc")
    if not compiler:
        raise JitUnavailable("mcc compiler not found in PATH")
    return compiler


def build_shared_object(
    *,
    name: str,
    sources: list[Path],
    extra_flags: list[str] | None = None,
    force: bool | None = None,
) -> JitResult:
    """Build a MUSA shared object with mcc.

    This is the Phase-0 proof-of-concept JIT path. It intentionally keeps the
    command simple and records a build manifest next to the generated library.
    """
    flags = extra_flags or []
    manifest = make_manifest(name, "mcc-shared", sources, flags)
    build_dir = cache_root() / name / manifest.source_hash
    lib_path = build_dir / f"lib{name}.so"
    manifest_path = build_dir / "manifest.json"

    if force is None:
        force = os.environ.get("MUSA_AITER_REBUILD", "0") == "1"

    if lib_path.exists() and not force:
        return JitResult(True, lib_path, build_dir, "cache hit")

    try:
        compiler = _compiler()
        musart = shutil.which("musart")
        if musart is not None:
            ldflags = ["-lmusart"]
        elif Path("/usr/local/musa/lib/libmusart.so").exists():
            ldflags = ["-L/usr/local/musa/lib", "-Wl,-rpath,/usr/local/musa/lib", "-lmusart"]
        else:
            ldflags = []
    except JitUnavailable as exc:
        return JitResult(False, None, build_dir, str(exc))

    build_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        compiler,
        "-shared",
        "-fPIC",
        "-O3",
        *[str(s) for s in sources],
        "-o",
        str(lib_path),
        *flags,
        *ldflags,
    ]
    try:
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=120,
            check=False,
        )
        manifest_path.write_text(manifest.to_json(), encoding="utf-8")
        log_path = build_dir / "build.log"
        log_path.write_text(" ".join(cmd) + "\n\n" + proc.stdout, encoding="utf-8")
        if proc.returncode != 0:
            return JitResult(False, None, build_dir, f"build failed, see {log_path}")
        return JitResult(True, lib_path, build_dir, "build ok")
    except Exception as exc:
        return JitResult(False, None, build_dir, repr(exc))
