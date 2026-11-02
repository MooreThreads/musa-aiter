from __future__ import annotations

import sys
from pathlib import Path


_PACKAGE_ROOT = Path(__file__).resolve().parents[1]


def resolve_project_data_path(
    *parts: str,
    package_root: Path | None = None,
    install_prefix: Path | None = None,
) -> Path:
    """Resolve repo-root data files in editable and wheel installs."""
    root = package_root or _PACKAGE_ROOT
    prefix = install_prefix or Path(sys.prefix)
    relative = Path(*parts)

    candidates = (
        root / relative,
        prefix / relative,
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def project_data_path(*parts: str) -> Path:
    return resolve_project_data_path(*parts)
