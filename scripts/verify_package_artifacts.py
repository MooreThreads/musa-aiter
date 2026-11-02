#!/usr/bin/env python3
"""Verify built package artifacts include runtime data and install cleanly."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tarfile
import tempfile
import zipfile
from email.parser import Parser
from pathlib import Path


REQUIRED_DATA_PATHS = (
    "csrc/smoke/add.mu",
    "csrc/kvcache/kv_transfer_float.mu",
    "configs/tuned/example_mp31.csv",
)
REQUIRED_LICENSE_PATHS = ("LICENSE", "ThirdPartyNotices.txt")


def _metadata_version(text: str) -> str:
    metadata = Parser().parsestr(text)
    version = metadata.get("Version")
    if not version:
        raise AssertionError("wheel METADATA is missing Version")
    return version


def _assert_contains_suffix(names: set[str], suffix: str) -> None:
    if not any(name.endswith(suffix) for name in names):
        raise AssertionError(f"artifact is missing {suffix}")


def verify_wheel_archive(path: Path, *, expected_version: str | None = None) -> None:
    with zipfile.ZipFile(path) as zf:
        names = set(zf.namelist())
        metadata_names = [name for name in names if name.endswith(".dist-info/METADATA")]
        if len(metadata_names) != 1:
            raise AssertionError(f"expected one wheel METADATA file, found {len(metadata_names)}")
        version = _metadata_version(zf.read(metadata_names[0]).decode("utf-8"))

    if expected_version is not None and version != expected_version:
        raise AssertionError(f"wheel version {version!r} != expected {expected_version!r}")
    _assert_contains_suffix(names, "musa_aiter/__init__.py")
    for suffix in REQUIRED_DATA_PATHS:
        _assert_contains_suffix(names, suffix)
    for suffix in REQUIRED_LICENSE_PATHS:
        _assert_contains_suffix(names, suffix)


def verify_sdist_archive(path: Path) -> None:
    with tarfile.open(path, "r:gz") as tf:
        names = set(tf.getnames())
    _assert_contains_suffix(names, "musa_aiter/__init__.py")
    for suffix in REQUIRED_DATA_PATHS:
        _assert_contains_suffix(names, suffix)
    for suffix in REQUIRED_LICENSE_PATHS:
        _assert_contains_suffix(names, suffix)


def _find_one(dist_dir: Path, pattern: str) -> Path:
    matches = sorted(dist_dir.glob(pattern))
    if len(matches) != 1:
        raise SystemExit(f"expected exactly one {pattern} in {dist_dir}, found {len(matches)}")
    return matches[0]


def _wheel_version_from_name(wheel: Path) -> str | None:
    match = re.match(r"musa_aiter-([^-]+)-", wheel.name)
    return match.group(1) if match else None


def verify_installed_wheel(wheel: Path) -> dict[str, object]:
    with tempfile.TemporaryDirectory(prefix="musa-aiter-wheel-") as tmp:
        target = Path(tmp) / "target"
        subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--target",
                str(target),
                "--no-deps",
                "--force-reinstall",
                str(wheel),
            ],
            check=True,
        )
        env = {**os.environ, "PYTHONPATH": str(target)}
        proc = subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "import json; "
                    "import musa_aiter; "
                    "from musa_aiter.registry import list_capabilities; "
                    "from musa_aiter.ops.smoke import _add_source_path; "
                    "from musa_aiter.tuning import bundled_example_tuned_csv; "
                    "add_path = _add_source_path(); "
                    "cfg_path = bundled_example_tuned_csv(); "
                    "print(json.dumps({"
                    "'version': musa_aiter.__version__, "
                    "'capabilities': len(list_capabilities()), "
                    "'package_file': musa_aiter.__file__, "
                    "'add_source': str(add_path), "
                    "'add_source_exists': add_path.is_file(), "
                    "'tuned_csv': str(cfg_path), "
                    "'tuned_csv_exists': cfg_path is not None and cfg_path.is_file(), "
                    "}, sort_keys=True))"
                ),
            ],
            cwd=tempfile.gettempdir(),
            env=env,
            capture_output=True,
            text=True,
            check=True,
        )
    data = json.loads(proc.stdout.strip().splitlines()[-1])
    if not data["add_source_exists"]:
        raise AssertionError(f"installed wheel cannot resolve csrc source: {data['add_source']}")
    if not data["tuned_csv_exists"]:
        raise AssertionError(f"installed wheel cannot resolve bundled tuned CSV: {data['tuned_csv']}")
    return data


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dist-dir", default="dist", help="Directory containing built artifacts")
    parser.add_argument("--skip-install", action="store_true", help="Only inspect archives")
    args = parser.parse_args()

    dist_dir = Path(args.dist_dir)
    wheel = _find_one(dist_dir, "musa_aiter-*.whl")
    sdist = _find_one(dist_dir, "musa_aiter-*.tar.gz")
    expected_version = _wheel_version_from_name(wheel)

    verify_wheel_archive(wheel, expected_version=expected_version)
    verify_sdist_archive(sdist)

    install_data = None if args.skip_install else verify_installed_wheel(wheel)
    payload = {
        "wheel": str(wheel),
        "sdist": str(sdist),
        "expected_version": expected_version,
        "install": install_data,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
