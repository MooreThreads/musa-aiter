from __future__ import annotations

import tarfile
import zipfile
from pathlib import Path

import pytest


def _write_wheel(path: Path, names: list[str], *, version: str = "0.5.0") -> None:
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr(
            f"musa_aiter-{version}.dist-info/METADATA",
            f"Metadata-Version: 2.4\nName: musa-aiter\nVersion: {version}\n",
        )
        for name in names:
            zf.writestr(name, "test\n")


def _write_sdist(path: Path, names: list[str]) -> None:
    with tarfile.open(path, "w:gz") as tf:
        for name in names:
            payload = b"test\n"
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            tf.addfile(info, fileobj=__import__("io").BytesIO(payload))


def test_archive_checks_accept_required_package_data(tmp_path):
    from scripts.verify_package_artifacts import verify_sdist_archive, verify_wheel_archive

    wheel_names = [
        "musa_aiter/__init__.py",
        "musa_aiter-0.5.0.data/data/csrc/smoke/add.mu",
        "musa_aiter-0.5.0.data/data/csrc/kvcache/kv_transfer_float.mu",
        "musa_aiter-0.5.0.data/data/configs/tuned/example_mp31.csv",
        "musa_aiter-0.5.0.dist-info/licenses/LICENSE",
        "musa_aiter-0.5.0.dist-info/licenses/ThirdPartyNotices.txt",
    ]
    sdist_names = [
        "musa_aiter-0.5.0/musa_aiter/__init__.py",
        "musa_aiter-0.5.0/csrc/smoke/add.mu",
        "musa_aiter-0.5.0/csrc/kvcache/kv_transfer_float.mu",
        "musa_aiter-0.5.0/configs/tuned/example_mp31.csv",
        "musa_aiter-0.5.0/LICENSE",
        "musa_aiter-0.5.0/ThirdPartyNotices.txt",
    ]
    wheel = tmp_path / "musa_aiter-0.5.0-py3-none-any.whl"
    sdist = tmp_path / "musa_aiter-0.5.0.tar.gz"
    _write_wheel(wheel, wheel_names)
    _write_sdist(sdist, sdist_names)

    verify_wheel_archive(wheel, expected_version="0.5.0")
    verify_sdist_archive(sdist)


def test_archive_checks_reject_missing_native_sources(tmp_path):
    from scripts.verify_package_artifacts import verify_wheel_archive

    wheel = tmp_path / "musa_aiter-0.5.0-py3-none-any.whl"
    _write_wheel(
        wheel,
        [
            "musa_aiter/__init__.py",
            "musa_aiter-0.5.0.data/data/configs/tuned/example_mp31.csv",
        ],
    )

    with pytest.raises(AssertionError, match="csrc/smoke/add.mu"):
        verify_wheel_archive(wheel, expected_version="0.5.0")


def test_archive_checks_reject_missing_third_party_notice(tmp_path):
    from scripts.verify_package_artifacts import verify_wheel_archive

    wheel = tmp_path / "musa_aiter-0.5.0-py3-none-any.whl"
    _write_wheel(
        wheel,
        [
            "musa_aiter/__init__.py",
            "musa_aiter-0.5.0.data/data/csrc/smoke/add.mu",
            "musa_aiter-0.5.0.data/data/csrc/kvcache/kv_transfer_float.mu",
            "musa_aiter-0.5.0.data/data/configs/tuned/example_mp31.csv",
            "musa_aiter-0.5.0.dist-info/licenses/LICENSE",
        ],
    )

    with pytest.raises(AssertionError, match="ThirdPartyNotices.txt"):
        verify_wheel_archive(wheel, expected_version="0.5.0")
