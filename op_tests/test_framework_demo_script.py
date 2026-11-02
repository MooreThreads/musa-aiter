"""Smoke-test ``examples/framework_demo.py`` like the SGLang integration §4 snippet."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SCRIPT = _REPO_ROOT / "examples" / "framework_demo.py"


def test_framework_demo_script_cpu_matches_guide() -> None:
    proc = subprocess.run(
        [sys.executable, str(_SCRIPT), "--device", "cpu"],
        cwd=str(_REPO_ROOT),
        env={**os.environ, "PYTHONPATH": str(_REPO_ROOT)},
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "prefill:" in proc.stdout and "decode:" in proc.stdout and "moe:" in proc.stdout


@pytest.mark.musa
def test_framework_demo_script_musa_if_available() -> None:
    import torch

    if not hasattr(torch, "musa") or not torch.musa.is_available():
        pytest.skip("no TorchMUSA device")
    proc = subprocess.run(
        [sys.executable, str(_SCRIPT), "--device", "musa"],
        cwd=str(_REPO_ROOT),
        env={**os.environ, "PYTHONPATH": str(_REPO_ROOT)},
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "prefill:" in proc.stdout
