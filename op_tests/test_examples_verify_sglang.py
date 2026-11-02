"""Ensure examples/verify_sglang_musa_aiter.py runs in --no-sglang mode."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def test_verify_sglang_script_no_sglang_mode():
    repo = Path(__file__).resolve().parents[1]
    script = repo / "examples" / "verify_sglang_musa_aiter.py"
    proc = subprocess.run(
        [sys.executable, str(script), "--no-sglang"],
        cwd=repo,
        env={**os.environ, "PYTHONPATH": str(repo)},
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
