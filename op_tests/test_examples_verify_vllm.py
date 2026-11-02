"""Ensure examples/verify_vllm_musa_aiter.py runs in --no-vllm mode."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def test_verify_script_no_vllm_mode():
    repo = Path(__file__).resolve().parents[1]
    script = repo / "examples" / "verify_vllm_musa_aiter.py"
    proc = subprocess.run(
        [sys.executable, str(script), "--no-vllm"],
        cwd=repo,
        env={**os.environ, "PYTHONPATH": str(repo)},
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr

