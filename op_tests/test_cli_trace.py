"""CLI ``musa-aiter trace`` (Phase P6 observability)."""

from __future__ import annotations

import subprocess
import sys


def test_trace_sets_dispatch_env():
    r = subprocess.run(
        [
            sys.executable,
            "-m",
            "musa_aiter.cli.main",
            "trace",
            "--",
            sys.executable,
            "-c",
            "import os; raise SystemExit(0 if os.environ.get('MUSA_AITER_TRACE_DISPATCH') == '1' else 1)",
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert r.returncode == 0, r.stderr + r.stdout


def test_trace_fails_without_command():
    r = subprocess.run(
        [sys.executable, "-m", "musa_aiter.cli.main", "trace"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert r.returncode == 2
