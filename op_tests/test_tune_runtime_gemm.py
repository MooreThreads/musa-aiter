"""Phase P9 — runtime GEMM tuning lookup (``tune_runtime``)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
import torch


def _example_csv() -> Path:
    return Path(__file__).resolve().parents[1] / "configs" / "tuned" / "example_mp31.csv"


def test_lookup_gemm_float16_matches_example_row(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MUSA_AITER_CONFIG_GEMM", str(_example_csv()))
    monkeypatch.setenv("MUSA_AITER_TUNING_ARCH", "mp31")
    from musa_aiter.tune_runtime import lookup_gemm_config

    a = torch.randn(256, 256, dtype=torch.float16)
    b = torch.randn(256, 256, dtype=torch.float16)
    row = lookup_gemm_config(a, b)
    assert row is not None
    assert row.kernel == "ref_matmul"
    assert row.backend == "torch-reference"


def test_lookup_gemm_no_env_returns_none(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MUSA_AITER_CONFIG_GEMM", raising=False)
    from musa_aiter.tune_runtime import lookup_gemm_config

    a = torch.randn(4, 4, dtype=torch.float32)
    b = torch.randn(4, 4, dtype=torch.float32)
    assert lookup_gemm_config(a, b) is None


def test_lookup_gemm_shape_miss(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MUSA_AITER_CONFIG_GEMM", str(_example_csv()))
    monkeypatch.setenv("MUSA_AITER_TUNING_ARCH", "mp31")
    from musa_aiter.tune_runtime import lookup_gemm_config

    a = torch.randn(7, 7, dtype=torch.float16)
    b = torch.randn(7, 7, dtype=torch.float16)
    assert lookup_gemm_config(a, b) is None


def test_gemm_tracementions_tuned_kernel(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MUSA_AITER_CONFIG_GEMM", str(_example_csv()))
    monkeypatch.setenv("MUSA_AITER_TUNING_ARCH", "mp31")
    monkeypatch.setenv("MUSA_AITER_TRACE_DISPATCH", "1")
    code = f"""
import os, torch, sys
os.environ["MUSA_AITER_CONFIG_GEMM"] = {str(_example_csv())!r}
os.environ["MUSA_AITER_TUNING_ARCH"] = "mp31"
os.environ["MUSA_AITER_TRACE_DISPATCH"] = "1"
import musa_aiter
a = torch.randn(256, 256, dtype=torch.float16)
b = torch.randn(256, 256, dtype=torch.float16)
musa_aiter.gemm(a, b)
"""
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stderr
    assert "tuned kernel=ref_matmul" in r.stderr
