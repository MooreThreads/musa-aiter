"""Runtime grouped GEMM tuning lookup (Phase P10)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
import torch


def _example_csv() -> Path:
    return Path(__file__).resolve().parents[1] / "configs" / "tuned" / "example_mp31.csv"


def _preset_tensors():
    tokens, hidden, out_dim, groups = 512, 128, 256, 4
    a = torch.randn(tokens, hidden, dtype=torch.float16)
    b = torch.randn(groups, hidden, out_dim, dtype=torch.float16)
    idx = torch.randint(0, groups, (tokens,))
    return a, b, idx


def test_grouped_gemm_shape_key_matches_example_placeholder() -> None:
    from musa_aiter.tune_runtime import grouped_gemm_shape_key

    a, b, _idx = _preset_tensors()
    assert grouped_gemm_shape_key(a, b) == "tokens=512,hidden=128,out=256,groups=4"


def test_lookup_grouped_gemm_example_row(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MUSA_AITER_CONFIG_GROUPED_GEMM", str(_example_csv()))
    monkeypatch.setenv("MUSA_AITER_TUNING_ARCH", "mp31")
    from musa_aiter.tune_runtime import lookup_grouped_gemm_config

    a, b, _idx = _preset_tensors()
    row = lookup_grouped_gemm_config(a, b)
    assert row is not None
    assert row.kernel == "ref_grouped_gemm_preset"
    assert row.backend == "torch-reference"


def test_grouped_gemm_trace_shows_tuned_kernel() -> None:
    code = f"""
import os, torch
os.environ["MUSA_AITER_CONFIG_GROUPED_GEMM"] = {str(_example_csv())!r}
os.environ["MUSA_AITER_TUNING_ARCH"] = "mp31"
os.environ["MUSA_AITER_TRACE_DISPATCH"] = "1"
import musa_aiter
tokens, hidden, out_dim, groups = 512, 128, 256, 4
a = torch.randn(tokens, hidden, dtype=torch.float16)
b = torch.randn(groups, hidden, out_dim, dtype=torch.float16)
idx = torch.randint(0, groups, (tokens,))
musa_aiter.grouped_gemm(a, b, idx)
"""
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr
    assert "tuned kernel=ref_grouped_gemm_preset" in r.stderr
