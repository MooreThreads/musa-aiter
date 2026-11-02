"""Phase P10 — runtime fused_moe tuning lookup."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
import torch


def _example_csv() -> Path:
    return Path(__file__).resolve().parents[1] / "configs" / "tuned" / "example_mp31.csv"


def _preset_tensors():
    tokens, hidden, inter, experts, topk = 128, 64, 128, 4, 2
    x = torch.randn(tokens, hidden, dtype=torch.float16)
    w1 = torch.randn(experts, inter * 2, hidden, dtype=torch.float16)
    w2 = torch.randn(experts, hidden, inter, dtype=torch.float16)
    tw = torch.randn(tokens, topk, dtype=torch.float16)
    ti = torch.randint(0, experts, (tokens, topk), dtype=torch.int32)
    return x, w1, w2, tw, ti


def test_fused_moe_shape_key_matches_placeholder() -> None:
    from musa_aiter.tune_runtime import fused_moe_shape_key

    x, w1, w2, tw, _ = _preset_tensors()
    assert fused_moe_shape_key(x, w1, w2, tw) == "tokens=128,hidden=64,inter=128,experts=4,topk=2"


def test_lookup_fused_moe_example_row(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MUSA_AITER_CONFIG_MOE", str(_example_csv()))
    monkeypatch.setenv("MUSA_AITER_TUNING_ARCH", "mp31")
    from musa_aiter.tune_runtime import lookup_fused_moe_config

    x, w1, w2, tw, _ti = _preset_tensors()
    row = lookup_fused_moe_config(x, w1, w2, tw)
    assert row is not None
    assert row.kernel == "ref_fused_moe_preset"


def test_fused_moe_shape_key_rejects_w2_mismatch() -> None:
    from musa_aiter.tune_runtime import fused_moe_shape_key

    x, w1, w2, tw, _ = _preset_tensors()
    w2_bad = torch.randn(4, 64, 127, dtype=torch.float16)
    with pytest.raises(ValueError, match="w2 shape"):
        fused_moe_shape_key(x, w1, w2_bad, tw)


def test_fused_moe_trace_shows_tuned_kernel() -> None:
    code = f"""
import os, torch
os.environ["MUSA_AITER_CONFIG_MOE"] = {str(_example_csv())!r}
os.environ["MUSA_AITER_TUNING_ARCH"] = "mp31"
os.environ["MUSA_AITER_TRACE_DISPATCH"] = "1"
import musa_aiter
tokens, hidden, inter, experts, topk = 128, 64, 128, 4, 2
x = torch.randn(tokens, hidden, dtype=torch.float16)
w1 = torch.randn(experts, inter * 2, hidden, dtype=torch.float16)
w2 = torch.randn(experts, hidden, inter, dtype=torch.float16)
tw = torch.randn(tokens, topk, dtype=torch.float16)
ti = torch.randint(0, experts, (tokens, topk), dtype=torch.int32)
musa_aiter.fused_moe(x, w1, w2, tw, ti)
"""
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr
    assert "tuned kernel=ref_fused_moe_preset" in r.stderr
