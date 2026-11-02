"""Phase P8 — ``musa-aiter benchmark`` and ``musa-aiter tune`` CLI."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def test_benchmark_gemm_json():
    r = subprocess.run(
        [
            sys.executable,
            "-m",
            "musa_aiter.cli.main",
            "benchmark",
            "--op",
            "gemm",
            "--iters",
            "2",
            "--warmup",
            "1",
            "--json",
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert r.returncode == 0, r.stderr
    data = json.loads(r.stdout)
    assert data["name"] == "gemm"
    assert "avg_ms" in data


def test_benchmark_softmax_last_dim_json():
    r = subprocess.run(
        [
            sys.executable,
            "-m",
            "musa_aiter.cli.main",
            "benchmark",
            "--op",
            "softmax_last_dim",
            "--iters",
            "2",
            "--warmup",
            "1",
            "--json",
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert r.returncode == 0, r.stderr
    data = json.loads(r.stdout)
    assert data["name"] == "softmax_last_dim"
    assert "avg_ms" in data


def test_benchmark_kv_paged_to_linear_json():
    r = subprocess.run(
        [
            sys.executable,
            "-m",
            "musa_aiter.cli.main",
            "benchmark",
            "--op",
            "kv_paged_to_linear",
            "--iters",
            "2",
            "--warmup",
            "1",
            "--json",
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert r.returncode == 0, r.stderr
    data = json.loads(r.stdout)
    assert data["name"] == "kv_paged_to_linear"
    assert "avg_ms" in data


def test_benchmark_kv_paged_to_linear_multi_layer_json():
    r = subprocess.run(
        [
            sys.executable,
            "-m",
            "musa_aiter.cli.main",
            "benchmark",
            "--op",
            "kv_paged_to_linear_multi_layer",
            "--iters",
            "2",
            "--warmup",
            "1",
            "--json",
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert r.returncode == 0, r.stderr
    data = json.loads(r.stdout)
    assert data["name"] == "kv_paged_to_linear_multi_layer"
    assert "avg_ms" in data


def test_tune_validate_example_csv():
    csv_path = _repo_root() / "configs" / "tuned" / "example_mp31.csv"
    r = subprocess.run(
        [
            sys.executable,
            "-m",
            "musa_aiter.cli.main",
            "tune",
            "--validate",
            "--config",
            str(csv_path),
            "--json",
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert r.returncode == 0, r.stderr
    assert json.loads(r.stdout)["count"] == 4


def test_tune_select_gemm_row():
    csv_path = _repo_root() / "configs" / "tuned" / "example_mp31.csv"
    r = subprocess.run(
        [
            sys.executable,
            "-m",
            "musa_aiter.cli.main",
            "tune",
            "--select",
            "--config",
            str(csv_path),
            "--op",
            "gemm",
            "--arch",
            "mp31",
            "--dtype",
            "float16",
            "--shape",
            "m=256,n=256,k=256",
            "--json",
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert r.returncode == 0, r.stderr
    row = json.loads(r.stdout)
    assert row["kernel"] == "ref_matmul"
