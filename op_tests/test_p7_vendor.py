"""Phase P7 — vendor registry, dlopen routing, and optional vendor trace lines."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from musa_aiter.registry import as_dict, get


def test_gemm_has_vendor_targets():
    cap = get("gemm")
    assert "mudnn" in cap.vendor_backends
    assert "mutlass" in cap.vendor_backends


def test_list_ops_json_includes_vendor_backends():
    blob = as_dict()["gemm"]
    assert blob["vendor_backends"] == list(get("gemm").vendor_backends)


def test_vendor_matrix_matches_registry_subset():
    root = Path(__file__).resolve().parents[1]
    matrix_path = root / "configs" / "backends" / "vendor_matrix.v1.json"
    matrix = json.loads(matrix_path.read_text())
    for op, expected in matrix["operators"].items():
        assert list(get(op).vendor_backends) == expected


def test_gemm_emits_vendor_line_under_trace():
    code = """
import os, torch
os.environ['MUSA_AITER_TRACE_DISPATCH'] = '1'
os.environ['MUSA_AITER_VENDOR_BACKEND'] = 'mudnn'
import musa_aiter
a = torch.randn(2, 3)
b = torch.randn(3, 2)
musa_aiter.gemm(a, b)
"""
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stderr
    assert "MUSA_AITER VENDOR" in r.stderr
    assert "mudnn" in r.stderr
    assert "vendor_targets=" in r.stderr


def test_gemm_dispatch_shows_vendor_channel_when_lib_resolves():
    code = """
import os, torch
os.environ['MUSA_AITER_TRACE_DISPATCH'] = '1'
os.environ['MUSA_AITER_VENDOR_BACKEND'] = 'mudnn'
import musa_aiter.vendors.routing as vr
from unittest.mock import MagicMock
vr.get_vendor_library = lambda: MagicMock(name='fake_vendor_lib')
import musa_aiter
a = torch.randn(2, 3)
b = torch.randn(3, 2)
musa_aiter.gemm(a, b)
"""
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stderr
    assert "backend=torch-reference+vendors" in r.stderr


def test_gemm_vendor_bad_request_stays_reference():
    code = """
import os, torch
os.environ['MUSA_AITER_TRACE_DISPATCH'] = '1'
os.environ['MUSA_AITER_VENDOR_BACKEND'] = 'not-a-vendor'
import musa_aiter.vendors.routing as vr
from unittest.mock import MagicMock
vr.get_vendor_library = lambda: MagicMock(name='fake_vendor_lib')
import musa_aiter
a = torch.randn(2, 3)
b = torch.randn(3, 2)
musa_aiter.gemm(a, b)
"""
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stderr
    assert "backend=torch-reference+vendors" not in r.stderr
    assert "vendor-request-not-in-matrix" in r.stderr


def test_planning_note_vendor_dlopen_ok_when_lib_present(monkeypatch, tmp_path):
    monkeypatch.setenv("MUSA_AITER_TRACE_DISPATCH", "1")
    monkeypatch.setenv("MUSA_AITER_VENDOR_BACKEND", "mudnn")
    so = tmp_path / "libfake.so"
    so.write_bytes(b"\x7fELFtest")
    monkeypatch.setenv("MUSA_AITER_VENDOR_LIB", str(so))

    from musa_aiter.vendors import loader

    loader.reset_vendor_library_cache()
    try:
        code = f"""
import os, torch
import musa_aiter.vendors.loader as L
L.reset_vendor_library_cache()
os.environ['MUSA_AITER_TRACE_DISPATCH']='1'
os.environ['MUSA_AITER_VENDOR_BACKEND']='mudnn'
os.environ['MUSA_AITER_VENDOR_LIB']={str(so)!r}

def fake_cdll(_path):
    class _H:
        pass
    return _H()

L.get_vendor_library = lambda: fake_cdll("x")
import musa_aiter
a=torch.randn(2,3)
b=torch.randn(3,2)
musa_aiter.gemm(a,b)
"""
        r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=60)
        assert r.returncode == 0, r.stderr
        assert "vendor_dlopen=ok" in r.stderr
    finally:
        loader.reset_vendor_library_cache()


def test_list_vendors_cli_json():
    r = subprocess.run(
        [sys.executable, "-m", "musa_aiter.cli.main", "list-vendors", "--json"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert r.returncode == 0
    data = json.loads(r.stdout)
    assert "mudnn" in data
