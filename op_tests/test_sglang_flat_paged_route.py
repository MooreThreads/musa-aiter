"""
L4 integration tests: DSV4 flat-paged route + patch script multi-injection (REQ-5).
Pure-Python: no MUSA device, no sglang install required.
"""

import types
from pathlib import Path

import pytest
import torch

from musa_aiter.integration.sglang_kv import (
    _can_route_sglang_flat_paged_hicache,
    try_sglang_hicache_flat_paged_transfer,
)


class _FakePool:
    def __init__(self, layers, pages, item_bytes, *, layout="layer_first", dtype=torch.uint8, dim=3):
        self.layout = layout
        if dim == 3:
            self.kv_buffer = torch.zeros(layers, pages, item_bytes, dtype=dtype)
        else:
            self.kv_buffer = torch.zeros(pages, item_bytes, dtype=dtype)


class TestFlatPagedRouteDecision:
    """T4.1: routing match/mismatch conditions."""

    def test_dsv4_pool_matches(self):
        host, device = _FakePool(43, 64, 1728), _FakePool(43, 64, 1728)
        decision = _can_route_sglang_flat_paged_hicache(host, device, op="test", require_env=False)
        assert decision.enabled, decision.reason

    def test_env_gate_blocks(self):
        host, device = _FakePool(2, 8, 64), _FakePool(2, 8, 64)
        decision = _can_route_sglang_flat_paged_hicache(host, device, op="test", require_env=True)
        assert not decision.enabled
        assert "MUSA_AITER_SGLANG_ADAPTER" in decision.reason

    def test_float_kv_rejected(self):
        # Standard MHA/MLA pools (float k_buffer) must NOT match this route
        host = _FakePool(2, 8, 64, dtype=torch.float16)
        device = _FakePool(2, 8, 64, dtype=torch.float16)
        decision = _can_route_sglang_flat_paged_hicache(host, device, op="test", require_env=False)
        assert not decision.enabled
        assert "uint8" in decision.reason

    def test_wrong_layout_rejected(self):
        host = _FakePool(2, 8, 64, layout="page_first")
        device = _FakePool(2, 8, 64)
        decision = _can_route_sglang_flat_paged_hicache(host, device, op="test", require_env=False)
        assert not decision.enabled

    def test_missing_kv_buffer_rejected(self):
        host = types.SimpleNamespace(layout="layer_first")  # k_buffer-only pool
        device = _FakePool(2, 8, 64)
        decision = _can_route_sglang_flat_paged_hicache(host, device, op="test", require_env=False)
        assert not decision.enabled

    def test_transfer_load_bit_exact(self):
        """Routed H2D copy produces bit-exact device rows (reference path)."""
        torch.manual_seed(3)
        host, device = _FakePool(4, 16, 256), _FakePool(4, 16, 256)
        host.kv_buffer.copy_(torch.randint(0, 256, host.kv_buffer.shape, dtype=torch.uint8))
        src_idx = torch.tensor([0, 3, 7, 11], dtype=torch.int64)
        dst_idx = torch.tensor([1, 2, 13, 15], dtype=torch.int64)

        routed = try_sglang_hicache_flat_paged_transfer(
            host, device, src_idx, dst_idx, direction="load_to_device", require_env=False
        )
        assert routed
        expected = torch.zeros_like(host.kv_buffer)
        expected.index_copy_(1, dst_idx, host.kv_buffer.index_select(1, src_idx))
        assert torch.equal(device.kv_buffer, expected)

    def test_transfer_backup_bit_exact(self):
        torch.manual_seed(4)
        host, device = _FakePool(3, 12, 128), _FakePool(3, 12, 128)
        device.kv_buffer.copy_(torch.randint(0, 256, device.kv_buffer.shape, dtype=torch.uint8))
        src_idx = torch.tensor([2, 5, 9], dtype=torch.int64)
        dst_idx = torch.tensor([0, 4, 8], dtype=torch.int64)

        routed = try_sglang_hicache_flat_paged_transfer(
            host, device, src_idx, dst_idx, direction="backup_from_device", require_env=False
        )
        assert routed
        # parallel semantics: host row src_idx[k] <- device row dst_idx[k]
        expected = torch.zeros_like(device.kv_buffer)
        expected.index_copy_(1, src_idx, device.kv_buffer.index_select(1, dst_idx))
        assert torch.equal(host.kv_buffer, expected)


class TestPatchScriptMultiInjection:
    """T4.2: patch script handles multiple hook points (DSV4 9-variant file)."""

    @pytest.fixture()
    def fake_memory_pool_host(self, tmp_path):
        """Synthesize a memory_pool_host.py with 2 MLA load + 2 MLA backup hooks."""
        needle_load = '''        if io_backend == "kernel":
            if self.layout == "layer_first":
                transfer_kv_per_layer_mla(
'''
        needle_backup = '''        if io_backend == "kernel":
            if self.layout == "layer_first":
                transfer_kv_all_layer_mla(
'''
        needle_mha_load = '''        if io_backend == "kernel":
            if self.layout == "layer_first":
                if self.can_use_jit:
                    jit_transfer_hicache_one_layer(
'''
        needle_mha_backup = '''        if io_backend == "kernel":
            if self.layout == "layer_first":
                if self.can_use_jit:
                    jit_transfer_hicache_all_layer(
'''
        lines = [
            "import logging\n",
            "logger = logging.getLogger(__name__)\n\n\n",
        ]
        def add_func(name, needle, count):
            for i in range(count):
                lines.append(f"def {name}_{i}():\n")
                lines.extend(line + "\n" for line in needle.rstrip("\n").split("\n"))
                lines.append("    pass\n\n")

        add_func("_mla_load", needle_load, 2)
        add_func("_mla_backup", needle_backup, 2)
        add_func("_mha_load", needle_mha_load, 1)
        add_func("_mha_backup", needle_mha_backup, 1)
        path = tmp_path / "memory_pool_host.py"
        path.write_text("".join(lines))
        return path

    def test_dry_run_reports_multi_hooks(self, fake_memory_pool_host, capsys):
        import sys

        script = str(Path(__file__).resolve().parents[1] / "scripts" / "patch_sglang_musa_aiter_hicache.py")
        result = 0
        import runpy

        orig_argv = sys.argv
        try:
            sys.argv = [
                script,
                "--sglang-root", str(fake_memory_pool_host.parent),
                "--file", fake_memory_pool_host.name,
                "--dry-run",
            ]
            runpy.run_path(script, run_name="__main__")
        except SystemExit as exc:
            result = exc.code
        finally:
            sys.argv = orig_argv
        assert result in (0, None)
        out = capsys.readouterr().out
        assert "MLA load: 2 match(es)" in out
        assert "dry-run" in out
        # dry-run must not modify the file
        assert "MUSA_AITER_SGLANG_HICACHE_ROUTE_PATCH_V4" not in fake_memory_pool_host.read_text()

    def test_patch_injects_all_hooks_and_reports_count(self, fake_memory_pool_host, capsys):
        import runpy
        import sys

        script = str(Path(__file__).resolve().parents[1] / "scripts" / "patch_sglang_musa_aiter_hicache.py")
        orig_argv = sys.argv
        try:
            sys.argv = [
                script,
                "--sglang-root", str(fake_memory_pool_host.parent),
                "--file", fake_memory_pool_host.name,
                "--no-backup",
            ]
            runpy.run_path(script, run_name="__main__")
        except SystemExit as exc:
            assert exc.code in (0, None), f"patch script exited with {exc.code}"
        finally:
            sys.argv = orig_argv
        text = fake_memory_pool_host.read_text()
        assert text.count("_musa_aiter_hicache_kernel_route(") >= 2  # both load hooks injected
        out = capsys.readouterr().out
        assert "injection point(s)" in out
