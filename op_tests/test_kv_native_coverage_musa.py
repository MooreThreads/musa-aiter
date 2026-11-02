from __future__ import annotations

import os

import pytest
import torch

import musa_aiter
from musa_aiter.jit.ctypes_loader import has_symbol, load_library
from musa_aiter.ops.kv_transfer import build_native_kv_transfer


def _require_c3_native(symbols: tuple[str, ...]) -> None:
    if not hasattr(torch, "musa") or not torch.musa.is_available():
        pytest.skip("TorchMUSA with a visible MUSA device is required")
    if os.environ.get("MUSA_AITER_NATIVE_KV_TRANSFER", "0") != "1":
        pytest.skip("set MUSA_AITER_NATIVE_KV_TRANSFER=1")
    result = build_native_kv_transfer()
    if not result.ok or result.library_path is None:
        pytest.skip(result.message)
    lib = load_library(result.library_path)
    if not all(has_symbol(lib, symbol) for symbol in symbols):
        pytest.skip("C3 native symbols are unavailable")


@pytest.mark.musa
@pytest.mark.native
def test_native_lmcache_float32_list_and_mla_roundtrip(monkeypatch: pytest.MonkeyPatch) -> None:
    _require_c3_native(
        (
            "lmcache_kv_buffer_to_paged_list_float",
            "lmcache_paged_to_kv_buffer_list_float",
            "lmcache_mla_buffer_to_paged_list_float",
            "lmcache_mla_paged_to_buffer_list_float",
        )
    )
    from musa_aiter.ops import lmcache_kv_transfer as lmcache_ops

    monkeypatch.setattr(
        lmcache_ops,
        "_slot_values",
        lambda _: (_ for _ in ()).throw(AssertionError("native path used Python slot loop")),
    )
    nl, nb, bs, nh, hs = 2, 2, 2, 1, 4
    slots = torch.tensor([0, 1, 2, -1], dtype=torch.int64, device="musa")
    memory = torch.arange(2 * nl * slots.numel() * nh * hs, dtype=torch.float32, device="musa").reshape(
        2, nl, slots.numel(), nh * hs
    )
    caches = [
        torch.zeros(2, nb, bs, nh, hs, dtype=torch.float32, device="musa")
        for _ in range(nl)
    ]
    assert musa_aiter.lmcache_kv_paged_to_buffer(
        caches, memory, slots, bs, nh, hs
    )
    restored = [torch.zeros_like(cache) for cache in caches]
    assert musa_aiter.lmcache_kv_buffer_to_paged(
        memory, restored, slots, 0, bs, nh, hs
    )

    mla_memory = torch.arange(nl * slots.numel() * hs, dtype=torch.float32, device="musa").reshape(
        1, nl, slots.numel(), hs
    )
    mla_caches = [
        torch.zeros(nb, bs, hs, dtype=torch.float32, device="musa") for _ in range(nl)
    ]
    assert musa_aiter.lmcache_mla_buffer_to_paged(
        mla_memory, mla_caches, slots, 0, bs, hs
    )
    mla_restored = torch.empty_like(mla_memory)
    assert musa_aiter.lmcache_mla_paged_to_buffer(
        mla_caches, mla_restored, slots, bs, hs
    )
    torch.musa.synchronize()
    torch.testing.assert_close(mla_restored[:, :, :3], mla_memory[:, :, :3])
    assert mla_restored[:, :, 3].abs().max().item() == 0


@pytest.mark.musa
@pytest.mark.native
def test_native_lmcache_non_mla_list_hidden_not_multiple_of_vec16(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression: the vec-16 list2 kernels have no scalar tail, so a non-MLA row
    whose hidden size is not a multiple of 16 (here hidden=4) must fall through to
    the tail-safe per-layer ``lmcache_*_list_u16`` path instead of silently
    dropping the trailing lanes."""

    _require_c3_native(
        (
            "lmcache_kv_buffer_to_paged_list_u16",
            "lmcache_paged_to_kv_buffer_list_u16",
        )
    )
    from musa_aiter.ops import lmcache_kv_transfer as lmcache_ops

    monkeypatch.setattr(
        lmcache_ops,
        "_slot_values",
        lambda _: (_ for _ in ()).throw(AssertionError("native path used Python slot loop")),
    )
    monkeypatch.setattr(
        lmcache_ops,
        "_call_non_mla_list2_u16_symbol",
        lambda *_, **__: (_ for _ in ()).throw(
            AssertionError("vec-16 list2 path must not run when hidden % 16 != 0")
        ),
    )
    list_u16_calls = {"count": 0}
    real_list_symbol = lmcache_ops._call_non_mla_list_symbol

    def _counting_list_symbol(*args: object, **kwargs: object) -> bool:
        list_u16_calls["count"] += 1
        return real_list_symbol(*args, **kwargs)

    monkeypatch.setattr(lmcache_ops, "_call_non_mla_list_symbol", _counting_list_symbol)

    nl, nb, bs, nh, hs = 2, 2, 2, 1, 4
    slots = torch.tensor([0, 1, 2, -1], dtype=torch.int64, device="musa")
    for dtype in (torch.float16, torch.bfloat16):
        memory = (
            torch.arange(2 * nl * slots.numel() * nh * hs, dtype=torch.float32, device="musa")
            .reshape(2, nl, slots.numel(), nh * hs)
            .to(dtype)
        )
        caches = [
            torch.zeros(2, nb, bs, nh, hs, dtype=dtype, device="musa") for _ in range(nl)
        ]
        assert musa_aiter.lmcache_kv_buffer_to_paged(memory, caches, slots, 0, bs, nh, hs)
        memory_rt = torch.empty_like(memory)
        assert musa_aiter.lmcache_kv_paged_to_buffer(caches, memory_rt, slots, bs, nh, hs)
        torch.musa.synchronize()
        torch.testing.assert_close(memory_rt[:, :, :3], memory[:, :, :3])
        assert memory_rt[:, :, 3].abs().max().item() == 0

    # 2 dtypes x 2 directions x nl layers, all served by the tail-safe list path.
    assert list_u16_calls["count"] == 2 * 2 * nl


@pytest.mark.musa
@pytest.mark.native
def test_native_lmcache_merged_and_mla_flat_support_half_and_bfloat16(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    for suffix in ("half", "bfloat16"):
        _require_c3_native(
            (
                f"musa_aiter_kv_paged_to_merged_linear_{suffix}",
                f"musa_aiter_kv_merged_linear_to_paged_{suffix}",
                f"musa_aiter_kv_paged_to_mla_linear_{suffix}",
                f"musa_aiter_kv_mla_linear_to_paged_{suffix}",
            )
        )
        dtype = torch.float16 if suffix == "half" else torch.bfloat16
        nb, bs, nh, hd, tokens = 2, 2, 1, 4, 3
        slots = torch.tensor([0, 1, 3], dtype=torch.int64, device="musa")
        merged = torch.arange(tokens * 2 * nh * hd, dtype=torch.float32, device="musa").reshape(
            tokens, 2, nh, hd
        ).to(dtype)
        key = torch.zeros(nb, nh, hd, bs, dtype=dtype, device="musa")
        value = torch.zeros_like(key)
        assert musa_aiter.kv_merged_linear_to_paged(merged, key, value, slots) is None
        recovered = musa_aiter.kv_paged_to_merged_linear(key, value, slots)

        kv_linear = torch.arange(tokens * 4, dtype=torch.float32, device="musa").reshape(tokens, 4).to(dtype)
        pe_linear = torch.arange(tokens * 2, dtype=torch.float32, device="musa").reshape(tokens, 2).to(dtype)
        kv_cache = torch.zeros(nb, bs, 4, dtype=dtype, device="musa")
        pe_cache = torch.zeros(nb, bs, 2, dtype=dtype, device="musa")
        assert musa_aiter.kv_mla_linear_to_paged(
            kv_linear, pe_linear, kv_cache, pe_cache, slots
        ) is None
        recovered_kv, recovered_pe = musa_aiter.kv_paged_to_mla_linear(
            kv_cache, pe_cache, slots
        )
        torch.musa.synchronize()
        torch.testing.assert_close(recovered[:2], merged[:2])
        torch.testing.assert_close(recovered_kv, kv_linear)
        torch.testing.assert_close(recovered_pe, pe_linear)

    assert "ctypes-musa-native" in capsys.readouterr().err
