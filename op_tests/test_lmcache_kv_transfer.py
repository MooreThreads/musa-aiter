import pytest
import torch

import musa_aiter
from musa_aiter.jit.ctypes_loader import has_symbol, load_library
from musa_aiter.ops.kv_transfer import build_native_kv_transfer


def test_lmcache_native_symbols_are_in_kv_transfer_shared_object():
    result = build_native_kv_transfer()
    if not result.ok and "compiler not found" in result.message:
        pytest.skip(result.message)
    assert result.ok, result.message
    assert result.library_path is not None
    lib = load_library(result.library_path)
    assert has_symbol(lib, "native_lmcache_kv_transfer_abi_version")
    assert has_symbol(lib, "lmcache_kv_buffer_to_paged")
    assert has_symbol(lib, "lmcache_paged_to_kv_buffer")
    assert has_symbol(lib, "lmcache_kv_buffer_to_paged_mla")
    assert has_symbol(lib, "lmcache_paged_to_kv_buffer_mla")
    assert has_symbol(lib, "lmcache_kv_buffer_to_paged_u16")
    assert has_symbol(lib, "lmcache_paged_to_kv_buffer_u16")
    assert has_symbol(lib, "lmcache_kv_buffer_to_paged_mla_u16")
    assert has_symbol(lib, "lmcache_paged_to_kv_buffer_mla_u16")
    assert has_symbol(lib, "lmcache_kv_buffer_to_paged_list_u16")
    assert has_symbol(lib, "lmcache_paged_to_kv_buffer_list_u16")
    assert has_symbol(lib, "lmcache_kv_buffer_to_paged_list_float")
    assert has_symbol(lib, "lmcache_paged_to_kv_buffer_list_float")
    assert has_symbol(lib, "lmcache_kv_buffer_to_paged_two_x_nl_float")
    assert has_symbol(lib, "lmcache_paged_to_kv_buffer_two_x_nl_float")
    assert has_symbol(lib, "lmcache_mla_buffer_to_paged_list_float")
    assert has_symbol(lib, "lmcache_mla_paged_to_buffer_list_float")
    assert has_symbol(lib, "lmcache_mla_buffer_to_paged_list_u16")
    assert has_symbol(lib, "lmcache_mla_paged_to_buffer_list_u16")
    assert has_symbol(lib, "lmcache_kv_buffer_to_paged_two_x_nl_u16")
    assert has_symbol(lib, "lmcache_paged_to_kv_buffer_two_x_nl_u16")
    for dtype_suffix in ("half", "bfloat16"):
        for base in (
            "musa_aiter_kv_paged_to_merged_linear",
            "musa_aiter_kv_merged_linear_to_paged",
            "musa_aiter_kv_paged_to_merged_linear_multi_layer",
            "musa_aiter_kv_merged_linear_to_paged_multi_layer",
            "musa_aiter_kv_paged_to_mla_linear",
            "musa_aiter_kv_mla_linear_to_paged",
        ):
            assert has_symbol(lib, f"{base}_{dtype_suffix}")


def test_lmcache_native_abi_symbols_are_exported():
    assert musa_aiter.native_lmcache_kv_transfer_abi_version() == 1
    assert callable(musa_aiter.lmcache_kv_buffer_to_paged)
    assert callable(musa_aiter.lmcache_paged_to_kv_buffer)
    assert callable(musa_aiter.lmcache_kv_buffer_to_paged_mla)
    assert callable(musa_aiter.lmcache_paged_to_kv_buffer_mla)
    assert callable(musa_aiter.lmcache_kv_paged_to_buffer)
    assert callable(musa_aiter.lmcache_kv_buffer_to_paged_two_x_nl)
    assert callable(musa_aiter.lmcache_kv_paged_to_buffer_two_x_nl)
    assert callable(musa_aiter.is_native_lmcache_two_x_nl_kv_transfer_available)
    assert callable(musa_aiter.lmcache_mla_buffer_to_paged)
    assert callable(musa_aiter.lmcache_mla_paged_to_buffer)


def test_lmcache_list_cache_alias_roundtrip_matches_current_stage2_adapter():
    nl, nb, bs, nh, hs = 2, 2, 4, 2, 3
    hidden = nh * hs
    slots = torch.tensor([0, 3, 5, -1, 7], dtype=torch.int64)
    source = [
        torch.arange(2 * nb * bs * nh * hs, dtype=torch.float32).reshape(2, nb, bs, nh, hs) + layer * 1000
        for layer in range(nl)
    ]
    destination = [torch.zeros_like(layer) for layer in source]
    memory = torch.empty(2, nl, slots.numel(), hidden, dtype=torch.float32)

    assert musa_aiter.lmcache_kv_paged_to_buffer(source, memory, slots, bs, nh, hs) is True
    assert musa_aiter.lmcache_kv_buffer_to_paged(memory, destination, slots, 0, bs, nh, hs) is True

    for src_layer, dst_layer in zip(source, destination, strict=True):
        src_k = src_layer[0].reshape(-1, hidden)
        src_v = src_layer[1].reshape(-1, hidden)
        dst_k = dst_layer[0].reshape(-1, hidden)
        dst_v = dst_layer[1].reshape(-1, hidden)
        valid_slots = slots[slots >= 0]
        torch.testing.assert_close(dst_k[valid_slots], src_k[valid_slots])
        torch.testing.assert_close(dst_v[valid_slots], src_v[valid_slots])


def test_lmcache_two_x_nl_roundtrip_matches_sglang_mp_format():
    nl, nb, bs, nh, hs = 3, 2, 4, 2, 3
    hidden = nh * hs
    slots = torch.tensor([0, 3, 5, -1, 7], dtype=torch.int64)
    source = [
        [
            torch.arange(nb * bs * nh * hs, dtype=torch.float32).reshape(nb, bs, nh, hs)
            + kv_axis * 10_000
            + layer * 1_000
            for layer in range(nl)
        ]
        for kv_axis in range(2)
    ]
    destination = [
        [torch.full_like(source[kv_axis][layer], -7.0) for layer in range(nl)]
        for kv_axis in range(2)
    ]
    memory = torch.empty(2, nl, slots.numel(), hidden, dtype=torch.float32)

    assert musa_aiter.lmcache_kv_paged_to_buffer(source, memory, slots, bs, nh, hs) is True
    assert musa_aiter.lmcache_kv_buffer_to_paged(memory, destination, slots, 0, bs, nh, hs) is True

    valid_slots = slots[slots >= 0]
    for kv_axis in range(2):
        for layer in range(nl):
            src_flat = source[kv_axis][layer].reshape(-1, hidden)
            dst_flat = destination[kv_axis][layer].reshape(-1, hidden)
            torch.testing.assert_close(dst_flat[valid_slots], src_flat[valid_slots])
    assert memory[:, :, 3].abs().max() == 0


def test_lmcache_two_x_nl_scatter_honors_prefix_skip():
    nl, nb, bs, nh, hs = 2, 2, 2, 1, 2
    slots = torch.tensor([0, 3], dtype=torch.int32)
    memory = torch.arange(2 * nl * 4 * nh * hs, dtype=torch.float32).reshape(
        2, nl, 4, nh * hs
    )
    destination = [
        [torch.full((nb, bs, nh, hs), -1.0) for _ in range(nl)] for _ in range(2)
    ]

    assert (
        musa_aiter.lmcache_kv_buffer_to_paged_two_x_nl(
            memory, destination, slots, 2, bs, nh, hs
        )
        is True
    )

    for kv_axis in range(2):
        for layer in range(nl):
            flat = destination[kv_axis][layer].reshape(-1, nh * hs)
            torch.testing.assert_close(flat[0], memory[kv_axis, layer, 2])
            torch.testing.assert_close(flat[3], memory[kv_axis, layer, 3])
            torch.testing.assert_close(flat[1], torch.full_like(flat[1], -1.0))


def test_lmcache_two_x_nl_rejects_invalid_nested_layout():
    memory = torch.zeros(2, 2, 2, 4)
    slots = torch.tensor([0, 1], dtype=torch.int64)
    mismatched_layers = [
        [torch.zeros(1, 2, 1, 4), torch.zeros(1, 2, 1, 4)],
        [torch.zeros(1, 2, 1, 4)],
    ]

    try:
        musa_aiter.lmcache_kv_paged_to_buffer(
            mismatched_layers, memory, slots, 2, 1, 4
        )
    except ValueError as exc:
        assert "layer" in str(exc)
    else:
        raise AssertionError("expected mismatched TWO_X_NL layer rejection")

    valid_caches = [
        [torch.zeros(1, 2, 1, 4) for _ in range(2)] for _ in range(2)
    ]
    with pytest.raises(ValueError, match="int32 or int64"):
        musa_aiter.lmcache_kv_paged_to_buffer(
            valid_caches, memory, slots.to(torch.float32), 2, 1, 4
        )
    with pytest.raises(ValueError, match="transfer range"):
        musa_aiter.lmcache_kv_buffer_to_paged_two_x_nl(
            memory, valid_caches, slots, 1, 2, 1, 4
        )


def test_lmcache_two_x_nl_empty_and_noncontiguous_buffer_contract():
    nl, nb, bs, nh, hs = 2, 1, 2, 1, 2
    caches = [
        [
            torch.arange(nb * bs * nh * hs, dtype=torch.float32).reshape(
                nb, bs, nh, hs
            )
            + kv_axis * 100
            + layer * 10
            for layer in range(nl)
        ]
        for kv_axis in range(2)
    ]
    backing = torch.empty(nl, 2, 2, nh * hs)
    noncontiguous_memory = backing.transpose(0, 1)
    assert not noncontiguous_memory.is_contiguous()
    slots = torch.tensor([0, 1], dtype=torch.int64)

    assert (
        musa_aiter.lmcache_kv_paged_to_buffer_two_x_nl(
            caches, noncontiguous_memory, slots, bs, nh, hs
        )
        is True
    )
    for kv_axis in range(2):
        for layer in range(nl):
            torch.testing.assert_close(
                noncontiguous_memory[kv_axis, layer],
                caches[kv_axis][layer].reshape(2, nh * hs),
            )

    empty_slots = torch.empty(0, dtype=torch.int64)
    empty_memory = torch.empty(2, nl, 0, nh * hs)
    assert (
        musa_aiter.lmcache_kv_paged_to_buffer_two_x_nl(
            caches, empty_memory, empty_slots, bs, nh, hs
        )
        is True
    )


def test_lmcache_mla_list_cache_alias_roundtrip_matches_current_stage2_adapter():
    nl, nb, bs, hs = 2, 2, 4, 5
    slots = torch.tensor([0, 4, -1, 7], dtype=torch.int64)
    source = [torch.arange(nb * bs * hs, dtype=torch.float32).reshape(nb, bs, hs) + layer * 1000 for layer in range(nl)]
    destination = [torch.zeros_like(layer) for layer in source]
    memory = torch.empty(1, nl, slots.numel(), hs, dtype=torch.float32)

    assert musa_aiter.lmcache_mla_paged_to_buffer(source, memory, slots, bs, hs) is True
    assert musa_aiter.lmcache_mla_buffer_to_paged(memory, destination, slots, 0, bs, hs) is True

    for src_layer, dst_layer in zip(source, destination, strict=True):
        src_flat = src_layer.reshape(-1, hs)
        dst_flat = dst_layer.reshape(-1, hs)
        valid_slots = slots[slots >= 0]
        torch.testing.assert_close(dst_flat[valid_slots], src_flat[valid_slots])


def test_lmcache_non_mla_roundtrip_matches_layout_contract():
    nl, nb, bs, nh, hs = 3, 2, 4, 2, 3
    slots = torch.tensor([0, 3, 5, -1, 7], dtype=torch.int32)
    tokens = int(slots.numel())
    kv_buffer = torch.arange(2 * nl * tokens * nh * hs, dtype=torch.float32).reshape(2, nl, tokens, nh * hs)
    paged = torch.zeros(nl, 2, nb, bs, nh, hs)

    assert musa_aiter.lmcache_kv_buffer_to_paged(kv_buffer, paged, slots) is True
    recovered = torch.empty_like(kv_buffer)
    assert musa_aiter.lmcache_paged_to_kv_buffer(paged, recovered, slots) is True

    torch.testing.assert_close(recovered[:, :, 0], kv_buffer[:, :, 0])
    torch.testing.assert_close(recovered[:, :, 1], kv_buffer[:, :, 1])
    torch.testing.assert_close(recovered[:, :, 2], kv_buffer[:, :, 2])
    torch.testing.assert_close(recovered[:, :, 4], kv_buffer[:, :, 4])
    assert recovered[:, :, 3].abs().max() == 0


def test_lmcache_non_mla_scatter_skips_negative_slots():
    nl, nb, bs, nh, hs = 1, 1, 4, 1, 2
    slots = torch.tensor([-1, 2], dtype=torch.int64)
    kv_buffer = torch.ones(2, nl, 2, nh * hs)
    paged = torch.full((nl, 2, nb, bs, nh, hs), 7.0)

    assert musa_aiter.lmcache_kv_buffer_to_paged(kv_buffer, paged, slots) is True

    torch.testing.assert_close(paged[:, :, 0, 0], torch.full_like(paged[:, :, 0, 0], 7.0))
    torch.testing.assert_close(paged[:, :, 0, 2], torch.ones_like(paged[:, :, 0, 2]))


def test_lmcache_mla_roundtrip_matches_layout_contract():
    nl, nb, bs, hs = 2, 2, 4, 5
    slots = torch.tensor([0, 4, -1, 7], dtype=torch.int32)
    tokens = int(slots.numel())
    kv_buffer = torch.arange(nl * tokens * hs, dtype=torch.float32).reshape(nl, tokens, hs)
    paged = torch.zeros(nl, nb, bs, hs)

    assert musa_aiter.lmcache_kv_buffer_to_paged_mla(kv_buffer, paged, slots) is True
    recovered = torch.empty_like(kv_buffer)
    assert musa_aiter.lmcache_paged_to_kv_buffer_mla(paged, recovered, slots) is True

    torch.testing.assert_close(recovered[:, 0], kv_buffer[:, 0])
    torch.testing.assert_close(recovered[:, 1], kv_buffer[:, 1])
    torch.testing.assert_close(recovered[:, 3], kv_buffer[:, 3])
    assert recovered[:, 2].abs().max() == 0


def test_lmcache_rejects_invalid_shapes_and_slots():
    slots = torch.tensor([0, 8], dtype=torch.int32)
    kv_buffer = torch.zeros(2, 1, 2, 4)
    paged = torch.zeros(1, 2, 1, 4, 1, 4)

    try:
        musa_aiter.lmcache_kv_buffer_to_paged(kv_buffer, paged, slots)
    except ValueError as exc:
        assert "out of range" in str(exc)
    else:
        raise AssertionError("expected out-of-range slot rejection")

    try:
        musa_aiter.lmcache_kv_buffer_to_paged(torch.zeros(1, 2, 3), paged, torch.tensor([0]))
    except ValueError as exc:
        assert "rank" in str(exc)
    else:
        raise AssertionError("expected rank rejection")
