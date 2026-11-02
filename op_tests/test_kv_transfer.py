import torch

import musa_aiter


def test_native_kv_transfer_opt_in_falls_back_to_reference(monkeypatch, capsys):
    monkeypatch.setenv("MUSA_AITER_NATIVE_KV_TRANSFER", "1")
    monkeypatch.setenv("MUSA_AITER_TRACE_DISPATCH", "1")

    nb, kvh, d, blk = 2, 2, 4, 4
    t = 3
    k_lin = torch.randn(t, kvh, d)
    v_lin = torch.randn(t, kvh, d)
    k_cache = torch.zeros(nb, kvh, d, blk)
    v_cache = torch.zeros(nb, kvh, d, blk)
    slots = torch.tensor([0, 3, 5], dtype=torch.int32)

    musa_aiter.kv_linear_to_paged(k_lin, v_lin, k_cache, v_cache, slots)
    kr, vr = musa_aiter.kv_paged_to_linear(k_cache, v_cache, slots)

    torch.testing.assert_close(kr, k_lin)
    torch.testing.assert_close(vr, v_lin)
    err = capsys.readouterr().err
    assert "native kv_linear_to_paged requested but kernel is not built yet" in err
    assert "native kv_paged_to_linear requested but kernel is not built yet" in err


def test_kv_linear_to_paged_then_paged_to_linear_matches():
    nb, kvh, d, blk = 2, 3, 8, 4
    t = 5
    k_lin = torch.randn(t, kvh, d)
    v_lin = torch.randn(t, kvh, d)
    k_cache = torch.zeros(nb, kvh, d, blk)
    v_cache = torch.zeros(nb, kvh, d, blk)
    slots = torch.tensor([0, 2, 5, 6, 7], dtype=torch.int32)

    musa_aiter.kv_linear_to_paged(k_lin, v_lin, k_cache, v_cache, slots)
    kr, vr = musa_aiter.kv_paged_to_linear(k_cache, v_cache, slots)

    torch.testing.assert_close(kr, k_lin)
    torch.testing.assert_close(vr, v_lin)


def test_kv_gather_agrees_copy_to_kv_cache():
    nb, kvh, d, blk = 4, 2, 6, 4
    t = 3
    k_tok = torch.randn(t, kvh, d)
    v_tok = torch.randn(t, kvh, d)
    k_cache = torch.zeros(nb, kvh, d, blk)
    v_cache = torch.zeros(nb, kvh, d, blk)
    slots = torch.tensor([1, blk + 3, blk * 2 + 2], dtype=torch.int64)

    musa_aiter.copy_to_kv_cache(k_tok, v_tok, k_cache, v_cache, slots)

    kr, vr = musa_aiter.kv_paged_to_linear(k_cache, v_cache, slots)

    torch.testing.assert_close(kr, k_tok)
    torch.testing.assert_close(vr, v_tok)


def test_negative_slot_zeros_gather_and_skip_scatter():
    nb, kvh, d, blk = 3, 2, 4, 8
    t = 4
    k_lin = torch.randn(t, kvh, d)
    v_lin = torch.randn(t, kvh, d)
    slots = torch.tensor([-1, 14, -1, 19], dtype=torch.int32)

    k_cache_rand = torch.randn(nb, kvh, d, blk)
    v_cache_rand = torch.randn(nb, kvh, d, blk)
    kr, vr = musa_aiter.kv_paged_to_linear(k_cache_rand, v_cache_rand, slots)
    assert kr[0].abs().max() == 0 and kr[2].abs().max() == 0
    assert vr[0].abs().max() == 0 and vr[2].abs().max() == 0

    kz = torch.zeros(nb, kvh, d, blk)
    vz = torch.zeros(nb, kvh, d, blk)
    musa_aiter.kv_linear_to_paged(k_lin, v_lin, kz, vz, slots)

    kr2, vr2 = musa_aiter.kv_paged_to_linear(kz, vz, slots)
    torch.testing.assert_close(kr2[1], k_lin[1])
    torch.testing.assert_close(kr2[3], k_lin[3])
    torch.testing.assert_close(vr2[1], v_lin[1])
    torch.testing.assert_close(vr2[3], v_lin[3])
    assert kr2[0].abs().max() == 0 and kr2[2].abs().max() == 0


def test_kv_multi_layer_roundtrip_matches_single_layer_loop():
    L, nb, kvh, d, blk = 3, 2, 2, 5, 4
    t = 5
    k_lin = torch.randn(L, t, kvh, d)
    v_lin = torch.randn(L, t, kvh, d)
    k_cache = torch.zeros(L, nb, kvh, d, blk)
    v_cache = torch.zeros(L, nb, kvh, d, blk)
    # Distinct flattened slots (avoid duplicate physical cells ↔ last writer on scatter).
    slots = torch.tensor([0, 2, blk + 2, blk + 1, nb * blk - 1], dtype=torch.int32)

    musa_aiter.kv_linear_to_paged_multi_layer(k_lin, v_lin, k_cache, v_cache, slots)
    kr, vr = musa_aiter.kv_paged_to_linear_multi_layer(k_cache, v_cache, slots)

    torch.testing.assert_close(kr, k_lin)
    torch.testing.assert_close(vr, v_lin)

    for layer in range(L):
        kc1 = torch.zeros(nb, kvh, d, blk)
        vc1 = torch.zeros(nb, kvh, d, blk)
        musa_aiter.kv_linear_to_paged(k_lin[layer], v_lin[layer], kc1, vc1, slots)
        r1k, r1v = musa_aiter.kv_paged_to_linear(kc1, vc1, slots)
        torch.testing.assert_close(r1k, k_lin[layer])
        torch.testing.assert_close(r1v, v_lin[layer])
        torch.testing.assert_close(k_cache[layer], kc1)
        torch.testing.assert_close(v_cache[layer], vc1)


def test_kv_multi_layer_negative_skips_gather_and_scatter():
    L = 2
    nb, kvh, d, blk = 2, 2, 4, 8
    t = 4
    slots = torch.tensor([-1, 10, -1, 15], dtype=torch.int32)

    k_cache = torch.randn(L, nb, kvh, d, blk)
    v_cache = torch.randn(L, nb, kvh, d, blk)

    kr, vr = musa_aiter.kv_paged_to_linear_multi_layer(k_cache, v_cache, slots)
    assert kr[:, 0].abs().max() == 0 and kr[:, 2].abs().max() == 0
    assert vr[:, 0].abs().max() == 0 and vr[:, 2].abs().max() == 0

    k_lin = torch.randn(L, t, kvh, d)
    v_lin = torch.randn(L, t, kvh, d)
    z = torch.zeros(L, nb, kvh, d, blk)
    w = torch.zeros(L, nb, kvh, d, blk)
    musa_aiter.kv_linear_to_paged_multi_layer(k_lin, v_lin, z, w, slots)
    rk2, rv2 = musa_aiter.kv_paged_to_linear_multi_layer(z, w, slots)
    torch.testing.assert_close(rk2[:, 1], k_lin[:, 1])
    torch.testing.assert_close(rk2[:, 3], k_lin[:, 3])
    torch.testing.assert_close(rv2[:, 1], v_lin[:, 1])
    torch.testing.assert_close(rv2[:, 3], v_lin[:, 3])
    assert rk2[:, 0].abs().max() == 0 and rk2[:, 2].abs().max() == 0


def test_kv_merged_linear_roundtrip_matches_separate_layout():
    nb, kvh, d, blk = 2, 2, 4, 4
    t = 4
    merged = torch.randn(t, 2, kvh, d)
    slots = torch.tensor([0, 3, -1, 7], dtype=torch.int32)
    k_cache = torch.zeros(nb, kvh, d, blk)
    v_cache = torch.zeros(nb, kvh, d, blk)

    musa_aiter.kv_merged_linear_to_paged(merged, k_cache, v_cache, slots)
    recovered = musa_aiter.kv_paged_to_merged_linear(k_cache, v_cache, slots)

    torch.testing.assert_close(recovered[0], merged[0])
    torch.testing.assert_close(recovered[1], merged[1])
    torch.testing.assert_close(recovered[3], merged[3])
    assert recovered[2].abs().max() == 0

    kr, vr = musa_aiter.kv_paged_to_linear(k_cache, v_cache, slots)
    torch.testing.assert_close(recovered[:, 0], kr)
    torch.testing.assert_close(recovered[:, 1], vr)


def test_kv_merged_linear_multi_layer_roundtrip_matches_separate_layout():
    L, nb, kvh, d, blk = 3, 2, 2, 4, 4
    t = 4
    merged = torch.randn(L, t, 2, kvh, d)
    slots = torch.tensor([0, 3, -1, 7], dtype=torch.int32)
    k_cache = torch.zeros(L, nb, kvh, d, blk)
    v_cache = torch.zeros(L, nb, kvh, d, blk)

    musa_aiter.kv_merged_linear_to_paged_multi_layer(merged, k_cache, v_cache, slots)
    recovered = musa_aiter.kv_paged_to_merged_linear_multi_layer(k_cache, v_cache, slots)

    torch.testing.assert_close(recovered[:, 0], merged[:, 0])
    torch.testing.assert_close(recovered[:, 1], merged[:, 1])
    torch.testing.assert_close(recovered[:, 3], merged[:, 3])
    assert recovered[:, 2].abs().max() == 0

    kr, vr = musa_aiter.kv_paged_to_linear_multi_layer(k_cache, v_cache, slots)
    torch.testing.assert_close(recovered[:, :, 0], kr)
    torch.testing.assert_close(recovered[:, :, 1], vr)


def test_kv_mla_asymmetric_roundtrip_and_negative_slots():
    nb, blk, kv_dim, pe_dim = 2, 4, 6, 3
    t = 4
    kv_linear = torch.randn(t, kv_dim)
    k_pe_linear = torch.randn(t, pe_dim)
    kv_cache = torch.zeros(nb, blk, kv_dim)
    k_pe_cache = torch.zeros(nb, blk, pe_dim)
    slots = torch.tensor([0, 3, -1, 7], dtype=torch.int32)

    musa_aiter.kv_mla_linear_to_paged(kv_linear, k_pe_linear, kv_cache, k_pe_cache, slots)
    recovered_kv, recovered_pe = musa_aiter.kv_paged_to_mla_linear(kv_cache, k_pe_cache, slots)

    torch.testing.assert_close(recovered_kv[0], kv_linear[0])
    torch.testing.assert_close(recovered_kv[1], kv_linear[1])
    torch.testing.assert_close(recovered_kv[3], kv_linear[3])
    torch.testing.assert_close(recovered_pe[0], k_pe_linear[0])
    torch.testing.assert_close(recovered_pe[1], k_pe_linear[1])
    torch.testing.assert_close(recovered_pe[3], k_pe_linear[3])
    assert recovered_kv[2].abs().max() == 0
    assert recovered_pe[2].abs().max() == 0


def test_kv_mla_asymmetric_rejects_mismatched_widths():
    kv_cache = torch.zeros(2, 4, 6)
    k_pe_cache = torch.zeros(2, 4, 3)
    kv_linear = torch.zeros(2, 7)
    k_pe_linear = torch.zeros(2, 3)
    slots = torch.tensor([0, 1], dtype=torch.int32)

    try:
        musa_aiter.kv_mla_linear_to_paged(kv_linear, k_pe_linear, kv_cache, k_pe_cache, slots)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for mismatched MLA KV width")


def test_kv_cachegen_encode_decode_roundtrip_and_cdf():
    tensor = torch.linspace(-1.0, 1.0, steps=17, dtype=torch.float32).reshape(1, 17)

    encoded, codebook, cdf = musa_aiter.kv_cachegen_encode(tensor, num_bins=16)
    decoded = musa_aiter.kv_cachegen_decode(encoded, codebook, dtype=tensor.dtype)

    assert encoded.dtype == torch.uint8
    assert codebook.shape == (16,)
    assert cdf.shape == (17,)
    assert cdf[0] == 0
    assert cdf[-1] == 1
    assert bool(torch.all(cdf[1:] >= cdf[:-1]))
    assert decoded.shape == tensor.shape
    assert torch.max(torch.abs(decoded - tensor)) <= (2.0 / 15.0)


def test_kv_cachegen_encode_constant_tensor():
    tensor = torch.full((2, 3), 5.0)

    encoded, codebook, cdf = musa_aiter.kv_cachegen_encode(tensor, num_bins=8)
    decoded = musa_aiter.kv_cachegen_decode(encoded, codebook)

    assert encoded.max() == 0
    torch.testing.assert_close(decoded, tensor)
    assert cdf[0] == 0
    assert bool(torch.all(cdf[1:] == 1))


def test_kv_cachegen_rejects_invalid_inputs():
    try:
        musa_aiter.kv_cachegen_encode(torch.arange(4, dtype=torch.int32))
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for integer encode input")

    try:
        musa_aiter.kv_cachegen_decode(torch.zeros(2, dtype=torch.int32), torch.zeros(4))
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for non-uint8 encoded input")


def test_kv_multi_layer_large_edge_roundtrip_with_sparse_slots():
    L, nb, kvh, d, blk = 5, 17, 3, 7, 8
    slots = torch.tensor([0, 3, -1, 15, 16, 31, -1, 63, 64, 95, 127, 135], dtype=torch.int32)
    t = int(slots.numel())
    k_lin = torch.randn(L, t, kvh, d)
    v_lin = torch.randn(L, t, kvh, d)
    k_cache = torch.zeros(L, nb, kvh, d, blk)
    v_cache = torch.zeros_like(k_cache)

    musa_aiter.kv_linear_to_paged_multi_layer(k_lin, v_lin, k_cache, v_cache, slots)
    k_recovered, v_recovered = musa_aiter.kv_paged_to_linear_multi_layer(k_cache, v_cache, slots)

    valid = slots >= 0
    torch.testing.assert_close(k_recovered[:, valid], k_lin[:, valid])
    torch.testing.assert_close(v_recovered[:, valid], v_lin[:, valid])
    assert k_recovered[:, ~valid].abs().max() == 0
    assert v_recovered[:, ~valid].abs().max() == 0


def test_kv_duplicate_slot_last_writer_wins_stress():
    nb, kvh, d, blk = 3, 2, 5, 4
    slots = torch.tensor([0, 1, 0, -1, 7, 7, 11], dtype=torch.int32)
    t = int(slots.numel())
    k_lin = torch.randn(t, kvh, d)
    v_lin = torch.randn(t, kvh, d)
    k_cache = torch.zeros(nb, kvh, d, blk)
    v_cache = torch.zeros_like(k_cache)

    musa_aiter.kv_linear_to_paged(k_lin, v_lin, k_cache, v_cache, slots)
    k_recovered, v_recovered = musa_aiter.kv_paged_to_linear(k_cache, v_cache, slots)

    torch.testing.assert_close(k_recovered[0], k_lin[2])
    torch.testing.assert_close(v_recovered[0], v_lin[2])
    torch.testing.assert_close(k_recovered[2], k_lin[2])
    torch.testing.assert_close(v_recovered[2], v_lin[2])
    torch.testing.assert_close(k_recovered[4], k_lin[5])
    torch.testing.assert_close(v_recovered[4], v_lin[5])
    torch.testing.assert_close(k_recovered[5], k_lin[5])
    torch.testing.assert_close(v_recovered[5], v_lin[5])
    assert k_recovered[3].abs().max() == 0
    assert v_recovered[3].abs().max() == 0


def test_kv_load_and_reshape_flash_matches_paged_blocks():
    nb, kvh, d, blk = 4, 2, 3, 4
    key_cache = torch.arange(nb * kvh * d * blk, dtype=torch.float32).reshape(nb, kvh, d, blk)
    value_cache = key_cache + 1000
    block_tables = torch.tensor([[1, 3], [0, 2]], dtype=torch.int32)
    seq_lens = torch.tensor([6, 3], dtype=torch.int32)

    k_dense, v_dense = musa_aiter.kv_load_and_reshape_flash(
        key_cache,
        value_cache,
        block_tables,
        seq_lens,
        max_seq_len=7,
    )

    assert k_dense.shape == (2, 7, kvh, d)
    expected0 = torch.cat(
        (key_cache[1].permute(2, 0, 1), key_cache[3].permute(2, 0, 1)),
        dim=0,
    )[:6]
    expected1 = key_cache[0].permute(2, 0, 1)[:3]
    torch.testing.assert_close(k_dense[0, :6], expected0)
    torch.testing.assert_close(v_dense[0, :6], expected0 + 1000)
    torch.testing.assert_close(k_dense[1, :3], expected1)
    torch.testing.assert_close(v_dense[1, :3], expected1 + 1000)
    assert k_dense[0, 6].abs().max() == 0
    assert k_dense[1, 3:].abs().max() == 0
    assert v_dense[1, 3:].abs().max() == 0


def test_kv_load_and_reshape_flash_uses_out_buffers_and_rejects_invalid_block():
    key_cache = torch.zeros(2, 1, 2, 4)
    value_cache = torch.zeros_like(key_cache)
    block_tables = torch.tensor([[0, 2]], dtype=torch.int32)
    seq_lens = torch.tensor([5], dtype=torch.int32)
    out_k = torch.empty(1, 5, 1, 2)
    out_v = torch.empty_like(out_k)

    try:
        musa_aiter.kv_load_and_reshape_flash(key_cache, value_cache, block_tables, seq_lens, out_k=out_k, out_v=out_v)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for invalid block id")


def test_kv_copy_paged_blocks_single_layer():
    src_k = torch.arange(3 * 2 * 4 * 5, dtype=torch.float32).reshape(3, 2, 4, 5)
    src_v = src_k + 1000
    dst_k = torch.zeros(4, 2, 4, 5)
    dst_v = torch.zeros_like(dst_k)
    mapping = torch.tensor([[0, 2], [2, 0], [-1, 1]], dtype=torch.int32)

    copied = musa_aiter.kv_copy_paged_blocks(src_k, src_v, dst_k, dst_v, mapping)

    assert copied == 2
    torch.testing.assert_close(dst_k[2], src_k[0])
    torch.testing.assert_close(dst_v[2], src_v[0])
    torch.testing.assert_close(dst_k[0], src_k[2])
    torch.testing.assert_close(dst_v[0], src_v[2])
    assert dst_k[1].abs().max() == 0
    assert dst_v[1].abs().max() == 0


def test_kv_copy_paged_blocks_multi_layer():
    src_k = torch.arange(2 * 3 * 2 * 4 * 5, dtype=torch.float32).reshape(2, 3, 2, 4, 5)
    src_v = src_k + 1000
    dst_k = torch.zeros(2, 4, 2, 4, 5)
    dst_v = torch.zeros_like(dst_k)
    mapping = torch.tensor([[1, 3], [2, 0]], dtype=torch.int64)

    copied = musa_aiter.kv_copy_paged_blocks(src_k, src_v, dst_k, dst_v, mapping)

    assert copied == 2
    torch.testing.assert_close(dst_k[:, 3], src_k[:, 1])
    torch.testing.assert_close(dst_v[:, 3], src_v[:, 1])
    torch.testing.assert_close(dst_k[:, 0], src_k[:, 2])
    torch.testing.assert_close(dst_v[:, 0], src_v[:, 2])
    assert dst_k[:, 1].abs().max() == 0
    assert dst_v[:, 1].abs().max() == 0


def test_kv_copy_paged_blocks_rejects_out_of_bounds():
    src_k = torch.zeros(2, 1, 2, 3)
    src_v = torch.zeros_like(src_k)
    dst_k = torch.zeros_like(src_k)
    dst_v = torch.zeros_like(dst_k)
    mapping = torch.tensor([[2, 0]], dtype=torch.int32)

    try:
        musa_aiter.kv_copy_paged_blocks(src_k, src_v, dst_k, dst_v, mapping)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for out-of-bounds block mapping")


def test_kv_fp8_reference_single_layer_roundtrip_and_negative_slots():
    nb, kvh, d, blk = 3, 2, 8, 4
    t = 5
    k_lin = torch.linspace(-1.5, 1.5, steps=t * kvh * d, dtype=torch.float32).reshape(t, kvh, d)
    v_lin = torch.linspace(1.25, -1.25, steps=t * kvh * d, dtype=torch.float32).reshape(t, kvh, d)
    k_cache = torch.zeros(nb, kvh, d, blk, dtype=torch.uint8)
    v_cache = torch.zeros_like(k_cache)
    slots = torch.tensor([0, 7, -1, 5, 9], dtype=torch.int32)
    k_scale = torch.tensor(0.125, dtype=torch.float32)
    v_scale = torch.tensor(0.125, dtype=torch.float32)

    musa_aiter.kv_linear_to_paged_fp8(k_lin, v_lin, k_cache, v_cache, slots, k_scale, v_scale, fp8_format="e4m3")
    rk, rv = musa_aiter.kv_paged_fp8_to_linear(k_cache, v_cache, slots, k_scale, v_scale, out_dtype=torch.float32, fp8_format="e4m3")

    assert k_cache.dtype == torch.uint8
    assert v_cache.dtype == torch.uint8
    assert rk.shape == k_lin.shape
    assert rv.shape == v_lin.shape
    valid = slots >= 0
    torch.testing.assert_close(rk[valid], k_lin[valid], atol=0.065, rtol=0.08)
    torch.testing.assert_close(rv[valid], v_lin[valid], atol=0.065, rtol=0.08)
    assert rk[~valid].abs().max() == 0
    assert rv[~valid].abs().max() == 0


def test_kv_fp8_reference_multi_layer_roundtrip_e5m2():
    L, nb, kvh, d, blk = 2, 3, 2, 8, 4
    t = 4
    k_lin = torch.linspace(-8.0, 8.0, steps=L * t * kvh * d, dtype=torch.float32).reshape(L, t, kvh, d)
    v_lin = torch.linspace(6.0, -6.0, steps=L * t * kvh * d, dtype=torch.float32).reshape(L, t, kvh, d)
    k_cache = torch.zeros(L, nb, kvh, d, blk, dtype=torch.uint8)
    v_cache = torch.zeros_like(k_cache)
    slots = torch.tensor([0, 4, -1, 11], dtype=torch.int64)
    k_scale = 0.25
    v_scale = 0.25

    musa_aiter.kv_linear_to_paged_fp8_multi_layer(k_lin, v_lin, k_cache, v_cache, slots, k_scale, v_scale, fp8_format="e5m2")
    rk, rv = musa_aiter.kv_paged_fp8_to_linear_multi_layer(k_cache, v_cache, slots, k_scale, v_scale, out_dtype=torch.float32, fp8_format="e5m2")

    valid = slots >= 0
    torch.testing.assert_close(rk[:, valid], k_lin[:, valid], atol=0.55, rtol=0.16)
    torch.testing.assert_close(rv[:, valid], v_lin[:, valid], atol=0.55, rtol=0.16)
    assert rk[:, ~valid].abs().max() == 0
    assert rv[:, ~valid].abs().max() == 0


def test_kv_fp8_reference_per_head_scales_roundtrip():
    nb, kvh, d, blk = 3, 2, 8, 4
    t = 5
    k_lin = torch.linspace(-1.5, 1.5, steps=t * kvh * d, dtype=torch.float32).reshape(t, kvh, d)
    v_lin = torch.linspace(1.25, -1.25, steps=t * kvh * d, dtype=torch.float32).reshape(t, kvh, d)
    k_cache = torch.zeros(nb, kvh, d, blk, dtype=torch.uint8)
    v_cache = torch.zeros_like(k_cache)
    slots = torch.tensor([0, 7, -1, 5, 9], dtype=torch.int32)
    k_scale = torch.tensor([0.125, 0.25], dtype=torch.float32)
    v_scale = torch.tensor([0.25, 0.125], dtype=torch.float32)

    musa_aiter.kv_linear_to_paged_fp8(k_lin, v_lin, k_cache, v_cache, slots, k_scale, v_scale, fp8_format="e4m3")
    rk, rv = musa_aiter.kv_paged_fp8_to_linear(k_cache, v_cache, slots, k_scale, v_scale, out_dtype=torch.float32, fp8_format="e4m3")

    valid = slots >= 0
    torch.testing.assert_close(rk[valid], k_lin[valid], atol=0.13, rtol=0.12)
    torch.testing.assert_close(rv[valid], v_lin[valid], atol=0.13, rtol=0.12)
    assert rk[~valid].abs().max() == 0
    assert rv[~valid].abs().max() == 0


def test_kv_fp8_reference_multi_layer_per_layer_head_scales_roundtrip():
    L, nb, kvh, d, blk = 2, 3, 2, 8, 4
    t = 4
    k_lin = torch.linspace(-8.0, 8.0, steps=L * t * kvh * d, dtype=torch.float32).reshape(L, t, kvh, d)
    v_lin = torch.linspace(6.0, -6.0, steps=L * t * kvh * d, dtype=torch.float32).reshape(L, t, kvh, d)
    k_cache = torch.zeros(L, nb, kvh, d, blk, dtype=torch.uint8)
    v_cache = torch.zeros_like(k_cache)
    slots = torch.tensor([0, 4, -1, 11], dtype=torch.int64)
    k_scale = torch.tensor([[0.25, 0.5], [0.5, 0.25]], dtype=torch.float32)
    v_scale = torch.tensor([[0.5, 0.25], [0.25, 0.5]], dtype=torch.float32)

    musa_aiter.kv_linear_to_paged_fp8_multi_layer(k_lin, v_lin, k_cache, v_cache, slots, k_scale, v_scale, fp8_format="e5m2")
    rk, rv = musa_aiter.kv_paged_fp8_to_linear_multi_layer(k_cache, v_cache, slots, k_scale, v_scale, out_dtype=torch.float32, fp8_format="e5m2")

    valid = slots >= 0
    torch.testing.assert_close(rk[:, valid], k_lin[:, valid], atol=1.1, rtol=0.20)
    torch.testing.assert_close(rv[:, valid], v_lin[:, valid], atol=1.1, rtol=0.20)
    assert rk[:, ~valid].abs().max() == 0
    assert rv[:, ~valid].abs().max() == 0


def test_kv_fp8_reference_rejects_invalid_inputs():
    k_lin = torch.zeros(2, 1, 4)
    v_lin = torch.zeros_like(k_lin)
    k_cache = torch.zeros(1, 1, 4, 2, dtype=torch.uint8)
    v_cache = torch.zeros_like(k_cache)
    slots = torch.tensor([0, 1], dtype=torch.int32)

    for bad_scale in (0.0, -1.0, float("inf")):
        try:
            musa_aiter.kv_linear_to_paged_fp8(k_lin, v_lin, k_cache, v_cache, slots, bad_scale, 1.0)
        except ValueError:
            pass
        else:
            raise AssertionError("expected ValueError for invalid FP8 scale")

    try:
        musa_aiter.kv_linear_to_paged_fp8(k_lin, v_lin, k_cache.float(), v_cache, slots, 1.0, 1.0)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for non-uint8 FP8 cache")

    try:
        musa_aiter.kv_linear_to_paged_fp8(k_lin, v_lin, k_cache, v_cache, slots, 1.0, 1.0, fp8_format="bad")
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for unsupported FP8 format")


def test_invalid_slot_raises():
    nb, kvh, d, blk = 2, 1, 4, 4
    kc = torch.zeros(nb, kvh, d, blk)
    vc = torch.zeros(nb, kvh, d, blk)
    slots_bad = torch.tensor([0, nb * blk])
    try:
        musa_aiter.kv_paged_to_linear(kc, vc, slots_bad)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for out-of-bounds slot")

    kl = torch.zeros(2, kvh, d)
    vl = torch.zeros(2, kvh, d)
    bad_scatter_slots = torch.tensor([0, nb * blk + 5])
    try:
        musa_aiter.kv_linear_to_paged(kl, vl, kc, vc, bad_scatter_slots)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError")

    L = 2
    k_big = torch.zeros(L, nb, kvh, d, blk)
    v_big = torch.zeros_like(k_big)
    kl4 = torch.zeros(L, 2, kvh, d)
    vl4 = torch.zeros_like(kl4)
    try:
        musa_aiter.kv_linear_to_paged_multi_layer(kl4, vl4, k_big, v_big, bad_scatter_slots)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for multi-layer OOB scatter")

    slots_bad2 = torch.tensor([0, nb * blk])
    try:
        musa_aiter.kv_paged_to_linear_multi_layer(k_big, v_big, slots_bad2)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for multi-layer OOB gather")

