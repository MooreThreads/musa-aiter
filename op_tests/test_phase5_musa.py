"""Phase 5: tests that require TorchMUSA and at least one MUSA device.

Skipped automatically on CPU-only CI; run on a Moore Threads lab:

    pytest op_tests/test_phase5_musa.py -v -m musa
"""

from __future__ import annotations

import os

import pytest
import torch

import musa_aiter


def _require_musa() -> None:
    if not hasattr(torch, "musa"):
        pytest.skip("install torch_musa for MUSA device tests")
    if not torch.musa.is_available():
        pytest.skip("no MUSA device visible to torch.musa")


@pytest.mark.musa
def test_rmsnorm_musa_tensor_shape() -> None:
    _require_musa()
    x = torch.randn(4, 64, device="musa", dtype=torch.float32)
    w = torch.ones(64, device="musa", dtype=torch.float32)
    y = musa_aiter.rmsnorm(x, w)
    assert y.shape == x.shape


@pytest.mark.musa
def test_gemm_musa_matches_reference() -> None:
    _require_musa()
    a = torch.randn(32, 16, device="musa", dtype=torch.float32)
    b = torch.randn(16, 32, device="musa", dtype=torch.float32)
    y = musa_aiter.gemm(a, b)
    ref = torch.matmul(a, b)
    torch.testing.assert_close(y.cpu(), ref.cpu(), rtol=1e-4, atol=1e-4)


@pytest.mark.musa
def test_flash_attn_musa_small() -> None:
    _require_musa()
    q = torch.randn(1, 4, 2, 8, device="musa", dtype=torch.float32)
    k = torch.randn(1, 4, 2, 8, device="musa", dtype=torch.float32)
    v = torch.randn(1, 4, 2, 8, device="musa", dtype=torch.float32)
    out = musa_aiter.flash_attn_func(q, k, v, causal=True)
    assert out.shape == q.shape


@pytest.mark.musa
@pytest.mark.native
def test_native_smoke_add_when_enabled() -> None:
    """ctypes smoke kernel; requires MUSA_AITER_NATIVE_SMOKE=1 and float32 musa tensors."""
    if os.environ.get("MUSA_AITER_NATIVE_SMOKE", "0") != "1":
        pytest.skip("set MUSA_AITER_NATIVE_SMOKE=1 to exercise ctypes native add")
    if not musa_aiter.is_native_add_available():
        pytest.skip("native smoke .so not built or symbols missing")
    _require_musa()
    a = torch.randn(256, device="musa", dtype=torch.float32)
    b = torch.randn(256, device="musa", dtype=torch.float32)
    out = musa_aiter.add(a, b, prefer_native=True)
    torch.testing.assert_close(out.cpu(), torch.add(a, b).cpu())


@pytest.mark.musa
@pytest.mark.native
def test_native_smoke_mul_when_enabled() -> None:
    if os.environ.get("MUSA_AITER_NATIVE_SMOKE", "0") != "1":
        pytest.skip("set MUSA_AITER_NATIVE_SMOKE=1 to exercise ctypes native mul")
    if not musa_aiter.is_native_mul_available():
        pytest.skip("native smoke mul .so not built or symbols missing")
    _require_musa()
    a = torch.randn(256, device="musa", dtype=torch.float32)
    b = torch.randn(256, device="musa", dtype=torch.float32)
    out = musa_aiter.multiply(a, b, prefer_native=True)
    torch.testing.assert_close(out.cpu(), torch.mul(a, b).cpu())


@pytest.mark.musa
@pytest.mark.native
def test_native_smoke_reduce_sum_when_enabled() -> None:
    if os.environ.get("MUSA_AITER_NATIVE_SMOKE", "0") != "1":
        pytest.skip("set MUSA_AITER_NATIVE_SMOKE=1 to exercise ctypes native reduce_sum")
    if not musa_aiter.is_native_reduce_sum_available():
        pytest.skip("native smoke reduce .so not built or symbols missing")
    _require_musa()
    x = torch.randn(12, 8, device="musa", dtype=torch.float32)
    out = musa_aiter.reduce_sum(x, prefer_native=True)
    torch.testing.assert_close(out.cpu(), torch.sum(x).cpu())


@pytest.mark.musa
@pytest.mark.native
def test_native_smoke_softmax_last_dim_when_enabled() -> None:
    if os.environ.get("MUSA_AITER_NATIVE_SMOKE", "0") != "1":
        pytest.skip("set MUSA_AITER_NATIVE_SMOKE=1 to exercise ctypes native softmax")
    if not musa_aiter.is_native_softmax_last_dim_available():
        pytest.skip("native softmax smoke .so not built or symbols missing")
    _require_musa()
    x = torch.randn(4, 64, device="musa", dtype=torch.float32)
    out = musa_aiter.softmax_last_dim(x, prefer_native=True)
    torch.testing.assert_close(out.cpu(), torch.softmax(x, dim=-1).cpu())


@pytest.mark.musa
@pytest.mark.native
def test_native_smoke_rmsnorm_when_enabled() -> None:
    if os.environ.get("MUSA_AITER_NATIVE_SMOKE", "0") != "1":
        pytest.skip("set MUSA_AITER_NATIVE_SMOKE=1 to exercise ctypes native rmsnorm")
    if not musa_aiter.is_native_rmsnorm_available():
        pytest.skip("native rmsnorm smoke .so not built or symbols missing")
    _require_musa()
    x = torch.randn(6, 48, device="musa", dtype=torch.float32)
    w = torch.randn(48, device="musa", dtype=torch.float32)
    out = musa_aiter.rmsnorm(x, w, eps=1e-6, prefer_native=True)
    ref = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + 1e-6) * w
    torch.testing.assert_close(out.cpu(), ref.cpu())


@pytest.mark.musa
def test_kv_transfer_musa_float32_roundtrip() -> None:
    _require_musa()
    nb, kvh, d, blk = 2, 2, 8, 8
    t = 4
    kl = torch.randn(t, kvh, d, device="musa", dtype=torch.float32)
    vl = torch.randn(t, kvh, d, device="musa", dtype=torch.float32)
    kc = torch.zeros(nb, kvh, d, blk, device="musa", dtype=torch.float32)
    vc = torch.zeros(nb, kvh, d, blk, device="musa", dtype=torch.float32)
    slots = torch.tensor([0, 15, 9, 3], dtype=torch.int32, device="musa")

    musa_aiter.kv_linear_to_paged(kl, vl, kc, vc, slots)
    rk, rv = musa_aiter.kv_paged_to_linear(kc, vc, slots)
    torch.testing.assert_close(rk.cpu(), kl.cpu())
    torch.testing.assert_close(rv.cpu(), vl.cpu())


@pytest.mark.musa
def test_kv_transfer_multi_layer_musa_roundtrip() -> None:
    _require_musa()
    L = 4
    nb, kvh, d, blk = 2, 2, 8, 8
    t = 4
    kl = torch.randn(L, t, kvh, d, device="musa", dtype=torch.float32)
    vl = torch.randn(L, t, kvh, d, device="musa", dtype=torch.float32)
    kc = torch.zeros(L, nb, kvh, d, blk, device="musa", dtype=torch.float32)
    vc = torch.zeros(L, nb, kvh, d, blk, device="musa", dtype=torch.float32)
    slots = torch.tensor([0, 15, 9, 3], dtype=torch.int32, device="musa")

    musa_aiter.kv_linear_to_paged_multi_layer(kl, vl, kc, vc, slots)
    rk, rv = musa_aiter.kv_paged_to_linear_multi_layer(kc, vc, slots)
    torch.testing.assert_close(rk.cpu(), kl.cpu())
    torch.testing.assert_close(rv.cpu(), vl.cpu())


@pytest.mark.musa
@pytest.mark.native
def test_native_kv_transfer_when_enabled() -> None:
    if os.environ.get("MUSA_AITER_NATIVE_KV_TRANSFER", "0") != "1":
        pytest.skip("set MUSA_AITER_NATIVE_KV_TRANSFER=1 to exercise native KV transfer")
    if not musa_aiter.is_native_kv_transfer_available():
        pytest.skip("native KV transfer .so not built or symbols missing")
    _require_musa()
    nb, kvh, d, blk = 2, 2, 8, 8
    t = 4
    kl = torch.randn(t, kvh, d, device="musa", dtype=torch.float32)
    vl = torch.randn(t, kvh, d, device="musa", dtype=torch.float32)
    kc = torch.zeros(nb, kvh, d, blk, device="musa", dtype=torch.float32)
    vc = torch.zeros(nb, kvh, d, blk, device="musa", dtype=torch.float32)
    slots = torch.tensor([0, 15, 9, 3], dtype=torch.int32, device="musa")

    musa_aiter.kv_linear_to_paged(kl, vl, kc, vc, slots)
    rk, rv = musa_aiter.kv_paged_to_linear(kc, vc, slots)
    torch.musa.synchronize()
    torch.testing.assert_close(rk.cpu(), kl.cpu())
    torch.testing.assert_close(rv.cpu(), vl.cpu())


@pytest.mark.musa
@pytest.mark.native
def test_native_kv_transfer_multi_layer_when_enabled() -> None:
    if os.environ.get("MUSA_AITER_NATIVE_KV_TRANSFER", "0") != "1":
        pytest.skip("set MUSA_AITER_NATIVE_KV_TRANSFER=1 to exercise native KV transfer")
    if not musa_aiter.is_native_kv_transfer_available():
        pytest.skip("native KV transfer .so not built or symbols missing")
    _require_musa()
    L = 4
    nb, kvh, d, blk = 2, 2, 8, 8
    t = 4
    kl = torch.randn(L, t, kvh, d, device="musa", dtype=torch.float32)
    vl = torch.randn(L, t, kvh, d, device="musa", dtype=torch.float32)
    kc = torch.zeros(L, nb, kvh, d, blk, device="musa", dtype=torch.float32)
    vc = torch.zeros(L, nb, kvh, d, blk, device="musa", dtype=torch.float32)
    slots = torch.tensor([0, 15, -1, 3], dtype=torch.int32, device="musa")

    musa_aiter.kv_linear_to_paged_multi_layer(kl, vl, kc, vc, slots)
    rk, rv = musa_aiter.kv_paged_to_linear_multi_layer(kc, vc, slots)
    torch.musa.synchronize()
    torch.testing.assert_close(rk[:, 0].cpu(), kl[:, 0].cpu())
    torch.testing.assert_close(rk[:, 1].cpu(), kl[:, 1].cpu())
    torch.testing.assert_close(rk[:, 3].cpu(), kl[:, 3].cpu())
    torch.testing.assert_close(rv[:, 0].cpu(), vl[:, 0].cpu())
    torch.testing.assert_close(rv[:, 1].cpu(), vl[:, 1].cpu())
    torch.testing.assert_close(rv[:, 3].cpu(), vl[:, 3].cpu())
    assert rk[:, 2].abs().max().item() == 0
    assert rv[:, 2].abs().max().item() == 0


@pytest.mark.musa
@pytest.mark.native
@pytest.mark.parametrize(
    ("fp8_format", "cache_dtype_name", "atol", "rtol"),
    [
        ("e4m3", "float8_e4m3fn", 0.13, 0.12),
        ("e5m2", "float8_e5m2", 1.1, 0.20),
        ("e5m2", "uint8", 1.1, 0.20),
    ],
)
def test_native_fp8_kv_multi_layer_per_layer_head_scales_when_enabled(
    fp8_format: str,
    cache_dtype_name: str,
    atol: float,
    rtol: float,
    capsys: pytest.CaptureFixture[str],
) -> None:
    if os.environ.get("MUSA_AITER_NATIVE_FP8_KV_CACHE", "0") != "1":
        pytest.skip("set MUSA_AITER_NATIVE_FP8_KV_CACHE=1 to exercise native FP8 KV cache")
    if not musa_aiter.is_native_fp8_kv_cache_available(fp8_format):
        pytest.skip(f"native FP8 {fp8_format.upper()} KV cache .so not built or symbols missing")
    _require_musa()
    if cache_dtype_name != "uint8" and not hasattr(torch, cache_dtype_name):
        pytest.skip(f"torch.{cache_dtype_name} is not available")
    nl, nb, kvh, d, blk = 2, 3, 2, 8, 4
    t = 5
    dtype = torch.bfloat16
    kl = torch.linspace(-2.0, 2.0, steps=nl * t * kvh * d, device="musa", dtype=torch.float32).reshape(nl, t, kvh, d).to(dtype)
    vl = torch.linspace(2.0, -2.0, steps=nl * t * kvh * d, device="musa", dtype=torch.float32).reshape(nl, t, kvh, d).to(dtype)
    cache_dtype = torch.uint8 if cache_dtype_name == "uint8" else getattr(torch, cache_dtype_name)
    kc = torch.empty(nl, nb, kvh, d, blk, device="musa", dtype=cache_dtype)
    vc = torch.empty_like(kc)
    slots = torch.tensor([0, 7, -1, 5, 9], dtype=torch.int32, device="musa")
    k_scale = torch.tensor([[0.125, 0.25], [0.25, 0.125]], dtype=torch.float32, device="musa")
    v_scale = torch.tensor([[0.25, 0.125], [0.125, 0.25]], dtype=torch.float32, device="musa")

    old_trace = os.environ.get("MUSA_AITER_TRACE_DISPATCH")
    os.environ["MUSA_AITER_TRACE_DISPATCH"] = "1"
    try:
        musa_aiter.kv_linear_to_paged_fp8_multi_layer(kl, vl, kc, vc, slots, k_scale, v_scale, fp8_format=fp8_format)
        rk, rv = musa_aiter.kv_paged_fp8_to_linear_multi_layer(kc, vc, slots, k_scale, v_scale, out_dtype=dtype, fp8_format=fp8_format)
        torch.musa.synchronize()
        captured = capsys.readouterr().err
    finally:
        if old_trace is None:
            os.environ.pop("MUSA_AITER_TRACE_DISPATCH", None)
        else:
            os.environ["MUSA_AITER_TRACE_DISPATCH"] = old_trace

    assert "op=kv_linear_to_paged_fp8_multi_layer backend=ctypes-musa-native-fp8" in captured
    assert "op=kv_paged_fp8_to_linear_multi_layer backend=ctypes-musa-native-fp8" in captured
    valid = slots.cpu() >= 0
    torch.testing.assert_close(rk.cpu()[:, valid], kl.cpu()[:, valid], atol=atol, rtol=rtol)
    torch.testing.assert_close(rv.cpu()[:, valid], vl.cpu()[:, valid], atol=atol, rtol=rtol)
    assert rk[:, 2].abs().max().item() == 0
    assert rv[:, 2].abs().max().item() == 0


@pytest.mark.musa
def test_kv_transfer_merged_musa_roundtrip() -> None:
    _require_musa()
    nb, kvh, d, blk = 2, 2, 8, 8
    t = 4
    merged = torch.randn(t, 2, kvh, d, device="musa", dtype=torch.float32)
    kc = torch.zeros(nb, kvh, d, blk, device="musa", dtype=torch.float32)
    vc = torch.zeros(nb, kvh, d, blk, device="musa", dtype=torch.float32)
    slots = torch.tensor([0, 15, -1, 3], dtype=torch.int32, device="musa")

    musa_aiter.kv_merged_linear_to_paged(merged, kc, vc, slots)
    recovered = musa_aiter.kv_paged_to_merged_linear(kc, vc, slots)
    torch.testing.assert_close(recovered[0].cpu(), merged[0].cpu())
    torch.testing.assert_close(recovered[1].cpu(), merged[1].cpu())
    torch.testing.assert_close(recovered[3].cpu(), merged[3].cpu())
    assert recovered[2].abs().max().item() == 0


@pytest.mark.musa
def test_kv_transfer_merged_multi_layer_musa_roundtrip() -> None:
    _require_musa()
    L = 4
    nb, kvh, d, blk = 2, 2, 8, 8
    t = 4
    merged = torch.randn(L, t, 2, kvh, d, device="musa", dtype=torch.float32)
    kc = torch.zeros(L, nb, kvh, d, blk, device="musa", dtype=torch.float32)
    vc = torch.zeros(L, nb, kvh, d, blk, device="musa", dtype=torch.float32)
    slots = torch.tensor([0, 15, -1, 3], dtype=torch.int32, device="musa")

    musa_aiter.kv_merged_linear_to_paged_multi_layer(merged, kc, vc, slots)
    recovered = musa_aiter.kv_paged_to_merged_linear_multi_layer(kc, vc, slots)
    torch.testing.assert_close(recovered[:, 0].cpu(), merged[:, 0].cpu())
    torch.testing.assert_close(recovered[:, 1].cpu(), merged[:, 1].cpu())
    torch.testing.assert_close(recovered[:, 3].cpu(), merged[:, 3].cpu())
    assert recovered[:, 2].abs().max().item() == 0


@pytest.mark.musa
@pytest.mark.native
def test_native_kv_transfer_merged_when_enabled() -> None:
    if os.environ.get("MUSA_AITER_NATIVE_KV_TRANSFER", "0") != "1":
        pytest.skip("set MUSA_AITER_NATIVE_KV_TRANSFER=1 to exercise native KV transfer")
    if not musa_aiter.is_native_kv_transfer_available():
        pytest.skip("native KV transfer .so not built or symbols missing")
    _require_musa()
    nb, kvh, d, blk = 2, 2, 8, 8
    t = 4
    merged = torch.randn(t, 2, kvh, d, device="musa", dtype=torch.float32)
    kc = torch.zeros(nb, kvh, d, blk, device="musa", dtype=torch.float32)
    vc = torch.zeros(nb, kvh, d, blk, device="musa", dtype=torch.float32)
    slots = torch.tensor([0, 15, -1, 3], dtype=torch.int32, device="musa")

    musa_aiter.kv_merged_linear_to_paged(merged, kc, vc, slots)
    recovered = musa_aiter.kv_paged_to_merged_linear(kc, vc, slots)
    torch.musa.synchronize()
    torch.testing.assert_close(recovered[0].cpu(), merged[0].cpu())
    torch.testing.assert_close(recovered[1].cpu(), merged[1].cpu())
    torch.testing.assert_close(recovered[3].cpu(), merged[3].cpu())
    assert recovered[2].abs().max().item() == 0


@pytest.mark.musa
@pytest.mark.native
def test_native_kv_transfer_merged_multi_layer_when_enabled() -> None:
    if os.environ.get("MUSA_AITER_NATIVE_KV_TRANSFER", "0") != "1":
        pytest.skip("set MUSA_AITER_NATIVE_KV_TRANSFER=1 to exercise native KV transfer")
    if not musa_aiter.is_native_kv_transfer_available():
        pytest.skip("native KV transfer .so not built or symbols missing")
    _require_musa()
    L = 4
    nb, kvh, d, blk = 2, 2, 8, 8
    t = 4
    merged = torch.randn(L, t, 2, kvh, d, device="musa", dtype=torch.float32)
    kc = torch.zeros(L, nb, kvh, d, blk, device="musa", dtype=torch.float32)
    vc = torch.zeros(L, nb, kvh, d, blk, device="musa", dtype=torch.float32)
    slots = torch.tensor([0, 15, -1, 3], dtype=torch.int32, device="musa")

    musa_aiter.kv_merged_linear_to_paged_multi_layer(merged, kc, vc, slots)
    recovered = musa_aiter.kv_paged_to_merged_linear_multi_layer(kc, vc, slots)
    torch.musa.synchronize()
    torch.testing.assert_close(recovered[:, 0].cpu(), merged[:, 0].cpu())
    torch.testing.assert_close(recovered[:, 1].cpu(), merged[:, 1].cpu())
    torch.testing.assert_close(recovered[:, 3].cpu(), merged[:, 3].cpu())
    assert recovered[:, 2].abs().max().item() == 0


@pytest.mark.musa
@pytest.mark.native
def test_native_kv_transfer_mla_when_enabled() -> None:
    if os.environ.get("MUSA_AITER_NATIVE_KV_TRANSFER", "0") != "1":
        pytest.skip("set MUSA_AITER_NATIVE_KV_TRANSFER=1 to exercise native KV transfer")
    if not musa_aiter.is_native_kv_transfer_available():
        pytest.skip("native KV transfer .so not built or symbols missing")
    _require_musa()
    nb, blk, kv_dim, pe_dim = 2, 8, 12, 5
    t = 4
    kv_linear = torch.randn(t, kv_dim, device="musa", dtype=torch.float32)
    k_pe_linear = torch.randn(t, pe_dim, device="musa", dtype=torch.float32)
    kv_cache = torch.zeros(nb, blk, kv_dim, device="musa", dtype=torch.float32)
    k_pe_cache = torch.zeros(nb, blk, pe_dim, device="musa", dtype=torch.float32)
    slots = torch.tensor([0, 15, -1, 3], dtype=torch.int32, device="musa")

    musa_aiter.kv_mla_linear_to_paged(kv_linear, k_pe_linear, kv_cache, k_pe_cache, slots)
    recovered_kv, recovered_pe = musa_aiter.kv_paged_to_mla_linear(kv_cache, k_pe_cache, slots)
    torch.musa.synchronize()
    torch.testing.assert_close(recovered_kv[0].cpu(), kv_linear[0].cpu())
    torch.testing.assert_close(recovered_kv[1].cpu(), kv_linear[1].cpu())
    torch.testing.assert_close(recovered_kv[3].cpu(), kv_linear[3].cpu())
    torch.testing.assert_close(recovered_pe[0].cpu(), k_pe_linear[0].cpu())
    torch.testing.assert_close(recovered_pe[1].cpu(), k_pe_linear[1].cpu())
    torch.testing.assert_close(recovered_pe[3].cpu(), k_pe_linear[3].cpu())
    assert recovered_kv[2].abs().max().item() == 0
    assert recovered_pe[2].abs().max().item() == 0


@pytest.mark.musa
def test_apply_rope_to_key_cache_musa() -> None:
    _require_musa()
    kc = torch.zeros(2, 1, 4, 4, device="musa", dtype=torch.float32)
    kc[0, :, :, 0] = torch.tensor([[1.0, 2.0, 3.0, 4.0]], device="musa")
    kc[1, :, :, 2] = torch.tensor([[5.0, 6.0, 7.0, 8.0]], device="musa")
    slots = torch.tensor([0, 6, -1], dtype=torch.int64, device="musa")
    positions = torch.tensor([0, 1, 2], dtype=torch.int64, device="musa")
    cos = torch.tensor([[1.0, 1.0], [0.0, 0.0], [1.0, 1.0]], device="musa")
    sin = torch.tensor([[0.0, 0.0], [1.0, 1.0], [0.0, 0.0]], device="musa")

    musa_aiter.apply_rope_to_key_cache(kc, slots, positions, cos, sin)
    torch.musa.synchronize()

    torch.testing.assert_close(kc[0, :, :, 0].cpu(), torch.tensor([[1.0, 2.0, 3.0, 4.0]]))
    torch.testing.assert_close(kc[1, :, :, 2].cpu(), torch.tensor([[-7.0, -8.0, 5.0, 6.0]]))
    assert kc[0, :, :, 1].abs().max().item() == 0


@pytest.mark.musa
def test_kv_cachegen_encode_decode_musa() -> None:
    _require_musa()
    tensor = torch.linspace(-1.0, 1.0, steps=17, dtype=torch.float32, device="musa").reshape(1, 17)

    encoded, codebook, cdf = musa_aiter.kv_cachegen_encode(tensor, num_bins=16)
    decoded = musa_aiter.kv_cachegen_decode(encoded, codebook, dtype=tensor.dtype)
    torch.musa.synchronize()

    assert encoded.dtype == torch.uint8
    assert codebook.shape == (16,)
    assert cdf.shape == (17,)
    assert cdf[-1].item() == 1
    assert torch.max(torch.abs(decoded - tensor)).item() <= (2.0 / 15.0)


@pytest.mark.musa
def test_kv_load_and_reshape_flash_musa() -> None:
    _require_musa()
    nb, kvh, d, blk = 4, 2, 3, 4
    key_cache = torch.arange(nb * kvh * d * blk, dtype=torch.float32, device="musa").reshape(nb, kvh, d, blk)
    value_cache = key_cache + 1000
    block_tables = torch.tensor([[1, 3], [0, 2]], dtype=torch.int32, device="musa")
    seq_lens = torch.tensor([6, 3], dtype=torch.int32, device="musa")

    k_dense, v_dense = musa_aiter.kv_load_and_reshape_flash(
        key_cache,
        value_cache,
        block_tables,
        seq_lens,
        max_seq_len=7,
    )
    torch.musa.synchronize()

    expected0 = torch.cat(
        (key_cache[1].permute(2, 0, 1), key_cache[3].permute(2, 0, 1)),
        dim=0,
    )[:6]
    expected1 = key_cache[0].permute(2, 0, 1)[:3]
    torch.testing.assert_close(k_dense[0, :6].cpu(), expected0.cpu())
    torch.testing.assert_close(v_dense[0, :6].cpu(), (expected0 + 1000).cpu())
    torch.testing.assert_close(k_dense[1, :3].cpu(), expected1.cpu())
    torch.testing.assert_close(v_dense[1, :3].cpu(), (expected1 + 1000).cpu())
    assert k_dense[1, 3:].abs().max().item() == 0
    assert v_dense[1, 3:].abs().max().item() == 0


@pytest.mark.musa
@pytest.mark.native
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_native_kv_transfer_half_dtypes_when_enabled(dtype: torch.dtype) -> None:
    if os.environ.get("MUSA_AITER_NATIVE_KV_TRANSFER", "0") != "1":
        pytest.skip("set MUSA_AITER_NATIVE_KV_TRANSFER=1 to exercise native KV transfer")
    if not musa_aiter.is_native_kv_transfer_available():
        pytest.skip("native KV transfer .so not built or symbols missing")
    _require_musa()
    nb, kvh, d, blk = 3, 2, 8, 8
    t = 5
    kl = torch.randn(t, kvh, d, device="musa", dtype=dtype)
    vl = torch.randn(t, kvh, d, device="musa", dtype=dtype)
    kc = torch.zeros(nb, kvh, d, blk, device="musa", dtype=dtype)
    vc = torch.zeros(nb, kvh, d, blk, device="musa", dtype=dtype)
    slots = torch.tensor([0, 15, -1, 9, 3], dtype=torch.int32, device="musa")

    musa_aiter.kv_linear_to_paged(kl, vl, kc, vc, slots)
    rk, rv = musa_aiter.kv_paged_to_linear(kc, vc, slots)
    torch.musa.synchronize()

    torch.testing.assert_close(rk[0].cpu(), kl[0].cpu())
    torch.testing.assert_close(rk[1].cpu(), kl[1].cpu())
    torch.testing.assert_close(rk[3].cpu(), kl[3].cpu())
    torch.testing.assert_close(rk[4].cpu(), kl[4].cpu())
    torch.testing.assert_close(rv[0].cpu(), vl[0].cpu())
    torch.testing.assert_close(rv[1].cpu(), vl[1].cpu())
    torch.testing.assert_close(rv[3].cpu(), vl[3].cpu())
    torch.testing.assert_close(rv[4].cpu(), vl[4].cpu())
    assert rk[2].abs().max().item() == 0
    assert rv[2].abs().max().item() == 0


@pytest.mark.musa
@pytest.mark.native
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_native_fp8_kv_e4m3_single_layer_when_enabled(dtype: torch.dtype) -> None:
    if os.environ.get("MUSA_AITER_NATIVE_FP8_KV_CACHE", "0") != "1":
        pytest.skip("set MUSA_AITER_NATIVE_FP8_KV_CACHE=1 to exercise native FP8 KV cache")
    if not musa_aiter.is_native_fp8_kv_cache_available():
        pytest.skip("native FP8 KV cache .so not built or symbols missing")
    _require_musa()
    if not hasattr(torch, "float8_e4m3fn"):
        pytest.skip("torch.float8_e4m3fn is not available")
    nb, kvh, d, blk = 3, 2, 8, 4
    t = 5
    kl = torch.linspace(-1.5, 1.5, steps=t * kvh * d, device="musa", dtype=torch.float32).reshape(t, kvh, d).to(dtype)
    vl = torch.linspace(1.25, -1.25, steps=t * kvh * d, device="musa", dtype=torch.float32).reshape(t, kvh, d).to(dtype)
    kc = torch.empty(nb, kvh, d, blk, device="musa", dtype=torch.float8_e4m3fn)
    vc = torch.empty_like(kc)
    slots = torch.tensor([0, 7, -1, 5, 9], dtype=torch.int32, device="musa")
    k_scale = torch.tensor(0.125, dtype=torch.float32, device="musa")
    v_scale = torch.tensor(0.125, dtype=torch.float32, device="musa")

    musa_aiter.kv_linear_to_paged_fp8(kl, vl, kc, vc, slots, k_scale, v_scale, fp8_format="e4m3")
    rk, rv = musa_aiter.kv_paged_fp8_to_linear(kc, vc, slots, k_scale, v_scale, out_dtype=dtype, fp8_format="e4m3")
    torch.musa.synchronize()

    valid = slots.cpu() >= 0
    torch.testing.assert_close(rk.cpu()[valid], kl.cpu()[valid], atol=0.085, rtol=0.10)
    torch.testing.assert_close(rv.cpu()[valid], vl.cpu()[valid], atol=0.085, rtol=0.10)
    assert rk[2].abs().max().item() == 0
    assert rv[2].abs().max().item() == 0


@pytest.mark.musa
@pytest.mark.native
def test_native_fp8_kv_e4m3_uint8_cache_when_enabled() -> None:
    if os.environ.get("MUSA_AITER_NATIVE_FP8_KV_CACHE", "0") != "1":
        pytest.skip("set MUSA_AITER_NATIVE_FP8_KV_CACHE=1 to exercise native FP8 KV cache")
    if not musa_aiter.is_native_fp8_kv_cache_available():
        pytest.skip("native FP8 KV cache .so not built or symbols missing")
    _require_musa()
    nb, kvh, d, blk = 2, 2, 8, 4
    t = 4
    dtype = torch.bfloat16
    kl = torch.linspace(-1.0, 1.0, steps=t * kvh * d, device="musa", dtype=torch.float32).reshape(t, kvh, d).to(dtype)
    vl = torch.linspace(1.0, -1.0, steps=t * kvh * d, device="musa", dtype=torch.float32).reshape(t, kvh, d).to(dtype)
    kc = torch.empty(nb, kvh, d, blk, device="musa", dtype=torch.uint8)
    vc = torch.empty_like(kc)
    slots = torch.tensor([0, 7, -1, 3], dtype=torch.int32, device="musa")
    k_scale = torch.tensor(0.125, dtype=torch.float32, device="musa")
    v_scale = torch.tensor(0.125, dtype=torch.float32, device="musa")

    musa_aiter.kv_linear_to_paged_fp8(kl, vl, kc, vc, slots, k_scale, v_scale, fp8_format="e4m3")
    rk, rv = musa_aiter.kv_paged_fp8_to_linear(kc, vc, slots, k_scale, v_scale, out_dtype=dtype, fp8_format="e4m3")
    torch.musa.synchronize()

    valid = slots.cpu() >= 0
    torch.testing.assert_close(rk.cpu()[valid], kl.cpu()[valid], atol=0.085, rtol=0.10)
    torch.testing.assert_close(rv.cpu()[valid], vl.cpu()[valid], atol=0.085, rtol=0.10)
    assert rk[2].abs().max().item() == 0
    assert rv[2].abs().max().item() == 0


@pytest.mark.musa
@pytest.mark.native
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_native_fp8_kv_e5m2_single_layer_when_enabled(dtype: torch.dtype) -> None:
    if os.environ.get("MUSA_AITER_NATIVE_FP8_KV_CACHE", "0") != "1":
        pytest.skip("set MUSA_AITER_NATIVE_FP8_KV_CACHE=1 to exercise native FP8 KV cache")
    if not musa_aiter.is_native_fp8_kv_cache_available("e5m2"):
        pytest.skip("native FP8 E5M2 KV cache .so not built or symbols missing")
    _require_musa()
    if not hasattr(torch, "float8_e5m2"):
        pytest.skip("torch.float8_e5m2 is not available")
    nb, kvh, d, blk = 3, 2, 8, 4
    t = 5
    kl = torch.linspace(-8.0, 8.0, steps=t * kvh * d, device="musa", dtype=torch.float32).reshape(t, kvh, d).to(dtype)
    vl = torch.linspace(6.0, -6.0, steps=t * kvh * d, device="musa", dtype=torch.float32).reshape(t, kvh, d).to(dtype)
    kc = torch.empty(nb, kvh, d, blk, device="musa", dtype=torch.float8_e5m2)
    vc = torch.empty_like(kc)
    slots = torch.tensor([0, 7, -1, 5, 9], dtype=torch.int32, device="musa")
    k_scale = torch.tensor(0.25, dtype=torch.float32, device="musa")
    v_scale = torch.tensor(0.25, dtype=torch.float32, device="musa")

    musa_aiter.kv_linear_to_paged_fp8(kl, vl, kc, vc, slots, k_scale, v_scale, fp8_format="e5m2")
    rk, rv = musa_aiter.kv_paged_fp8_to_linear(kc, vc, slots, k_scale, v_scale, out_dtype=dtype, fp8_format="e5m2")
    torch.musa.synchronize()

    valid = slots.cpu() >= 0
    torch.testing.assert_close(rk.cpu()[valid], kl.cpu()[valid], atol=0.65, rtol=0.18)
    torch.testing.assert_close(rv.cpu()[valid], vl.cpu()[valid], atol=0.65, rtol=0.18)
    assert rk[2].abs().max().item() == 0
    assert rv[2].abs().max().item() == 0


@pytest.mark.musa
@pytest.mark.native
def test_native_fp8_kv_e5m2_uint8_cache_when_enabled() -> None:
    if os.environ.get("MUSA_AITER_NATIVE_FP8_KV_CACHE", "0") != "1":
        pytest.skip("set MUSA_AITER_NATIVE_FP8_KV_CACHE=1 to exercise native FP8 KV cache")
    if not musa_aiter.is_native_fp8_kv_cache_available("e5m2"):
        pytest.skip("native FP8 E5M2 KV cache .so not built or symbols missing")
    _require_musa()
    nb, kvh, d, blk = 2, 2, 8, 4
    t = 4
    dtype = torch.bfloat16
    kl = torch.linspace(-4.0, 4.0, steps=t * kvh * d, device="musa", dtype=torch.float32).reshape(t, kvh, d).to(dtype)
    vl = torch.linspace(4.0, -4.0, steps=t * kvh * d, device="musa", dtype=torch.float32).reshape(t, kvh, d).to(dtype)
    kc = torch.empty(nb, kvh, d, blk, device="musa", dtype=torch.uint8)
    vc = torch.empty_like(kc)
    slots = torch.tensor([0, 7, -1, 3], dtype=torch.int32, device="musa")
    k_scale = torch.tensor(0.25, dtype=torch.float32, device="musa")
    v_scale = torch.tensor(0.25, dtype=torch.float32, device="musa")

    musa_aiter.kv_linear_to_paged_fp8(kl, vl, kc, vc, slots, k_scale, v_scale, fp8_format="e5m2")
    rk, rv = musa_aiter.kv_paged_fp8_to_linear(kc, vc, slots, k_scale, v_scale, out_dtype=dtype, fp8_format="e5m2")
    torch.musa.synchronize()

    valid = slots.cpu() >= 0
    torch.testing.assert_close(rk.cpu()[valid], kl.cpu()[valid], atol=0.65, rtol=0.18)
    torch.testing.assert_close(rv.cpu()[valid], vl.cpu()[valid], atol=0.65, rtol=0.18)
    assert rk[2].abs().max().item() == 0
    assert rv[2].abs().max().item() == 0


@pytest.mark.musa
@pytest.mark.native
def test_native_fp8_kv_uses_current_musa_stream_when_enabled() -> None:
    if os.environ.get("MUSA_AITER_NATIVE_FP8_KV_CACHE", "0") != "1":
        pytest.skip("set MUSA_AITER_NATIVE_FP8_KV_CACHE=1 to exercise native FP8 KV cache")
    if not musa_aiter.is_native_fp8_kv_cache_available("e4m3"):
        pytest.skip("native FP8 E4M3 KV cache .so not built or symbols missing")
    _require_musa()
    if not hasattr(torch, "float8_e4m3fn"):
        pytest.skip("torch.float8_e4m3fn is not available")
    nb, kvh, d, blk = 3, 2, 8, 4
    t = 5
    dtype = torch.bfloat16
    kl = torch.linspace(-2.0, 2.0, steps=t * kvh * d, device="musa", dtype=torch.float32).reshape(t, kvh, d).to(dtype)
    vl = torch.linspace(2.0, -2.0, steps=t * kvh * d, device="musa", dtype=torch.float32).reshape(t, kvh, d).to(dtype)
    kc = torch.empty(nb, kvh, d, blk, device="musa", dtype=torch.float8_e4m3fn)
    vc = torch.empty_like(kc)
    slots = torch.tensor([0, 7, -1, 5, 9], dtype=torch.int32, device="musa")
    k_scale = torch.tensor(0.125, dtype=torch.float32, device="musa")
    v_scale = torch.tensor(0.125, dtype=torch.float32, device="musa")
    stream = torch.musa.Stream()

    with torch.musa.stream(stream):
        musa_aiter.kv_linear_to_paged_fp8(kl, vl, kc, vc, slots, k_scale, v_scale, fp8_format="e4m3")
        rk, rv = musa_aiter.kv_paged_fp8_to_linear(kc, vc, slots, k_scale, v_scale, out_dtype=dtype, fp8_format="e4m3")
    stream.synchronize()

    valid = slots.cpu() >= 0
    torch.testing.assert_close(rk.cpu()[valid], kl.cpu()[valid], atol=0.085, rtol=0.10)
    torch.testing.assert_close(rv.cpu()[valid], vl.cpu()[valid], atol=0.085, rtol=0.10)
    assert rk[2].abs().max().item() == 0
    assert rv[2].abs().max().item() == 0


@pytest.mark.musa
@pytest.mark.native
@pytest.mark.parametrize(
    ("fp8_format", "cache_dtype_name", "scale", "head_dim", "block_size", "atol", "rtol"),
    [
        ("e4m3", "float8_e4m3fn", 0.125, 64, 16, 0.085, 0.10),
        ("e4m3", "float8_e4m3fn", 0.125, 128, 16, 0.085, 0.10),
        ("e4m3", "float8_e4m3fn", 0.125, 128, 32, 0.085, 0.10),
        ("e5m2", "float8_e5m2", 0.25, 64, 16, 0.65, 0.18),
        ("e5m2", "float8_e5m2", 0.25, 128, 16, 0.65, 0.18),
        ("e5m2", "float8_e5m2", 0.25, 128, 32, 0.65, 0.18),
        ("e5m2", "uint8", 0.25, 64, 16, 0.65, 0.18),
        ("e5m2", "uint8", 0.25, 128, 16, 0.65, 0.18),
        ("e5m2", "uint8", 0.25, 128, 32, 0.65, 0.18),
    ],
)
def test_native_fp8_kv_shape_specialized_gather_when_enabled(
    fp8_format: str,
    cache_dtype_name: str,
    scale: float,
    head_dim: int,
    block_size: int,
    atol: float,
    rtol: float,
    capsys: pytest.CaptureFixture[str],
) -> None:
    if os.environ.get("MUSA_AITER_NATIVE_FP8_KV_CACHE", "0") != "1" or os.environ.get("MUSA_AITER_NATIVE_FP8_KV_VECTORIZE", "0") != "1":
        pytest.skip("set MUSA_AITER_NATIVE_FP8_KV_CACHE=1 and MUSA_AITER_NATIVE_FP8_KV_VECTORIZE=1 to exercise shape-specialized native FP8 KV gather")
    if not musa_aiter.is_native_fp8_kv_vectorized_available():
        pytest.skip("native vectorized FP8 KV .so not built or symbols missing")
    _require_musa()
    if cache_dtype_name != "uint8" and not hasattr(torch, cache_dtype_name):
        pytest.skip(f"torch.{cache_dtype_name} is not available")
    old_trace = os.environ.get("MUSA_AITER_TRACE_DISPATCH")
    os.environ["MUSA_AITER_TRACE_DISPATCH"] = "1"
    try:
        nb, kvh, d, blk = 8, 4, head_dim, block_size
        t = 10
        dtype = torch.bfloat16
        kl = torch.linspace(-2.0, 2.0, steps=t * kvh * d, device="musa", dtype=torch.float32).reshape(t, kvh, d).to(dtype)
        vl = torch.linspace(2.0, -2.0, steps=t * kvh * d, device="musa", dtype=torch.float32).reshape(t, kvh, d).to(dtype)
        cache_dtype = torch.uint8 if cache_dtype_name == "uint8" else getattr(torch, cache_dtype_name)
        kc = torch.empty(nb, kvh, d, blk, device="musa", dtype=cache_dtype)
        vc = torch.empty_like(kc)
        slots = torch.tensor([0, 7, -1, 5, 9, 12, 33, 44, 71, 103], dtype=torch.int32, device="musa")
        k_scale = torch.tensor(scale, dtype=torch.float32, device="musa")
        v_scale = torch.tensor(scale, dtype=torch.float32, device="musa")

        musa_aiter.kv_linear_to_paged_fp8(kl, vl, kc, vc, slots, k_scale, v_scale, fp8_format=fp8_format)
        rk, rv = musa_aiter.kv_paged_fp8_to_linear(kc, vc, slots, k_scale, v_scale, out_dtype=dtype, fp8_format=fp8_format)
        torch.musa.synchronize()
        captured = capsys.readouterr().err
    finally:
        if old_trace is None:
            os.environ.pop("MUSA_AITER_TRACE_DISPATCH", None)
        else:
            os.environ["MUSA_AITER_TRACE_DISPATCH"] = old_trace

    assert "op=kv_paged_fp8_to_linear backend=ctypes-musa-native-fp8-shape" in captured
    valid = slots.cpu() >= 0
    torch.testing.assert_close(rk.cpu()[valid], kl.cpu()[valid], atol=atol, rtol=rtol)
    torch.testing.assert_close(rv.cpu()[valid], vl.cpu()[valid], atol=atol, rtol=rtol)
    assert rk[2].abs().max().item() == 0
    assert rv[2].abs().max().item() == 0


@pytest.mark.musa
@pytest.mark.native
@pytest.mark.parametrize(
    ("fp8_format", "cache_dtype_name", "scale", "atol", "rtol"),
    [
        ("e4m3", "float8_e4m3fn", 0.125, 0.085, 0.10),
        ("e5m2", "float8_e5m2", 0.25, 0.65, 0.18),
        ("e5m2", "uint8", 0.25, 0.65, 0.18),
    ],
)
def test_native_fp8_kv_vectorized_scatter_when_enabled(
    fp8_format: str,
    cache_dtype_name: str,
    scale: float,
    atol: float,
    rtol: float,
    capsys: pytest.CaptureFixture[str],
) -> None:
    if os.environ.get("MUSA_AITER_NATIVE_FP8_KV_CACHE", "0") != "1" or os.environ.get("MUSA_AITER_NATIVE_FP8_KV_VECTORIZE", "0") != "1":
        pytest.skip("set MUSA_AITER_NATIVE_FP8_KV_CACHE=1 and MUSA_AITER_NATIVE_FP8_KV_VECTORIZE=1 to exercise vectorized native FP8 KV scatter")
    if not musa_aiter.is_native_fp8_kv_vectorized_available():
        pytest.skip("native vectorized FP8 KV .so not built or symbols missing")
    _require_musa()
    if cache_dtype_name != "uint8" and not hasattr(torch, cache_dtype_name):
        pytest.skip(f"torch.{cache_dtype_name} is not available")
    old_trace = os.environ.get("MUSA_AITER_TRACE_DISPATCH")
    os.environ["MUSA_AITER_TRACE_DISPATCH"] = "1"
    try:
        nb, kvh, d, blk = 4, 2, 8, 4
        t = 6
        dtype = torch.bfloat16
        kl = torch.linspace(-2.0, 2.0, steps=t * kvh * d, device="musa", dtype=torch.float32).reshape(t, kvh, d).to(dtype)
        vl = torch.linspace(2.0, -2.0, steps=t * kvh * d, device="musa", dtype=torch.float32).reshape(t, kvh, d).to(dtype)
        cache_dtype = torch.uint8 if cache_dtype_name == "uint8" else getattr(torch, cache_dtype_name)
        kc = torch.empty(nb, kvh, d, blk, device="musa", dtype=cache_dtype)
        vc = torch.empty_like(kc)
        slots = torch.tensor([0, 7, -1, 5, 9, 12], dtype=torch.int32, device="musa")
        k_scale = torch.tensor(scale, dtype=torch.float32, device="musa")
        v_scale = torch.tensor(scale, dtype=torch.float32, device="musa")

        musa_aiter.kv_linear_to_paged_fp8(kl, vl, kc, vc, slots, k_scale, v_scale, fp8_format=fp8_format)
        rk, rv = musa_aiter.kv_paged_fp8_to_linear(kc, vc, slots, k_scale, v_scale, out_dtype=dtype, fp8_format=fp8_format)
        torch.musa.synchronize()
        captured = capsys.readouterr().err
    finally:
        if old_trace is None:
            os.environ.pop("MUSA_AITER_TRACE_DISPATCH", None)
        else:
            os.environ["MUSA_AITER_TRACE_DISPATCH"] = old_trace

    assert "op=kv_linear_to_paged_fp8 backend=ctypes-musa-native-fp8-vec" in captured
    valid = slots.cpu() >= 0
    torch.testing.assert_close(rk.cpu()[valid], kl.cpu()[valid], atol=atol, rtol=rtol)
    torch.testing.assert_close(rv.cpu()[valid], vl.cpu()[valid], atol=atol, rtol=rtol)
    assert rk[2].abs().max().item() == 0
    assert rv[2].abs().max().item() == 0


@pytest.mark.musa
@pytest.mark.native
@pytest.mark.parametrize(
    ("fp8_format", "cache_dtype_name", "scale", "atol", "rtol"),
    [
        ("e4m3", "float8_e4m3fn", 0.125, 0.085, 0.10),
        ("e5m2", "float8_e5m2", 0.25, 0.65, 0.18),
        ("e5m2", "uint8", 0.25, 0.65, 0.18),
    ],
)
def test_native_fp8_kv_vectorized_gather_when_enabled(
    fp8_format: str,
    cache_dtype_name: str,
    scale: float,
    atol: float,
    rtol: float,
    capsys: pytest.CaptureFixture[str],
) -> None:
    if os.environ.get("MUSA_AITER_NATIVE_FP8_KV_CACHE", "0") != "1" or os.environ.get("MUSA_AITER_NATIVE_FP8_KV_VECTORIZE", "0") != "1":
        pytest.skip("set MUSA_AITER_NATIVE_FP8_KV_CACHE=1 and MUSA_AITER_NATIVE_FP8_KV_VECTORIZE=1 to exercise vectorized native FP8 KV gather")
    if not musa_aiter.is_native_fp8_kv_vectorized_available():
        pytest.skip("native vectorized FP8 KV .so not built or symbols missing")
    _require_musa()
    if cache_dtype_name != "uint8" and not hasattr(torch, cache_dtype_name):
        pytest.skip(f"torch.{cache_dtype_name} is not available")
    old_trace = os.environ.get("MUSA_AITER_TRACE_DISPATCH")
    os.environ["MUSA_AITER_TRACE_DISPATCH"] = "1"
    try:
        nb, kvh, d, blk = 4, 2, 8, 4
        t = 6
        dtype = torch.bfloat16
        kl = torch.linspace(-2.0, 2.0, steps=t * kvh * d, device="musa", dtype=torch.float32).reshape(t, kvh, d).to(dtype)
        vl = torch.linspace(2.0, -2.0, steps=t * kvh * d, device="musa", dtype=torch.float32).reshape(t, kvh, d).to(dtype)
        cache_dtype = torch.uint8 if cache_dtype_name == "uint8" else getattr(torch, cache_dtype_name)
        kc = torch.empty(nb, kvh, d, blk, device="musa", dtype=cache_dtype)
        vc = torch.empty_like(kc)
        slots = torch.tensor([0, 7, -1, 5, 9, 12], dtype=torch.int32, device="musa")
        k_scale = torch.tensor(scale, dtype=torch.float32, device="musa")
        v_scale = torch.tensor(scale, dtype=torch.float32, device="musa")

        musa_aiter.kv_linear_to_paged_fp8(kl, vl, kc, vc, slots, k_scale, v_scale, fp8_format=fp8_format)
        rk, rv = musa_aiter.kv_paged_fp8_to_linear(kc, vc, slots, k_scale, v_scale, out_dtype=dtype, fp8_format=fp8_format)
        torch.musa.synchronize()
        captured = capsys.readouterr().err
    finally:
        if old_trace is None:
            os.environ.pop("MUSA_AITER_TRACE_DISPATCH", None)
        else:
            os.environ["MUSA_AITER_TRACE_DISPATCH"] = old_trace

    assert "op=kv_paged_fp8_to_linear backend=ctypes-musa-native-fp8-vec" in captured
    valid = slots.cpu() >= 0
    torch.testing.assert_close(rk.cpu()[valid], kl.cpu()[valid], atol=atol, rtol=rtol)
    torch.testing.assert_close(rv.cpu()[valid], vl.cpu()[valid], atol=atol, rtol=rtol)
    assert rk[2].abs().max().item() == 0
    assert rv[2].abs().max().item() == 0


@pytest.mark.musa
@pytest.mark.native
def test_native_fp8_kv_per_head_scales_merged_and_flash_when_enabled() -> None:
    if os.environ.get("MUSA_AITER_NATIVE_FP8_KV_CACHE", "0") != "1":
        pytest.skip("set MUSA_AITER_NATIVE_FP8_KV_CACHE=1 to exercise native FP8 KV cache")
    if not musa_aiter.is_native_fp8_kv_cache_available("e4m3"):
        pytest.skip("native FP8 E4M3 KV cache .so not built or symbols missing")
    _require_musa()
    if not hasattr(torch, "float8_e4m3fn"):
        pytest.skip("torch.float8_e4m3fn is not available")
    nb, kvh, d, blk = 3, 2, 8, 4
    t = 8
    dtype = torch.bfloat16
    merged = torch.linspace(-2.0, 2.0, steps=t * 2 * kvh * d, device="musa", dtype=torch.float32).reshape(t, 2, kvh, d).to(dtype)
    kc = torch.empty(nb, kvh, d, blk, device="musa", dtype=torch.float8_e4m3fn)
    vc = torch.empty_like(kc)
    slots = torch.arange(t, dtype=torch.int32, device="musa")
    k_scale = torch.tensor([0.125, 0.25], dtype=torch.float32, device="musa")
    v_scale = torch.tensor([0.25, 0.125], dtype=torch.float32, device="musa")

    musa_aiter.kv_merged_linear_to_paged_fp8(merged, kc, vc, slots, k_scale, v_scale, fp8_format="e4m3")
    recovered = musa_aiter.kv_paged_fp8_to_merged_linear(kc, vc, slots, k_scale, v_scale, out_dtype=dtype, fp8_format="e4m3")
    dense_k, dense_v = musa_aiter.kv_load_and_dequantize_fp8_flash(
        kc,
        vc,
        torch.tensor([[0, 1], [0, 1]], dtype=torch.int32, device="musa"),
        torch.tensor([6, 3], dtype=torch.int32, device="musa"),
        k_scale,
        v_scale,
        max_seq_len=7,
        out_dtype=dtype,
        fp8_format="e4m3",
    )
    torch.musa.synchronize()

    torch.testing.assert_close(recovered.cpu(), merged.cpu(), atol=0.13, rtol=0.12)
    torch.testing.assert_close(dense_k.cpu()[0, :6], merged.cpu()[:6, 0], atol=0.13, rtol=0.12)
    torch.testing.assert_close(dense_v.cpu()[1, :3], merged.cpu()[:3, 1], atol=0.13, rtol=0.12)
    assert dense_k[0, 6:].abs().max().item() == 0
    assert dense_v[1, 3:].abs().max().item() == 0


@pytest.mark.musa
@pytest.mark.native
@pytest.mark.parametrize(
    ("fp8_format", "cache_dtype_name", "scale", "atol", "rtol"),
    [
        ("e4m3", "float8_e4m3fn", 0.125, 0.085, 0.10),
        ("e5m2", "float8_e5m2", 0.25, 0.65, 0.18),
        ("e5m2", "uint8", 0.25, 0.65, 0.18),
    ],
)
def test_native_fp8_kv_multi_layer_when_enabled(
    fp8_format: str,
    cache_dtype_name: str,
    scale: float,
    atol: float,
    rtol: float,
    capsys: pytest.CaptureFixture[str],
) -> None:
    if os.environ.get("MUSA_AITER_NATIVE_FP8_KV_CACHE", "0") != "1":
        pytest.skip("set MUSA_AITER_NATIVE_FP8_KV_CACHE=1 to exercise native FP8 KV cache")
    if not musa_aiter.is_native_fp8_kv_cache_available(fp8_format):
        pytest.skip(f"native FP8 {fp8_format.upper()} KV cache .so not built or symbols missing")
    _require_musa()
    if cache_dtype_name != "uint8" and not hasattr(torch, cache_dtype_name):
        pytest.skip(f"torch.{cache_dtype_name} is not available")
    old_trace = os.environ.get("MUSA_AITER_TRACE_DISPATCH")
    os.environ["MUSA_AITER_TRACE_DISPATCH"] = "1"
    try:
        nl, nb, kvh, d, blk = 2, 3, 2, 8, 4
        t = 5
        dtype = torch.bfloat16
        kl = torch.linspace(-2.0, 2.0, steps=nl * t * kvh * d, device="musa", dtype=torch.float32).reshape(nl, t, kvh, d).to(dtype)
        vl = torch.linspace(2.0, -2.0, steps=nl * t * kvh * d, device="musa", dtype=torch.float32).reshape(nl, t, kvh, d).to(dtype)
        cache_dtype = torch.uint8 if cache_dtype_name == "uint8" else getattr(torch, cache_dtype_name)
        kc = torch.empty(nl, nb, kvh, d, blk, device="musa", dtype=cache_dtype)
        vc = torch.empty_like(kc)
        slots = torch.tensor([0, 7, -1, 5, 9], dtype=torch.int32, device="musa")
        k_scale = torch.tensor(scale, dtype=torch.float32, device="musa")
        v_scale = torch.tensor(scale, dtype=torch.float32, device="musa")

        musa_aiter.kv_linear_to_paged_fp8_multi_layer(kl, vl, kc, vc, slots, k_scale, v_scale, fp8_format=fp8_format)
        rk, rv = musa_aiter.kv_paged_fp8_to_linear_multi_layer(kc, vc, slots, k_scale, v_scale, out_dtype=dtype, fp8_format=fp8_format)
        torch.musa.synchronize()
        captured = capsys.readouterr().err
    finally:
        if old_trace is None:
            os.environ.pop("MUSA_AITER_TRACE_DISPATCH", None)
        else:
            os.environ["MUSA_AITER_TRACE_DISPATCH"] = old_trace

    assert "op=kv_linear_to_paged_fp8_multi_layer backend=ctypes-musa-native-fp8" in captured
    assert "op=kv_paged_fp8_to_linear_multi_layer backend=ctypes-musa-native-fp8" in captured
    valid = slots.cpu() >= 0
    torch.testing.assert_close(rk.cpu()[:, valid], kl.cpu()[:, valid], atol=atol, rtol=rtol)
    torch.testing.assert_close(rv.cpu()[:, valid], vl.cpu()[:, valid], atol=atol, rtol=rtol)
    assert rk[:, 2].abs().max().item() == 0
    assert rv[:, 2].abs().max().item() == 0


@pytest.mark.musa
@pytest.mark.native
@pytest.mark.parametrize(
    ("fp8_format", "cache_dtype_name", "atol", "rtol"),
    [
        ("e4m3", "float8_e4m3fn", 0.13, 0.12),
        ("e5m2", "float8_e5m2", 1.1, 0.20),
        ("e5m2", "uint8", 1.1, 0.20),
    ],
)
def test_native_fp8_kv_per_head_scales_when_enabled(
    fp8_format: str,
    cache_dtype_name: str,
    atol: float,
    rtol: float,
    capsys: pytest.CaptureFixture[str],
) -> None:
    if os.environ.get("MUSA_AITER_NATIVE_FP8_KV_CACHE", "0") != "1":
        pytest.skip("set MUSA_AITER_NATIVE_FP8_KV_CACHE=1 to exercise native FP8 KV cache")
    if os.environ.get("MUSA_AITER_NATIVE_FP8_KV_VECTORIZE", "0") != "1":
        pytest.skip("set MUSA_AITER_NATIVE_FP8_KV_VECTORIZE=1 to exercise vectorized per-head FP8 KV cache")
    if not musa_aiter.is_native_fp8_kv_vectorized_available():
        pytest.skip(f"native vectorized FP8 {fp8_format.upper()} KV cache .so not built or symbols missing")
    _require_musa()
    if cache_dtype_name != "uint8" and not hasattr(torch, cache_dtype_name):
        pytest.skip(f"torch.{cache_dtype_name} is not available")
    nb, kvh, d, blk = 3, 2, 8, 4
    t = 5
    dtype = torch.bfloat16
    kl = torch.linspace(-2.0, 2.0, steps=t * kvh * d, device="musa", dtype=torch.float32).reshape(t, kvh, d).to(dtype)
    vl = torch.linspace(2.0, -2.0, steps=t * kvh * d, device="musa", dtype=torch.float32).reshape(t, kvh, d).to(dtype)
    cache_dtype = torch.uint8 if cache_dtype_name == "uint8" else getattr(torch, cache_dtype_name)
    kc = torch.empty(nb, kvh, d, blk, device="musa", dtype=cache_dtype)
    vc = torch.empty_like(kc)
    slots = torch.tensor([0, 7, -1, 5, 9], dtype=torch.int32, device="musa")
    k_scale = torch.tensor([0.125, 0.25], dtype=torch.float32, device="musa")
    v_scale = torch.tensor([0.25, 0.125], dtype=torch.float32, device="musa")

    old_trace = os.environ.get("MUSA_AITER_TRACE_DISPATCH")
    os.environ["MUSA_AITER_TRACE_DISPATCH"] = "1"
    try:
        musa_aiter.kv_linear_to_paged_fp8(kl, vl, kc, vc, slots, k_scale, v_scale, fp8_format=fp8_format)
        rk, rv = musa_aiter.kv_paged_fp8_to_linear(kc, vc, slots, k_scale, v_scale, out_dtype=dtype, fp8_format=fp8_format)
        torch.musa.synchronize()
        captured = capsys.readouterr().err
    finally:
        if old_trace is None:
            os.environ.pop("MUSA_AITER_TRACE_DISPATCH", None)
        else:
            os.environ["MUSA_AITER_TRACE_DISPATCH"] = old_trace

    assert "op=kv_linear_to_paged_fp8 backend=ctypes-musa-native-fp8-vec" in captured
    assert "op=kv_paged_fp8_to_linear backend=ctypes-musa-native-fp8-vec" in captured

    valid = slots.cpu() >= 0
    torch.testing.assert_close(rk.cpu()[valid], kl.cpu()[valid], atol=atol, rtol=rtol)
    torch.testing.assert_close(rv.cpu()[valid], vl.cpu()[valid], atol=atol, rtol=rtol)
    assert rk[2].abs().max().item() == 0
    assert rv[2].abs().max().item() == 0


@pytest.mark.musa
@pytest.mark.native
@pytest.mark.parametrize(
    ("fp8_format", "cache_dtype_name", "scale", "atol", "rtol"),
    [
        ("e4m3", "float8_e4m3fn", 0.125, 0.085, 0.10),
        ("e5m2", "float8_e5m2", 0.25, 0.65, 0.18),
        ("e5m2", "uint8", 0.25, 0.65, 0.18),
    ],
)
def test_native_fp8_kv_merged_when_enabled(
    fp8_format: str,
    cache_dtype_name: str,
    scale: float,
    atol: float,
    rtol: float,
    capsys: pytest.CaptureFixture[str],
) -> None:
    if os.environ.get("MUSA_AITER_NATIVE_FP8_KV_CACHE", "0") != "1":
        pytest.skip("set MUSA_AITER_NATIVE_FP8_KV_CACHE=1 to exercise native FP8 KV cache")
    if not musa_aiter.is_native_fp8_kv_cache_available(fp8_format):
        pytest.skip(f"native FP8 {fp8_format.upper()} KV cache .so not built or symbols missing")
    _require_musa()
    if cache_dtype_name != "uint8" and not hasattr(torch, cache_dtype_name):
        pytest.skip(f"torch.{cache_dtype_name} is not available")
    old_trace = os.environ.get("MUSA_AITER_TRACE_DISPATCH")
    os.environ["MUSA_AITER_TRACE_DISPATCH"] = "1"
    try:
        nb, kvh, d, blk = 3, 2, 8, 4
        t = 5
        dtype = torch.bfloat16
        merged = torch.linspace(-2.0, 2.0, steps=t * 2 * kvh * d, device="musa", dtype=torch.float32).reshape(t, 2, kvh, d).to(dtype)
        cache_dtype = torch.uint8 if cache_dtype_name == "uint8" else getattr(torch, cache_dtype_name)
        kc = torch.empty(nb, kvh, d, blk, device="musa", dtype=cache_dtype)
        vc = torch.empty_like(kc)
        slots = torch.tensor([0, 7, -1, 5, 9], dtype=torch.int32, device="musa")
        k_scale = torch.tensor(scale, dtype=torch.float32, device="musa")
        v_scale = torch.tensor(scale, dtype=torch.float32, device="musa")

        musa_aiter.kv_merged_linear_to_paged_fp8(merged, kc, vc, slots, k_scale, v_scale, fp8_format=fp8_format)
        recovered = musa_aiter.kv_paged_fp8_to_merged_linear(kc, vc, slots, k_scale, v_scale, out_dtype=dtype, fp8_format=fp8_format)
        torch.musa.synchronize()
        captured = capsys.readouterr().err
    finally:
        if old_trace is None:
            os.environ.pop("MUSA_AITER_TRACE_DISPATCH", None)
        else:
            os.environ["MUSA_AITER_TRACE_DISPATCH"] = old_trace

    assert "op=kv_merged_linear_to_paged_fp8 backend=ctypes-musa-native-fp8" in captured
    assert "op=kv_paged_fp8_to_merged_linear backend=ctypes-musa-native-fp8" in captured
    valid = slots.cpu() >= 0
    torch.testing.assert_close(recovered.cpu()[valid], merged.cpu()[valid], atol=atol, rtol=rtol)
    assert recovered[2].abs().max().item() == 0


@pytest.mark.musa
@pytest.mark.native
@pytest.mark.parametrize(
    ("fp8_format", "cache_dtype_name", "scale", "atol", "rtol"),
    [
        ("e4m3", "float8_e4m3fn", 0.125, 0.085, 0.10),
        ("e5m2", "float8_e5m2", 0.25, 0.65, 0.18),
    ],
)
def test_native_fp8_kv_merged_multi_layer_when_enabled(
    fp8_format: str,
    cache_dtype_name: str,
    scale: float,
    atol: float,
    rtol: float,
    capsys: pytest.CaptureFixture[str],
) -> None:
    if os.environ.get("MUSA_AITER_NATIVE_FP8_KV_CACHE", "0") != "1":
        pytest.skip("set MUSA_AITER_NATIVE_FP8_KV_CACHE=1 to exercise native FP8 KV cache")
    if not musa_aiter.is_native_fp8_kv_cache_available(fp8_format):
        pytest.skip(f"native FP8 {fp8_format.upper()} KV cache .so not built or symbols missing")
    _require_musa()
    if not hasattr(torch, cache_dtype_name):
        pytest.skip(f"torch.{cache_dtype_name} is not available")
    old_trace = os.environ.get("MUSA_AITER_TRACE_DISPATCH")
    os.environ["MUSA_AITER_TRACE_DISPATCH"] = "1"
    try:
        nl, nb, kvh, d, blk = 2, 3, 2, 8, 4
        t = 5
        dtype = torch.bfloat16
        merged = torch.linspace(-2.0, 2.0, steps=nl * t * 2 * kvh * d, device="musa", dtype=torch.float32).reshape(nl, t, 2, kvh, d).to(dtype)
        cache_dtype = getattr(torch, cache_dtype_name)
        kc = torch.empty(nl, nb, kvh, d, blk, device="musa", dtype=cache_dtype)
        vc = torch.empty_like(kc)
        slots = torch.tensor([0, 7, -1, 5, 9], dtype=torch.int32, device="musa")
        k_scale = torch.tensor(scale, dtype=torch.float32, device="musa")
        v_scale = torch.tensor(scale, dtype=torch.float32, device="musa")

        musa_aiter.kv_merged_linear_to_paged_fp8_multi_layer(merged, kc, vc, slots, k_scale, v_scale, fp8_format=fp8_format)
        recovered = musa_aiter.kv_paged_fp8_to_merged_linear_multi_layer(kc, vc, slots, k_scale, v_scale, out_dtype=dtype, fp8_format=fp8_format)
        torch.musa.synchronize()
        captured = capsys.readouterr().err
    finally:
        if old_trace is None:
            os.environ.pop("MUSA_AITER_TRACE_DISPATCH", None)
        else:
            os.environ["MUSA_AITER_TRACE_DISPATCH"] = old_trace

    assert "op=kv_merged_linear_to_paged_fp8_multi_layer backend=ctypes-musa-native-fp8" in captured
    assert "op=kv_paged_fp8_to_merged_linear_multi_layer backend=ctypes-musa-native-fp8" in captured
    valid = slots.cpu() >= 0
    torch.testing.assert_close(recovered.cpu()[:, valid], merged.cpu()[:, valid], atol=atol, rtol=rtol)
    assert recovered[:, 2].abs().max().item() == 0


@pytest.mark.musa
@pytest.mark.native
@pytest.mark.parametrize(
    ("fp8_format", "cache_dtype_name", "scale", "atol", "rtol"),
    [
        ("e4m3", "float8_e4m3fn", 0.125, 0.085, 0.10),
        ("e5m2", "float8_e5m2", 0.25, 0.65, 0.18),
        ("e5m2", "uint8", 0.25, 0.65, 0.18),
    ],
)
def test_native_fp8_kv_flash_load_and_copy_when_enabled(
    fp8_format: str,
    cache_dtype_name: str,
    scale: float,
    atol: float,
    rtol: float,
    capsys: pytest.CaptureFixture[str],
) -> None:
    if os.environ.get("MUSA_AITER_NATIVE_FP8_KV_CACHE", "0") != "1":
        pytest.skip("set MUSA_AITER_NATIVE_FP8_KV_CACHE=1 to exercise native FP8 KV cache")
    if not musa_aiter.is_native_fp8_kv_cache_available(fp8_format):
        pytest.skip(f"native FP8 {fp8_format.upper()} KV cache .so not built or symbols missing")
    _require_musa()
    if cache_dtype_name != "uint8" and not hasattr(torch, cache_dtype_name):
        pytest.skip(f"torch.{cache_dtype_name} is not available")
    old_trace = os.environ.get("MUSA_AITER_TRACE_DISPATCH")
    os.environ["MUSA_AITER_TRACE_DISPATCH"] = "1"
    try:
        nb, kvh, d, blk = 4, 2, 8, 4
        t = 8
        dtype = torch.bfloat16
        kl = torch.linspace(-2.0, 2.0, steps=t * kvh * d, device="musa", dtype=torch.float32).reshape(t, kvh, d).to(dtype)
        vl = torch.linspace(2.0, -2.0, steps=t * kvh * d, device="musa", dtype=torch.float32).reshape(t, kvh, d).to(dtype)
        cache_dtype = torch.uint8 if cache_dtype_name == "uint8" else getattr(torch, cache_dtype_name)
        kc = torch.empty(nb, kvh, d, blk, device="musa", dtype=cache_dtype)
        vc = torch.empty_like(kc)
        slots = torch.arange(t, dtype=torch.int32, device="musa")
        k_scale = torch.tensor(scale, dtype=torch.float32, device="musa")
        v_scale = torch.tensor(scale, dtype=torch.float32, device="musa")
        musa_aiter.kv_linear_to_paged_fp8(kl, vl, kc, vc, slots, k_scale, v_scale, fp8_format=fp8_format)

        block_tables = torch.tensor([[0, 1], [0, 1]], dtype=torch.int32, device="musa")
        seq_lens = torch.tensor([6, 3], dtype=torch.int32, device="musa")
        dense_k, dense_v = musa_aiter.kv_load_and_dequantize_fp8_flash(kc, vc, block_tables, seq_lens, k_scale, v_scale, max_seq_len=7, out_dtype=dtype, fp8_format=fp8_format)
        dst_k = torch.empty_like(kc)
        dst_v = torch.empty_like(vc)
        copied = musa_aiter.kv_copy_paged_blocks_fp8(kc, vc, dst_k, dst_v, torch.tensor([[1, 3], [0, 2], [-1, 1]], dtype=torch.int32, device="musa"))
        mkc = torch.empty((2, *kc.shape), device="musa", dtype=kc.dtype)
        mvc = torch.empty_like(mkc)
        mkc[0].copy_(kc)
        mkc[1].copy_(kc)
        mvc[0].copy_(vc)
        mvc[1].copy_(vc)
        mdst_k = torch.empty_like(mkc)
        mdst_v = torch.empty_like(mvc)
        copied_multi = musa_aiter.kv_copy_paged_blocks_fp8(mkc, mvc, mdst_k, mdst_v, torch.tensor([[1, 3], [0, 2], [-1, 1]], dtype=torch.int32, device="musa"))
        torch.musa.synchronize()
        captured = capsys.readouterr().err
    finally:
        if old_trace is None:
            os.environ.pop("MUSA_AITER_TRACE_DISPATCH", None)
        else:
            os.environ["MUSA_AITER_TRACE_DISPATCH"] = old_trace

    assert "op=kv_load_and_dequantize_fp8_flash backend=ctypes-musa-native-fp8" in captured
    assert "op=kv_copy_paged_blocks_fp8 backend=ctypes-musa-native-fp8" in captured
    torch.testing.assert_close(dense_k.cpu()[0, :6], kl.cpu()[:6], atol=atol, rtol=rtol)
    torch.testing.assert_close(dense_v.cpu()[0, :6], vl.cpu()[:6], atol=atol, rtol=rtol)
    torch.testing.assert_close(dense_k.cpu()[1, :3], kl.cpu()[:3], atol=atol, rtol=rtol)
    torch.testing.assert_close(dense_v.cpu()[1, :3], vl.cpu()[:3], atol=atol, rtol=rtol)
    assert dense_k[0, 6:].abs().max().item() == 0
    assert dense_v[1, 3:].abs().max().item() == 0
    assert copied == 2
    assert copied_multi == 2
    torch.testing.assert_close(dst_k[3].cpu(), kc[1].cpu())
    torch.testing.assert_close(dst_v[3].cpu(), vc[1].cpu())
    torch.testing.assert_close(dst_k[2].cpu(), kc[0].cpu())
    torch.testing.assert_close(dst_v[2].cpu(), vc[0].cpu())
    torch.testing.assert_close(mdst_k[:, 3].cpu(), mkc[:, 1].cpu())
    torch.testing.assert_close(mdst_v[:, 3].cpu(), mvc[:, 1].cpu())
    torch.testing.assert_close(mdst_k[:, 2].cpu(), mkc[:, 0].cpu())
    torch.testing.assert_close(mdst_v[:, 2].cpu(), mvc[:, 0].cpu())


@pytest.mark.musa
@pytest.mark.native
@pytest.mark.parametrize("cache_dtype_name", ["float8_e4m3fn", "float8_e5m2", "uint8"])
def test_native_fp8_kv_vectorized_copy_when_enabled(cache_dtype_name: str, capsys: pytest.CaptureFixture[str]) -> None:
    if os.environ.get("MUSA_AITER_NATIVE_FP8_KV_CACHE", "0") != "1" or os.environ.get("MUSA_AITER_NATIVE_FP8_KV_VECTORIZE", "0") != "1":
        pytest.skip("set MUSA_AITER_NATIVE_FP8_KV_CACHE=1 and MUSA_AITER_NATIVE_FP8_KV_VECTORIZE=1 to exercise vectorized native FP8 KV copy")
    if not musa_aiter.is_native_fp8_kv_vectorized_available():
        pytest.skip("native vectorized FP8 KV copy .so not built or symbol missing")
    _require_musa()
    if cache_dtype_name != "uint8" and not hasattr(torch, cache_dtype_name):
        pytest.skip(f"torch.{cache_dtype_name} is not available")
    old_trace = os.environ.get("MUSA_AITER_TRACE_DISPATCH")
    os.environ["MUSA_AITER_TRACE_DISPATCH"] = "1"
    try:
        layers, blocks, kvh, d, blk = 2, 5, 2, 8, 4
        total = layers * blocks * kvh * d * blk
        cache_dtype = torch.uint8 if cache_dtype_name == "uint8" else getattr(torch, cache_dtype_name)
        src_bits = torch.arange(total, dtype=torch.uint8).reshape(layers, blocks, kvh, d, blk).to("musa")
        src_k = src_bits.clone() if cache_dtype is torch.uint8 else src_bits.view(cache_dtype)
        src_v_bits = (src_bits.cpu() + 17).to(torch.uint8).to("musa")
        src_v = src_v_bits.clone() if cache_dtype is torch.uint8 else src_v_bits.view(cache_dtype)
        dst_k_bits = torch.full((layers, blocks, kvh, d, blk), 0xA5, dtype=torch.uint8).to("musa")
        dst_v_bits = torch.full((layers, blocks, kvh, d, blk), 0x5A, dtype=torch.uint8).to("musa")
        dst_k = dst_k_bits if cache_dtype is torch.uint8 else dst_k_bits.view(cache_dtype)
        dst_v = dst_v_bits if cache_dtype is torch.uint8 else dst_v_bits.view(cache_dtype)
        mapping = torch.tensor([[1, 4], [3, 0], [-1, 2]], dtype=torch.int32, device="musa")

        copied = musa_aiter.kv_copy_paged_blocks_fp8(src_k, src_v, dst_k, dst_v, mapping)
        torch.musa.synchronize()
        captured = capsys.readouterr().err
    finally:
        if old_trace is None:
            os.environ.pop("MUSA_AITER_TRACE_DISPATCH", None)
        else:
            os.environ["MUSA_AITER_TRACE_DISPATCH"] = old_trace

    assert copied == 2
    assert "op=kv_copy_paged_blocks_fp8 backend=ctypes-musa-native-fp8-vec" in captured
    dst_k_raw = dst_k if dst_k.dtype is torch.uint8 else dst_k.view(torch.uint8)
    dst_v_raw = dst_v if dst_v.dtype is torch.uint8 else dst_v.view(torch.uint8)
    src_k_raw = src_k if src_k.dtype is torch.uint8 else src_k.view(torch.uint8)
    src_v_raw = src_v if src_v.dtype is torch.uint8 else src_v.view(torch.uint8)
    torch.testing.assert_close(dst_k_raw[:, 4].cpu(), src_k_raw[:, 1].cpu())
    torch.testing.assert_close(dst_v_raw[:, 4].cpu(), src_v_raw[:, 1].cpu())
    torch.testing.assert_close(dst_k_raw[:, 0].cpu(), src_k_raw[:, 3].cpu())
    torch.testing.assert_close(dst_v_raw[:, 0].cpu(), src_v_raw[:, 3].cpu())
    assert torch.equal(dst_k_raw[:, 2].cpu(), torch.full_like(dst_k_raw[:, 2].cpu(), 0xA5))
    assert torch.equal(dst_v_raw[:, 2].cpu(), torch.full_like(dst_v_raw[:, 2].cpu(), 0x5A))


@pytest.mark.musa
@pytest.mark.native
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_native_kv_multi_layer_flash_and_copy_half_dtypes_when_enabled(dtype: torch.dtype) -> None:
    if os.environ.get("MUSA_AITER_NATIVE_KV_TRANSFER", "0") != "1":
        pytest.skip("set MUSA_AITER_NATIVE_KV_TRANSFER=1 to exercise native KV transfer")
    if not musa_aiter.is_native_kv_transfer_available():
        pytest.skip("native KV transfer .so not built or symbols missing")
    _require_musa()
    nl, nb, kvh, d, blk = 2, 4, 2, 8, 4
    t = 5
    kl = torch.randn(nl, t, kvh, d, device="musa", dtype=dtype)
    vl = torch.randn(nl, t, kvh, d, device="musa", dtype=dtype)
    kc = torch.zeros(nl, nb, kvh, d, blk, device="musa", dtype=dtype)
    vc = torch.zeros_like(kc)
    slots = torch.tensor([0, 7, -1, 5, 3], dtype=torch.int32, device="musa")

    musa_aiter.kv_linear_to_paged_multi_layer(kl, vl, kc, vc, slots)
    rk, rv = musa_aiter.kv_paged_to_linear_multi_layer(kc, vc, slots)
    torch.musa.synchronize()

    torch.testing.assert_close(rk[:, 0].cpu(), kl[:, 0].cpu())
    torch.testing.assert_close(rk[:, 1].cpu(), kl[:, 1].cpu())
    torch.testing.assert_close(rk[:, 3].cpu(), kl[:, 3].cpu())
    torch.testing.assert_close(rk[:, 4].cpu(), kl[:, 4].cpu())
    torch.testing.assert_close(rv[:, 0].cpu(), vl[:, 0].cpu())
    torch.testing.assert_close(rv[:, 1].cpu(), vl[:, 1].cpu())
    torch.testing.assert_close(rv[:, 3].cpu(), vl[:, 3].cpu())
    torch.testing.assert_close(rv[:, 4].cpu(), vl[:, 4].cpu())
    assert rk[:, 2].abs().max().item() == 0
    assert rv[:, 2].abs().max().item() == 0

    dense_k, dense_v = musa_aiter.kv_load_and_reshape_flash(kc[0], vc[0], torch.tensor([[0, 1], [2, 3]], dtype=torch.int32, device="musa"), torch.tensor([5, 3], dtype=torch.int32, device="musa"), max_seq_len=5)
    torch.musa.synchronize()
    assert dense_k.dtype == dtype
    assert dense_v.dtype == dtype
    assert dense_k[1, 3:].abs().max().item() == 0
    assert dense_v[1, 3:].abs().max().item() == 0

    dst_k = torch.zeros_like(kc)
    dst_v = torch.zeros_like(vc)
    copied = musa_aiter.kv_copy_paged_blocks(kc, vc, dst_k, dst_v, torch.tensor([[1, 3], [2, 0], [-1, 2]], dtype=torch.int32, device="musa"))
    torch.musa.synchronize()
    assert copied == 2
    torch.testing.assert_close(dst_k[:, 3].cpu(), kc[:, 1].cpu())
    torch.testing.assert_close(dst_v[:, 3].cpu(), vc[:, 1].cpu())
    torch.testing.assert_close(dst_k[:, 0].cpu(), kc[:, 2].cpu())
    torch.testing.assert_close(dst_v[:, 0].cpu(), vc[:, 2].cpu())


@pytest.mark.musa
@pytest.mark.native
def test_native_kv_cachegen_encode_decode_when_enabled() -> None:
    if os.environ.get("MUSA_AITER_NATIVE_KV_TRANSFER", "0") != "1":
        pytest.skip("set MUSA_AITER_NATIVE_KV_TRANSFER=1 to exercise native KV transfer")
    if not musa_aiter.is_native_kv_transfer_available():
        pytest.skip("native KV transfer .so not built or symbols missing")
    _require_musa()
    tensor = torch.linspace(-1.0, 1.0, steps=257, dtype=torch.float32, device="musa").reshape(1, 257)

    encoded, codebook, cdf = musa_aiter.kv_cachegen_encode(tensor, num_bins=32)
    decoded = musa_aiter.kv_cachegen_decode(encoded, codebook, dtype=tensor.dtype)
    torch.musa.synchronize()

    assert encoded.dtype == torch.uint8
    assert codebook.shape == (32,)
    assert cdf.shape == (33,)
    assert cdf[-1].item() == 1
    assert torch.max(torch.abs(decoded - tensor)).item() <= (2.0 / 31.0)


@pytest.mark.musa
@pytest.mark.native
def test_native_kv_load_and_reshape_flash_when_enabled() -> None:
    if os.environ.get("MUSA_AITER_NATIVE_KV_TRANSFER", "0") != "1":
        pytest.skip("set MUSA_AITER_NATIVE_KV_TRANSFER=1 to exercise native KV transfer")
    if not musa_aiter.is_native_kv_transfer_available():
        pytest.skip("native KV transfer .so not built or symbols missing")
    _require_musa()
    nb, kvh, d, blk = 4, 2, 3, 4
    key_cache = torch.arange(nb * kvh * d * blk, dtype=torch.float32, device="musa").reshape(nb, kvh, d, blk)
    value_cache = key_cache + 1000
    block_tables = torch.tensor([[1, 3], [0, 2]], dtype=torch.int32, device="musa")
    seq_lens = torch.tensor([6, 3], dtype=torch.int32, device="musa")

    k_dense, v_dense = musa_aiter.kv_load_and_reshape_flash(
        key_cache,
        value_cache,
        block_tables,
        seq_lens,
        max_seq_len=7,
    )
    torch.musa.synchronize()

    expected0 = torch.cat(
        (key_cache[1].permute(2, 0, 1), key_cache[3].permute(2, 0, 1)),
        dim=0,
    )[:6]
    expected1 = key_cache[0].permute(2, 0, 1)[:3]
    torch.testing.assert_close(k_dense[0, :6].cpu(), expected0.cpu())
    torch.testing.assert_close(v_dense[0, :6].cpu(), (expected0 + 1000).cpu())
    torch.testing.assert_close(k_dense[1, :3].cpu(), expected1.cpu())
    torch.testing.assert_close(v_dense[1, :3].cpu(), (expected1 + 1000).cpu())
    assert k_dense[1, 3:].abs().max().item() == 0
    assert v_dense[1, 3:].abs().max().item() == 0


@pytest.mark.musa
def test_kv_copy_paged_blocks_musa_multi_layer() -> None:
    _require_musa()
    src_k = torch.arange(2 * 3 * 2 * 4 * 5, dtype=torch.float32, device="musa").reshape(2, 3, 2, 4, 5)
    src_v = src_k + 1000
    dst_k = torch.zeros(2, 4, 2, 4, 5, device="musa", dtype=torch.float32)
    dst_v = torch.zeros_like(dst_k)
    mapping = torch.tensor([[1, 3], [2, 0], [-1, 2]], dtype=torch.int32, device="musa")

    copied = musa_aiter.kv_copy_paged_blocks(src_k, src_v, dst_k, dst_v, mapping)
    torch.musa.synchronize()

    assert copied == 2
    torch.testing.assert_close(dst_k[:, 3].cpu(), src_k[:, 1].cpu())
    torch.testing.assert_close(dst_v[:, 3].cpu(), src_v[:, 1].cpu())
    torch.testing.assert_close(dst_k[:, 0].cpu(), src_k[:, 2].cpu())
    torch.testing.assert_close(dst_v[:, 0].cpu(), src_v[:, 2].cpu())
    assert dst_k[:, 2].abs().max().item() == 0
    assert dst_v[:, 2].abs().max().item() == 0


@pytest.mark.musa
@pytest.mark.native
def test_native_kv_copy_paged_blocks_when_enabled() -> None:
    if os.environ.get("MUSA_AITER_NATIVE_KV_TRANSFER", "0") != "1":
        pytest.skip("set MUSA_AITER_NATIVE_KV_TRANSFER=1 to exercise native KV transfer")
    if not musa_aiter.is_native_kv_transfer_available():
        pytest.skip("native KV transfer .so not built or symbols missing")
    _require_musa()
    src_k = torch.arange(2 * 3 * 2 * 4 * 5, dtype=torch.float32, device="musa").reshape(2, 3, 2, 4, 5)
    src_v = src_k + 1000
    dst_k = torch.zeros(2, 4, 2, 4, 5, device="musa", dtype=torch.float32)
    dst_v = torch.zeros_like(dst_k)
    mapping = torch.tensor([[1, 3], [2, 0], [-1, 2]], dtype=torch.int32, device="musa")

    copied = musa_aiter.kv_copy_paged_blocks(src_k, src_v, dst_k, dst_v, mapping)
    torch.musa.synchronize()

    assert copied == 2
    torch.testing.assert_close(dst_k[:, 3].cpu(), src_k[:, 1].cpu())
    torch.testing.assert_close(dst_v[:, 3].cpu(), src_v[:, 1].cpu())
    torch.testing.assert_close(dst_k[:, 0].cpu(), src_k[:, 2].cpu())
    torch.testing.assert_close(dst_v[:, 0].cpu(), src_v[:, 2].cpu())
    assert dst_k[:, 2].abs().max().item() == 0
    assert dst_v[:, 2].abs().max().item() == 0


@pytest.mark.musa
def test_flash_attn_musa_bfloat16_small() -> None:
    """SDPA backends prefer bf16/half — exercises MUSA attention without fp32 sdpa churn."""
    _require_musa()
    dtype = torch.bfloat16
    q = torch.randn(1, 4, 2, 8, device="musa", dtype=dtype)
    k = torch.randn(1, 4, 2, 8, device="musa", dtype=dtype)
    v = torch.randn(1, 4, 2, 8, device="musa", dtype=dtype)
    out = musa_aiter.flash_attn_func(q, k, v, causal=True)
    assert out.shape == q.shape


@pytest.mark.musa
def test_paged_attention_musa_bf16_matches_cpu_reference() -> None:
    """Parity versus CPU tensors through the same musa_aiter path (GPU vs CPU kernels)."""

    _require_musa()
    num_seqs, qh, kvh, d, block, num_blocks = 2, 4, 2, 8, 4, 4
    dtype = torch.bfloat16
    query_cpu = torch.randn(num_seqs, qh, d, dtype=dtype)
    key_cpu = torch.randn(num_blocks, kvh, d, block, dtype=dtype)
    val_cpu = torch.randn(num_blocks, kvh, d, block, dtype=dtype)
    bt = torch.tensor([[0, 1], [2, 3]], dtype=torch.int32)
    seq_lens = torch.tensor([5, 7], dtype=torch.int32)
    args = dict(num_kv_heads=kvh, block_size=block)

    out_c = musa_aiter.paged_attention(query_cpu, key_cpu, val_cpu, bt, seq_lens, **args)
    query_dev = query_cpu.to(device="musa")
    key_dev = key_cpu.to(device="musa")
    val_dev = val_cpu.to(device="musa")
    bt_dev = bt.to(device="musa")
    sl_dev = seq_lens.to(device="musa")
    out_dev = musa_aiter.paged_attention(query_dev, key_dev, val_dev, bt_dev, sl_dev, **args)
    torch.testing.assert_close(out_dev.cpu(), out_c, rtol=8e-2, atol=8e-2)


@pytest.mark.musa
def test_fused_moe_musa_bf16_matches_cpu() -> None:
    """Fused MoE reference should match CPU when routing tensors are mirrored on MUSA."""

    _require_musa()
    dtype = torch.bfloat16
    tokens, hidden, inter, experts, top_k = 8, 16, 32, 4, 2

    logits_cpu = torch.randn(tokens, experts, dtype=dtype)
    w_cpu, ids_cpu = musa_aiter.topk_softmax(logits_cpu, top_k)

    x_c = torch.randn(tokens, hidden, dtype=dtype)
    w1_c = torch.randn(experts, inter * 2, hidden, dtype=dtype)
    w2_c = torch.randn(experts, hidden, inter, dtype=dtype)

    exp_c = musa_aiter.fused_moe(x_c, w1_c, w2_c, w_cpu, ids_cpu)

    x_m = x_c.to(device="musa")
    w1_m = w1_c.to(device="musa")
    w2_m = w2_c.to(device="musa")
    w_m = w_cpu.to(device="musa")
    ids_m = ids_cpu.to(device="musa")
    exp_m = musa_aiter.fused_moe(x_m, w1_m, w2_m, w_m, ids_m)

    torch.testing.assert_close(exp_m.cpu(), exp_c, rtol=8e-2, atol=8e-2)


@pytest.mark.musa
def test_rmsnorm_musa_matches_cpu_reference() -> None:
    """RMSNorm on MUSA tensors should stay close to the CPU reference path."""

    _require_musa()
    dtype = torch.float32
    x_cpu = torch.randn(4, 128, dtype=dtype)
    w = torch.ones(128, dtype=dtype)
    y_cpu = musa_aiter.rmsnorm(x_cpu, w)
    x_m = x_cpu.to(device="musa")
    w_m = w.to(device="musa")
    y_m = musa_aiter.rmsnorm(x_m, w_m)
    torch.testing.assert_close(y_m.cpu(), y_cpu, rtol=1e-4, atol=1e-4)


@pytest.mark.musa
def test_grouped_gemm_musa_bf16_matches_cpu() -> None:
    """Grouped GEMM parity: same algorithm on CPU vs MUSA."""

    _require_musa()
    dtype = torch.bfloat16
    tokens, hidden, out_dim, groups = 64, 32, 48, 4
    a_c = torch.randn(tokens, hidden, dtype=dtype)
    b_c = torch.randn(groups, hidden, out_dim, dtype=dtype)
    idx = torch.randint(0, groups, (tokens,))
    out_c = musa_aiter.grouped_gemm(a_c, b_c, idx)
    a_m = a_c.to(device="musa")
    b_m = b_c.to(device="musa")
    idx_m = idx.to(device="musa")
    out_m = musa_aiter.grouped_gemm(a_m, b_m, idx_m)
    torch.testing.assert_close(out_m.cpu(), out_c, rtol=6e-2, atol=6e-2)


@pytest.mark.musa
def test_gemm_musa_bf16_matches_reference() -> None:
    _require_musa()
    dtype = torch.bfloat16
    a = torch.randn(24, 48, device="musa", dtype=dtype)
    b = torch.randn(48, 32, device="musa", dtype=dtype)
    y = musa_aiter.gemm(a, b)
    ref = torch.matmul(a, b)
    torch.testing.assert_close(y.cpu(), ref.cpu(), rtol=2e-2, atol=2e-2)


@pytest.mark.musa
def test_topk_softmax_musa_matches_cpu() -> None:
    _require_musa()
    dtype = torch.float16
    logits_cpu = torch.randn(16, 8, dtype=dtype)
    wc, ic = musa_aiter.topk_softmax(logits_cpu, 3)
    logits_m = logits_cpu.to(device="musa")
    wm, im = musa_aiter.topk_softmax(logits_m, 3)
    torch.testing.assert_close(wm.cpu(), wc, rtol=5e-3, atol=5e-3)
    assert torch.equal(im.cpu(), ic)


@pytest.mark.musa
def test_flash_attn_prefill_musa_bf16_matches_cpu() -> None:
    """Prefill attention: compare MUSA execution against CPU tensors."""

    _require_musa()
    dtype = torch.bfloat16
    q_c = torch.randn(1, 4, 8, 16, dtype=dtype)
    k_c = torch.randn(1, 4, 8, 16, dtype=dtype)
    v_c = torch.randn(1, 4, 8, 16, dtype=dtype)
    out_c = musa_aiter.flash_attn_func(q_c, k_c, v_c, causal=True)
    q_m = q_c.to(device="musa")
    k_m = k_c.to(device="musa")
    v_m = v_c.to(device="musa")
    out_m = musa_aiter.flash_attn_func(q_m, k_m, v_m, causal=True)
    torch.testing.assert_close(out_m.cpu(), out_c, rtol=8e-2, atol=8e-2)


@pytest.mark.musa
@pytest.mark.native
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_native_lmcache_kv_transfer_non_mla_when_enabled(dtype: torch.dtype) -> None:
    if os.environ.get("MUSA_AITER_NATIVE_LMCACHE_KV_TRANSFER", "0") != "1":
        pytest.skip("set MUSA_AITER_NATIVE_LMCACHE_KV_TRANSFER=1 to exercise native LMCache KV transfer")
    if not musa_aiter.is_native_lmcache_kv_transfer_available():
        pytest.skip("native LMCache KV transfer symbols missing")
    _require_musa()
    nl, nb, bs, nh, hs = 2, 2, 4, 2, 3
    slots = torch.tensor([0, 3, 5, -1, 7], dtype=torch.int64, device="musa")
    kv_buffer = torch.arange(2 * nl * slots.numel() * nh * hs, dtype=torch.float32, device="musa").reshape(2, nl, slots.numel(), nh * hs).to(dtype=dtype)
    paged = torch.zeros(nl, 2, nb, bs, nh, hs, dtype=dtype, device="musa")

    assert musa_aiter.lmcache_kv_buffer_to_paged(kv_buffer, paged, slots) is True
    recovered = torch.empty_like(kv_buffer)
    assert musa_aiter.lmcache_paged_to_kv_buffer(paged, recovered, slots) is True
    torch.musa.synchronize()

    torch.testing.assert_close(recovered[:, :, 0].cpu(), kv_buffer[:, :, 0].cpu())
    torch.testing.assert_close(recovered[:, :, 1].cpu(), kv_buffer[:, :, 1].cpu())
    torch.testing.assert_close(recovered[:, :, 2].cpu(), kv_buffer[:, :, 2].cpu())
    torch.testing.assert_close(recovered[:, :, 4].cpu(), kv_buffer[:, :, 4].cpu())
    assert recovered[:, :, 3].abs().max().item() == 0


@pytest.mark.musa
@pytest.mark.native
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_native_lmcache_two_x_nl_sglang_mp_roundtrip_when_enabled(
    dtype: torch.dtype, monkeypatch: pytest.MonkeyPatch
) -> None:
    if os.environ.get("MUSA_AITER_NATIVE_LMCACHE_KV_TRANSFER", "0") != "1":
        pytest.skip(
            "set MUSA_AITER_NATIVE_LMCACHE_KV_TRANSFER=1 to exercise native "
            "TWO_X_NL_X_NB_BS_NH_HS transfer"
        )
    _require_musa()
    if not musa_aiter.is_native_lmcache_two_x_nl_kv_transfer_available():
        pytest.skip("native TWO_X_NL_X_NB_BS_NH_HS symbols missing")

    from musa_aiter.ops import lmcache_kv_transfer as lmcache_ops

    def _reference_path_was_used(_: torch.Tensor) -> list[int]:
        raise AssertionError("native TWO_X_NL path fell back to Python slot iteration")

    monkeypatch.setattr(lmcache_ops, "_slot_values", _reference_path_was_used)
    nl, nb, bs, nh, hs = 3, 2, 4, 2, 4
    slots = torch.tensor([0, 3, 5, -1, 7], dtype=torch.int64, device="musa")
    memory = torch.arange(
        2 * nl * slots.numel() * nh * hs, dtype=torch.float32, device="musa"
    ).reshape(2, nl, slots.numel(), nh * hs).to(dtype=dtype)
    caches = [
        [torch.zeros(nb, bs, nh, hs, dtype=dtype, device="musa") for _ in range(nl)]
        for _ in range(2)
    ]

    assert (
        musa_aiter.lmcache_kv_buffer_to_paged_two_x_nl(
            memory, caches, slots, 0, bs, nh, hs
        )
        is True
    )
    recovered = torch.empty_like(memory)
    assert (
        musa_aiter.lmcache_kv_paged_to_buffer_two_x_nl(
            caches, recovered, slots, bs, nh, hs
        )
        is True
    )
    torch.musa.synchronize()

    valid = slots >= 0
    torch.testing.assert_close(recovered[:, :, valid].cpu(), memory[:, :, valid].cpu())
    assert recovered[:, :, ~valid].abs().max().item() == 0


@pytest.mark.musa
@pytest.mark.native
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_native_lmcache_kv_transfer_mla_when_enabled(dtype: torch.dtype) -> None:
    if os.environ.get("MUSA_AITER_NATIVE_LMCACHE_KV_TRANSFER", "0") != "1":
        pytest.skip("set MUSA_AITER_NATIVE_LMCACHE_KV_TRANSFER=1 to exercise native LMCache KV transfer")
    if not musa_aiter.is_native_lmcache_kv_transfer_available():
        pytest.skip("native LMCache KV transfer symbols missing")
    _require_musa()
    nl, nb, bs, hs = 2, 2, 4, 5
    slots = torch.tensor([0, 4, -1, 7], dtype=torch.int64, device="musa")
    kv_buffer = torch.arange(nl * slots.numel() * hs, dtype=torch.float32, device="musa").reshape(nl, slots.numel(), hs).to(dtype=dtype)
    paged = torch.zeros(nl, nb, bs, hs, dtype=dtype, device="musa")

    assert musa_aiter.lmcache_kv_buffer_to_paged_mla(kv_buffer, paged, slots) is True
    recovered = torch.empty_like(kv_buffer)
    assert musa_aiter.lmcache_paged_to_kv_buffer_mla(paged, recovered, slots) is True
    torch.musa.synchronize()

    torch.testing.assert_close(recovered[:, 0].cpu(), kv_buffer[:, 0].cpu())
    torch.testing.assert_close(recovered[:, 1].cpu(), kv_buffer[:, 1].cpu())
    torch.testing.assert_close(recovered[:, 3].cpu(), kv_buffer[:, 3].cpu())
    assert recovered[:, 2].abs().max().item() == 0
