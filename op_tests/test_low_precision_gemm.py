import torch
import musa_aiter


def test_gemm_a8w8_shape():
    a = torch.randn(8, 16)
    b = torch.randn(16, 32)
    out = musa_aiter.gemm_a8w8(a, b)
    assert out.shape == (8, 32)


def test_gemm_a8w8_per_token_shape():
    a = torch.randn(4, 8)
    b = torch.randn(8, 12)
    out = musa_aiter.gemm_a8w8_per_token(a, b)
    assert out.shape == (4, 12)


def test_gemm_a4w4_shape():
    a = torch.randn(4, 8)
    b = torch.randn(8, 12)
    out = musa_aiter.gemm_a4w4(a, b)
    assert out.shape == (4, 12)


def test_grouped_gemm_a8w8_shape():
    a = torch.randn(10, 8)
    b = torch.randn(3, 8, 6)
    idx = torch.randint(0, 3, (10,))
    out = musa_aiter.grouped_gemm_a8w8(a, b, idx)
    assert out.shape == (10, 6)
