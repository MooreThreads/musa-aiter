import torch
import musa_aiter


def test_gemm_all_reduce():
    a = torch.randn(3, 4)
    b = torch.randn(4, 5)
    out = musa_aiter.gemm_all_reduce(a, b)
    torch.testing.assert_close(out, a @ b)


def test_gemm_reduce_scatter():
    a = torch.randn(3, 4)
    b = torch.randn(4, 5)
    out = musa_aiter.gemm_reduce_scatter(a, b)
    torch.testing.assert_close(out, a @ b)


def test_all_gather_gemm():
    a = torch.randn(3, 4)
    b = torch.randn(4, 5)
    out = musa_aiter.all_gather_gemm(a, b)
    torch.testing.assert_close(out, a @ b)
