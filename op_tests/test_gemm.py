import torch

import musa_aiter


def test_gemm_matches_matmul():
    a=torch.randn(3,4)
    b=torch.randn(4,5)
    out=musa_aiter.gemm(a,b)
    torch.testing.assert_close(out, a @ b)


def test_gemm_out():
    a=torch.randn(2,3)
    b=torch.randn(3,4)
    out=torch.empty(2,4)
    ret=musa_aiter.gemm(a,b,out=out)
    assert ret is out
    torch.testing.assert_close(out, a @ b)
