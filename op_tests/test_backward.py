import torch
import musa_aiter


def test_rmsnorm_backward_ref_shapes():
    x = torch.randn(2, 4)
    w = torch.ones(4)
    grad = torch.randn(2, 4)
    dx, dw = musa_aiter.rmsnorm_backward_ref(grad, x, w)
    assert dx.shape == x.shape
    assert dw.shape == w.shape


def test_gemm_backward_ref():
    a = torch.randn(3, 4)
    b = torch.randn(4, 5)
    grad = torch.randn(3, 5)
    da, db = musa_aiter.gemm_backward_ref(grad, a, b)
    torch.testing.assert_close(da, grad @ b.T)
    torch.testing.assert_close(db, a.T @ grad)


def test_attention_backward_ref_shapes():
    q = torch.randn(1, 3, 2, 4)
    k = torch.randn(1, 3, 2, 4)
    v = torch.randn(1, 3, 2, 4)
    grad = torch.randn(1, 3, 2, 4)
    dq, dk, dv = musa_aiter.attention_backward_ref(grad, q, k, v)
    assert dq.shape == q.shape
    assert dk.shape == k.shape
    assert dv.shape == v.shape
