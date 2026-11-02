import torch
import musa_aiter


def test_grouped_gemm_matches_manual():
    tokens, hidden, out_dim, experts = 6, 4, 3, 2
    a = torch.randn(tokens, hidden)
    b = torch.randn(experts, hidden, out_dim)
    idx = torch.tensor([0, 1, 0, 1, 1, 0], dtype=torch.int64)
    out = musa_aiter.grouped_gemm(a, b, idx)
    ref = torch.empty_like(out)
    for i in range(tokens):
        ref[i] = a[i] @ b[idx[i]]
    torch.testing.assert_close(out, ref)


def test_grouped_gemm_out():
    a = torch.randn(3, 4)
    b = torch.randn(2, 4, 5)
    idx = torch.tensor([0, 1, 0])
    out = torch.empty(3, 5)
    ret = musa_aiter.grouped_gemm(a, b, idx, out=out)
    assert ret is out


def test_ragged_gemm():
    a_list = [torch.randn(2, 4), torch.randn(3, 4)]
    b = torch.randn(2, 4, 5)
    outs = musa_aiter.ragged_gemm(a_list, b)
    assert len(outs) == 2
    torch.testing.assert_close(outs[0], a_list[0] @ b[0])
    torch.testing.assert_close(outs[1], a_list[1] @ b[1])
