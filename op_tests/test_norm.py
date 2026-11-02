import torch

import musa_aiter


def test_rmsnorm_matches_reference():
    x = torch.randn(4, 8, dtype=torch.float32)
    w = torch.randn(8, dtype=torch.float32)
    out = musa_aiter.rmsnorm(x, w, eps=1e-6)
    ref = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + 1e-6) * w
    torch.testing.assert_close(out, ref)


def test_rmsnorm_validates_last_dim():
    x = torch.randn(2, 4)
    w = torch.randn(3)
    try:
        musa_aiter.rmsnorm(x, w)
    except ValueError as exc:
        assert "last dim mismatch" in str(exc)
    else:
        raise AssertionError("expected ValueError")
