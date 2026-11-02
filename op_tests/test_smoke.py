import torch

import musa_aiter


def test_reduce_sum_cpu_matches_torch():
    x = torch.randn(3, 4)
    got = musa_aiter.reduce_sum(x, prefer_native=False)
    torch.testing.assert_close(got, torch.sum(x))


def test_softmax_last_dim_cpu_matches_torch():
    x = torch.randn(2, 5, 17)
    got = musa_aiter.softmax_last_dim(x, dim=-1, prefer_native=False)
    torch.testing.assert_close(got, torch.softmax(x, dim=-1))


def test_multiply_cpu_matches_torch():
    a = torch.randn(16)
    b = torch.randn(16)
    out = musa_aiter.multiply(a, b, prefer_native=False)
    torch.testing.assert_close(out, a * b)


def test_add_cpu_matches_torch():
    a = torch.randn(16)
    b = torch.randn(16)
    out = musa_aiter.add(a, b, prefer_native=False)
    torch.testing.assert_close(out, a + b)


def test_add_validates_shape():
    a = torch.randn(2)
    b = torch.randn(3)
    try:
        musa_aiter.add(a, b, prefer_native=False)
    except ValueError as exc:
        assert "shape mismatch" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_version_present():
    assert isinstance(musa_aiter.__version__, str)
    assert musa_aiter.__version__
