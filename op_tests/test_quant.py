import torch
import musa_aiter


def test_quantize_per_token():
    x = torch.randn(4, 8)
    x_q, scale = musa_aiter.quantize_per_token(x)
    assert x_q.shape == x.shape
    assert scale.shape == (4,)
    x_d = musa_aiter.dequantize_per_token(x_q, scale, dtype=torch.float32)
    torch.testing.assert_close(x_d, x, rtol=0.1, atol=0.1)


def test_quantize_per_tensor():
    x = torch.randn(32, 32)
    x_q, scale = musa_aiter.quantize_per_tensor(x)
    assert x_q.shape == x.shape
    x_d = musa_aiter.dequantize_per_tensor(x_q, scale, dtype=torch.float32)
    torch.testing.assert_close(x_d, x, rtol=0.1, atol=0.1)


def test_rmsnorm_quant():
    x = torch.randn(4, 16)
    w = torch.ones(16)
    x_q, scale = musa_aiter.rmsnorm_quant(x, w)
    assert x_q.shape == x.shape
    assert scale.shape == (4,)


def test_gated_rmsnorm_quant():
    x = torch.randn(2, 256)
    w = torch.ones(256)
    x_q, scale = musa_aiter.gated_rmsnorm_quant(x, w, group_size=128)
    assert x_q.shape == x.shape
