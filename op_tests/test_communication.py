import torch
import pytest
import musa_aiter


def test_init_destroy_dist_env():
    ctx = musa_aiter.init_dist_env(tensor_model_parallel_size=1, rank=0)
    assert ctx.world_size == 1
    assert ctx.rank == 0
    assert ctx.initialized
    musa_aiter.destroy_dist_env(ctx)
    assert not ctx.initialized


def test_all_reduce_single_rank():
    x = torch.randn(4, 8)
    out = musa_aiter.all_reduce(x)
    torch.testing.assert_close(out, x)
    assert out is not x


def test_reduce_scatter_single_rank():
    x = torch.randn(4, 8)
    out = musa_aiter.reduce_scatter(x)
    torch.testing.assert_close(out, x)


def test_all_gather_single_rank():
    x = torch.randn(4, 8)
    out = musa_aiter.all_gather(x)
    torch.testing.assert_close(out, x)


def test_multi_rank_reference_not_supported():
    ctx = musa_aiter.init_dist_env(tensor_model_parallel_size=2, rank=0)
    with pytest.raises(NotImplementedError):
        musa_aiter.all_reduce(torch.randn(2, 2), ctx=ctx)


def test_fused_all_reduce_rmsnorm():
    x = torch.randn(2, 4)
    w = torch.ones(4)
    out = musa_aiter.fused_all_reduce_rmsnorm(x, w)
    ref = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + 1e-6)
    torch.testing.assert_close(out, ref)
