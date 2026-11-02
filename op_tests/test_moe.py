import torch
import musa_aiter


def test_topk_softmax_shape_and_values():
    gating = torch.randn(4, 8)
    w, idx = musa_aiter.topk_softmax(gating, top_k=2)
    assert w.shape == (4, 2)
    assert idx.shape == (4, 2)
    torch.testing.assert_close(w.sum(dim=-1), torch.ones(4))


def test_topk_sigmoid_shape():
    gating = torch.randn(2, 4)
    w, idx = musa_aiter.topk_sigmoid(gating, top_k=3)
    assert w.shape == (2, 3)
    assert idx.shape == (2, 3)


def test_moe_sorting():
    topk_ids = torch.tensor([[0, 1], [2, 0], [1, 1]], dtype=torch.int64)
    sorted_ids, expert_ids, tokens_per_expert, _ = musa_aiter.moe_sorting(topk_ids, num_experts=3, block_size=16)
    assert sorted_ids.numel() == 6
    assert expert_ids.numel() == 6
    assert tokens_per_expert.shape == (3,)


def test_fused_moe_shape():
    num_tokens, hidden, inter, num_experts = 8, 16, 32, 4
    x = torch.randn(num_tokens, hidden)
    w1 = torch.randn(num_experts, inter * 2, hidden)
    w2 = torch.randn(num_experts, hidden, inter)
    w = torch.randn(num_tokens, 2)
    ids = torch.randint(0, num_experts, (num_tokens, 2))
    out = musa_aiter.fused_moe(x, w1, w2, w, ids)
    assert out.shape == (num_tokens, hidden)
