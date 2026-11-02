import torch
import math
import musa_aiter


def test_mla_decode_shape():
    num_seqs, qh, d = 2, 4, 16
    block_size, num_blocks = 4, 8
    q = torch.randn(num_seqs, qh, d)
    kv = torch.randn(num_blocks, block_size, d)
    k_pe = torch.randn(num_blocks, block_size, d // 2)
    block_tables = torch.tensor([[0, 1], [2, 3]], dtype=torch.int32)
    seq_lens = torch.tensor([5, 7], dtype=torch.int32)
    out = musa_aiter.mla_decode_fwd(q, kv, k_pe, block_tables, seq_lens, num_q_heads=qh, block_size=block_size)
    assert out.shape == q.shape


def test_mla_prefill_shape():
    q = torch.randn(1, 4, 2, 16)
    k = torch.randn(1, 4, 2, 16)
    v = torch.randn(1, 4, 2, 16)
    out = musa_aiter.mla_prefill_fwd(q, k, v, causal=True)
    assert out.shape == q.shape
