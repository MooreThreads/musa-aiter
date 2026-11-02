from __future__ import annotations

import argparse
import torch
import musa_aiter


class MusaAiterAttentionBackend:
    def prefill(self, q, k, v, causal=True):
        at = (
            musa_aiter.AttentionType.DECODER
            if causal
            else musa_aiter.AttentionType.ENCODER_ONLY
        )
        return musa_aiter.flash_attn_func(q, k, v, causal=causal, attention_type=at)

    def decode(self, query, key_cache, value_cache, block_tables, seq_lens, *, num_kv_heads, block_size):
        return musa_aiter.paged_attention(
            query,
            key_cache,
            value_cache,
            block_tables,
            seq_lens,
            num_kv_heads=num_kv_heads,
            block_size=block_size,
            attention_type=musa_aiter.AttentionType.DECODER,
        )


class MusaAiterMoEBackend:
    def forward(self, x, w1, w2, router_logits, top_k=2):
        weights, ids = musa_aiter.topk_softmax(router_logits, top_k)
        return musa_aiter.fused_moe(x, w1, w2, weights, ids)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    device = args.device

    attn = MusaAiterAttentionBackend()
    moe = MusaAiterMoEBackend()

    q = torch.randn(1, 8, 4, 16, device=device)
    k = torch.randn(1, 8, 4, 16, device=device)
    v = torch.randn(1, 8, 4, 16, device=device)
    out = attn.prefill(q, k, v)
    print("prefill:", tuple(out.shape))

    query = torch.randn(2, 4, 16, device=device)
    key_cache = torch.randn(4, 2, 16, 4, device=device)
    value_cache = torch.randn(4, 2, 16, 4, device=device)
    block_tables = torch.tensor([[0, 1], [2, 3]], dtype=torch.int32, device=device)
    seq_lens = torch.tensor([5, 7], dtype=torch.int32, device=device)
    dec = attn.decode(query, key_cache, value_cache, block_tables, seq_lens, num_kv_heads=2, block_size=4)
    print("decode:", tuple(dec.shape))

    tokens, hidden, inter, experts = 8, 16, 32, 4
    x = torch.randn(tokens, hidden, device=device)
    w1 = torch.randn(experts, inter * 2, hidden, device=device)
    w2 = torch.randn(experts, hidden, inter, device=device)
    logits = torch.randn(tokens, experts, device=device)
    moe_out = moe.forward(x, w1, w2, logits)
    print("moe:", tuple(moe_out.shape))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
