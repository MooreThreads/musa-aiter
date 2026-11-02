"""Fixed-shape micro-workloads for ``musa-aiter benchmark`` (Phase P8 / §9.5)."""

from __future__ import annotations

from typing import Callable

import torch

import musa_aiter


def benchmark_ops() -> tuple[str, ...]:
    """Registered benchmark op ids (CLI ``--op`` choices)."""
    return (
        "add",
        "rmsnorm",
        "flash_attn_func",
        "gemm",
        "grouped_gemm",
        "fused_moe",
        "gemm_all_reduce",
        "kv_paged_to_linear",
        "kv_linear_to_paged",
        "kv_paged_to_linear_multi_layer",
        "kv_linear_to_paged_multi_layer",
        "kv_load_and_reshape_flash",
        "kv_copy_paged_blocks",
        "kv_cachegen_encode",
        "kv_cachegen_decode",
        "apply_rope_to_key_cache",
        "rmsnorm_backward",
        "softmax_last_dim",
    )


def make_benchmark_workload(op: str, device: str) -> tuple[Callable[[], object], str]:
    """Return a zero-arg callable and a dtype string for timing iterations."""
    key = op.strip().lower().replace("-", "_")
    if key == "add":
        a = torch.randn(4096, device=device)
        b = torch.randn(4096, device=device)
        return lambda: musa_aiter.add(a, b, prefer_native=False), str(a.dtype)
    if key == "rmsnorm":
        x = torch.randn(8, 128, device=device)
        w = torch.ones(128, device=device)
        return lambda: musa_aiter.rmsnorm(x, w), str(x.dtype)
    if key == "flash_attn_func":
        q = torch.randn(1, 4, 2, 8, device=device)
        k = torch.randn(1, 4, 2, 8, device=device)
        v = torch.randn(1, 4, 2, 8, device=device)
        return lambda: musa_aiter.flash_attn_func(q, k, v, causal=True), str(q.dtype)
    if key == "gemm":
        a = torch.randn(256, 256, device=device)
        b = torch.randn(256, 256, device=device)
        return lambda: musa_aiter.gemm(a, b), str(a.dtype)
    if key == "grouped_gemm":
        a = torch.randn(512, 128, device=device)
        b = torch.randn(4, 128, 256, device=device)
        idx = torch.randint(0, 4, (512,), device=device)
        return lambda: musa_aiter.grouped_gemm(a, b, idx), str(a.dtype)
    if key == "fused_moe":
        tokens, hidden, inter, experts = 128, 64, 128, 4
        x = torch.randn(tokens, hidden, device=device)
        w1 = torch.randn(experts, inter * 2, hidden, device=device)
        w2 = torch.randn(experts, hidden, inter, device=device)
        tw = torch.randn(tokens, 2, device=device)
        ti = torch.randint(0, experts, (tokens, 2), device=device)
        return lambda: musa_aiter.fused_moe(x, w1, w2, tw, ti), str(x.dtype)
    if key == "gemm_all_reduce":
        a = torch.randn(256, 256, device=device)
        b = torch.randn(256, 256, device=device)
        return lambda: musa_aiter.gemm_all_reduce(a, b), str(a.dtype)
    if key == "rmsnorm_backward":
        x = torch.randn(64, 128, device=device)
        w = torch.ones(128, device=device)
        g = torch.randn(64, 128, device=device)
        return lambda: musa_aiter.rmsnorm_backward_ref(g, x, w), str(x.dtype)
    if key == "softmax_last_dim":
        x = torch.randn(64, 128, device=device)
        return lambda: musa_aiter.softmax_last_dim(x, prefer_native=False), str(x.dtype)
    if key == "kv_paged_to_linear":
        nb, kvh, d, blk = 16, 8, 128, 16
        slots = torch.randint(0, nb * blk, (256,), device=device, dtype=torch.int32)
        kc = torch.randn(nb, kvh, d, blk, device=device)
        vc = torch.randn(nb, kvh, d, blk, device=device)
        return lambda: musa_aiter.kv_paged_to_linear(kc, vc, slots), str(kc.dtype)
    if key == "kv_linear_to_paged":
        nb, kvh, d, blk = 16, 8, 128, 16
        tokens = 256
        slots = torch.randint(0, nb * blk, (tokens,), device=device, dtype=torch.int32)
        kl = torch.randn(tokens, kvh, d, device=device)
        vl = torch.randn(tokens, kvh, d, device=device)
        kc = torch.zeros(nb, kvh, d, blk, device=device)
        vc = torch.zeros(nb, kvh, d, blk, device=device)
        return lambda: musa_aiter.kv_linear_to_paged(kl, vl, kc, vc, slots), str(kl.dtype)
    if key == "kv_paged_to_linear_multi_layer":
        nl, nb, kvh, d, blk = 4, 16, 8, 128, 16
        slots = torch.randint(0, nb * blk, (256,), device=device, dtype=torch.int32)
        kc = torch.randn(nl, nb, kvh, d, blk, device=device)
        vc = torch.randn(nl, nb, kvh, d, blk, device=device)
        return lambda: musa_aiter.kv_paged_to_linear_multi_layer(kc, vc, slots), str(kc.dtype)
    if key == "kv_linear_to_paged_multi_layer":
        nl, nb, kvh, d, blk = 4, 16, 8, 128, 16
        tokens = 256
        slots = torch.randint(0, nb * blk, (tokens,), device=device, dtype=torch.int32)
        kl = torch.randn(nl, tokens, kvh, d, device=device)
        vl = torch.randn(nl, tokens, kvh, d, device=device)
        kc = torch.zeros(nl, nb, kvh, d, blk, device=device)
        vc = torch.zeros(nl, nb, kvh, d, blk, device=device)
        return lambda: musa_aiter.kv_linear_to_paged_multi_layer(kl, vl, kc, vc, slots), str(kl.dtype)
    if key == "kv_load_and_reshape_flash":
        nb, kvh, d, blk = 32, 8, 128, 16
        batch, seq_len = 8, 128
        blocks_per_seq = (seq_len + blk - 1) // blk
        kc = torch.randn(nb, kvh, d, blk, device=device)
        vc = torch.randn(nb, kvh, d, blk, device=device)
        block_tables = torch.randint(0, nb, (batch, blocks_per_seq), device=device, dtype=torch.int32)
        seq_lens = torch.full((batch,), seq_len, device=device, dtype=torch.int32)
        return lambda: musa_aiter.kv_load_and_reshape_flash(kc, vc, block_tables, seq_lens, max_seq_len=seq_len), str(kc.dtype)
    if key == "kv_copy_paged_blocks":
        nl, nb, kvh, d, blk = 4, 32, 8, 128, 16
        src_k = torch.randn(nl, nb, kvh, d, blk, device=device)
        src_v = torch.randn(nl, nb, kvh, d, blk, device=device)
        dst_k = torch.zeros_like(src_k)
        dst_v = torch.zeros_like(src_v)
        mapping = torch.stack((torch.arange(16, device=device), torch.arange(15, -1, -1, device=device)), dim=1).to(torch.int32)
        return lambda: musa_aiter.kv_copy_paged_blocks(src_k, src_v, dst_k, dst_v, mapping), str(src_k.dtype)
    if key == "kv_cachegen_encode":
        x = torch.randn(4, 256, 8, 128, device=device)
        return lambda: musa_aiter.kv_cachegen_encode(x, num_bins=256), str(x.dtype)
    if key == "kv_cachegen_decode":
        x = torch.randn(4, 256, 8, 128, device=device)
        encoded, codebook, _ = musa_aiter.kv_cachegen_encode(x, num_bins=256)
        return lambda: musa_aiter.kv_cachegen_decode(encoded, codebook, dtype=x.dtype), str(encoded.dtype)
    if key == "apply_rope_to_key_cache":
        nb, kvh, d, blk = 16, 8, 128, 16
        tokens = 256
        kc = torch.randn(nb, kvh, d, blk, device=device)
        slots = torch.arange(tokens, device=device, dtype=torch.int32) % (nb * blk)
        positions = torch.arange(tokens, device=device, dtype=torch.int32)
        cos = torch.ones(tokens, d // 2, device=device)
        sin = torch.zeros(tokens, d // 2, device=device)
        return lambda: musa_aiter.apply_rope_to_key_cache(kc, slots, positions, cos, sin), str(kc.dtype)
    raise ValueError(f"unknown benchmark op: {op!r}; choose from {benchmark_ops()}")
