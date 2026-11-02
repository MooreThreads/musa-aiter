"""Isolate native kernel perf: per-layer vs layer_table vs torch baseline (musa)."""
import os

os.environ.setdefault("MUSA_AITER_NATIVE_KV_TRANSFER", "1")

import time

import torch

import musa_aiter

L, P, IB = 43, 1024, 37440

srcs = [torch.randint(0, 256, (P, IB), dtype=torch.uint8).to("musa") for _ in range(L)]
dsts = [torch.zeros_like(t) for t in srcs]
s_idx = torch.randperm(P).to(torch.int64).to("musa")
d_idx = torch.randperm(P).to(torch.int64).to("musa")
nbytes = L * P * IB


def timed(fn, iters=5):
    fn()
    torch.musa.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.musa.synchronize()
    return (time.perf_counter() - t0) / iters


# 1) torch baseline all layers
def torch_all():
    for s, d in zip(srcs, dsts):
        d.index_copy_(0, d_idx, s.index_select(0, s_idx))


# 2) torch contiguous copy (upper bound)
big_src = torch.randint(0, 256, (L * P, IB), dtype=torch.uint8).to("musa")
big_dst = torch.zeros_like(big_src)


def torch_contig():
    big_dst.copy_(big_src)


# 3) native per-layer loop
def native_per_layer():
    for s, d in zip(srcs, dsts):
        musa_aiter.kv_copy_paged_rows(s, d, s_idx, d_idx)


# 4) native layer_table
def native_table():
    musa_aiter.kv_copy_paged_rows_layer_table(srcs, dsts, s_idx, d_idx)


t = timed(torch_all)
print(f"torch index_copy all-layer : {t*1e3:8.2f} ms  {nbytes/t/1e9:8.1f} GB/s")
t = timed(torch_contig)
print(f"torch contiguous copy      : {t*1e3:8.2f} ms  {nbytes/t/1e9:8.1f} GB/s")
t = timed(native_per_layer)
print(f"native per-layer x43       : {t*1e3:8.2f} ms  {nbytes/t/1e9:8.1f} GB/s")
t = timed(native_table)
print(f"native layer_table         : {t*1e3:8.2f} ms  {nbytes/t/1e9:8.1f} GB/s")
