"""Verify native dispatch + bit-exactness on a MUSA device (run in container)."""
import os

os.environ.setdefault("MUSA_AITER_NATIVE_KV_TRANSFER", "1")
os.environ.setdefault("MUSA_AITER_TRACE_DISPATCH", "1")

import torch

assert torch.musa.is_available(), "no MUSA device"
import musa_aiter
from musa_aiter.ops.kv_transfer import is_native_kv_transfer_available

print("native_kv_transfer_available:", is_native_kv_transfer_available())

# per-layer, non-aligned item_bytes
src = torch.randint(0, 256, (64, 1729), dtype=torch.uint8).to("musa")
dst = torch.zeros(64, 1729, dtype=torch.uint8).to("musa")
idx = torch.randperm(64)[:16].to(torch.int64).to("musa")
musa_aiter.kv_copy_paged_rows(src, dst, idx, idx)
torch.musa.synchronize()
ref = torch.zeros(64, 1729, dtype=torch.uint8)
ref.index_copy_(0, idx.cpu(), src.cpu().index_select(0, idx.cpu()))
print("native per-layer (1729B) bit-exact:", torch.equal(dst.cpu(), ref))

# all-layer ptr-table, DSV4-scale item_bytes
srcs = [torch.randint(0, 256, (64, 37440), dtype=torch.uint8).to("musa") for _ in range(4)]
dsts = [torch.zeros_like(t) for t in srcs]
musa_aiter.kv_copy_paged_rows_layer_table(srcs, dsts, idx, idx)
torch.musa.synchronize()
ok = True
for s, d in zip(srcs, dsts):
    ref = torch.zeros_like(s, device="cpu")
    ref.index_copy_(0, idx.cpu(), s.cpu().index_select(0, idx.cpu()))
    ok = ok and torch.equal(d.cpu(), ref)
print("native layer-table (4L x 37440B) bit-exact:", ok)

# partial-index gather/scatter (src != dst indices)
src2 = torch.randint(0, 256, (128, 1728), dtype=torch.uint8).to("musa")
dst2 = torch.zeros(128, 1728, dtype=torch.uint8).to("musa")
s_idx = torch.randperm(128)[:32].to(torch.int64).to("musa")
d_idx = torch.randperm(128)[:32].to(torch.int64).to("musa")
musa_aiter.kv_copy_paged_rows(src2, dst2, s_idx, d_idx)
torch.musa.synchronize()
ref2 = torch.zeros(128, 1728, dtype=torch.uint8)
ref2.index_copy_(0, d_idx.cpu(), src2.cpu().index_select(0, s_idx.cpu()))
print("native per-layer gather/scatter bit-exact:", torch.equal(dst2.cpu(), ref2))
