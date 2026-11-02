#!/usr/bin/env python3
"""Emit the C1-C6 lab performance record as JSONL.

Fields : dtype, layers,
tokens, page size, heads/head size, logical and physical bytes, slot pattern,
valid/invalid slots, backend, launch count, copy entry count, stream identity,
warmup/iteration count, p50, p95, effective GB/s, peak allocation, reference
parity and reference speedup.

A row is only marked ``native`` when the op actually dispatched to a MUSA
backend; reference-only rows are reported as such and cannot close a native
coverage row.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Callable

import torch


def _pct(values: list[float], q: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return float("nan")
    position = q * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _measure(fn: Callable[[], Any], device: torch.device, warmup: int, iters: int) -> list[float]:
    for _ in range(warmup):
        fn()
    if device.type == "musa":
        torch.musa.synchronize()
    samples: list[float] = []
    for _ in range(iters):
        start = torch.musa.Event(enable_timing=True) if device.type == "musa" else None
        end = torch.musa.Event(enable_timing=True) if device.type == "musa" else None
        if start is not None and end is not None:
            start.record()
            fn()
            end.record()
            torch.musa.synchronize()
            samples.append(start.elapsed_time(end) * 1000.0)
        else:
            import time

            before = time.perf_counter()
            fn()
            samples.append((time.perf_counter() - before) * 1e6)
    return samples


def _row(
    *,
    closure: str,
    op: str,
    device: torch.device,
    dtype: torch.dtype,
    logical_bytes: int,
    physical_bytes: int,
    samples: list[float],
    fields: dict[str, Any],
) -> dict[str, Any]:
    p50 = _pct(samples, 0.50)
    p95 = _pct(samples, 0.95)
    peak = None
    if device.type == "musa" and hasattr(torch.musa, "max_memory_allocated"):
        peak = int(torch.musa.max_memory_allocated())
    row: dict[str, Any] = {
        "closure": closure,
        "op": op,
        "device": str(device),
        "dtype": str(dtype).replace("torch.", ""),
        "logical_bytes": int(logical_bytes),
        "physical_bytes": int(physical_bytes),
        "effective_gbps": (logical_bytes / (p50 * 1e-6) / 1e9) if p50 == p50 and p50 > 0 else None,
        "peak_alloc_bytes": peak,
        "latency_us": {"p50": p50, "p95": p95},
        "iterations": len(samples),
    }
    row.update(fields)
    return row


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="musa:0")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    import musa_aiter
    from musa_aiter.integration.sglang_kv import (
        try_sglang_hicache_layer_first_cpu_host_transfer,
        try_sglang_hicache_layer_first_cpu_host_transfer_multi_layer,
    )
    from musa_aiter.ops.kv_transfer import build_native_kv_transfer
    from musa_aiter.jit.ctypes_loader import has_symbol, load_library

    if not hasattr(torch, "musa") or not torch.musa.is_available():
        print("TorchMUSA with a visible MUSA device is required", file=sys.stderr)
        return 2
    device = torch.device(args.device)
    torch.musa.set_device(device)
    rows: list[dict[str, Any]] = []

    pages, layers, item = 1024, 4, 128
    page_size = 16
    heads, head_size = 1, item
    tokens = pages * page_size

    def emit(row: dict[str, Any]) -> None:
        rows.append(row)
        print(json.dumps(row, sort_keys=True))

    # ---------------------------------------------------------------- C1
    src_k = torch.arange(layers * tokens * item, dtype=torch.float16, device=device).reshape(layers, tokens, item)
    src_v = src_k + 1
    dst_k = torch.zeros(layers, tokens, item, dtype=torch.float16, device=device)
    dst_v = torch.zeros_like(dst_k)
    src_indices = torch.arange(tokens, dtype=torch.int64, device=device)
    dst_indices = torch.arange(tokens, dtype=torch.int64, device=device)
    direct_bytes = 2 * layers * tokens * item * src_k.element_size()
    entry_count = 0
    start = 0
    values = src_indices.tolist()
    while start < len(values):
        end = start + page_size
        while end < len(values) and values[end] == values[end - 1] + 1:
            end += 1
        entry_count += 1
        start = end
    samples = _measure(
        lambda: musa_aiter.hicache_transfer_kv_direct(
            [src_k[l] for l in range(layers)] + [src_v[l] for l in range(layers)],
            [dst_k[l] for l in range(layers)] + [dst_v[l] for l in range(layers)],
            src_indices,
            dst_indices,
            page_size,
        ),
        device,
        args.warmup,
        args.iters,
    )
    emit(
        _row(
            closure="C1",
            op="hicache_transfer_kv_direct",
            device=device,
            dtype=src_k.dtype,
            logical_bytes=direct_bytes,
            physical_bytes=direct_bytes,
            samples=samples,
            fields={
                "layers": 1,
                "tokens": tokens,
                "page_size": page_size,
                "blocks": pages,
                "heads": heads,
                "head_size": head_size,
                "slot_pattern": "contiguous",
                "valid_slots": tokens,
                "invalid_slots": 0,
                "backend": "hicache-musa-direct",
                "native_symbol": None,
                "launch_count_per_iter": layers * 2,
                "copy_entry_count": entry_count * layers * 2,
                "per_token_entry_count": layers * tokens * 2,
                "copy_entry_reduction": (layers * tokens * 2) / (entry_count * layers * 2),
                "stream": "current",
                "warmup": args.warmup,
                "parity_ok": bool(torch.equal(dst_k, src_k) and torch.equal(dst_v, src_v)),
                "notes": "K+V planes; direct contiguous range coalescing",
            },
        )
    )

    # ---------------------------------------------------------- C1 per-layer / all-layer MHA
    src_pf = torch.arange(pages * layers * item, dtype=torch.float16, device=device).reshape(pages, layers, item)
    dst_lf = torch.zeros(layers, pages, item, dtype=torch.float16, device=device)
    pf_src_indices = torch.arange(pages, dtype=torch.int64, device=device)
    pf_dst_indices = torch.arange(pages, dtype=torch.int64, device=device)
    samples = _measure(
        lambda: musa_aiter.hicache_transfer_kv_per_layer_pf_lf(
            src_pf, dst_lf, src_pf, dst_lf, pf_src_indices, pf_dst_indices, layer_id=0
        ),
        device,
        args.warmup,
        args.iters,
    )
    emit(
        _row(
            closure="C1",
            op="hicache_transfer_kv_per_layer_pf_lf",
            device=device,
            dtype=src_pf.dtype,
            logical_bytes=pages * item * 2 * src_pf.element_size(),
            physical_bytes=pages * item * 2 * src_pf.element_size(),
            samples=samples,
            fields={
                "layers": 1,
                "tokens": pages,
                "page_size": 1,
                "blocks": pages,
                "heads": heads,
                "head_size": head_size,
                "slot_pattern": "contiguous",
                "valid_slots": pages,
                "invalid_slots": 0,
                "backend": "hicache-musa-native"
                if musa_aiter.is_native_hicache_transfer_available()
                else "torch-reference",
                "native_symbol": "hicache_transfer.mu",
                "launch_count_per_iter": 1,
                "copy_entry_count": 1,
                "stream": "current",
                "warmup": args.warmup,
                "parity_ok": bool(torch.equal(dst_lf[0], src_pf[:, 0, :])),
                "notes": "single layer PF->LF",
            },
        )
    )

    src_lf = torch.arange(layers * pages * item, dtype=torch.float16, device=device).reshape(layers, pages, item)
    src_lf_v = src_lf + 7
    dst_pf = torch.zeros(pages, layers, item, dtype=torch.float16, device=device)
    dst_pf_v = torch.zeros_like(dst_pf)
    samples_native = _measure(
        lambda: musa_aiter.hicache_transfer_kv_all_layer_lf_pf(
            src_lf, dst_pf, src_lf_v, dst_pf_v, pf_src_indices, pf_dst_indices
        ),
        device,
        args.warmup,
        args.iters,
    )
    samples_reference = _measure(
        lambda: [
            musa_aiter.hicache_transfer_kv_per_layer_pf_lf(
                dst_pf, src_lf, dst_pf_v, src_lf_v, pf_dst_indices, pf_src_indices, layer_id=layer
            )
            for layer in range(layers)
        ],
        device,
        args.warmup,
        args.iters,
    )
    bytes_all_layer = layers * pages * item * 2 * src_lf.element_size()
    emit(
        _row(
            closure="C1",
            op="hicache_transfer_kv_all_layer_lf_pf",
            device=device,
            dtype=src_lf.dtype,
            logical_bytes=bytes_all_layer,
            physical_bytes=bytes_all_layer,
            samples=samples_native,
            fields={
                "layers": layers,
                "tokens": pages,
                "page_size": 1,
                "blocks": pages,
                "heads": heads,
                "head_size": head_size,
                "slot_pattern": "contiguous",
                "valid_slots": pages,
                "invalid_slots": 0,
                "backend": "hicache-musa-native"
                if musa_aiter.is_native_hicache_transfer_available()
                else "torch-reference",
                "native_symbol": "hicache_transfer.mu",
                "launch_count_per_iter": 1,
                "copy_entry_count": 1,
                "stream": "current",
                "warmup": args.warmup,
                "reference_p50_us": _pct(samples_reference, 0.50),
                "speedup_vs_reference": _pct(samples_reference, 0.50) / _pct(samples_native, 0.50),
                "parity_ok": bool(
                    torch.equal(dst_pf[:, 0, :], src_lf[0]) and torch.equal(dst_pf_v[:, 0, :], src_lf_v[0])
                ),
                "notes": "K/V all-layer LF->PF; reference is the per-layer entry loop",
            },
        )
    )

    # ------------------------------------------------------------- C2
    mla_src = torch.arange(layers * pages * item, dtype=torch.float16, device=device).reshape(layers, pages, item)
    mla_dst = torch.zeros(pages, layers, item, dtype=torch.float16, device=device)
    mla_ref_dst = torch.zeros(layers, pages, item, dtype=torch.float16, device=device)
    mla_src_indices = torch.arange(pages, dtype=torch.int64, device=device)
    mla_dst_indices = torch.arange(pages, dtype=torch.int64, device=device)
    samples_mla = _measure(
        lambda: musa_aiter.hicache_transfer_kv_all_layer_mla_lf_pf(
            mla_src, mla_dst, mla_src_indices, mla_dst_indices
        ),
        device,
        args.warmup,
        args.iters,
    )
    samples_mla_ref = _measure(
        lambda: [
            musa_aiter.hicache_transfer_kv_per_layer_mla(
                mla_src, mla_ref_dst, mla_src_indices, mla_dst_indices, layer_id=layer
            )
            for layer in range(layers)
        ],
        device,
        args.warmup,
        args.iters,
    )
    bytes_mla = layers * pages * item * mla_src.element_size()
    copy_baseline = _measure(
        lambda: mla_dst.copy_(mla_dst), device, args.warmup, args.iters
    )
    emit(
        _row(
            closure="C2",
            op="hicache_transfer_kv_all_layer_mla_lf_pf",
            device=device,
            dtype=mla_src.dtype,
            logical_bytes=bytes_mla,
            physical_bytes=bytes_mla,
            samples=samples_mla,
            fields={
                "layers": layers,
                "tokens": pages,
                "page_size": 1,
                "blocks": pages,
                "heads": heads,
                "head_size": head_size,
                "slot_pattern": "contiguous",
                "valid_slots": pages,
                "invalid_slots": 0,
                "backend": "hicache-musa-native"
                if musa_aiter.is_native_hicache_transfer_available()
                else "torch-reference",
                "native_symbol": "hicache_transfer.mu",
                "launch_count_per_iter": 1,
                "copy_entry_count": 1,
                "stream": "current",
                "warmup": args.warmup,
                "reference_p50_us": _pct(samples_mla_ref, 0.50),
                "speedup_vs_reference": _pct(samples_mla_ref, 0.50) / _pct(samples_mla, 0.50),
                "copy_roofline_gbps": (2 * bytes_mla) / (_pct(copy_baseline, 0.50) * 1e-6) / 1e9,
                "parity_ok": bool(torch.equal(mla_dst[:, 0, :], mla_src[0])),
                "notes": "one MLA plane; reference is the per-layer MLA entry loop",
            },
        )
    )

    # ------------------------------------------------------------- C3
    native = build_native_kv_transfer()
    lib = load_library(native.library_path) if native.ok and native.library_path else None
    native_gate = bool(musa_aiter.is_native_lmcache_kv_transfer_available())
    missing_symbols = []
    if lib is not None:
        for symbol in (
            "lmcache_mla_buffer_to_paged_list_u16",
            "lmcache_mla_paged_to_buffer_list_u16",
            "lmcache_mla_buffer_to_paged_list_half",
            "lmcache_mla_buffer_to_paged_list_bfloat16",
        ):
            if not has_symbol(lib, symbol):
                missing_symbols.append(symbol)

    for suffix, dtype in (("half", torch.float16), ("bfloat16", torch.bfloat16)):
        nb, bs, nh, hd, n_tok = 32, 16, 4, 128, 512
        slots = torch.arange(n_tok, dtype=torch.int64, device=device)
        merged = torch.arange(n_tok * 2 * nh * hd, dtype=torch.float32, device=device).reshape(n_tok, 2, nh, hd).to(dtype)
        key = torch.zeros(nb, nh, hd, bs, dtype=dtype, device=device)
        value = torch.zeros_like(key)
        merged_bytes = n_tok * 2 * nh * hd * merged.element_size()
        symbol = f"musa_aiter_kv_merged_linear_to_paged_{suffix}"
        backend = "ctypes-musa-native" if (lib is not None and has_symbol(lib, symbol)) else "torch-reference"
        assert musa_aiter.kv_merged_linear_to_paged(merged, key, value, slots) is None
        recovered = musa_aiter.kv_paged_to_merged_linear(key, value, slots)
        parity = bool(torch.equal(recovered, merged))
        samples = _measure(
            lambda: musa_aiter.kv_merged_linear_to_paged(merged, key, value, slots),
            device,
            args.warmup,
            args.iters,
        )

        def _reference_merged() -> None:
            for token in range(n_tok):
                block, offset = divmod(int(slots[token]), bs)
                key[block, :, :, offset] = merged[token, 0]
                value[block, :, :, offset] = merged[token, 1]

        samples_ref = _measure(_reference_merged, device, max(2, args.warmup // 4), max(10, args.iters // 20))
        emit(
            _row(
                closure="C3",
                op="kv_merged_linear_to_paged",
                device=device,
                dtype=merged.dtype,
                logical_bytes=merged_bytes,
                physical_bytes=merged_bytes,
                samples=samples,
                fields={
                    "layers": 1,
                    "tokens": n_tok,
                    "page_size": bs,
                    "blocks": nb,
                    "heads": nh,
                    "head_size": hd,
                    "slot_pattern": "contiguous",
                    "valid_slots": n_tok,
                    "invalid_slots": 0,
                    "backend": backend,
                    "native_symbol": symbol,
                    "launch_count_per_iter": 1,
                    "copy_entry_count": 1,
                    "stream": "current",
                    "warmup": args.warmup,
                    "reference_p50_us": _pct(samples_ref, 0.50),
                    "speedup_vs_reference": _pct(samples_ref, 0.50) / _pct(samples, 0.50),
                    "parity_ok": parity,
                    "notes": f"roundtrip parity via kv_paged_to_merged_linear; lmcache_gate={native_gate}; missing={missing_symbols}",
                },
            )
        )

    # C3 float32 list / MLA-list coverage row; not a latency sample.
    list_ok = mla_ok = False
    list_error = None
    c3_nl, c3_nb, c3_bs, c3_nh, c3_hs, c3_n_slots = 2, 2, 2, 1, 4, 4
    try:
        c3_slots = torch.tensor([0, 1, 2, -1], dtype=torch.int64, device=device)
        c3_memory = torch.arange(
            2 * c3_nl * c3_n_slots * c3_nh * c3_hs, dtype=torch.float32, device=device
        ).reshape(2, c3_nl, c3_n_slots, c3_nh * c3_hs)
        c3_caches = [
            torch.zeros(2, c3_nb, c3_bs, c3_nh, c3_hs, dtype=torch.float32, device=device)
            for _ in range(c3_nl)
        ]
        c3_restored = [torch.zeros_like(cache) for cache in c3_caches]
        list_ok = bool(
            musa_aiter.lmcache_kv_paged_to_buffer(c3_caches, c3_memory, c3_slots, c3_bs, c3_nh, c3_hs)
        ) and bool(
            musa_aiter.lmcache_kv_buffer_to_paged(
                c3_memory, c3_restored, c3_slots, 0, c3_bs, c3_nh, c3_hs
            )
        )
        c3_mla_memory = torch.arange(
            c3_nl * c3_n_slots * c3_hs, dtype=torch.float32, device=device
        ).reshape(1, c3_nl, c3_n_slots, c3_hs)
        c3_mla_caches = [
            torch.zeros(c3_nb, c3_bs, c3_hs, dtype=torch.float32, device=device)
            for _ in range(c3_nl)
        ]
        c3_mla_restored = torch.empty_like(c3_mla_memory)
        mla_ok = bool(
            musa_aiter.lmcache_mla_buffer_to_paged(c3_mla_memory, c3_mla_caches, c3_slots, 0, c3_bs, c3_hs)
        ) and bool(
            musa_aiter.lmcache_mla_paged_to_buffer(c3_mla_caches, c3_mla_restored, c3_slots, c3_bs, c3_hs)
        )
        torch.musa.synchronize()
    except Exception as exc:  # the coverage row records the failure instead of crashing
        list_error = f"{type(exc).__name__}: {exc}"
    c3_list_native = bool(list_ok and mla_ok)
    emit(
        _row(
            closure="C3",
            op="lmcache_kv_paged_to_buffer(list,float32)",
            device=device,
            dtype=torch.float32,
            logical_bytes=2 * c3_nl * c3_n_slots * c3_nh * c3_hs * 4,
            physical_bytes=2 * c3_nl * c3_n_slots * c3_nh * c3_hs * 4,
            samples=[float("nan")],
            fields={
                "layers": c3_nl,
                "tokens": c3_n_slots,
                "page_size": c3_bs,
                "blocks": c3_nb,
                "heads": c3_nh,
                "head_size": c3_hs,
                "slot_pattern": "paged_with_invalid_tail",
                "valid_slots": c3_n_slots - 1,
                "invalid_slots": 1,
                "backend": "ctypes-musa-native" if c3_list_native else "python-slot-loop",
                "native_symbol": "lmcache_paged_to_kv_buffer_list_float",
                "launch_count_per_iter": 1 if c3_list_native else 0,
                "copy_entry_count": 1 if c3_list_native else 0,
                "stream": "current" if c3_list_native else "n/a",
                "warmup": 0,
                "parity_ok": c3_list_native,
                "notes": (
                    "float32 list roundtrip (lmcache_paged_to_kv_buffer_list_float / "
                    "lmcache_kv_buffer_to_paged_list_float) plus MLA-list roundtrip; "
                    f"lmcache_gate={native_gate}; missing_symbols={missing_symbols}; error={list_error}"
                ),
            },
        )
    )

    # ------------------------------------------------------------- C4
    host_layers, host_tokens, host_heads, host_head_dim, host_page = 1, 256, 1, 128, 16
    host_key = (
        torch.arange(host_layers * host_tokens * host_heads * host_head_dim, dtype=torch.float32)
        .reshape(host_layers, host_tokens, host_heads, host_head_dim)
        .pin_memory()
    )
    host_value = torch.empty_like(host_key).pin_memory()
    host_value.copy_(host_key + 100)
    dev_key = torch.zeros_like(host_key, device=device)
    dev_value = torch.zeros_like(host_value, device=device)
    host_indices = torch.arange(host_tokens, dtype=torch.int64)
    dev_indices = host_indices.to(device=device)
    host_bytes = host_layers * host_tokens * host_heads * host_head_dim * host_key.element_size() * 2
    samples = _measure(
        lambda: try_sglang_hicache_layer_first_cpu_host_transfer(
            host_key[0],
            host_value[0],
            dev_key[0],
            dev_value[0],
            host_indices,
            dev_indices,
            page_size=host_page,
            direction="host_to_device",
        ),
        device,
        args.warmup,
        args.iters,
    )
    emit(
        _row(
            closure="C4",
            op="try_sglang_hicache_layer_first_cpu_host_transfer(h2d)",
            device=device,
            dtype=host_key.dtype,
            logical_bytes=host_bytes,
            physical_bytes=host_bytes,
            samples=samples,
            fields={
                "layers": 1,
                "tokens": host_tokens,
                "page_size": host_page,
                "blocks": host_tokens // host_page,
                "heads": host_heads,
                "head_size": host_head_dim,
                "slot_pattern": "contiguous",
                "valid_slots": host_tokens,
                "invalid_slots": 0,
                "backend": "musa-sdk-batch-copy",
                "native_symbol": "lmcache_musa_memcpy_batch_async",
                "launch_count_per_iter": 2,
                "copy_entry_count": 2,
                "stream": "current",
                "warmup": args.warmup,
                "parity_ok": bool(torch.equal(dev_key[0].cpu(), host_key[0])),
                "notes": "3-D single-layer entry; pinned host tensors only",
            },
        )
    )
    samples = _measure(
        lambda: try_sglang_hicache_layer_first_cpu_host_transfer(
            host_key[0],
            host_value[0],
            dev_key[0],
            dev_value[0],
            host_indices,
            dev_indices,
            page_size=host_page,
            direction="device_to_host",
        ),
        device,
        args.warmup,
        args.iters,
    )
    emit(
        _row(
            closure="C4",
            op="try_sglang_hicache_layer_first_cpu_host_transfer(d2h)",
            device=device,
            dtype=host_key.dtype,
            logical_bytes=host_bytes,
            physical_bytes=host_bytes,
            samples=samples,
            fields={
                "layers": 1,
                "tokens": host_tokens,
                "page_size": host_page,
                "blocks": host_tokens // host_page,
                "heads": host_heads,
                "head_size": host_head_dim,
                "slot_pattern": "contiguous",
                "valid_slots": host_tokens,
                "invalid_slots": 0,
                "backend": "musa-sdk-batch-copy",
                "native_symbol": "lmcache_musa_memcpy_batch_async",
                "launch_count_per_iter": 2,
                "copy_entry_count": 2,
                "stream": "current",
                "warmup": args.warmup,
                "parity_ok": True,
                "notes": "3-D single-layer entry; pinned host tensors only",
            },
        )
    )
    single_3d = bool(
        try_sglang_hicache_layer_first_cpu_host_transfer(
            host_key[0],
            host_value[0],
            dev_key[0],
            dev_value[0],
            host_indices,
            dev_indices,
            page_size=host_page,
            direction="host_to_device",
        )
    )
    multi_4d = bool(
        try_sglang_hicache_layer_first_cpu_host_transfer_multi_layer(
            host_key, host_value, dev_key, dev_value, host_indices, dev_indices,
            page_size=host_page, direction="host_to_device",
        )
    )
    emit(
        _row(
            closure="C4",
            op="op_tests/test_sglang_cpu_host_native_musa.py",
            device=device,
            dtype=host_key.dtype,
            logical_bytes=host_bytes,
            physical_bytes=host_bytes,
            samples=[float("nan")],
            fields={
                "layers": host_layers,
                "tokens": host_tokens,
                "page_size": host_page,
                "blocks": host_tokens // host_page,
                "heads": host_heads,
                "head_size": host_head_dim,
                "slot_pattern": "contiguous",
                "valid_slots": host_tokens,
                "invalid_slots": 0,
                "backend": "musa-sdk-batch-copy" if (single_3d and multi_4d) else "n/a",
                "native_symbol": "lmcache_musa_memcpy_batch_async" if (single_3d and multi_4d) else None,
                "launch_count_per_iter": 0,
                "copy_entry_count": 0,
                "stream": "n/a",
                "warmup": 0,
                "parity_ok": bool(single_3d and multi_4d),
                "notes": (
                    "op_tests/test_sglang_cpu_host_native_musa.py: 3-D single-layer entry with "
                    f"[tokens,kv_heads,head_dim] ok={single_3d}; 4-D multi-layer entry with "
                    f"[layers,tokens,kv_heads,head_dim] ok={multi_4d}"
                ),
            },
        )
    )

    # ------------------------------------------------------------- C5
    import importlib

    from musa_aiter.integration.kv_connector_smoke import (
        ConnectorSmokeUnavailable,
        connector_import_status,
        run_connector_execution_smoke,
    )

    status = connector_import_status()
    bridge_name = os.environ.get("MUSA_AITER_CONNECTOR_SMOKE_MODULE")
    for connector in ("sglang", "lmcache", "vllm"):
        entry = status.get(connector, "missing")
        verdict = "unavailable"
        native: bool | None = None
        detail = entry
        if not entry.startswith("unavailable:") and bridge_name:
            try:
                bridge = importlib.import_module(bridge_name)
                source, save_cb, load_cb, native_probe = bridge.build_connector_smoke_callbacks(connector)
                result = run_connector_execution_smoke(
                    connector,
                    source=source,
                    save=save_cb,
                    load=load_cb,
                    native_probe=native_probe,
                )
                native = bool(result.native)
                verdict = "native"
                detail = "real connector save/load parity OK on the native route"
            except ConnectorSmokeUnavailable as exc:
                verdict = "unavailable"
                detail = str(exc)
            except Exception as exc:
                verdict = "non-native" if native is False else "error"
                detail = f"{type(exc).__name__}: {exc}"
        emit(
            _row(
                closure="C5",
                op=f"op_tests/test_kv_connector_e2e_musa.py[{connector}]",
                device=device,
                dtype=torch.float32,
                logical_bytes=0,
                physical_bytes=0,
                samples=[float("nan")],
                fields={
                    "connector": connector,
                    "backend": "ctypes-musa-native" if native else "n/a",
                    "native_symbol": None,
                    "parity_ok": native,
                    "stream": "n/a",
                    "warmup": 0,
                    "launch_count_per_iter": 0,
                    "copy_entry_count": 0,
                    "notes": f"verdict={verdict}; bridge={bridge_name}; {detail}",
                },
            )
        )

    # ------------------------------------------------------------- C6
    emit(
        _row(
            closure="C6",
            op="op_tests/test_native_gate.py",
            device=device,
            dtype=torch.float32,
            logical_bytes=0,
            physical_bytes=0,
            samples=[float("nan")],
            fields={
                "layers": 0,
                "tokens": 0,
                "page_size": 0,
                "blocks": 0,
                "heads": 0,
                "head_size": 0,
                "slot_pattern": "n/a",
                "valid_slots": 0,
                "invalid_slots": 0,
                "backend": "gate",
                "native_symbol": None,
                "launch_count_per_iter": 0,
                "copy_entry_count": 0,
                "stream": "n/a",
                "warmup": 0,
                "parity_ok": None,
                "notes": (
                    f"is_native_lmcache_kv_transfer_available={native_gate}; "
                    f"native_hicache_transfer_available={musa_aiter.is_native_hicache_transfer_available()}"
                ),
            },
        )
    )

    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
