#!/usr/bin/env python3
"""CacheGen MUSA benefit benchmark.

Measures the end-to-end benefit of the musa-aiter CacheGen-style codec
(``kv_cachegen_encode`` / ``kv_cachegen_decode``) on MUSA:

- compression ratio (encoded bytes + metadata vs source tensor bytes);
- encode/decode latency (min/median/p95/max) and effective throughput (GB/s);
- accuracy cost (MSE, MAE, max abs error, SNR);
- a parametric transfer-link benefit model (PCIe / NVMe / NIC) so the
  volume reduction can be turned into saved transfer time minus codec cost.

Run inside the MUSA container. The native data path is opt-in via
``MUSA_AITER_NATIVE_KV_TRANSFER=1``; the reference Torch path runs when the
gate is off. Use ``--path`` to run one or both.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import statistics
import time

import torch


# Link bandwidths used only for the parametric benefit model (GB/s, decimal).
# Labeled as reference points; not measured in this run.
DEFAULT_LINK_MODEL = {
    "pcie_gen4_x16": 25.0,
    "pcie_gen5_x16": 50.0,
    "nvme_ssd": 7.0,
    "nic_200gbe": 25.0,
}

# Peak working-set estimate multiplier: input(4B) + encoded(1B) + decoded(4B)
# plus reference-path temporaries (int64 index/round tensors).
_PEAK_MULTIPLIER = 2.5


def _stats(samples_ms: list[float]) -> dict:
    ordered = sorted(samples_ms)
    n = len(ordered)

    def _pct(q: float) -> float:
        idx = min(n - 1, max(0, int(round(q * (n - 1)))))
        return ordered[idx]

    return {
        "min_ms": ordered[0],
        "median_ms": statistics.median(ordered),
        "p95_ms": _pct(0.95),
        "max_ms": ordered[-1],
        "mean_ms": statistics.fmean(ordered),
    }


def _sync() -> None:
    torch.musa.synchronize()


def _timed(fn, iters: int, warmup: int) -> dict:
    for _ in range(warmup):
        fn()
    _sync()
    samples = []
    for _ in range(iters):
        start = time.perf_counter()
        fn()
        _sync()
        samples.append((time.perf_counter() - start) * 1e3)
    return _stats(samples)


def _make_input(torch, shape, kind: str, seed: int):
    gen = torch.Generator(device="cpu").manual_seed(seed)
    if kind == "narrow":
        base = torch.randn(shape, generator=gen, dtype=torch.float32) * 0.01
    elif kind == "wide":
        base = torch.randn(shape, generator=gen, dtype=torch.float32) * 1000.0
    elif kind == "const":
        base = torch.full(shape, 0.5, dtype=torch.float32)
    else:
        base = torch.randn(shape, generator=gen, dtype=torch.float32)
    return base.to("musa")


def _accuracy(src, decoded) -> dict:
    diff = (src - decoded).float()
    mse = (diff**2).mean().item()
    mae = diff.abs().mean().item()
    max_abs = diff.abs().max().item()
    sig = (src.float() ** 2).mean().item()
    if mse == 0.0:
        snr_db = float("inf")
    else:
        snr_db = 10.0 * math.log10(sig / mse)
    return {"mse": mse, "mae": mae, "max_abs_error": max_abs, "snr_db": snr_db}


def _cases():
    return [
        ("tiny", (1, 1, 1, 1), 256, "random"),
        ("chunk_mla_256x576", (1, 256, 576), 256, "random"),
        ("chunk_mla_narrow", (1, 256, 576), 256, "narrow"),
        ("chunk_mla_bins16", (1, 256, 576), 16, "random"),
        ("chunk_mla_bins128", (1, 256, 576), 128, "random"),
        ("layers64_chunk128", (64, 128, 576), 256, "random"),
        ("layers129_chunk128", (129, 128, 576), 256, "random"),
        ("layer_ctx32k", (1, 32768, 576), 256, "random"),
        ("all_layers_chunk1k", (129, 1024, 576), 256, "random"),
        ("preset_4x256x8x128", (4, 256, 8, 128), 256, "random"),
        ("odd_2x4x17x33", (2, 4, 17, 33), 64, "random"),
        ("const_256x576", (1, 256, 576), 256, "const"),
        ("wide_dyn_256x1024", (256, 1024), 256, "wide"),
    ]


def _run_phase(kv_transfer, label: str, native_available: bool, iters: int, warmup: int, budget_bytes: int):
    import torch

    cases = []
    skipped = []
    for name, shape, bins, kind in _cases():
        numel = 1
        for dim in shape:
            numel *= dim
        peak = int(numel * 9 * _PEAK_MULTIPLIER)
        if peak > budget_bytes:
            skipped.append({"case": name, "shape": list(shape), "peak_bytes_est": peak})
            continue
        src = _make_input(torch, shape, kind, seed=7)
        encoded, codebook, cdf = kv_transfer.kv_cachegen_encode(src, num_bins=bins)
        enc_stats = _timed(lambda: kv_transfer.kv_cachegen_encode(src, num_bins=bins), iters, warmup)
        dec_stats = _timed(lambda: kv_transfer.kv_cachegen_decode(encoded, codebook, dtype=torch.float32), iters, warmup)
        decoded = kv_transfer.kv_cachegen_decode(encoded, codebook, dtype=torch.float32)
        input_bytes = src.numel() * src.element_size()
        encoded_bytes = encoded.numel() * encoded.element_size()
        metadata_bytes = codebook.numel() * codebook.element_size() + cdf.numel() * cdf.element_size()
        compressed_bytes = encoded_bytes + metadata_bytes
        enc_s = enc_stats["median_ms"] / 1e3
        dec_s = dec_stats["median_ms"] / 1e3
        cases.append({
            "case": name,
            "shape": list(shape),
            "bins": bins,
            "kind": kind,
            "numel": numel,
            "input_bytes": input_bytes,
            "encoded_bytes": encoded_bytes,
            "metadata_bytes": metadata_bytes,
            "compressed_bytes": compressed_bytes,
            "ratio_x": (input_bytes / compressed_bytes) if compressed_bytes else None,
            "encode": {**enc_stats, "gbps": (input_bytes / enc_s / 1e9) if enc_s > 0 else None},
            "decode": {**dec_stats, "gbps": (input_bytes / dec_s / 1e9) if dec_s > 0 else None},
            "accuracy": _accuracy(src, decoded),
        })
        del src, encoded, codebook, cdf, decoded
        torch.musa.empty_cache()
    return {"path": label, "native_available": native_available, "cases": cases, "skipped": skipped}


def _link_model(phases, link_model: dict):
    model = []
    for phase in phases:
        rows = []
        for case in phase["cases"]:
            raw_b = case["input_bytes"]
            comp_b = case["compressed_bytes"]
            enc_ms = case["encode"]["median_ms"]
            dec_ms = case["decode"]["median_ms"]
            entry = {"path": phase["path"], "case": case["case"], "raw_bytes": raw_b, "compressed_bytes": comp_b, "links": {}}
            for link, gbps in link_model.items():
                bw = gbps * 1e9
                raw_ms = raw_b / bw * 1e3
                comp_ms = comp_b / bw * 1e3
                entry["links"][link] = {
                    "raw_ms": raw_ms,
                    "compressed_ms": comp_ms,
                    "store_net_ms": raw_ms - comp_ms - enc_ms,
                    "retrieve_net_ms": raw_ms - comp_ms - dec_ms,
                    "store_speedup_x": (raw_ms / (comp_ms + enc_ms)) if (comp_ms + enc_ms) > 0 else None,
                    "retrieve_speedup_x": (raw_ms / (comp_ms + dec_ms)) if (comp_ms + dec_ms) > 0 else None,
                }
            rows.append(entry)
        model.append({"path": phase["path"], "rows": rows})
    return model


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="musa:0")
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--path", choices=["both", "native", "reference"], default="both")
    parser.add_argument("--budget-mib", type=int, default=600, help="max estimated peak bytes per case")
    args = parser.parse_args()

    import torch  # noqa: F401
    import torch_musa  # noqa: F401
    from musa_aiter.ops import kv_transfer

    phases = []
    if args.path in ("both", "reference"):
        os.environ["MUSA_AITER_NATIVE_KV_TRANSFER"] = "0"
        phases.append(_run_phase(kv_transfer, "reference", False, args.iters, args.warmup, args.budget_mib * 1024 * 1024))
    if args.path in ("both", "native"):
        os.environ["MUSA_AITER_NATIVE_KV_TRANSFER"] = "1"
        build = kv_transfer.build_native_kv_transfer()
        native_available = bool(build.ok and kv_transfer.is_native_kv_transfer_available())
        phases.append(_run_phase(kv_transfer, "native", native_available, args.iters, args.warmup, args.budget_mib * 1024 * 1024))

    payload = {
        "schema": "musa_aiter.cachegen_benefit_benchmark.v1",
        "created_at_unix": int(time.time()),
        "device": args.device,
        "iters": args.iters,
        "warmup": args.warmup,
        "budget_mib": args.budget_mib,
        "torch": torch.__version__,
        "torch_musa": torch_musa.__version__,
        "link_model_gbps": DEFAULT_LINK_MODEL,
        "phases": phases,
        "link_model": _link_model(phases, DEFAULT_LINK_MODEL),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "phases": [p["path"] for p in phases]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
