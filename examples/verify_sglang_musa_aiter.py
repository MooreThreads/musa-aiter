#!/usr/bin/env python3
"""Verify musa_aiter can live in the same environment as SGLang (+ TorchMUSA / vendor stack).

**This repository targets Moore Threads MUSA** (``torch.musa``). Prefer **vendor-published**
SGLang + TorchMUSA images that include a compatible **musa-aiter** version; see
``docs/SGLANG_INTEGRATION_GUIDE.md``.

The optional ``--engine-smoke`` path loads a small Hugging Face model through
``sglang.Engine`` (downloads weights, launches SGLang subprocesses). Use it only
when your stack documents a working **CUDA** or **MUSA** runtime for SGLang; skip
it in CPU-only CI.

Checklist:

  - ``import sglang`` (unless ``--no-sglang``).
  - musa_aiter attention + paged decode + fused MoE on the chosen **device**
    (``cpu``, ``musa:0``, ``cuda:0``).

Run from repository root::

    pip install -e .
    # Then install SGLang + torch_musa per your vendor image / upstream docs
    PYTHONPATH=. python examples/verify_sglang_musa_aiter.py --device musa:0

CI / no SGLang installed::

    PYTHONPATH=. python examples/verify_sglang_musa_aiter.py --no-sglang
"""

from __future__ import annotations

import argparse
import os
import sys

from framework_verify_common import assert_device_available, run_musa_aiter_ops


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--no-sglang",
        action="store_true",
        help="Skip SGLang import/version checks (only exercise musa_aiter).",
    )
    p.add_argument(
        "--device",
        default="cpu",
        help="Torch device for musa_aiter micro-benchmarks: cpu | musa:0 | cuda:0 (default: cpu).",
    )
    p.add_argument(
        "--engine-smoke",
        action="store_true",
        help=(
            "Run a tiny offline sglang.Engine.generate smoke (HF download, subprocesses). "
            "Requires CUDA or MUSA per your SGLang build; not for CPU-only CI."
        ),
    )
    p.add_argument(
        "--engine-model",
        default="facebook/opt-125m",
        help="HF model id for --engine-smoke (default: facebook/opt-125m).",
    )
    return p.parse_args()


def _check_sglang_version() -> str:
    import sglang

    return getattr(sglang, "__version__", "unknown")


def _engine_smoke(model_id: str, *, device: str) -> None:
    import torch

    dl = device.lower()
    if dl == "cpu":
        print(
            "Skipping --engine-smoke: chosen --device is cpu; "
            "use --device musa:0 or cuda:0 for SGLang offline engine.",
        )
        return

    if dl.startswith("cuda") and not torch.cuda.is_available():
        print(
            "Skipping --engine-smoke: torch.cuda.is_available() is False.",
        )
        return
    if dl.startswith("musa"):
        if not hasattr(torch, "musa") or not torch.musa.is_available():
            print(
                "Skipping --engine-smoke: TorchMUSA unavailable (torch.musa.is_available() is False).",
            )
            return

    try:
        import sglang as sgl
    except ImportError as e:
        raise SystemExit(f"SGLang import failed ({e}); install SGLang first.") from e

    # Bounded memory / context for smoke; kwargs map to sglang.srt.server_args.ServerArgs
    engine_kw: dict = {
        "model_path": model_id,
        "mem_fraction_static": 0.35,
        "context_length": 256,
        "max_total_tokens": 4096,
        "max_running_requests": 2,
        "trust_remote_code": False,
        "log_level": "error",
    }
    if dl.startswith("musa"):
        engine_kw["device"] = "musa"
    elif dl.startswith("cuda"):
        engine_kw["device"] = "cuda"

    llm = sgl.Engine(**engine_kw)
    try:
        out = llm.generate(
            prompt="Hello",
            sampling_params={"temperature": 0.0, "max_new_tokens": 8},
        )
        if not out:
            raise AssertionError("empty engine.generate output")
        # SGLang versions differ: accept dict with text-like payload or str
        ok = False
        if isinstance(out, str):
            ok = bool(out.strip())
        elif isinstance(out, dict):
            ok = any(
                isinstance(out.get(k), str) and out.get(k, "").strip()
                for k in ("text", "output", "completion")
            ) or bool(out)
        else:
            ok = True
        if not ok:
            raise AssertionError(out)
        print(f"SGLang Engine smoke OK ({model_id!r}); raw output keys/type: {type(out).__name__}")
    finally:
        llm.shutdown()


def main() -> int:
    args = _parse_args()

    if not args.no_sglang:
        try:
            sver = _check_sglang_version()
            print(f"SGLang import OK, version={sver!r}")
        except ImportError:
            print(
                "SGLang is not installed. Install SGLang into this environment "
                "(see docs/SGLANG_INTEGRATION_GUIDE.md) or pass --no-sglang.",
                file=sys.stderr,
            )
            return 1
    else:
        print("--no-sglang: skipping SGLang import checks.")

    assert_device_available(args.device)

    import musa_aiter as ma

    print(f"musa_aiter version={ma.__version__!r}")

    trace = os.environ.get("MUSA_AITER_TRACE_DISPATCH")
    print(f"MUSA_AITER_TRACE_DISPATCH={trace!r}")
    run_musa_aiter_ops(args.device)
    print(f"musa_aiter ops OK on device={args.device!r}")

    if args.engine_smoke:
        if args.no_sglang:
            print(
                "ERROR: --engine-smoke requires SGLang; do not combine with --no-sglang.",
                file=sys.stderr,
            )
            return 1
        _engine_smoke(args.engine_model, device=args.device)

    print("All verification steps passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
