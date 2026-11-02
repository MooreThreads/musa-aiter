#!/usr/bin/env python3
"""Verify musa_aiter can live in the same environment as vLLM 0.20.x.

**This repository targets Moore Threads MUSA** (TorchMUSA, ``torch.musa``). Use the
vendor trees **[torch_musa](https://github.com/MooreThreads/torch_musa)** and
**[vllm-musa](https://github.com/MooreThreads/vllm-musa)** for deployment; see
``docs/VLLM_0_20_INTEGRATION_GUIDE.md``. CPU and **MUSA** device strings are the
primary validation path for our operators.

The optional ``--llm-smoke`` probe exercises **stock** PyPI ``vllm.LLM`` and expects a
typical **CUDA-capable** PyTorch build (NVIDIA reference only). Skip it on pure
MUSA setups or when using vllm-musa unless their docs describe an equivalent smoke.

Checklist:

  - Imports resolve (Torch ABI compatible enough to import both packages).
  - Optional: vLLM reports a 0.20.x version.
  - musa_aiter attention + MoE reference ops run on the chosen **device**
    (``cpu``, ``musa:0``, or ``cuda:0`` for cross-stack checks).

Run from repository root::

    pip install -e .
    # Then install torch_musa + vllm-musa per https://github.com/MooreThreads/vllm-musa
    PYTHONPATH=. python examples/verify_vllm_musa_aiter.py --device musa:0

CI / no vLLM installed::

    PYTHONPATH=. python examples/verify_vllm_musa_aiter.py --no-vllm
"""

from __future__ import annotations

import argparse
import os
import sys

from framework_verify_common import assert_device_available, run_musa_aiter_ops


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--no-vllm",
        action="store_true",
        help="Skip vLLM import/version checks (only exercise musa_aiter).",
    )
    p.add_argument(
        "--device",
        default="cpu",
        help="Torch device for musa_aiter micro-benchmarks: cpu | musa:0 | cuda:0 (default: cpu).",
    )
    p.add_argument(
        "--llm-smoke",
        action="store_true",
        help=(
            "Run a tiny vLLM LLM.generate smoke (HF weights download). "
            "Requires stock vLLM's usual **CUDA** PyTorch stack; skipped if unavailable."
        ),
    )
    p.add_argument(
        "--llm-model",
        default="facebook/opt-125m",
        help="HF model id for --llm-smoke (default: facebook/opt-125m).",
    )
    return p.parse_args()


def _check_vllm_version() -> str:
    import vllm

    ver = getattr(vllm, "__version__", "0.0.0")
    parts = ver.split(".")
    if len(parts) < 2 or parts[0] != "0" or parts[1] != "20":
        print(
            f"WARNING: expected vLLM 0.20.x, got {ver!r}. "
            "Pin with: pip install 'vllm>=0.20,<0.21'",
            file=sys.stderr,
        )
    return ver


def _llm_smoke(model_id: str) -> None:
    import torch

    if not torch.cuda.is_available():
        print(
            "Skipping --llm-smoke: no CUDA runtime (stock vLLM 0.20 expects NVIDIA CUDA here). "
            "For MUSA-first workflows, rely on --device musa:0 and TorchMUSA validation instead."
        )
        return

    try:
        from vllm import LLM, SamplingParams
    except ImportError as e:
        raise SystemExit(f"vLLM import failed ({e}); install vLLM first.") from e

    sampling = SamplingParams(temperature=0.0, max_tokens=8)
    llm = LLM(
        model=model_id,
        enforce_eager=True,
        gpu_memory_utilization=0.3,
        max_model_len=256,
        trust_remote_code=False,
    )
    out = llm.generate(["Hello"], sampling)
    if not out or not out[0].outputs:
        raise AssertionError(out)
    text = out[0].outputs[0].text
    print(f"LLM smoke OK ({model_id!r}); sample continuation: {text!r}")


def main() -> int:
    args = _parse_args()

    if not args.no_vllm:
        try:
            vver = _check_vllm_version()
            print(f"vLLM import OK, version={vver!r}")
        except ImportError:
            print(
                "vLLM is not installed. Install vLLM 0.20.x into this environment "
                "(see docs/VLLM_0_20_INTEGRATION_GUIDE.md) or pass --no-vllm.",
                file=sys.stderr,
            )
            return 1
    else:
        print("--no-vllm: skipping vLLM import checks.")

    assert_device_available(args.device)

    import musa_aiter as ma

    print(f"musa_aiter version={ma.__version__!r}")

    trace = os.environ.get("MUSA_AITER_TRACE_DISPATCH")
    print(f"MUSA_AITER_TRACE_DISPATCH={trace!r}")
    run_musa_aiter_ops(args.device)
    print(f"musa_aiter ops OK on device={args.device!r}")

    if args.llm_smoke:
        if args.no_vllm:
            print("ERROR: --llm-smoke requires vLLM; do not combine with --no-vllm.", file=sys.stderr)
            return 1
        _llm_smoke(args.llm_model)

    print("All verification steps passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
