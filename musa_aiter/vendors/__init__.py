"""Phase P7 — vendor backend catalog (identifiers only; optional linkage is environment-specific).

Moore Threads stacks may expose **MuDNN**, **MUTLASS**/template libraries, **MUBIN** bundles, or in-repo **handwritten** ``.mu`` / **MCC JIT** builds. The runtime still defaults to **torch-reference** until native/vendor paths are wired per operator.
"""

from __future__ import annotations

# Stable string ids used in registry ``vendor_backends`` and ``configs/backends/``.
VENDOR_IDS: tuple[str, ...] = (
    "torch-reference",
    "mudnn",
    "mutlass",
    "mubin",
    "handwritten-mu",
    "mcc-jit",
)

VENDOR_DESCRIPTIONS: dict[str, str] = {
    "torch-reference": "PyTorch reference implementation (CPU/MUSA tensors per TorchMUSA).",
    "mudnn": "Vendor DNN / attention integration (planned; requires linked runtime).",
    "mutlass": "Template GEMM/attention-style library (planned).",
    "mubin": "Prebuilt tuned kernel bundle (planned).",
    "handwritten-mu": "In-tree MUSA C++ / ``.mu`` sources (JIT or packaged).",
    "mcc-jit": "JIT compile via ``mcc`` (smoke and future ops).",
}


def vendor_catalog() -> dict[str, str]:
    """Human-readable id → description (subset of vendor ecosystem)."""
    return dict(VENDOR_DESCRIPTIONS)
