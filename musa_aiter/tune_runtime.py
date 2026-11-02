"""Runtime tuning table lookup (Phases P9–P10) — wires CSV/JSON configs into dispatch metadata.

**GEMM** — ``MUSA_AITER_CONFIG_GEMM``. Shape key:
``m=<a.shape[-2]>,n=<b.shape[-1]>,k=<a.shape[-1]>``.

**Fused MoE** — ``MUSA_AITER_CONFIG_MOE``. Shape key:
``tokens=T,hidden=H,inter=I,experts=E,topk=K`` with ``inter = w1.shape[1] // 2`` for the SiLU
gate+up split (see ``musa_aiter.ops.moe.fused_moe``).

**Grouped GEMM** — ``MUSA_AITER_CONFIG_GROUPED_GEMM``. Shape key:
``tokens=T,hidden=K,out=N,groups=G`` (``K`` contraction / ``hidden_dim``, ``N`` ``out_dim``,
``G`` ``b.shape[0]``).

Architecture: ``MUSA_AITER_TUNING_ARCH`` or first ``MUSA_AITER_ARCHS`` entry (default ``mp31``).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch

from musa_aiter.tuning import TunedConfig, load_csv_configs, load_json_configs, select_config

_CACHE: dict[str, tuple[float, list[TunedConfig]]] = {}


def tuning_arch() -> str:
    explicit = os.environ.get("MUSA_AITER_TUNING_ARCH", "").strip()
    if explicit:
        return explicit
    archs = os.environ.get("MUSA_AITER_ARCHS", "mp31")
    first = archs.split(",")[0].strip()
    return first if first else "mp31"


def dtype_token(dt: Any) -> str:
    if dt is torch.float16:
        return "float16"
    if dt is torch.bfloat16:
        return "bfloat16"
    if dt is torch.float32:
        return "float32"
    if dt is torch.float64:
        return "float64"
    name = str(dt).lower()
    if name.startswith("torch."):
        name = name.replace("torch.", "")
    return name


def gemm_shape_key(a: torch.Tensor, b: torch.Tensor) -> str:
    m = int(a.shape[-2])
    k = int(a.shape[-1])
    n = int(b.shape[-1])
    return f"m={m},n={n},k={k}"


def _load_cached(path: Path) -> list[TunedConfig]:
    key = str(path.resolve())
    try:
        mtime = path.stat().st_mtime
    except OSError:
        return []
    hit = _CACHE.get(key)
    if hit is not None and hit[0] == mtime:
        return hit[1]
    suf = path.suffix.lower()
    if suf == ".csv":
        rows = load_csv_configs(path)
    elif suf == ".json":
        rows = load_json_configs(path)
    else:
        return []
    _CACHE[key] = (mtime, rows)
    return rows


def tuning_config_path_gemm() -> Path | None:
    raw = os.environ.get("MUSA_AITER_CONFIG_GEMM", "").strip()
    if not raw:
        return None
    p = Path(raw)
    return p if p.is_file() else None


def lookup_gemm_config(a: torch.Tensor, b: torch.Tensor) -> TunedConfig | None:
    """Return the selected tuning row for this GEMM contract, or ``None``."""
    path = tuning_config_path_gemm()
    if path is None:
        return None
    configs = _load_cached(path)
    if not configs:
        return None
    shape = gemm_shape_key(a, b)
    arch = tuning_arch()
    dt = dtype_token(a.dtype)
    return select_config(configs, op="gemm", arch=arch, dtype=dt, shape=shape)


def fused_moe_shape_key(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weight: torch.Tensor,
) -> str:
    """Stable shape token aligned with ``configs/model_configs`` MoE placeholders."""
    num_tokens, hidden_dim = hidden_states.shape
    num_experts, inter_x2, w1_hidden = w1.shape
    if w1_hidden != hidden_dim:
        raise ValueError("w1 last dim must match hidden_states")
    if inter_x2 % 2 != 0:
        raise ValueError("w1 inter dimension must be even (gate+up split)")
    inter_dim = inter_x2 // 2
    e2, h2, i2 = w2.shape
    if (e2, h2, i2) != (num_experts, hidden_dim, inter_dim):
        raise ValueError("w2 shape must be [experts, hidden, inter]")
    top_k = int(topk_weight.shape[-1])
    return (
        f"tokens={int(num_tokens)},hidden={int(hidden_dim)},inter={inter_dim},"
        f"experts={int(num_experts)},topk={top_k}"
    )


def tuning_config_path_moe() -> Path | None:
    raw = os.environ.get("MUSA_AITER_CONFIG_MOE", "").strip()
    if not raw:
        return None
    p = Path(raw)
    return p if p.is_file() else None


def lookup_fused_moe_config(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weight: torch.Tensor,
) -> TunedConfig | None:
    path = tuning_config_path_moe()
    if path is None:
        return None
    configs = _load_cached(path)
    if not configs:
        return None
    shape = fused_moe_shape_key(hidden_states, w1, w2, topk_weight)
    arch = tuning_arch()
    dt = dtype_token(hidden_states.dtype)
    return select_config(configs, op="fused_moe", arch=arch, dtype=dt, shape=shape)


def grouped_gemm_shape_key(a: torch.Tensor, b: torch.Tensor) -> str:
    """Shape token for grouped GEMM (``a @ b[g]`` per token group)."""
    if a.dim() != 2 or b.dim() != 3:
        raise ValueError("a must be 2D and b 3D for shape key")
    num_tokens, hidden = a.shape
    num_groups, k_dim, out_dim = b.shape
    if int(hidden) != int(k_dim):
        raise ValueError("a.shape[1] must equal b.shape[1] (K)")
    return (
        f"tokens={int(num_tokens)},hidden={int(hidden)},out={int(out_dim)},groups={int(num_groups)}"
    )


def tuning_config_path_grouped_gemm() -> Path | None:
    raw = os.environ.get("MUSA_AITER_CONFIG_GROUPED_GEMM", "").strip()
    if not raw:
        return None
    p = Path(raw)
    return p if p.is_file() else None


def lookup_grouped_gemm_config(a: torch.Tensor, b: torch.Tensor) -> TunedConfig | None:
    path = tuning_config_path_grouped_gemm()
    if path is None:
        return None
    configs = _load_cached(path)
    if not configs:
        return None
    shape = grouped_gemm_shape_key(a, b)
    arch = tuning_arch()
    dt = dtype_token(a.dtype)
    return select_config(configs, op="grouped_gemm", arch=arch, dtype=dt, shape=shape)


def tuning_dispatch_reason(base: str, row: TunedConfig | None) -> str:
    if row is None:
        return base
    lat = row.latency_us
    lat_s = f"{lat}" if lat is not None else "none"
    return f"{base}; tuned kernel={row.kernel} backend={row.backend} latency_us={lat_s}"


def gemm_dispatch_reason(base: str, row: TunedConfig | None) -> str:
    return tuning_dispatch_reason(base, row)


def grouped_gemm_dispatch_reason(base: str, row: TunedConfig | None) -> str:
    return tuning_dispatch_reason(base, row)
