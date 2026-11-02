"""Execution smoke helpers for real framework KV connectors.

The harness deliberately receives bound connector callbacks from the caller.
This keeps framework lifecycle/configuration ownership in SGLang, LMCache, and
vLLM while making parity and native-dispatch assertions reusable.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Callable

import torch


class ConnectorSmokeUnavailable(RuntimeError):
    """Raised when a real framework connector is not installed or importable."""


@dataclass(frozen=True)
class ConnectorSmokeResult:
    connector: str
    module: str
    native: bool


_CONNECTOR_MODULE_CANDIDATES: dict[str, tuple[str, ...]] = {
    "sglang": (
        "sglang.srt.mem_cache.memory_pool_host",
        "sgl_kernel.kvcacheio",
    ),
    "lmcache": ("lmcache.v1.gpu_connector",),
    "vllm": (
        "lmcache.integration.vllm.vllm_v1_adapter",
        "vllm.distributed.kv_transfer.kv_connector.v1.lmcache_connector",
    ),
}


def _import_connector_module(connector: str):
    try:
        candidates = _CONNECTOR_MODULE_CANDIDATES[connector]
    except KeyError as exc:
        raise ValueError(f"unsupported connector={connector!r}") from exc
    errors: list[str] = []
    for module_name in candidates:
        try:
            return importlib.import_module(module_name)
        except (ImportError, ModuleNotFoundError) as exc:
            errors.append(f"{module_name}: {exc}")
    raise ConnectorSmokeUnavailable(
        f"{connector} connector is not installed/importable; tried " + "; ".join(errors)
    )


def connector_import_status() -> dict[str, str]:
    status: dict[str, str] = {}
    for connector in _CONNECTOR_MODULE_CANDIDATES:
        try:
            module = _import_connector_module(connector)
        except ConnectorSmokeUnavailable as exc:
            status[connector] = f"unavailable: {exc}"
        else:
            status[connector] = f"imported: {module.__name__}"
    return status


def run_connector_execution_smoke(
    connector: str,
    *,
    source: torch.Tensor,
    save: Callable[[torch.Tensor], object],
    load: Callable[[], torch.Tensor],
    native_probe: Callable[[], bool],
) -> ConnectorSmokeResult:
    """Execute a save/load pair owned by an imported real connector.

    ``save`` and ``load`` must be bound to the framework connector under test;
    they are not AITER reference callbacks. ``native_probe`` must be backed by
    a dispatch trace or native-required assertion from that connector route.
    """

    module = _import_connector_module(connector)
    if not isinstance(source, torch.Tensor):
        raise TypeError("connector smoke source must be a torch.Tensor")
    save(source)
    restored = load()
    if not isinstance(restored, torch.Tensor):
        raise TypeError("connector smoke load callback must return a torch.Tensor")
    torch.testing.assert_close(restored, source)
    native = bool(native_probe())
    if not native:
        raise AssertionError(f"{connector} connector execution did not hit a native route")
    return ConnectorSmokeResult(connector, module.__name__, native)


__all__ = [
    "ConnectorSmokeResult",
    "ConnectorSmokeUnavailable",
    "connector_import_status",
    "run_connector_execution_smoke",
]
