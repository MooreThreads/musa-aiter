from __future__ import annotations

import importlib
import os

import pytest
import torch

from musa_aiter.integration.kv_connector_smoke import (
    ConnectorSmokeUnavailable,
    connector_import_status,
    run_connector_execution_smoke,
)


def _require_connector_bridge(connector: str):
    status = connector_import_status()
    if status[connector].startswith("unavailable:"):
        pytest.skip(status[connector])
    bridge_name = os.environ.get("MUSA_AITER_CONNECTOR_SMOKE_MODULE")
    if not bridge_name:
        pytest.skip(
            "set MUSA_AITER_CONNECTOR_SMOKE_MODULE to a lab bridge that binds "
            "the real connector save/load callbacks"
        )
    bridge = importlib.import_module(bridge_name)
    factory = getattr(bridge, "build_connector_smoke_callbacks", None)
    if not callable(factory):
        pytest.fail(
            f"{bridge_name} must export build_connector_smoke_callbacks(connector)"
        )
    return factory(connector)


@pytest.mark.musa
@pytest.mark.native
@pytest.mark.parametrize("connector", ["sglang", "lmcache", "vllm"])
def test_real_connector_execution_hits_native(connector: str) -> None:
    if not hasattr(torch, "musa") or not torch.musa.is_available():
        pytest.skip("TorchMUSA with a visible MUSA device is required")
    source, save, load, native_probe = _require_connector_bridge(connector)
    try:
        result = run_connector_execution_smoke(
            connector,
            source=source,
            save=save,
            load=load,
            native_probe=native_probe,
        )
    except ConnectorSmokeUnavailable as exc:
        pytest.skip(str(exc))
    assert result.native is True
