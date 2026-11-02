from __future__ import annotations

from types import SimpleNamespace

import torch

from musa_aiter.integration.kv_connector_smoke import (
    ConnectorSmokeUnavailable,
    run_connector_execution_smoke,
)


def test_connector_smoke_requires_real_connector_import(monkeypatch):
    monkeypatch.setattr(
        "musa_aiter.integration.kv_connector_smoke._import_connector_module",
        lambda _: (_ for _ in ()).throw(ConnectorSmokeUnavailable("missing")),
    )
    try:
        run_connector_execution_smoke(
            "lmcache",
            source=torch.ones(2),
            save=lambda _: None,
            load=lambda: torch.ones(2),
            native_probe=lambda: True,
        )
    except ConnectorSmokeUnavailable as exc:
        assert "missing" in str(exc)
    else:
        raise AssertionError("connector smoke must require an imported real connector")


def test_connector_smoke_executes_save_load_and_native_probe(monkeypatch):
    calls = []
    monkeypatch.setattr(
        "musa_aiter.integration.kv_connector_smoke._import_connector_module",
        lambda _: SimpleNamespace(__name__="real.connector"),
    )
    source = torch.arange(4, dtype=torch.float32)
    restored = source.clone()

    result = run_connector_execution_smoke(
        "vllm",
        source=source,
        save=lambda tensor: calls.append(("save", tensor.clone())),
        load=lambda: restored,
        native_probe=lambda: True,
    )

    assert result.connector == "vllm"
    assert result.module == "real.connector"
    assert result.native is True
    assert calls[0][0] == "save"
