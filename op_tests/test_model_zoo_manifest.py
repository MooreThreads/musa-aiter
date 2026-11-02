"""Phase P6 model zoo manifest sanity checks."""

from __future__ import annotations

import json
from pathlib import Path


def test_model_zoo_manifest_parses():
    root = Path(__file__).resolve().parents[1]
    path = root / "configs" / "model_zoo" / "manifest.v1.json"
    data = json.loads(path.read_text())
    assert data.get("schema") == "musa_aiter.model_zoo.v1"
    assert data.get("version")
    assert isinstance(data.get("models"), list)
    assert len(data["models"]) >= 1
    for m in data["models"]:
        assert m.get("id")
        assert "hidden_size" in m
