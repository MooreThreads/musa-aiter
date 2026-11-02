from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from musa_aiter._paths import project_data_path


@dataclass(frozen=True)
class TunedConfig:
    op: str
    arch: str
    dtype: str
    shape: str
    backend: str
    kernel: str
    latency_us: float | None = None
    extra: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_json_configs(path: str | Path) -> list[TunedConfig]:
    data = json.loads(Path(path).read_text())
    if isinstance(data, dict):
        data = data.get("configs", [])
    return [TunedConfig(**item) for item in data]


def write_json_configs(configs: list[TunedConfig], path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"configs": [c.to_dict() for c in configs]}, indent=2, sort_keys=True))
    return p


def load_csv_configs(path: str | Path) -> list[TunedConfig]:
    rows = []
    with Path(path).open(newline="") as f:
        for row in csv.DictReader(f):
            rows.append(
                TunedConfig(
                    op=row["op"],
                    arch=row["arch"],
                    dtype=row["dtype"],
                    shape=row["shape"],
                    backend=row["backend"],
                    kernel=row["kernel"],
                    latency_us=float(row["latency_us"])
                    if (row.get("latency_us") or "").strip()
                    else None,
                    extra={},
                )
            )
    return rows


def bundled_example_tuned_csv() -> Path | None:
    """Resolved path to ``configs/tuned/example_mp31.csv`` next to the repo root (editable installs)."""
    p = project_data_path("configs", "tuned", "example_mp31.csv")
    return p if p.is_file() else None


def load_bundled_example_tuned_configs() -> list[TunedConfig]:
    """Load packaged example CSV when present (Phase 5 tuning pipeline hook)."""
    p = bundled_example_tuned_csv()
    if p is None:
        return []
    return load_csv_configs(p)


def select_config(
    configs: list[TunedConfig],
    *,
    op: str,
    arch: str,
    dtype: str,
    shape: str,
) -> TunedConfig | None:
    exact = [c for c in configs if c.op == op and c.arch == arch and c.dtype == dtype and c.shape == shape]
    if not exact:
        return None
    return min(exact, key=lambda c: float("inf") if c.latency_us is None else c.latency_us)
