from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from musa_aiter.registry import list_capabilities
from musa_aiter.version import __version__


@dataclass(frozen=True)
class AotPlan:
    project: str
    version: str
    archs: list[str]
    prebuild_level: int
    operators: list[str]
    output_dir: str
    native_build_enabled: bool

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True)


def create_plan(
    *,
    archs: list[str],
    prebuild_level: int,
    output_dir: str | Path,
    operators: list[str] | None = None,
    native_build_enabled: bool = False,
) -> AotPlan:
    if operators is None:
        operators = [cap.name for cap in list_capabilities()]
    if not archs:
        archs = ["native"]
    return AotPlan(
        project="musa-aiter",
        version=__version__,
        archs=archs,
        prebuild_level=prebuild_level,
        operators=sorted(operators),
        output_dir=str(output_dir),
        native_build_enabled=native_build_enabled,
    )


def write_plan(plan: AotPlan) -> Path:
    out_dir = Path(plan.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "musa_aiter_aot_plan.json"
    path.write_text(plan.to_json(), encoding="utf-8")
    return path
