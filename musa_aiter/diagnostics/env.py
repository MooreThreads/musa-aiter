from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from typing import Optional


@dataclass
class CheckItem:
    name: str
    ok: bool
    message: str
    detail: Optional[str] = None


@dataclass
class EnvironmentReport:
    python: CheckItem
    torch: CheckItem
    torch_musa: CheckItem
    musa_available: CheckItem
    mcc: CheckItem
    mthreads_gmi: CheckItem
    cache_dir: CheckItem

    @property
    def hard_ok(self) -> bool:
        # Phase 0: import/static tests may run without MUSA. Hard readiness for native execution
        # requires all checks except cache_dir to pass.
        return all(
            item.ok
            for item in [
                self.python,
                self.torch,
                self.torch_musa,
                self.musa_available,
                self.mcc,
                self.mthreads_gmi,
                self.cache_dir,
            ]
        )

    def to_dict(self) -> dict:
        data = asdict(self)
        data["hard_ok"] = self.hard_ok
        return data


def _run_command(cmd: list[str], timeout: int = 10) -> tuple[bool, str]:
    try:
        proc = subprocess.run(
            cmd,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=timeout,
        )
        return proc.returncode == 0, proc.stdout.strip()
    except Exception as exc:
        return False, repr(exc)


def _check_python() -> CheckItem:
    ok = sys.version_info >= (3, 9)
    return CheckItem(
        "python",
        ok,
        sys.version.split()[0],
        "requires >=3.9",
    )


def _check_torch() -> tuple[CheckItem, object | None]:
    try:
        import torch

        return CheckItem("torch", True, torch.__version__), torch
    except Exception as exc:
        return CheckItem("torch", False, "not importable", repr(exc)), None


def _check_torch_musa() -> CheckItem:
    try:
        import torch_musa  # noqa: F401

        version = getattr(torch_musa, "__version__", "unknown")
        return CheckItem("torch_musa", True, str(version))
    except Exception as exc:
        return CheckItem("torch_musa", False, "not importable", repr(exc))


def _check_musa_available(torch_mod: object | None) -> CheckItem:
    if torch_mod is None:
        return CheckItem("musa_available", False, "torch unavailable")
    try:
        musa = getattr(torch_mod, "musa", None)
        if musa is not None and hasattr(musa, "is_available"):
            ok = bool(musa.is_available())
            count = musa.device_count() if ok and hasattr(musa, "device_count") else 0
            return CheckItem("musa_available", ok, f"available={ok}, device_count={count}")
        # Some ROCm/CUDA-compatible stacks expose AMD-like devices as cuda. For MUSA AITER
        # we still prefer torch.musa and mark missing as not ready.
        return CheckItem("musa_available", False, "torch.musa is not available")
    except Exception as exc:
        return CheckItem("musa_available", False, "check failed", repr(exc))


def _check_executable(name: str, args: list[str]) -> CheckItem:
    path = shutil.which(name)
    if not path:
        return CheckItem(name, False, "not found in PATH")
    ok, output = _run_command([path, *args])
    return CheckItem(name, ok, path, output[:2000])


def _check_cache_dir() -> CheckItem:
    cache_dir = os.environ.get(
        "MUSA_AITER_CACHE_DIR",
        os.path.expanduser("~/.cache/musa_aiter"),
    )
    try:
        os.makedirs(cache_dir, exist_ok=True)
        test_path = os.path.join(cache_dir, ".write_test")
        with open(test_path, "w", encoding="utf-8") as f:
            f.write("ok")
        os.remove(test_path)
        return CheckItem("cache_dir", True, cache_dir)
    except Exception as exc:
        return CheckItem("cache_dir", False, cache_dir, repr(exc))


def collect_environment() -> EnvironmentReport:
    torch_item, torch_mod = _check_torch()
    return EnvironmentReport(
        python=_check_python(),
        torch=torch_item,
        torch_musa=_check_torch_musa(),
        musa_available=_check_musa_available(torch_mod),
        mcc=_check_executable("mcc", ["--version"]),
        mthreads_gmi=_check_executable("mthreads-gmi", []),
        cache_dir=_check_cache_dir(),
    )


def format_report(report: EnvironmentReport, json_output: bool = False) -> str:
    if json_output:
        return json.dumps(report.to_dict(), indent=2, sort_keys=True)
    lines = ["MUSA AITER environment check", "=" * 34]
    for item in [
        report.python,
        report.torch,
        report.torch_musa,
        report.musa_available,
        report.mcc,
        report.mthreads_gmi,
        report.cache_dir,
    ]:
        status = "OK" if item.ok else "WARN"
        lines.append(f"[{status}] {item.name}: {item.message}")
        if item.detail:
            detail = item.detail.replace("\n", "\n    ")
            lines.append(f"    {detail}")
    lines.append(f"hard_ready={report.hard_ok}")
    return "\n".join(lines)


def relevant_env() -> dict[str, str]:
    keys = [
        "MUSA_AITER_ARCHS",
        "MUSA_AITER_PREBUILD_KERNELS",
        "MUSA_AITER_REBUILD",
        "MUSA_AITER_CACHE_DIR",
        "MUSA_AITER_LOG_LEVEL",
        "MUSA_AITER_TRACE_DISPATCH",
        "MUSA_AITER_CONFIG_GEMM",
        "MUSA_AITER_CONFIG_MOE",
        "MUSA_AITER_CONFIG_GROUPED_GEMM",
        "MUSA_AITER_TUNING_ARCH",
        "MUSA_AITER_VENDOR_BACKEND",
        "MUSA_AITER_VENDOR_LIB",
        "MUSA_AITER_FALLBACK_POLICY",
        "MUSA_VISIBLE_DEVICES",
        "MUSA_HOME",
        "MUSA_PATH",
    ]
    return {key: os.environ.get(key, "") for key in keys}
