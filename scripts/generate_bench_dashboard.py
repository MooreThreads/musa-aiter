#!/usr/bin/env python3
"""Run CPU micro-benchmarks and refresh docs/benchmarks/dashboard.md + latest.json."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def _run_bench(repo: Path, script: str, op: str, *, iters: int, device: str) -> dict:
    cmd = [
        sys.executable,
        str(repo / "benchmarks" / script),
        "--op",
        op,
        "--iters",
        str(iters),
        "--device",
        device,
        "--json",
    ]
    proc = subprocess.run(
        cmd,
        cwd=repo,
        env={**os.environ, "PYTHONPATH": str(repo)},
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"Benchmark failed ({script} {op}): {proc.stderr or proc.stdout}"
        )
    return json.loads(proc.stdout.strip())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iters", type=int, default=8, help="Iterations per benchmark")
    parser.add_argument("--device", default="cpu", help="torch device string")
    args = parser.parse_args()

    repo = Path(__file__).resolve().parents[1]
    out_dir = repo / "docs" / "benchmarks"
    out_dir.mkdir(parents=True, exist_ok=True)

    phase2_ops = ["gemm", "grouped_gemm", "fused_moe"]
    phase3_ops = [
        "gemm",
        "grouped_gemm",
        "fused_moe",
        "gemm_all_reduce",
        "rmsnorm_backward",
    ]

    rows: list[dict] = []
    for op in phase2_ops:
        rows.append(
            {
                "suite": "bench_phase2.py",
                "op": op,
                **_run_bench(repo, "bench_phase2.py", op, iters=args.iters, device=args.device),
            }
        )
    for op in phase3_ops:
        rows.append(
            {
                "suite": "bench_phase3.py",
                "op": op,
                **_run_bench(repo, "bench_phase3.py", op, iters=args.iters, device=args.device),
            }
        )

    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "suites_note": (
            "`bench_phase2.py`/`bench_phase3.py` above; also see `benchmarks/README.md` and "
            "`benchmarks/bench_smoke.py` for JIT smoke timings (run separately)."
        ),
        "device": args.device,
        "iterations": args.iters,
        "rows": rows,
    }
    (out_dir / "latest.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    lines = [
        "# Benchmark dashboard",
        "",
        f"_Generated at (UTC): `{payload['generated_at_utc']}` — device `{args.device}`, "
        f"iterations `{args.iters}`._",
        "",
        "Regenerate:",
        "",
        "```bash",
        "PYTHONPATH=. python3 scripts/generate_bench_dashboard.py",
        "```",
        "",
        "_" + payload["suites_note"] + "_",
        "",
        "| Suite | Operator | Dtype | Avg ms | Min ms | Max ms |",
        "| --- | --- | --- | ---: | ---: | ---: |",
    ]
    for r in rows:
        lines.append(
            f"| `{r['suite']}` | `{r['op']}` | `{r['dtype']}` | "
            f"{r['avg_ms']:.4f} | {r['min_ms']:.4f} | {r['max_ms']:.4f} |"
        )
    lines.append("")
    (out_dir / "dashboard.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
