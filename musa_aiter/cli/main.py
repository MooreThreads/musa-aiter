from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from musa_aiter import __version__
from musa_aiter.aot import create_plan, write_plan
from musa_aiter.benchmarking import benchmark_callable
from musa_aiter.diagnostics.benchmark_presets import benchmark_ops, make_benchmark_workload
from musa_aiter.diagnostics.env import collect_environment, format_report, relevant_env
from musa_aiter.ops.smoke import (
    build_native_add,
    build_native_mul,
    native_smoke_abi_version,
    native_smoke_mul_abi_version,
)
from musa_aiter.ops.smoke_reduce import (
    build_native_reduce_sum,
    native_smoke_reduce_abi_version,
)
from musa_aiter.ops.smoke_softmax import (
    build_native_softmax_last_dim,
    native_smoke_softmax_abi_version,
)
from musa_aiter.ops.smoke_rmsnorm import (
    build_native_rmsnorm,
    native_smoke_rmsnorm_abi_version,
)
from musa_aiter.ops.kv_transfer import (
    build_native_kv_transfer,
    native_kv_transfer_abi_version,
)
from musa_aiter.registry import as_dict, list_capabilities
from musa_aiter.tuning import load_csv_configs, load_json_configs, select_config
from musa_aiter.vendors import VENDOR_DESCRIPTIONS


def cmd_check(args: argparse.Namespace) -> int:
    report = collect_environment()
    print(format_report(report, json_output=args.json))
    return 0 if (report.hard_ok or args.soft) else 1


def cmd_env(args: argparse.Namespace) -> int:
    env = relevant_env()
    if args.json:
        print(json.dumps(env, indent=2, sort_keys=True))
    else:
        for key, value in env.items():
            print(f"{key}={value}")
    return 0


def cmd_show_config(args: argparse.Namespace) -> int:
    data = {
        "version": __version__,
        "environment": collect_environment().to_dict(),
        "env": relevant_env(),
        "operators": as_dict(),
    }
    if args.json:
        print(json.dumps(data, indent=2, sort_keys=True))
    else:
        print(f"MUSA AITER version: {__version__}")
        print(format_report(collect_environment()))
    return 0


def cmd_list_ops(args: argparse.Namespace) -> int:
    caps = list_capabilities(family=args.family)
    if args.json:
        print(json.dumps(as_dict(), indent=2, sort_keys=True))
        return 0
    for cap in caps:
        vb = ",".join(cap.vendor_backends) if cap.vendor_backends else ""
        extra = f"\tvendor_targets={vb}" if vb else ""
        print(f"{cap.name}\t{cap.family}\t{cap.status}\tbackends={','.join(cap.backends)}{extra}")
    return 0


def cmd_list_vendors(args: argparse.Namespace) -> int:
    if args.json:
        print(json.dumps(VENDOR_DESCRIPTIONS, indent=2, sort_keys=True))
        return 0
    for vid in sorted(VENDOR_DESCRIPTIONS):
        print(f"{vid}\t{VENDOR_DESCRIPTIONS[vid]}")
    return 0


def cmd_build_smoke(args: argparse.Namespace) -> int:
    result_add = build_native_add()
    result_mul = build_native_mul()
    result_reduce = build_native_reduce_sum()
    result_softmax = build_native_softmax_last_dim()
    result_rmsnorm = build_native_rmsnorm()
    result_kv_transfer = build_native_kv_transfer()
    ok_all = bool(
        result_add.ok
        and result_mul.ok
        and result_reduce.ok
        and result_softmax.ok
        and result_rmsnorm.ok
        and result_kv_transfer.ok,
    )
    payload = {
        "ok": ok_all,
        "library_path": str(result_add.library_path) if result_add.library_path else None,
        "build_dir": str(result_add.build_dir),
        "message": (
            "build ok"
            if ok_all
            else f"add={'ok' if result_add.ok else result_add.message}; "
            f"mul={'ok' if result_mul.ok else result_mul.message}; "
            f"reduce_sum={'ok' if result_reduce.ok else result_reduce.message}; "
            f"softmax_ld={'ok' if result_softmax.ok else result_softmax.message}; "
            f"rmsnorm={'ok' if result_rmsnorm.ok else result_rmsnorm.message}; "
            f"kv_transfer={'ok' if result_kv_transfer.ok else result_kv_transfer.message}"
        ),
        "abi_version": native_smoke_abi_version() if result_add.ok else None,
        "abi_version_mul": native_smoke_mul_abi_version() if result_mul.ok else None,
        "abi_version_reduce": native_smoke_reduce_abi_version() if result_reduce.ok else None,
        "abi_version_softmax_ld": native_smoke_softmax_abi_version()
        if result_softmax.ok
        else None,
        "abi_version_rmsnorm": native_smoke_rmsnorm_abi_version()
        if result_rmsnorm.ok
        else None,
        "abi_version_kv_transfer": native_kv_transfer_abi_version()
        if result_kv_transfer.ok
        else None,
        "add_ok": result_add.ok,
        "mul_ok": result_mul.ok,
        "reduce_ok": result_reduce.ok,
        "softmax_ld_ok": result_softmax.ok,
        "rmsnorm_ok": result_rmsnorm.ok,
        "kv_transfer_ok": result_kv_transfer.ok,
        "library_path_mul": str(result_mul.library_path) if result_mul.library_path else None,
        "library_path_reduce": str(result_reduce.library_path) if result_reduce.library_path else None,
        "library_path_softmax_ld": str(result_softmax.library_path)
        if result_softmax.library_path
        else None,
        "library_path_rmsnorm": str(result_rmsnorm.library_path)
        if result_rmsnorm.library_path
        else None,
        "library_path_kv_transfer": str(result_kv_transfer.library_path)
        if result_kv_transfer.library_path
        else None,
        "build_dir_mul": str(result_mul.build_dir),
        "build_dir_reduce": str(result_reduce.build_dir),
        "build_dir_softmax_ld": str(result_softmax.build_dir),
        "build_dir_rmsnorm": str(result_rmsnorm.build_dir),
        "build_dir_kv_transfer": str(result_kv_transfer.build_dir),
    }
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(f"ok={ok_all}")
        print(
            "add_ok="
            f"{result_add.ok} mul_ok={result_mul.ok} reduce_ok={result_reduce.ok}"
            f" softmax_ld_ok={result_softmax.ok} rmsnorm_ok={result_rmsnorm.ok}"
            f" kv_transfer_ok={result_kv_transfer.ok}",
        )
        print(f"message={payload['message']}")
        print(f"build_dir={result_add.build_dir}")
        print(f"build_dir_mul={result_mul.build_dir}")
        print(f"build_dir_reduce={result_reduce.build_dir}")
        print(f"build_dir_softmax_ld={result_softmax.build_dir}")
        print(f"build_dir_rmsnorm={result_rmsnorm.build_dir}")
        print(f"build_dir_kv_transfer={result_kv_transfer.build_dir}")
        print(f"abi_version={payload['abi_version']}")
        print(f"abi_version_mul={payload['abi_version_mul']}")
        print(f"abi_version_reduce={payload['abi_version_reduce']}")
        print(f"abi_version_softmax_ld={payload['abi_version_softmax_ld']}")
        print(f"abi_version_rmsnorm={payload['abi_version_rmsnorm']}")
        print(f"abi_version_kv_transfer={payload['abi_version_kv_transfer']}")
        if result_add.library_path:
            print(f"library_path={result_add.library_path}")
        if result_mul.library_path:
            print(f"library_path_mul={result_mul.library_path}")
        if result_reduce.library_path:
            print(f"library_path_reduce={result_reduce.library_path}")
        if result_softmax.library_path:
            print(f"library_path_softmax_ld={result_softmax.library_path}")
        if result_rmsnorm.library_path:
            print(f"library_path_rmsnorm={result_rmsnorm.library_path}")
        if result_kv_transfer.library_path:
            print(f"library_path_kv_transfer={result_kv_transfer.library_path}")
    return 0 if ok_all else 1


def cmd_trace(args: argparse.Namespace) -> int:
    """Run a subprocess with MUSA_AITER_TRACE_DISPATCH=1 (see system design §9.5)."""
    cmd = list(args.cmd)
    while cmd and cmd[0] == "--":
        cmd = cmd[1:]
    if not cmd:
        print(
            "musa-aiter trace: missing command\n"
            "Example: musa-aiter trace -- python -c \"import musa_aiter; musa_aiter.rmsnorm\"",
            file=sys.stderr,
        )
        return 2
    exe = cmd[0]
    if not os.path.isabs(exe) and os.sep not in exe:
        found = shutil.which(exe)
        if found:
            cmd[0] = found
    env = os.environ.copy()
    env["MUSA_AITER_TRACE_DISPATCH"] = "1"
    try:
        proc = subprocess.run(cmd, env=env)
    except FileNotFoundError:
        print(f"musa-aiter trace: command not found: {exe}", file=sys.stderr)
        return 127
    return int(proc.returncode)


def cmd_plan_aot(args: argparse.Namespace) -> int:
    archs = [item.strip() for item in args.archs.split(",") if item.strip()]
    operators = None if not args.ops else [item.strip() for item in args.ops.split(",") if item.strip()]
    plan = create_plan(
        archs=archs,
        prebuild_level=args.prebuild_level,
        output_dir=args.output_dir,
        operators=operators,
        native_build_enabled=args.native_build,
    )
    path = write_plan(plan)
    if args.json:
        payload = plan.to_dict()
        payload["plan_path"] = str(path)
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(f"AOT plan written: {path}")
        print(plan.to_json())
    return 0


def cmd_benchmark(args: argparse.Namespace) -> int:
    """§9.5 micro-benchmark entry point."""
    try:
        fn, dtype = make_benchmark_workload(args.op, args.device)
    except ValueError as exc:
        print(f"musa-aiter benchmark: {exc}", file=sys.stderr)
        return 2
    result = benchmark_callable(
        args.op,
        fn,
        iterations=args.iters,
        warmup=args.warmup,
        device=args.device,
        dtype=dtype,
    )
    if args.json:
        print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
    else:
        print(result)
    return 0


def _load_tune_file(path: Path) -> list:
    suf = path.suffix.lower()
    if suf == ".csv":
        return load_csv_configs(path)
    if suf == ".json":
        return load_json_configs(path)
    raise ValueError(f"unsupported file type {suf!r} (use .csv or .json)")


def cmd_tune(args: argparse.Namespace) -> int:
    """Validate tuning tables or select one row (``tuning.py``)."""
    cfg_path = Path(args.config)
    if not cfg_path.is_file():
        print(f"musa-aiter tune: file not found: {cfg_path}", file=sys.stderr)
        return 2
    if args.validate and args.select:
        print("musa-aiter tune: use only one of --validate or --select", file=sys.stderr)
        return 2
    if not args.validate and not args.select:
        print("musa-aiter tune: specify --validate or --select", file=sys.stderr)
        return 2

    if args.validate:
        try:
            configs = _load_tune_file(cfg_path)
        except ValueError as exc:
            print(f"musa-aiter tune: {exc}", file=sys.stderr)
            return 2
        if args.json:
            print(json.dumps({"path": str(cfg_path), "count": len(configs)}, indent=2, sort_keys=True))
        else:
            print(f"ok: {len(configs)} config row(s) in {cfg_path}")
        return 0

    try:
        configs = _load_tune_file(cfg_path)
    except ValueError as exc:
        print(f"musa-aiter tune: {exc}", file=sys.stderr)
        return 2
    required = (args.op, args.arch, args.dtype, args.shape)
    if not all(required):
        print("musa-aiter tune --select requires --op --arch --dtype --shape", file=sys.stderr)
        return 2
    sel = select_config(
        configs,
        op=args.op,
        arch=args.arch,
        dtype=args.dtype,
        shape=args.shape,
    )
    if sel is None:
        print(
            f"musa-aiter tune: no row for op={args.op!r} arch={args.arch!r} dtype={args.dtype!r} shape={args.shape!r}",
            file=sys.stderr,
        )
        return 1
    if args.json:
        print(json.dumps(sel.to_dict(), indent=2, sort_keys=True))
    else:
        lat = sel.latency_us
        print(f"kernel={sel.kernel} backend={sel.backend} latency_us={lat}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="musa-aiter", description="MUSA AITER CLI")
    parser.add_argument("--version", action="version", version=f"musa-aiter {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("check", help="Check MUSA AITER runtime environment")
    p.add_argument("--json", action="store_true")
    p.add_argument("--soft", action="store_true", help="return 0 even when native MUSA readiness checks warn")
    p.set_defaults(func=cmd_check)

    p = sub.add_parser("env", help="Show MUSA AITER-related environment variables")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_env)

    p = sub.add_parser("show-config", help="Show package and environment configuration")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_show_config)

    p = sub.add_parser("list-ops", help="List registered operators and capabilities")
    p.add_argument("--json", action="store_true")
    p.add_argument("--family", default=None)
    p.set_defaults(func=cmd_list_ops)

    p = sub.add_parser("list-vendors", help="List vendor backend identifiers (Phase P7 catalog)")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_list_vendors)

    p = sub.add_parser("build-smoke", help="JIT build smoke kernels (add, mul_float, reduction/sum)")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_build_smoke)

    p = sub.add_parser(
        "trace",
        help="Run a command with MUSA_AITER_TRACE_DISPATCH=1 (dispatch log to stderr)",
    )
    p.add_argument(
        "cmd",
        nargs=argparse.REMAINDER,
        metavar="COMMAND",
        help="Place -- before the program, e.g. musa-aiter trace -- python script.py",
    )
    p.set_defaults(func=cmd_trace)

    p = sub.add_parser("plan-aot", help="Create an AOT/prebuild plan manifest")
    p.add_argument("--json", action="store_true")
    p.add_argument("--archs", default="native", help="comma-separated arch list, e.g. mp31,native")
    p.add_argument("--prebuild-level", type=int, default=0)
    p.add_argument("--ops", default="", help="comma-separated op names; default all registered ops")
    p.add_argument("--output-dir", default="build/aot-plan")
    p.add_argument("--native-build", action="store_true", help="mark native build enabled in plan; does not compile yet")
    p.set_defaults(func=cmd_plan_aot)

    p = sub.add_parser("benchmark", help="Operator micro-benchmark with fixed presets (design §9.5)")
    p.add_argument("--op", required=True, choices=sorted(benchmark_ops()))
    p.add_argument("--device", default="cpu")
    p.add_argument("--iters", type=int, default=20)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_benchmark)

    p = sub.add_parser("tune", help="Validate tuning CSV/JSON or select a row (see tuning.py)")
    p.add_argument("--config", required=True, metavar="PATH", help="Tuning CSV or JSON file")
    p.add_argument("--validate", action="store_true", help="Load file and report row count")
    p.add_argument("--select", action="store_true", help="Select row for op/arch/dtype/shape")
    p.add_argument("--op", default=None, help="with --select: operator name")
    p.add_argument("--arch", default=None)
    p.add_argument("--dtype", default=None)
    p.add_argument("--shape", default=None, help="exact shape string as stored in the table")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_tune)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
