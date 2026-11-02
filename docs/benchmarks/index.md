# Benchmarks

Micro-benchmarks live under the repository `benchmarks/` directory. They use the shared harness in `musa_aiter.benchmarking` and default to **CPU** reference execution unless you pass a different `--device`.

## Running locally

```bash
cd /path/to/musa-aiter
PYTHONPATH=. python3 benchmarks/bench_add.py --n 10000 --iters 3
PYTHONPATH=. python3 benchmarks/bench_phase2.py --op gemm --iters 20 --json
PYTHONPATH=. python3 benchmarks/bench_phase3.py --op gemm_all_reduce --iters 20 --json
```

## Generate a snapshot

Generate a fresh benchmark snapshot locally when comparing code or environments:

```bash
PYTHONPATH=. python3 scripts/generate_bench_dashboard.py
```

Optional flags:

```text
--iters N     Iterations per op (default: 8)
--device cpu  Device string passed through to benchmarks
```
