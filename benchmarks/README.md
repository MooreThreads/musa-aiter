# Benchmark scripts

Standalone timing scripts using `torch` and `musa_aiter`. They complement the CLI benchmark:

```bash
PYTHONPATH=. python3 -m musa_aiter.cli.main benchmark --op gemm --device cpu --iters 10 --json
```

## Scripts

| File | Purpose |
| --- | --- |
| **`bench_smoke.py`** | Elementwise/reduction/smoke operators: **`add`**, **`multiply`**, **`reduce_sum`**, **`softmax_last_dim`**, **`rmsnorm`**. Use **`--prefer-native`** on MUSA with **`export MUSA_AITER_NATIVE_SMOKE=1`** to time ctypes JIT paths. |
| **`bench_add.py`** | Legacy focused harness for **`musa_aiter.add`** with **`--n`** vector length (reference path by default). |
| **`bench_phase2.py`** | **`gemm`**, **`grouped_gemm`**, **`fused_moe`** presets (`--device`, `--json`). |
| **`bench_phase3.py`** | **`gemm_all_reduce`** and **`rmsnorm_backward`** presets (`--device`, `--json`). |

## Typical usage

```bash
cd /path/to/musa-aiter
pip install -e .

# Smoke suite (CPU reference timings)
PYTHONPATH=. python3 benchmarks/bench_smoke.py --op softmax_last_dim --rows 128 --cols 128 --device cpu --iters 30

# MUSA native smoke timings (TorchMUSA + mcc-built `.so`)
export MUSA_AITER_NATIVE_SMOKE=1
PYTHONPATH=. python3 benchmarks/bench_smoke.py --op rmsnorm --device musa --prefer-native --iters 20
```

The optional snapshot generator is **`scripts/generate_bench_dashboard.py`**.

See also **`docs/benchmarks/`** after running the dashboard generator.
