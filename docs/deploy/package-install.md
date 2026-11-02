# Package and install guide

This guide covers how to build, verify, and install the `musa-aiter` Python package from this repository.

Use the CPU/dev path for local development and CI-style checks. Use the MUSA lab path only on a machine or container with a working Moore Threads MUSA stack.

## 1. Prerequisites

### 1.1 Common prerequisites

Required on every machine:

- Python **3.9 or newer**.
- `pip`, `setuptools`, `wheel`, and `build`.
- A PyTorch build installed before importing `musa_aiter`.
- A clean checkout of this repository.

Recommended packaging tools:

```bash
python3 -m pip install -U pip
python3 -m pip install "setuptools>=77" wheel build
```

`musa-aiter` keeps PyTorch out of `pyproject.toml` dependencies so framework and vendor stacks can pin their own compatible Torch/TorchMUSA builds. Install the correct PyTorch stack for your target environment first. For MUSA, do not use a generic `pip install torch` command from the public PyPI index.

For a CPU/reference install from PyPI, use the convenience extra:

```bash
python3 -m pip install "musa-aiter[cpu]"
```

For MUSA, install the vendor-matched `torch` and `torch_musa` stack first, then install the base `musa-aiter` package without the CPU extra and without dependency resolution:

```bash
python3 -m pip install --no-deps musa-aiter
python3 -c "import torch, torch_musa; print(torch.__version__); print(torch.musa.is_available())"
```

The `--no-deps` flag is intentional. It protects the already-installed vendor stack from replacement or downgrade. Do not use `musa-aiter[cpu]` on a MUSA host.

### 1.2 CPU/dev prerequisite stack

For a CPU-only development machine, install CPU PyTorch and dev tools:

```bash
python3 -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python3 -m pip install -e ".[dev,docs]"
```

Expected limitations on CPU/dev hosts:

- `torch_musa` may be absent.
- `torch.musa` may be unavailable.
- `mcc` and `mthreads-gmi` may be absent.
- MUSA-marked tests should skip.
- Native compile-symbol tests require `mcc`; run them only on a MUSA toolkit host.

### 1.3 MUSA lab prerequisite stack

Required for device and native validation:

- Visible MUSA GPU; `mthreads-gmi` should report the expected device count.
- MUSA Toolkit with `mcc` on `PATH`.
- PyTorch and `torch_musa` versions matched to the installed driver/toolkit.
- `python3 -c "import torch, torch_musa; print(torch.musa.is_available())"` prints `True`.

Recommended sanity checks:

```bash
mthreads-gmi
mcc --version
python3 -c "import torch; import torch_musa; print(torch.__version__); print(torch.musa.is_available()); print(torch.musa.device_count())"
```

## 2. Build package artifacts

From the repository root:

```bash
python3 -m pip install -e ".[dev]"
python3 -m build --wheel --sdist
```

Expected output:

```text
dist/musa_aiter-<version>-py3-none-any.whl
dist/musa_aiter-<version>.tar.gz
```

Verify that the wheel and sdist include runtime data files and that the wheel imports from an isolated install target:

```bash
python3 scripts/verify_package_artifacts.py --dist-dir dist
```

The verifier checks:

- wheel metadata version,
- sdist contents,
- bundled `csrc/` native source files,
- bundled `configs/` files,
- isolated wheel install,
- `musa_aiter` import,
- registered capability count,
- installed native-source and tuning-config path resolution.

## 3. Install from the wheel

Create a fresh environment and install the wheel:

```bash
python3 -m venv /tmp/musa_aiter_env
source /tmp/musa_aiter_env/bin/activate
python3 -m pip install -U pip
python3 -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python3 -m pip install --no-deps dist/musa_aiter-*.whl
```

For a MUSA lab environment, install the lab-approved `torch` and `torch_musa` stack instead of CPU PyTorch first, then install the `musa-aiter` wheel with `--no-deps`:

```bash
python3 -m pip install --no-deps dist/musa_aiter-*.whl
```

Verify the installed package:

```bash
musa-aiter check --soft
musa-aiter list-ops
python3 -c "import musa_aiter; print(musa_aiter.__version__)"
```

On a healthy MUSA lab stack, run the hard check:

```bash
musa-aiter check
python3 scripts/validate_native_musa_env.py --strict
```

## 4. Install from source for development

Use editable install when changing code:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -U pip
python3 -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python3 -m pip install -e ".[dev,docs]"
```

Run the CPU-safe checks:

```bash
python3 -m pytest op_tests -q -k "not test_build_smoke and not lmcache_native_symbols_are_in_kv_transfer_shared_object"
PYTHONPATH=. python3 -m musa_aiter.cli.main check --soft
PYTHONPATH=. python3 -m musa_aiter.cli.main list-ops --json
mkdocs build --strict
```

If `mcc` is available and native smoke builds are expected to work, run the full suite or the native build checks explicitly:

```bash
python3 -m pytest op_tests -q
PYTHONPATH=. MUSA_AITER_REBUILD=1 python3 -m musa_aiter.cli.main build-smoke --json
```

## 5. MUSA lab install and validation

On a real MUSA lab host or container:

```bash
python3 -m venv ~/venvs/musa-aiter-lab
source ~/venvs/musa-aiter-lab/bin/activate
python3 -m pip install -U pip wheel
# Install the lab-approved torch + torch_musa stack here.
python3 -m pip install --no-deps dist/musa_aiter-*.whl
```

Validate the stack:

```bash
python3 scripts/validate_native_musa_env.py --strict
PYTHONPATH=. python3 -m musa_aiter.cli.main check
python3 -m pytest op_tests -q -m musa
```

Validate native smoke paths when `mcc` and TorchMUSA are available:

```bash
PYTHONPATH=. MUSA_AITER_REBUILD=1 python3 -m musa_aiter.cli.main build-smoke --json
MUSA_AITER_NATIVE_SMOKE=1 python3 -m pytest op_tests/test_phase5_musa.py -q -m "musa and native"
```

## 6. Moore Threads Docker runtime notes

On hosts where Docker reports `Default Runtime: mthreads`, prefer the Moore Threads runtime flags. Do not rely on NVIDIA-style `--gpus all` as the only GPU flag unless your site explicitly documents that mode for MUSA.

Minimal pattern:

```bash
IMG="registry.example.com/repo/image:YOUR_TAG"
WS="/ABS/PATH/on/remote/host/musa-aiter"

docker run --rm \
  --runtime mthreads \
  --network host \
  --ipc host \
  --security-opt label=disable \
  --shm-size=1g \
  --ulimit memlock=-1:-1 \
  -v "$WS:$WS" \
  -w "$WS" \
  -v /usr/bin/mthreads-gmi:/usr/bin/mthreads-gmi:ro \
  "$IMG" \
  bash -lc '
    test -f /path/to/vendor/venv/bin/activate && source /path/to/vendor/venv/bin/activate
    python3 -m pip install -U pip wheel
    python3 -m pip install --no-deps dist/musa_aiter-*.whl
    PYTHONPATH=. python3 -m musa_aiter.cli.main check
    python3 scripts/validate_native_musa_env.py --strict
  '
```

## 7. Troubleshooting

| Symptom | Likely cause | Action |
| --- | --- | --- |
| `ModuleNotFoundError: torch` on import | PyTorch is not installed in the target environment | Install the correct CPU or TorchMUSA PyTorch stack before installing/running `musa-aiter`. |
| `torch_musa: not importable` in `check --soft` | CPU/dev environment | Expected on non-MUSA hosts; use MUSA lab for device validation. |
| `mcc compiler not found in PATH` | MUSA Toolkit not installed or not on `PATH` | Install/activate the MUSA Toolkit or skip native compile checks on CPU/dev machines. |
| Full `pytest op_tests -q` fails on native build-symbol tests | Running native compile tests without `mcc` | Use the CPU-safe selector above or run on a toolkit host. |
| Installed wheel cannot find `csrc/` or `configs/` | Broken artifact or old wheel | Rebuild with `python3 -m build --wheel --sdist` and run `scripts/verify_package_artifacts.py`. |
| `musa-aiter check` fails in lab | Torch/TorchMUSA/driver/toolkit mismatch | Re-run stack sanity checks and align versions with site/vendor documentation. |

## 8. Release handoff checklist

Before giving artifacts to a user:

1. Build wheel and sdist with `python3 -m build --wheel --sdist`.
2. Run `python3 scripts/verify_package_artifacts.py --dist-dir dist`.
3. Record SHA-256 hashes:

   ```bash
   shasum -a 256 dist/musa_aiter-*.whl dist/musa_aiter-*.tar.gz
   ```

4. Include the target environment note: CPU/reference only, or MUSA lab/device validated.
5. If native or device support is claimed, attach sanitized lab results from `pytest -m musa`, `musa-aiter check`, and `scripts/validate_native_musa_env.py --strict`.

See also [Docker](../DOCKER.md) and [Docker + TorchMUSA](../DOCKER_TORCHMUSA.md).
