# `op_tests` — operator and integration tests

## Run

From the repo root (with **`pip install -e ".[dev]"`** or **`pytest`** available):

```bash
python3 -m pytest op_tests -q
```

## Markers (`pyproject.toml`)

| Marker | When |
| --- | --- |
| **`musa`** | TorchMUSA + visible MUSA device; **`pytest -m musa`** on a Moore Threads lab machine. Skipped on CPU CI. |
| **`native`** | Optional ctypes / JIT smoke execution; combine with **`MUSA_AITER_NATIVE_SMOKE=1`** + **`pytest -m native`** where **`mcc`** built the smoke `.so` files. |

Examples:

```bash
pytest op_tests/test_phase5_musa.py -m musa -q
pytest op_tests -q -m "musa and native"
```

## Layout

| Area | Tests |
| --- | --- |
| Operator and integration behavior | **`test_attention.py`**, **`test_norm.py`**, **`test_backward.py`**, and related operator tests |
| Registry / CLI / AOT | **`test_registry.py`**, **`test_cli_*.py`**, **`test_aot.py`** |
| Native JIT ABI | **`test_native_smoke.py`** — asserts **`build-smoke`** artifacts compile (**`mcc`** on **`PATH`** in that environment). |

Native ABI checks require **`mcc`** on `PATH`; run them on a MUSA toolkit host.
