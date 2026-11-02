# Examples

Small, runnable scripts that show how **musa-aiter** operators sit at the PyTorch boundary—useful when wiring **[TorchMUSA](https://github.com/MooreThreads/torch_musa)**, **[vllm-musa](https://github.com/MooreThreads/vllm-musa)**, or **[SGLang](https://github.com/sgl-project/sglang)** workloads.

## Prerequisites

From the repository root, install the package (editable is typical for development):

```bash
cd /path/to/musa-aiter
pip install -e .
```

Run examples with the repo on `PYTHONPATH` if you are not using an installed package:

```bash
PYTHONPATH=. python examples/<script>.py ...
```

For MUSA tensors (`--device musa` / `musa:0`), you need **TorchMUSA** (`import torch_musa`, `torch.musa.is_available()`).

## Scripts

| Script | Purpose |
| --- | --- |
| [`framework_demo.py`](framework_demo.py) | Minimal “framework adapter” for prefill attention, paged decode, and fused MoE (`--device` defaults to `cpu`). |
| [`implement_flash_attn4_musa.py`](implement_flash_attn4_musa.py) | **FA4-aligned skeleton:** optional native hook (`MUSA_AITER_NATIVE_FA4`) + parity with `flash_attn_func`. |
| [`verify_vllm_musa_aiter.py`](verify_vllm_musa_aiter.py) | Coexistence check: `musa_aiter` next to **vLLM 0.20.x** (or `--no-vllm` for CI); optional `--llm-smoke` on **CUDA** reference stacks. |
| [`verify_sglang_musa_aiter.py`](verify_sglang_musa_aiter.py) | Coexistence check: `musa_aiter` next to **SGLang** (or `--no-sglang` for CI); optional `--engine-smoke` on **CUDA/MUSA** stacks (HF download). |

### `framework_demo.py`

```bash
PYTHONPATH=. python examples/framework_demo.py
PYTHONPATH=. python examples/framework_demo.py --device musa
```

### `implement_flash_attn4_musa.py`

```bash
PYTHONPATH=. python examples/implement_flash_attn4_musa.py
PYTHONPATH=. python examples/implement_flash_attn4_musa.py --device musa:0
PYTHONPATH=. python -m musa_aiter.cli.main trace -- python examples/implement_flash_attn4_musa.py
```

### `verify_sglang_musa_aiter.py`

```bash
# CPU / no SGLang (CI-friendly)
PYTHONPATH=. python examples/verify_sglang_musa_aiter.py --no-sglang --device cpu

# MUSA lab: SGLang + TorchMUSA per vendor / upstream
PYTHONPATH=. python examples/verify_sglang_musa_aiter.py --device musa:0

# Optional offline engine (downloads HF weights; not for CPU --device)
PYTHONPATH=. python examples/verify_sglang_musa_aiter.py --device musa:0 --engine-smoke --engine-model facebook/opt-125m
```

### `verify_vllm_musa_aiter.py`

```bash
# CPU / no vLLM (CI-friendly)
PYTHONPATH=. python examples/verify_vllm_musa_aiter.py --no-vllm --device cpu

# MUSA lab: operators on device (install vllm-musa + TorchMUSA per upstream)
PYTHONPATH=. python examples/verify_vllm_musa_aiter.py --device musa:0
```

## Integration documentation

| Topic | Doc |
| --- | --- |
| vLLM 0.20 + vllm-musa | [`docs/VLLM_0_20_INTEGRATION_GUIDE.md`](../docs/VLLM_0_20_INTEGRATION_GUIDE.md) |
| SGLang | [`docs/SGLANG_INTEGRATION_GUIDE.md`](../docs/SGLANG_INTEGRATION_GUIDE.md) |
| Examples hub in built docs (MkDocs) | [`docs/examples_index.md`](../docs/examples_index.md) |

Dispatch tracing for any script:

```bash
PYTHONPATH=. python -m musa_aiter.cli.main trace -- python examples/framework_demo.py
```
