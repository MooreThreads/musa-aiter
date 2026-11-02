# Step-by-step: MUSA AITER with vLLM 0.20.x

**musa-aiter is a MUSA-first project** (Moore Threads accelerators, **TorchMUSA**, `torch.musa`, `mcc`, `mthreads-gmi`). CPU and **MUSA** devices are the primary targets for operator validation.

Runnable scripts: **[Examples](examples_index.md)** (links to the **`examples/README.md`** file in a source checkout).

## Official integration path (selected)

**Product default:** Moore Threads **[vllm-musa](https://github.com/MooreThreads/vllm-musa)** **absorbs** **musa-aiter** as a **declared dependency**, wires operators internally, and ships **pinned** combinations of `vllm-musa` + **TorchMUSA** + `musa_aiter` (wheels or container images). End users install **vendor artifacts**; they do **not** replace vLLM attention kernels by hand for standard serving.

The procedural sections below remain useful for **device validation**, **fork bring-up**, and **parity debugging** until vendor docs supersede them.

## Moore Threads upstream repositories

Use the vendor-maintained trees for PyTorch and vLLM on MUSA before mixing in generic pip packages:

| Stack | Repository | Purpose |
| --- | --- | --- |
| **TorchMUSA** | [github.com/MooreThreads/torch_musa](https://github.com/MooreThreads/torch_musa) | PyTorch extension for MUSA (`import torch_musa`, `torch.musa`, `musa:N` tensors). Install/build per that README for your driver and Python version. |
| **vLLM-MUSA** | [github.com/MooreThreads/vllm-musa](https://github.com/MooreThreads/vllm-musa) | vLLM adapted for Moore Threads GPUs; this is the **intended** vLLM line for MUSA inference (version alignment with upstream vLLM, e.g. 0.20.x, is documented there). |

For **architecture and API shape** context, the upstream **[vLLM 0.20](https://docs.vllm.ai/en/v0.20.1/)** docs still help. The PyPI package `pip install vllm` targets **NVIDIA/CUDA**; on MUSA hardware prefer **vLLM-MUSA** plus **TorchMUSA**, then layer **musa-aiter** operators where you replace kernel math.

Official vLLM plugin overview: [`Plugin System`](https://docs.vllm.ai/en/v0.20.1/design/plugin_system/) and [`vllm.plugins` API summary](https://docs.vllm.ai/en/v0.20.1/api/vllm/plugins/).

---

## 1. What you will achieve

| Level | Result |
| --- | --- |
| **Baseline (required)** | `import musa_aiter` succeeds; reference attention + fused MoE ops run on **`cpu`** or **`musa:0`** (with TorchMUSA installed). |
| **MUSA + vLLM (target UX)** | Install **vendor-published** **[vllm-musa](https://github.com/MooreThreads/vllm-musa)** + **[torch_musa](https://github.com/MooreThreads/torch_musa)** bundles so **`musa_aiter` is pulled in automatically** where the stack declares it—not a manual “swap kernels” exercise for operators. |
| **Optional NVIDIA reference** | `pip install vllm` on a **CUDA** box for upstream-only comparison; not the MUSA deployment path. |
| **Fork / device validation** | If you patch **vllm-musa** yourself, replace FA/paged-attn/MoE math with `musa_aiter.*` at the Torch boundary, guided by [vLLM 0.20 design docs](https://docs.vllm.ai/en/v0.20.1/)—until the vendor distribution provides that wiring. |
| **Optional `--llm-smoke`** | Quick `LLM.generate` check: on **CUDA**, exercises stock PyPI vLLM if installed; with **vllm-musa**, follow that project’s supported runtime (see script help text). |

Native MUSA execution depends on TorchMUSA and vendor kernels; use `musa-aiter check` and the MUSA-marked tests before serving.

---

## 2. Prerequisites

1. **Python**: for the vLLM side, 0.20 supports **Python 3.10–3.13**; **3.12** matches upstream CI defaults ([Developing](https://docs.vllm.ai/en/v0.20.1/design/plugin_system/design/#developing)). For MUSA-only work, follow your TorchMUSA release notes.
2. **Accelerator**
   - **MUSA (primary):** Moore Threads GPU + driver + **[TorchMUSA](https://github.com/MooreThreads/torch_musa)**; verify with `python -c "import torch; import torch_musa; print(torch.musa.is_available())"`.
   - **vLLM on MUSA:** install **[vllm-musa](https://github.com/MooreThreads/vllm-musa)** per its README (branch/tag matching the vLLM line you need, e.g. 0.20).
   - **NVIDIA-only reference (optional):** PyPI `vllm` + CUDA PyTorch per [Installation](https://docs.vllm.ai/en/v0.20.1/getting_started/installation/).
3. **Disk / network**: optional `--llm-smoke` downloads HF weights (~250MB class for OPT-125M).
4. **Build tools:** if building vLLM from source, use vLLM’s documented toolchain ([Build wheel from source](https://docs.vllm.ai/en/v0.20.1/getting_started/installation/)) — usually **CUDA** on x86_64, not MUSA.

---

## 3. Create an isolated virtual environment

Use a fresh venv so Framework A’s pinned PyTorch does not clash with Framework B’s.

```bash
python3.12 -m venv ~/venvs/musa-aiter-work
source ~/venvs/musa-aiter-work/bin/activate
python -m pip install --upgrade pip setuptools wheel
```

On MUSA machines, install **[TorchMUSA](https://github.com/MooreThreads/torch_musa)** first (see that repository’s build and wheel instructions), then continue.

---

## 4. Install musa-aiter (editable)

From the **musa-aiter** repository root:

```bash
cd /path/to/musa-aiter
pip install -e .
```

### Verify operators on MUSA hardware

```bash
PYTHONPATH=. python examples/verify_vllm_musa_aiter.py --no-vllm --device musa:0
```

(Use `--device cpu` on laptops without a card.)

---

## 5. Install vLLM for MUSA (**vllm-musa**) — recommended

For inference on Moore Threads hardware, follow **[MooreThreads/vllm-musa](https://github.com/MooreThreads/vllm-musa)**:

- Clone the branch or release tag that matches your target vLLM line (e.g. **0.20**), install **TorchMUSA** from **[MooreThreads/torch_musa](https://github.com/MooreThreads/torch_musa)** with compatible versions, then build/install `vllm-musa` as documented upstream.

```bash
# Illustrative — exact steps are in vllm-musa README (submodules, torch pin, etc.)
git clone https://github.com/MooreThreads/vllm-musa.git
cd vllm-musa
# … follow README: environment, torch_musa, pip install -e .
```

Then install **musa-aiter** in the same (or compatible) environment:

```bash
cd /path/to/musa-aiter
pip install -e .
```

Validate:

```bash
python -c "import vllm, musa_aiter; print(vllm.__version__, musa_aiter.__version__)"
```

---

## 6. Optional: PyPI **vllm** on NVIDIA/CUDA (reference only)

Use this only for **CUDA** comparison or rebasing work, **not** for MUSA deployment:

```bash
pip install 'vllm>=0.20,<0.21'
```

See [vLLM 0.20.1](https://docs.vllm.ai/en/v0.20.1/) docs. Official wheels expect **CUDA** PyTorch; they generally do **not** install on a **pure MUSA** torch stack.

**If PyTorch conflicts appear:** install vLLM’s expected **torch** build from the [installation](https://docs.vllm.ai/en/v0.20.1/getting_started/installation/) page, then `pip install -e .` again for musa-aiter.

### Combined dry check (when both import)

```bash
python -c "import musa_aiter, vllm; print(musa_aiter.__version__, vllm.__version__)"
```

---

## 7. Understand the integration surface (before editing vLLM)

vLLM routes attention and MoE through its **model executor / kernels** layers, not through a single “swap backend” toggle. Practical approaches:

### 7.A Reference integration (recommended first)

Treat **musa_aiter** as the Torch-facing operator library for **MUSA tensors**:

1. Locate Python call sites for prefill/decode attention, paged KV, MoE routing (design context: [`Paged Attention`](https://docs.vllm.ai/en/v0.20.1/design/design/paged_attention/), [`Fused MoE Modular Kernel`](https://docs.vllm.ai/en/v0.20.1/design/design/fused_moe_modular_kernel/), [`CustomOp`](https://docs.vllm.ai/en/v0.20.1/design/design/custom_op/)).

2. Replace **only the tensor math** with `musa_aiter` entry points (`flash_attn_func`, `paged_attention`, `topk_softmax`, `fused_moe`, …).

3. Keep vLLM’s batching, scheduler, KV layout, and parallelism assumptions intact — but run your **integration tests on `musa:N`**, not NVIDIA-only assumptions.

Adapter illustration: `examples/framework_demo.py` at the repository root.

### 7.B vLLM plugin / custom-model path (distribution)

Use setuptools **entry points** expected by [vLLM’s plugin loaders](https://docs.vllm.ai/en/v0.20.1/api/vllm/plugins/). After installing vLLM, inspect locally:

```bash
python -c "from importlib import metadata as m
eps = sorted({e.group for e in m.entry_points() if e.group.startswith('vllm.')})
print('\n'.join(eps))"
```

Expose custom model registrations from a small shim module (patterns in [contributing docs](https://docs.vllm.ai/en/v0.20.1/contributing/model/registration/)).

### 7.C Device model on MUSA

Prefer **`device="musa"` / `musa:N`** tensors from TorchMUSA. If the stack exposes CUDA-name aliases, keep compatibility shims **inside** your fork or plugin, not inside musa-aiter’s public API surface. Align distributed collectives with **MUSA/McCL** (or vendor equivalents), not NCCL, for production.

---

## 8. Minimal code pattern (copy into your model layer)

Same minimal pattern as `examples/framework_demo.py` — run tensors on **`musa:0`** in production:

```python
import musa_aiter

class MusaAiterAttentionBackend:
    def prefill(self, q, k, v, causal=True):
        return musa_aiter.flash_attn_func(q, k, v, causal=causal)

    def decode(self, query, key_cache, value_cache, block_tables, seq_lens, *, num_kv_heads, block_size):
        return musa_aiter.paged_attention(
            query, key_cache, value_cache, block_tables, seq_lens,
            num_kv_heads=num_kv_heads, block_size=block_size,
        )

class MusaAiterMoEBackend:
    def forward(self, x, w1, w2, router_logits, top_k=2):
        weights, ids = musa_aiter.topk_softmax(router_logits, top_k)
        return musa_aiter.fused_moe(x, w1, w2, weights, ids)
```

Use `MUSA_AITER_TRACE_DISPATCH=1` while debugging.

---

## 9. Verify the installation (required checks)

Script: `examples/verify_vllm_musa_aiter.py` (repository root).

### 9.1 MUSA hardware (recommended)

```bash
cd /path/to/musa-aiter
PYTHONPATH=. python examples/verify_vllm_musa_aiter.py --device musa:0
```

Expected:

- `musa_aiter ops OK on device='musa:0'`
- If vLLM is installed: `vLLM import OK, version='0.20.x'`

### 9.2 CPU-only (CI)

```bash
PYTHONPATH=. python examples/verify_vllm_musa_aiter.py --no-vllm --device cpu
```

### 9.3 Optional `LLM.generate` smoke (NVIDIA / reference)

Stock **PyPI** `vllm` + **`--llm-smoke`** expects **CUDA**. On a CUDA-capable machine:

```bash
PYTHONPATH=. python examples/verify_vllm_musa_aiter.py --llm-smoke --llm-model facebook/opt-125m
```

This checks the **upstream NVIDIA** stack bundled with PyPI vLLM, not Moore Threads MUSA kernels.

### 9.4 vLLM-MUSA

After you install **[vllm-musa](https://github.com/MooreThreads/vllm-musa)** and **[torch_musa](https://github.com/MooreThreads/torch_musa)** per those repositories, run the same `examples/verify_vllm_musa_aiter.py` **without** `--llm-smoke` first (`--device musa:0`). Enable `--llm-smoke` only if that project documents a CUDA-compatible smoke path, or add a model/serve check per **vllm-musa** instructions.

---

## 10. Common failures

| Symptom | Likely fix |
| --- | --- |
| `torch_musa` / `torch.musa` missing | Install **[TorchMUSA](https://github.com/MooreThreads/torch_musa)** before running `--device musa:0`. |
| `Torch not compiled with CUDA enabled` when installing **PyPI** `vllm` | Expected on **pure MUSA** stacks; use **[vllm-musa](https://github.com/MooreThreads/vllm-musa)** instead, or install PyPI vLLM only on a **CUDA** reference machine. |
| `torch` version conflict | Install the stack that **your accelerator** requires first (TorchMUSA pin vs vLLM’s CUDA torch). |
| Attention shape mismatch | Match vLLM’s head layout on **MUSA tensors**; add small tensor tests. |

---

## 11. Next steps toward production

1. Run operator parity on **`musa:0`** against CPU references in this repo.
2. Gate native paths on TorchMUSA and verify them on the target device.
3. Integrate operators inside **[vllm-musa](https://github.com/MooreThreads/vllm-musa)** (or track it while rebasing); use [vLLM 0.20 design](https://docs.vllm.ai/en/v0.20.1/) and PyPI vLLM on CUDA only as a **porting reference**.

---

## Reference links

### Moore Threads (MUSA runtime)

- [torch_musa (TorchMUSA)](https://github.com/MooreThreads/torch_musa)
- [vllm-musa](https://github.com/MooreThreads/vllm-musa)

### Upstream vLLM 0.20.1 documentation (design reference)

- [Getting started / Installation](https://docs.vllm.ai/en/v0.20.1/getting_started/installation/)
- [Plugin system](https://docs.vllm.ai/en/v0.20.1/design/plugin_system/)
- [Model registration (contributing)](https://docs.vllm.ai/en/v0.20.1/contributing/model/registration/)
- [CustomOp design](https://docs.vllm.ai/en/v0.20.1/design/design/custom_op/)
- [Paged Attention design](https://docs.vllm.ai/en/v0.20.1/design/design/paged_attention/)
