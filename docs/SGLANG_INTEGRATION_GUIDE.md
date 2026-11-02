# MUSA AITER with SGLang

This guide describes the public integration boundary between [SGLang](https://github.com/sgl-project/sglang), [TorchMUSA](https://github.com/MooreThreads/torch_musa), and MUSA AITER. MUSA AITER supplies operators and KV-cache movement; SGLang remains responsible for scheduling, model execution, memory pools, and serving.

## Recommended installation model

For production serving, use a vendor-published SGLang + TorchMUSA distribution that declares `musa-aiter` as a dependency and pins compatible versions. This keeps framework wiring and device-specific version selection in the distribution that owns them.

For development or a custom integration, install the three projects in one environment according to their upstream documentation:

```bash
python3 -m pip install -e .
# Install TorchMUSA and SGLang using versions compatible with your MUSA runtime.
```

Check that the package can be imported alongside SGLang:

```bash
PYTHONPATH=. python3 examples/verify_sglang_musa_aiter.py --no-sglang --device cpu
PYTHONPATH=. python3 examples/verify_sglang_musa_aiter.py --device musa:0
```

The second command requires `torch_musa`, a visible MUSA device, and an installed SGLang distribution.

## Integration points

Use the public Python APIs at the framework’s tensor/operator boundary:

| SGLang workload | MUSA AITER surface |
| --- | --- |
| Dense or decode attention | `flash_attn_func`, `paged_attention`, and `paged_sparse_attention` |
| GEMM and MoE | `gemm`, `grouped_gemm`, and `fused_moe` |
| Normalization and quantization | `rmsnorm`, `rmsnorm_quant`, and quantization helpers |
| KV-cache movement | `musa_aiter.integration.sglang_kv` and the KV transfer functions exported by `musa_aiter` |
| Capability discovery | `musa-aiter list-ops` and `musa-aiter trace` |

Keep framework-owned metadata (block tables, slot mappings, sequence lengths, and layer ownership) in SGLang. Convert it once at the adapter boundary, then call the corresponding MUSA AITER operation.

## KV-cache movement

The SGLang adapter supports reference-safe conversion and transfer for common paged, linear, merged, MLA, FP8, and multi-layer layouts. Preserve the framework’s layout and scale metadata when adding a new route.

Adapter paths are opt-in through environment variables so a framework can roll out one operation at a time. The most commonly used switches are:

```bash
export MUSA_AITER_SGLANG_ADAPTER=1
export MUSA_AITER_NATIVE_KV_TRANSFER=1
export MUSA_AITER_NATIVE_CPU_HOST_KV_TRANSFER=1
```

Native routes only activate when device, dtype, shape, layout, and runtime checks pass. Otherwise the adapter returns a clear unavailable decision and the caller can use its existing reference path.

## Attention-side integration

Attention replacement should be enabled only after parity is established against the SGLang backend for the target model and sequence-length range. The public patch helper can be used by a controlled custom deployment:

```bash
python3 scripts/patch_sglang_musa_aiter_hicache.py --help
```

Keep output replacement and paged/sparse replacement disabled while validating a new model. Enable them explicitly with the corresponding `MUSA_AITER_SGLANG_*` switches after correctness and memory behavior are understood. Run `musa-aiter check` and the MUSA-marked tests for device smoke coverage.

## Dispatch and diagnostics

Use scoped tracing when diagnosing a framework call:

```bash
MUSA_AITER_TRACE_DISPATCH=1 \
  PYTHONPATH=. python3 -m musa_aiter.cli.main trace -- \
  python3 examples/verify_sglang_musa_aiter.py --no-sglang --device cpu
```

Trace records identify the operation, selected backend, device, dtype, and fallback reason. Do not enable verbose route logging for production traffic; follow [Security for artifacts and traces](SECURITY_ARTIFACTS.md) before sharing logs.

## Validation checklist

1. Install compatible PyTorch, TorchMUSA, and SGLang versions.
2. Run `musa-aiter check` and `scripts/validate_native_musa_env.py --strict` on a MUSA host.
3. Run `python3 -m pytest op_tests -q` and `python3 -m pytest op_tests -q -m musa`.
4. Run the SGLang coexistence example on CPU first, then on `musa:0`.
5. Compare framework output with the original SGLang path before enabling replacement routes.
6. Capture only sanitized traces and benchmark summaries when reporting an integration issue.

## Troubleshooting

| Symptom | Likely cause | Action |
| --- | --- | --- |
| `torch_musa` cannot be imported | TorchMUSA is not installed or is incompatible with PyTorch | Install the vendor-matched stack before installing MUSA AITER. |
| `torch.musa.is_available()` is false | No visible MUSA device or driver mismatch | Check the host runtime and `mthreads-gmi`; retry on a supported device. |
| Adapter decision says unavailable | An opt-in switch or layout/device precondition is missing | Inspect the decision reason and use the reference path until the precondition is met. |
| Numerical mismatch after replacement | Framework metadata or dtype/layout conversion differs | Disable replacement, compare the converted tensors and scales, then re-enable one route at a time. |

For a runnable coexistence check and optional engine smoke, see [`examples/verify_sglang_musa_aiter.py`](../examples/verify_sglang_musa_aiter.py).
