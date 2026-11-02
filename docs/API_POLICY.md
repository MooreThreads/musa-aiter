# MUSA AITER API Policy

This policy describes the compatibility expectations for the public API.

## API Stability Levels

MUSA AITER uses explicit API stability levels so framework integrators can distinguish production-facing APIs from experimental or internal ones.

| Level | Meaning | Compatibility promise |
| --- | --- | --- |
| Stable | Public API intended for framework/application integration | Backward compatible within a major version |
| Experimental | Public but may change as native kernels mature | Best-effort compatibility; changes documented |
| Reference | CPU/Torch reference implementation for semantics and tests | API may be promoted to stable after native backend validation |
| Internal | Implementation detail | No compatibility promise |

## Current API Classification

### Stable Candidate APIs

These are intended to become stable once validated on TorchMUSA/native kernels:

- `musa_aiter.rmsnorm` — default PyTorch recipe; optional **`prefer_native=True`** JIT/ctypes smoke on MUSA **`float32`** (1-D **`weight`**, last-dim cap **`RMSNORM_NATIVE_MAX_DIM`**) when **`MUSA_AITER_NATIVE_SMOKE=1`**.
- `musa_aiter.flash_attn_func`
- `musa_aiter.AttentionType` (labels aligned with vLLM / SGLang ``AttentionType`` string enums)
- `musa_aiter.paged_attention`
- `musa_aiter.gemm`
- `musa_aiter.fused_moe`
- `musa_aiter.mla_decode_fwd`
- `musa_aiter.mla_prefill_fwd`

### Experimental APIs

- `musa_aiter.gemm_a8w8`
- `musa_aiter.kv_paged_to_linear`, `kv_linear_to_paged`, `kv_paged_to_linear_multi_layer`, `kv_linear_to_paged_multi_layer` (reference paths; native coverage depends on the installed MUSA stack)
- `musa_aiter.gemm_a8w8_per_token`
- `musa_aiter.gemm_a4w4`
- `musa_aiter.grouped_gemm_a8w8`
- `musa_aiter.rmsnorm_quant`
- `musa_aiter.gated_rmsnorm_quant`
- communication and tensor-parallel APIs

### Reference-only APIs

- `rmsnorm_backward_ref`
- `gemm_backward_ref`
- `attention_backward_ref`

### Internal APIs

- `musa_aiter.jit.*`
- `musa_aiter.dispatch.*`
- `musa_aiter.policy.*`
- `musa_aiter.tuning.*` except documented config schema helpers
- `musa_aiter.add`, `musa_aiter.multiply`, `musa_aiter.reduce_sum`, `musa_aiter.softmax_last_dim` (JIT/ctypes **smoke** teaching operators; not inference-stack stable surfaces)

## Deprecation Policy

1. Stable APIs require one minor-release deprecation notice before removal.
2. Experimental APIs may change with changelog notice.
3. Reference-only APIs may be renamed when native equivalents are introduced.
4. Internal APIs may change at any time.

## Native Backend Promotion Rules

An API can be promoted from Reference/Experimental to Stable when:

1. CPU/Torch reference tests pass.
2. MUSA tensor tests pass in a TorchMUSA environment.
3. Native backend correctness passes against reference.
4. Dispatch tracing identifies backend selection clearly.
5. Benchmark numbers are recorded for at least one supported MUSA architecture.
6. API docs include shape, dtype, device, and fallback policy.
