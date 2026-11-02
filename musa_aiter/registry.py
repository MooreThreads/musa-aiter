from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable


@dataclass(frozen=True)
class OperatorCapability:
    name: str
    family: str
    status: str
    dtypes: tuple[str, ...]
    devices: tuple[str, ...]
    backends: tuple[str, ...]
    description: str
    notes: tuple[str, ...] = field(default_factory=tuple)
    #: Planned vendor or template backends (Phase P7); not implied to be linked at runtime.
    vendor_backends: tuple[str, ...] = field(default_factory=tuple)


_REGISTRY: dict[str, OperatorCapability] = {}


def register(capability: OperatorCapability) -> OperatorCapability:
    if capability.name in _REGISTRY:
        raise ValueError(f"duplicate operator capability: {capability.name}")
    _REGISTRY[capability.name] = capability
    return capability


def get(name: str) -> OperatorCapability:
    return _REGISTRY[name]


def list_capabilities(family: str | None = None) -> list[OperatorCapability]:
    values: Iterable[OperatorCapability] = _REGISTRY.values()
    if family is not None:
        values = [cap for cap in values if cap.family == family]
    return sorted(values, key=lambda cap: (cap.family, cap.name))


def as_dict() -> dict[str, dict]:
    return {
        name: {
            "name": cap.name,
            "family": cap.family,
            "status": cap.status,
            "dtypes": list(cap.dtypes),
            "devices": list(cap.devices),
            "backends": list(cap.backends),
            "description": cap.description,
            "notes": list(cap.notes),
            "vendor_backends": list(cap.vendor_backends),
        }
        for name, cap in sorted(_REGISTRY.items())
    }


def _reg(name, family, status, dtypes, backends, description, notes=(), vendor_backends=()):
    register(
        OperatorCapability(
            name,
            family,
            status,
            dtypes,
            ("cpu", "musa"),
            backends,
            description,
            notes,
            vendor_backends,
        )
    )


# Phase 0
_reg("add", "smoke", "phase0", ("float32", "float16", "bfloat16", "int32", "int64"), ("torch", "mcc-jit-compile-probe"), "Phase-0 smoke add operator.", vendor_backends=("handwritten-mu", "mcc-jit"))
_reg("multiply", "smoke", "phase0", ("float32", "float16", "bfloat16", "int32", "int64"), ("torch", "mcc-jit-compile-probe"), "Phase-1 smoke elementwise multiply (JIT teaching kernel).", vendor_backends=("handwritten-mu", "mcc-jit"))
_reg("reduce_sum", "smoke", "phase0", ("float32", "float16", "bfloat16", "int32", "int64"), ("torch", "mcc-jit-compile-probe"), "Phase-2 ctypes parallel reduction sum smoke (single-CTA teaching kernel).", vendor_backends=("handwritten-mu", "mcc-jit"))
_reg("softmax_last_dim", "smoke", "phase0", ("float32", "float16", "bfloat16"), ("torch", "mcc-jit-compile-probe"), "Phase-3 last-dim softmax smoke (Flash-style stable reduce; one CTA per row).", vendor_backends=("handwritten-mu", "mcc-jit", "mutlass"))

# Phase 1
_reg("rmsnorm", "norm", "phase1-reference", ("float32", "float16", "bfloat16"), ("torch-reference", "mcc-jit-compile-probe"), "RMSNorm reference implementation.", vendor_backends=("mudnn", "handwritten-mu", "mcc-jit"))
_reg("flash_attn_func", "attention", "phase1-reference", ("float32", "float16", "bfloat16"), ("torch-sdpa-reference",), "Dense FlashAttention-style API.", vendor_backends=("mudnn", "mubin", "mutlass"))
_reg("paged_attention_v1", "attention", "phase1-reference", ("float32", "float16", "bfloat16"), ("torch-reference",), "Paged Attention v1 reference implementation.", vendor_backends=("mudnn", "mubin"))
_reg("apply_rope", "cache", "phase1-reference", ("float32", "float16", "bfloat16"), ("torch-reference",), "RoPE reference implementation.")
_reg("apply_rope_to_key_cache", "cache", "phase-pkv-reference", ("float32", "float16", "bfloat16"), ("torch-reference",), "Apply RoPE in-place to selected paged key-cache slots for MUSA KV repositioning.")
_reg("kv_cachegen_encode", "kvcache", "phase-pkv-reference", ("float32", "float16", "bfloat16"), ("torch-reference",), "CacheGen-style uniform-bin KV tensor encode with empirical CDF metadata for MUSA offload experiments.")
_reg("kv_cachegen_decode", "kvcache", "phase-pkv-reference", ("uint8",), ("torch-reference",), "CacheGen-style uint8 codebook decode for compressed KV tensor experiments.")
_reg("kv_copy_paged_blocks", "kvcache", "phase-pkv-reference", ("float32", "float16", "bfloat16"), ("torch-reference",), "Copy selected paged KV cache blocks between same-layout source and destination caches for MUSA staged H2D/D2H transfers.", vendor_backends=("handwritten-mu", "mubin", "mudnn"))
_reg("kv_copy_paged_blocks_fp8", "kvcache", "phase-fp8-kv-reference", ("uint8", "float8_e4m3fn", "float8_e5m2"), ("torch-reference",), "Raw FP8 paged KV cache block copy between same-layout source and destination caches.", vendor_backends=("handwritten-mu", "mubin", "mudnn"))
_reg("kv_load_and_reshape_flash", "kvcache", "phase-pkv-reference", ("float32", "float16", "bfloat16"), ("torch-reference",), "Load paged KV caches into dense Flash-style [batch, seq, kv_heads, head_dim] tensors for MUSA attention interop.", vendor_backends=("handwritten-mu", "mubin", "mudnn"))
_reg("kv_load_and_dequantize_fp8_flash", "kvcache", "phase-fp8-kv-reference", ("uint8", "float8_e4m3fn", "float8_e5m2"), ("torch-reference",), "Load paged FP8 KV caches into dense dequantized Flash-style tensors for MUSA attention interop.", vendor_backends=("handwritten-mu", "mubin", "mudnn"))
_reg("kv_reshape_and_cache_back_flash", "kvcache", "phase-pkv-reference", ("float32", "float16", "bfloat16"), ("torch-reference",), "Scatter dense Flash-style KV tensors back into paged KV caches.", vendor_backends=("handwritten-mu", "mubin", "mudnn"))
_reg("kv_paged_to_linear", "kvcache", "phase-pkv-reference", ("float32", "float16", "bfloat16"), ("torch-reference",), "Paged KV→token-major linear gather (single layer; Phase PKV).", vendor_backends=("handwritten-mu", "mubin", "mudnn"))
_reg("kv_paged_fp8_to_linear", "kvcache", "phase-fp8-kv-reference", ("uint8", "float16", "bfloat16", "float32"), ("torch-reference",), "Paged FP8 KV cache→linear dequantize gather with scalar K/V scales.", vendor_backends=("handwritten-mu", "mubin", "mudnn"))
_reg("kv_linear_to_paged", "kvcache", "phase-pkv-reference", ("float32", "float16", "bfloat16"), ("torch-reference",), "Token-major KV→paged scatter (single layer; Phase PKV).", vendor_backends=("handwritten-mu", "mubin", "mudnn"))
_reg("kv_linear_to_paged_fp8", "kvcache", "phase-fp8-kv-reference", ("float16", "bfloat16", "float32", "uint8"), ("torch-reference",), "Linear KV→paged FP8 cache quantize scatter with scalar K/V scales.", vendor_backends=("handwritten-mu", "mubin", "mudnn"))
_reg("kv_paged_to_linear_multi_layer", "kvcache", "phase-pkv-reference", ("float32", "float16", "bfloat16"), ("torch-reference",), "Paged KV→linear gather (multi-layer stack; Phase PKV).", vendor_backends=("handwritten-mu", "mubin", "mudnn"))
_reg("kv_paged_fp8_to_linear_multi_layer", "kvcache", "phase-fp8-kv-reference", ("uint8", "float16", "bfloat16", "float32"), ("torch-reference",), "Paged FP8 KV cache→linear dequantize gather (multi-layer stack) with scalar K/V scales.", vendor_backends=("handwritten-mu", "mubin", "mudnn"))
_reg("kv_paged_fp8_to_merged_linear", "kvcache", "phase-fp8-kv-reference", ("uint8", "float16", "bfloat16", "float32"), ("torch-reference",), "Paged FP8 KV cache→merged linear dequantize gather with scalar K/V scales.", vendor_backends=("handwritten-mu", "mubin", "mudnn"))
_reg("kv_paged_fp8_to_merged_linear_multi_layer", "kvcache", "phase-fp8-kv-reference", ("uint8", "float16", "bfloat16", "float32"), ("torch-reference",), "Paged FP8 KV cache→merged linear dequantize gather (multi-layer stack) with scalar K/V scales.", vendor_backends=("handwritten-mu", "mubin", "mudnn"))
_reg("kv_linear_to_paged_multi_layer", "kvcache", "phase-pkv-reference", ("float32", "float16", "bfloat16"), ("torch-reference",), "Linear→paged scatter (multi-layer stack; Phase PKV).", vendor_backends=("handwritten-mu", "mubin", "mudnn"))
_reg("kv_linear_to_paged_fp8_multi_layer", "kvcache", "phase-fp8-kv-reference", ("float16", "bfloat16", "float32", "uint8"), ("torch-reference",), "Linear KV→paged FP8 cache quantize scatter (multi-layer stack) with scalar K/V scales.", vendor_backends=("handwritten-mu", "mubin", "mudnn"))
_reg("kv_merged_linear_to_paged_fp8", "kvcache", "phase-fp8-kv-reference", ("float16", "bfloat16", "float32", "uint8"), ("torch-reference",), "Merged linear KV→paged FP8 cache quantize scatter with scalar K/V scales.", vendor_backends=("handwritten-mu", "mubin", "mudnn"))
_reg("kv_merged_linear_to_paged_fp8_multi_layer", "kvcache", "phase-fp8-kv-reference", ("float16", "bfloat16", "float32", "uint8"), ("torch-reference",), "Merged linear KV→paged FP8 cache quantize scatter (multi-layer stack) with scalar K/V scales.", vendor_backends=("handwritten-mu", "mubin", "mudnn"))
_reg("kv_paged_to_merged_linear", "kvcache", "phase-pkv-reference", ("float32", "float16", "bfloat16"), ("torch-reference",), "Paged KV→merged token-major linear gather [T, 2, kv_heads, head_dim] (single layer; Phase PKV).", vendor_backends=("handwritten-mu", "mubin", "mudnn"))
_reg("kv_merged_linear_to_paged", "kvcache", "phase-pkv-reference", ("float32", "float16", "bfloat16"), ("torch-reference",), "Merged token-major linear KV→paged scatter [T, 2, kv_heads, head_dim] (single layer; Phase PKV).", vendor_backends=("handwritten-mu", "mubin", "mudnn"))
_reg("kv_paged_to_merged_linear_multi_layer", "kvcache", "phase-pkv-reference", ("float32", "float16", "bfloat16"), ("torch-reference",), "Paged KV→merged linear gather [L, T, 2, kv_heads, head_dim] (multi-layer stack; Phase PKV).", vendor_backends=("handwritten-mu", "mubin", "mudnn"))
_reg("kv_merged_linear_to_paged_multi_layer", "kvcache", "phase-pkv-reference", ("float32", "float16", "bfloat16"), ("torch-reference",), "Merged linear KV→paged scatter [L, T, 2, kv_heads, head_dim] (multi-layer stack; Phase PKV).", vendor_backends=("handwritten-mu", "mubin", "mudnn"))
_reg("kv_paged_to_mla_linear", "kvcache", "phase-pkv-reference", ("float32", "float16", "bfloat16"), ("torch-reference",), "MLA/DSA asymmetric flat paged caches→linear gather [T, kv_hidden_dim] and [T, pe_hidden_dim].", vendor_backends=("handwritten-mu", "mubin", "mudnn"))
_reg("kv_mla_linear_to_paged", "kvcache", "phase-pkv-reference", ("float32", "float16", "bfloat16"), ("torch-reference",), "MLA/DSA asymmetric linear→flat paged cache scatter for kv and positional-key widths.", vendor_backends=("handwritten-mu", "mubin", "mudnn"))
_reg("kv_paged_to_mla_linear_multi_layer", "kvcache", "phase-pkv-reference", ("float32", "float16", "bfloat16"), ("torch-reference",), "Stacked MLA/DSA asymmetric flat paged caches→layer-major linear buffers.", vendor_backends=("handwritten-mu", "mubin", "mudnn"))
_reg("kv_mla_linear_to_paged_multi_layer", "kvcache", "phase-pkv-reference", ("float32", "float16", "bfloat16"), ("torch-reference",), "Layer-major MLA/DSA asymmetric linear buffers→stacked flat paged caches.", vendor_backends=("handwritten-mu", "mubin", "mudnn"))
_reg("hicache_transfer_kv_per_layer", "kvcache", "phase-kv-v05-hicache", ("float16", "bfloat16"), ("torch-reference", "ctypes-musa-native"), "SGLang HiCache per-layer layer-first transfer with two index tables.", vendor_backends=("handwritten-mu",))
_reg("hicache_transfer_kv_per_layer_pf_lf", "kvcache", "phase-kv-v05-hicache", ("float16", "bfloat16"), ("torch-reference", "ctypes-musa-native"), "SGLang HiCache per-layer page-first→layer-first transfer.", vendor_backends=("handwritten-mu",))
_reg("hicache_transfer_kv_all_layer", "kvcache", "phase-kv-v05-hicache", ("float16", "bfloat16"), ("torch-reference", "ctypes-musa-native"), "SGLang HiCache all-layer layer-first transfer.", vendor_backends=("handwritten-mu",))
_reg("hicache_transfer_kv_all_layer_lf_pf", "kvcache", "phase-kv-v05-hicache", ("float16", "bfloat16"), ("torch-reference", "ctypes-musa-native"), "SGLang HiCache all-layer layer-first→page-first transfer.", vendor_backends=("handwritten-mu",))
_reg("hicache_transfer_kv_per_layer_mla", "kvcache", "phase-kv-v05-hicache", ("float16", "bfloat16"), ("torch-reference", "ctypes-musa-native"), "SGLang HiCache per-layer MLA layer-first transfer.", vendor_backends=("handwritten-mu",))
_reg("hicache_transfer_kv_per_layer_mla_pf_lf", "kvcache", "phase-kv-v05-hicache", ("float16", "bfloat16"), ("torch-reference", "ctypes-musa-native"), "SGLang HiCache per-layer MLA page-first→layer-first transfer.", vendor_backends=("handwritten-mu",))
_reg("lmcache_kv_paged_to_buffer_two_x_nl", "kvcache", "phase-pkv-lmcache-sglang-mp", ("float32", "float16", "bfloat16"), ("torch-reference", "torchmusa-cpp-extension-coalesced-u16", "ctypes-musa-native-u16"), "Gather SGLang MP 2 x NL x [NB, BS, NH, HS] split K/V caches into an LMCache [2, NL, T, NH*HS] buffer.", vendor_backends=("handwritten-mu",))
_reg("lmcache_kv_buffer_to_paged_two_x_nl", "kvcache", "phase-pkv-lmcache-sglang-mp", ("float32", "float16", "bfloat16"), ("torch-reference", "torchmusa-cpp-extension-coalesced-u16", "ctypes-musa-native-u16"), "Scatter an LMCache [2, NL, T, NH*HS] buffer into SGLang MP 2 x NL x [NB, BS, NH, HS] split K/V caches.", vendor_backends=("handwritten-mu",))
_reg("lmcache_musa_memcpy_batch_async", "kvcache", "phase-pkv-lmcache-musa-batch-copy-experimental", ("uint8",), ("musa-sdk-5.1-batch-copy",), "Opt-in asynchronous host/device byte-copy batch for LMCache direct transfer through MUSA SDK 5.1+ musaMemcpyBatchAsync.")
_reg("gemm", "gemm", "phase1-reference", ("float32", "float16", "bfloat16"), ("torch-reference",), "GEMM wrapper.", vendor_backends=("mudnn", "mutlass", "mubin"))

# Phase 2 GEMM
_reg("grouped_gemm", "gemm", "phase2-reference", ("float32", "float16", "bfloat16"), ("torch-reference",), "Grouped GEMM reference implementation.", vendor_backends=("mudnn", "mutlass", "mubin"))
_reg("ragged_gemm", "gemm", "phase2-reference", ("float32", "float16", "bfloat16"), ("torch-reference",), "Ragged GEMM reference implementation.")
_reg("gemm_a8w8", "gemm", "phase2-reference", ("float32", "float16", "bfloat16"), ("torch-reference",), "A8W8 GEMM reference implementation.")
_reg("gemm_a8w8_per_token", "gemm", "phase2-reference", ("float32", "float16", "bfloat16"), ("torch-reference",), "Per-token A8W8 GEMM reference.")
_reg("gemm_a4w4", "gemm", "phase2-reference", ("float32", "float16", "bfloat16"), ("torch-reference",), "A4W4 GEMM reference implementation.")
_reg("grouped_gemm_a8w8", "gemm", "phase2-reference", ("float32", "float16", "bfloat16"), ("torch-reference",), "Grouped A8W8 GEMM reference implementation.")

# Phase 2 MoE
_reg("topk_softmax", "moe", "phase2-reference", ("float32", "float16", "bfloat16"), ("torch-reference",), "MoE top-k softmax routing.")
_reg("topk_sigmoid", "moe", "phase2-reference", ("float32", "float16", "bfloat16"), ("torch-reference",), "MoE top-k sigmoid routing.")
_reg("moe_sorting", "moe", "phase2-reference", ("int64",), ("torch-reference",), "MoE token sorting.")
_reg("fused_moe", "moe", "phase2-reference", ("float32", "float16", "bfloat16"), ("torch-reference",), "Fused MoE reference implementation.", vendor_backends=("mudnn", "mutlass", "mubin"))

# Phase 2 Quant
_reg("quantize_per_token", "quant", "phase2-reference", ("float32", "float16", "bfloat16"), ("torch-reference",), "Per-token quantization.")
_reg("dequantize_per_token", "quant", "phase2-reference", ("int8", "float32"), ("torch-reference",), "Per-token dequantization.")
_reg("quantize_per_tensor", "quant", "phase2-reference", ("float32", "float16", "bfloat16"), ("torch-reference",), "Per-tensor quantization.")
_reg("dequantize_per_tensor", "quant", "phase2-reference", ("int8", "float32"), ("torch-reference",), "Per-tensor dequantization.")
_reg("rmsnorm_quant", "quant", "phase2-reference", ("float32", "float16", "bfloat16"), ("torch-reference",), "Fused RMSNorm + quantization.")
_reg("gated_rmsnorm_quant", "quant", "phase2-reference", ("float32", "float16", "bfloat16"), ("torch-reference",), "Gated RMSNorm + group quantization.")

# Phase 2 MLA
_reg("mla_decode_fwd", "mla", "phase2-reference", ("float32", "float16", "bfloat16"), ("torch-reference",), "MLA decode reference implementation.")
_reg("mla_prefill_fwd", "mla", "phase2-reference", ("float32", "float16", "bfloat16"), ("torch-reference",), "MLA prefill reference implementation.")

# Phase 3 Communication
_reg("all_reduce", "communication", "phase3-reference", ("float32", "float16", "bfloat16", "int32", "int64"), ("torch-reference",), "Single-rank all-reduce reference API.")
_reg("reduce_scatter", "communication", "phase3-reference", ("float32", "float16", "bfloat16", "int32", "int64"), ("torch-reference",), "Single-rank reduce-scatter reference API.")
_reg("all_gather", "communication", "phase3-reference", ("float32", "float16", "bfloat16", "int32", "int64"), ("torch-reference",), "Single-rank all-gather reference API.")
_reg("fused_all_reduce_rmsnorm", "communication", "phase3-reference", ("float32", "float16", "bfloat16"), ("torch-reference",), "Fused all-reduce + RMSNorm reference API.")
_reg("gemm_all_reduce", "communication", "phase3-reference", ("float32", "float16", "bfloat16"), ("torch-reference",), "Fused GEMM + all-reduce reference API.")
_reg("gemm_reduce_scatter", "communication", "phase3-reference", ("float32", "float16", "bfloat16"), ("torch-reference",), "Fused GEMM + reduce-scatter reference API.")
_reg("all_gather_gemm", "communication", "phase3-reference", ("float32", "float16", "bfloat16"), ("torch-reference",), "Fused all-gather + GEMM reference API.")

# Phase 3 Backward
_reg("rmsnorm_backward_ref", "backward", "phase3-reference", ("float32", "float16", "bfloat16"), ("torch-autograd-reference",), "RMSNorm backward reference API.")
_reg("gemm_backward_ref", "backward", "phase3-reference", ("float32", "float16", "bfloat16"), ("torch-reference",), "GEMM backward reference API.")
_reg("attention_backward_ref", "backward", "phase3-reference", ("float32", "float16", "bfloat16"), ("torch-autograd-reference",), "Dense attention backward reference API.")
