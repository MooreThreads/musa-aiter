#include <stdint.h>
#include <math.h>
#include "../common/musa_compat.h"

static __device__ float musa_aiter_u32_to_float(uint32_t bits) {
  union {
    uint32_t u;
    float f;
  } v;
  v.u = bits;
  return v.f;
}

static __device__ uint32_t musa_aiter_float_to_u32(float value) {
  union {
    float f;
    uint32_t u;
  } v;
  v.f = value;
  return v.u;
}

static __device__ float musa_aiter_half_to_float(uint16_t h) {
  uint32_t sign = (static_cast<uint32_t>(h & 0x8000)) << 16;
  uint32_t exp = (h >> 10) & 0x1f;
  uint32_t mant = h & 0x03ff;
  if (exp == 0) {
    if (mant == 0) {
      return musa_aiter_u32_to_float(sign);
    }
    while ((mant & 0x0400) == 0) {
      mant <<= 1;
      exp--;
    }
    exp++;
    mant &= 0x03ff;
  } else if (exp == 31) {
    return musa_aiter_u32_to_float(sign | 0x7f800000u | (mant << 13));
  }
  exp = exp + (127 - 15);
  return musa_aiter_u32_to_float(sign | (exp << 23) | (mant << 13));
}

static __device__ uint16_t musa_aiter_float_to_half(float value) {
  uint32_t bits = musa_aiter_float_to_u32(value);
  uint32_t sign = (bits >> 16) & 0x8000;
  int32_t exp = static_cast<int32_t>((bits >> 23) & 0xff) - 127 + 15;
  uint32_t mant = bits & 0x7fffff;
  if (exp <= 0) {
    return static_cast<uint16_t>(sign);
  }
  if (exp >= 31) {
    return static_cast<uint16_t>(sign | 0x7c00);
  }
  mant = mant + 0x1000;
  if (mant & 0x800000) {
    mant = 0;
    exp += 1;
    if (exp >= 31) {
      return static_cast<uint16_t>(sign | 0x7c00);
    }
  }
  return static_cast<uint16_t>(sign | (static_cast<uint32_t>(exp) << 10) | (mant >> 13));
}

static __device__ float musa_aiter_bfloat16_to_float(uint16_t b) {
  return musa_aiter_u32_to_float(static_cast<uint32_t>(b) << 16);
}

static __device__ uint16_t musa_aiter_float_to_bfloat16(float value) {
  uint32_t bits = musa_aiter_float_to_u32(value);
  uint32_t lsb = (bits >> 16) & 1;
  bits += 0x7fffu + lsb;
  return static_cast<uint16_t>(bits >> 16);
}

static __device__ float musa_aiter_load_f16_or_bf16(const void* ptr, int64_t idx, int32_t dtype) {
  const uint16_t* p = static_cast<const uint16_t*>(ptr);
  if (dtype == 1) {
    return musa_aiter_half_to_float(p[idx]);
  }
  return musa_aiter_bfloat16_to_float(p[idx]);
}

static __device__ void musa_aiter_store_f16_or_bf16(void* ptr, int64_t idx, int32_t dtype, float value) {
  uint16_t* p = static_cast<uint16_t*>(ptr);
  if (dtype == 1) {
    p[idx] = musa_aiter_float_to_half(value);
  } else {
    p[idx] = musa_aiter_float_to_bfloat16(value);
  }
}

static __device__ unsigned char musa_aiter_float_to_e4m3fn(float value) {
  if (!isfinite(value)) {
    value = 0.0f;
  }
  unsigned char sign = value < 0.0f ? 0x80 : 0x00;
  float ax = fabsf(value);
  if (ax < 0.015625f) {
    return sign;
  }
  if (ax > 448.0f) {
    ax = 448.0f;
  }
  int exp = static_cast<int>(floorf(log2f(ax)));
  if (exp < -6) {
    exp = -6;
  }
  if (exp > 8) {
    exp = 8;
  }
  float base = ldexpf(1.0f, exp);
  int mant = static_cast<int>(floorf((ax / base - 1.0f) * 8.0f + 0.5f));
  if (mant >= 8) {
    mant = 0;
    exp += 1;
    if (exp > 8) {
      exp = 8;
      mant = 7;
    }
  }
  int exp_field = exp + 7;
  if (exp_field < 0) {
    exp_field = 0;
    mant = 0;
  }
  if (exp_field > 15) {
    exp_field = 15;
    mant = 7;
  }
  return static_cast<unsigned char>(sign | (exp_field << 3) | (mant & 0x7));
}

static __device__ float musa_aiter_e4m3fn_to_float(unsigned char bits) {
  int sign = (bits >> 7) & 1;
  int exp_field = (bits >> 3) & 0xf;
  int mant = bits & 0x7;
  if (exp_field == 0) {
    return sign ? -0.0f : 0.0f;
  }
  float value = (1.0f + static_cast<float>(mant) / 8.0f) * ldexpf(1.0f, exp_field - 7);
  return sign ? -value : value;
}

extern "C" __global__ void musa_aiter_kv_linear_to_paged_fp8_e4m3_kernel(
    const void* key_linear,
    const void* value_linear,
    const int64_t* slot_mapping,
    unsigned char* key_cache,
    unsigned char* value_cache,
    const float* k_scale,
    const float* v_scale,
    int64_t num_tokens,
    int64_t num_blocks,
    int64_t num_kv_heads,
    int64_t head_dim,
    int64_t block_size,
    int32_t src_dtype) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  int64_t elems = num_tokens * num_kv_heads * head_dim;
  if (idx >= elems) {
    return;
  }
  int64_t dim = idx % head_dim;
  int64_t tmp = idx / head_dim;
  int64_t head = tmp % num_kv_heads;
  int64_t token = tmp / num_kv_heads;
  int64_t slot = slot_mapping[token];
  if (slot < 0) {
    return;
  }
  int64_t block = slot / block_size;
  int64_t offset = slot - block * block_size;
  if (block < 0 || block >= num_blocks) {
    return;
  }
  int64_t cache_idx = (((block * num_kv_heads + head) * head_dim + dim) * block_size + offset);
  key_cache[cache_idx] = musa_aiter_float_to_e4m3fn(musa_aiter_load_f16_or_bf16(key_linear, idx, src_dtype) / k_scale[0]);
  value_cache[cache_idx] = musa_aiter_float_to_e4m3fn(musa_aiter_load_f16_or_bf16(value_linear, idx, src_dtype) / v_scale[0]);
}

extern "C" __global__ void musa_aiter_kv_paged_fp8_to_linear_e4m3_kernel(
    const unsigned char* key_cache,
    const unsigned char* value_cache,
    const int64_t* slot_mapping,
    const float* k_scale,
    const float* v_scale,
    void* out_k,
    void* out_v,
    int64_t num_tokens,
    int64_t num_blocks,
    int64_t num_kv_heads,
    int64_t head_dim,
    int64_t block_size,
    int32_t out_dtype) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  int64_t elems = num_tokens * num_kv_heads * head_dim;
  if (idx >= elems) {
    return;
  }
  int64_t dim = idx % head_dim;
  int64_t tmp = idx / head_dim;
  int64_t head = tmp % num_kv_heads;
  int64_t token = tmp / num_kv_heads;
  int64_t slot = slot_mapping[token];
  if (slot < 0) {
    musa_aiter_store_f16_or_bf16(out_k, idx, out_dtype, 0.0f);
    musa_aiter_store_f16_or_bf16(out_v, idx, out_dtype, 0.0f);
    return;
  }
  int64_t block = slot / block_size;
  int64_t offset = slot - block * block_size;
  if (block < 0 || block >= num_blocks) {
    musa_aiter_store_f16_or_bf16(out_k, idx, out_dtype, 0.0f);
    musa_aiter_store_f16_or_bf16(out_v, idx, out_dtype, 0.0f);
    return;
  }
  int64_t cache_idx = (((block * num_kv_heads + head) * head_dim + dim) * block_size + offset);
  musa_aiter_store_f16_or_bf16(out_k, idx, out_dtype, musa_aiter_e4m3fn_to_float(key_cache[cache_idx]) * k_scale[0]);
  musa_aiter_store_f16_or_bf16(out_v, idx, out_dtype, musa_aiter_e4m3fn_to_float(value_cache[cache_idx]) * v_scale[0]);
}

extern "C" void musa_aiter_kv_linear_to_paged_fp8_e4m3(
    const void* key_linear,
    const void* value_linear,
    const int64_t* slot_mapping,
    unsigned char* key_cache,
    unsigned char* value_cache,
    const float* k_scale,
    const float* v_scale,
    int64_t num_tokens,
    int64_t num_blocks,
    int64_t num_kv_heads,
    int64_t head_dim,
    int64_t block_size,
    int32_t src_dtype,
    gpuStream_t stream) {
#ifdef __MUSACC__
  int threads = 256;
  int64_t elems = num_tokens * num_kv_heads * head_dim;
  int blocks = static_cast<int>((elems + threads - 1) / threads);
  musa_aiter_kv_linear_to_paged_fp8_e4m3_kernel<<<blocks, threads, 0, stream>>>(
      key_linear, value_linear, slot_mapping, key_cache, value_cache, k_scale, v_scale,
      num_tokens, num_blocks, num_kv_heads, head_dim, block_size, src_dtype);
#endif
}

extern "C" void musa_aiter_kv_paged_fp8_to_linear_e4m3(
    const unsigned char* key_cache,
    const unsigned char* value_cache,
    const int64_t* slot_mapping,
    const float* k_scale,
    const float* v_scale,
    void* out_k,
    void* out_v,
    int64_t num_tokens,
    int64_t num_blocks,
    int64_t num_kv_heads,
    int64_t head_dim,
    int64_t block_size,
    int32_t out_dtype,
    gpuStream_t stream) {
#ifdef __MUSACC__
  int threads = 256;
  int64_t elems = num_tokens * num_kv_heads * head_dim;
  int blocks = static_cast<int>((elems + threads - 1) / threads);
  musa_aiter_kv_paged_fp8_to_linear_e4m3_kernel<<<blocks, threads, 0, stream>>>(
      key_cache, value_cache, slot_mapping, k_scale, v_scale, out_k, out_v,
      num_tokens, num_blocks, num_kv_heads, head_dim, block_size, out_dtype);
#endif
}

static __device__ unsigned char musa_aiter_float_to_e5m2(float value) {
  if (!isfinite(value)) {
    value = 0.0f;
  }
  unsigned char sign = value < 0.0f ? 0x80 : 0x00;
  float ax = fabsf(value);
  if (ax < 0.00006103515625f) {
    return sign;
  }
  if (ax > 57344.0f) {
    ax = 57344.0f;
  }
  int exp = static_cast<int>(floorf(log2f(ax)));
  if (exp < -14) {
    exp = -14;
  }
  if (exp > 15) {
    exp = 15;
  }
  float base = ldexpf(1.0f, exp);
  int mant = static_cast<int>(floorf((ax / base - 1.0f) * 4.0f + 0.5f));
  if (mant >= 4) {
    mant = 0;
    exp += 1;
    if (exp > 15) {
      exp = 15;
      mant = 3;
    }
  }
  int exp_field = exp + 15;
  if (exp_field < 0) {
    exp_field = 0;
    mant = 0;
  }
  if (exp_field > 31) {
    exp_field = 31;
    mant = 3;
  }
  return static_cast<unsigned char>(sign | (exp_field << 2) | (mant & 0x3));
}

static __device__ float musa_aiter_e5m2_to_float(unsigned char bits) {
  int sign = (bits >> 7) & 1;
  int exp_field = (bits >> 2) & 0x1f;
  int mant = bits & 0x3;
  if (exp_field == 0) {
    return sign ? -0.0f : 0.0f;
  }
  float value = (1.0f + static_cast<float>(mant) / 4.0f) * ldexpf(1.0f, exp_field - 15);
  return sign ? -value : value;
}

extern "C" __global__ void musa_aiter_kv_linear_to_paged_fp8_e5m2_kernel(
    const void* key_linear,
    const void* value_linear,
    const int64_t* slot_mapping,
    unsigned char* key_cache,
    unsigned char* value_cache,
    const float* k_scale,
    const float* v_scale,
    int64_t num_tokens,
    int64_t num_blocks,
    int64_t num_kv_heads,
    int64_t head_dim,
    int64_t block_size,
    int32_t src_dtype) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  int64_t elems = num_tokens * num_kv_heads * head_dim;
  if (idx >= elems) {
    return;
  }
  int64_t dim = idx % head_dim;
  int64_t tmp = idx / head_dim;
  int64_t head = tmp % num_kv_heads;
  int64_t token = tmp / num_kv_heads;
  int64_t slot = slot_mapping[token];
  if (slot < 0) {
    return;
  }
  int64_t block = slot / block_size;
  int64_t offset = slot - block * block_size;
  if (block < 0 || block >= num_blocks) {
    return;
  }
  int64_t cache_idx = (((block * num_kv_heads + head) * head_dim + dim) * block_size + offset);
  key_cache[cache_idx] = musa_aiter_float_to_e5m2(musa_aiter_load_f16_or_bf16(key_linear, idx, src_dtype) / k_scale[0]);
  value_cache[cache_idx] = musa_aiter_float_to_e5m2(musa_aiter_load_f16_or_bf16(value_linear, idx, src_dtype) / v_scale[0]);
}

extern "C" __global__ void musa_aiter_kv_paged_fp8_to_linear_e5m2_kernel(
    const unsigned char* key_cache,
    const unsigned char* value_cache,
    const int64_t* slot_mapping,
    const float* k_scale,
    const float* v_scale,
    void* out_k,
    void* out_v,
    int64_t num_tokens,
    int64_t num_blocks,
    int64_t num_kv_heads,
    int64_t head_dim,
    int64_t block_size,
    int32_t out_dtype) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  int64_t elems = num_tokens * num_kv_heads * head_dim;
  if (idx >= elems) {
    return;
  }
  int64_t dim = idx % head_dim;
  int64_t tmp = idx / head_dim;
  int64_t head = tmp % num_kv_heads;
  int64_t token = tmp / num_kv_heads;
  int64_t slot = slot_mapping[token];
  if (slot < 0) {
    musa_aiter_store_f16_or_bf16(out_k, idx, out_dtype, 0.0f);
    musa_aiter_store_f16_or_bf16(out_v, idx, out_dtype, 0.0f);
    return;
  }
  int64_t block = slot / block_size;
  int64_t offset = slot - block * block_size;
  if (block < 0 || block >= num_blocks) {
    musa_aiter_store_f16_or_bf16(out_k, idx, out_dtype, 0.0f);
    musa_aiter_store_f16_or_bf16(out_v, idx, out_dtype, 0.0f);
    return;
  }
  int64_t cache_idx = (((block * num_kv_heads + head) * head_dim + dim) * block_size + offset);
  musa_aiter_store_f16_or_bf16(out_k, idx, out_dtype, musa_aiter_e5m2_to_float(key_cache[cache_idx]) * k_scale[0]);
  musa_aiter_store_f16_or_bf16(out_v, idx, out_dtype, musa_aiter_e5m2_to_float(value_cache[cache_idx]) * v_scale[0]);
}

extern "C" void musa_aiter_kv_linear_to_paged_fp8_e5m2(
    const void* key_linear,
    const void* value_linear,
    const int64_t* slot_mapping,
    unsigned char* key_cache,
    unsigned char* value_cache,
    const float* k_scale,
    const float* v_scale,
    int64_t num_tokens,
    int64_t num_blocks,
    int64_t num_kv_heads,
    int64_t head_dim,
    int64_t block_size,
    int32_t src_dtype,
    gpuStream_t stream) {
#ifdef __MUSACC__
  int threads = 256;
  int64_t elems = num_tokens * num_kv_heads * head_dim;
  int blocks = static_cast<int>((elems + threads - 1) / threads);
  musa_aiter_kv_linear_to_paged_fp8_e5m2_kernel<<<blocks, threads, 0, stream>>>(
      key_linear, value_linear, slot_mapping, key_cache, value_cache, k_scale, v_scale,
      num_tokens, num_blocks, num_kv_heads, head_dim, block_size, src_dtype);
#endif
}

extern "C" void musa_aiter_kv_paged_fp8_to_linear_e5m2(
    const unsigned char* key_cache,
    const unsigned char* value_cache,
    const int64_t* slot_mapping,
    const float* k_scale,
    const float* v_scale,
    void* out_k,
    void* out_v,
    int64_t num_tokens,
    int64_t num_blocks,
    int64_t num_kv_heads,
    int64_t head_dim,
    int64_t block_size,
    int32_t out_dtype,
    gpuStream_t stream) {
#ifdef __MUSACC__
  int threads = 256;
  int64_t elems = num_tokens * num_kv_heads * head_dim;
  int blocks = static_cast<int>((elems + threads - 1) / threads);
  musa_aiter_kv_paged_fp8_to_linear_e5m2_kernel<<<blocks, threads, 0, stream>>>(
      key_cache, value_cache, slot_mapping, k_scale, v_scale, out_k, out_v,
      num_tokens, num_blocks, num_kv_heads, head_dim, block_size, out_dtype);
#endif
}

extern "C" __global__ void musa_aiter_kv_linear_to_paged_fp8_multi_layer_e4m3_kernel(
    const void* key_linear,
    const void* value_linear,
    const int64_t* slot_mapping,
    unsigned char* key_cache,
    unsigned char* value_cache,
    const float* k_scale,
    const float* v_scale,
    int64_t num_layers,
    int64_t num_tokens,
    int64_t num_blocks,
    int64_t num_kv_heads,
    int64_t head_dim,
    int64_t block_size,
    int32_t src_dtype,
    int32_t scale_mode) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  int64_t elems = num_layers * num_tokens * num_kv_heads * head_dim;
  if (idx >= elems) {
    return;
  }
  int64_t dim = idx % head_dim;
  int64_t tmp = idx / head_dim;
  int64_t head = tmp % num_kv_heads;
  tmp = tmp / num_kv_heads;
  int64_t token = tmp % num_tokens;
  int64_t layer = tmp / num_tokens;
  int64_t slot = slot_mapping[token];
  if (slot < 0) {
    return;
  }
  int64_t block = slot / block_size;
  int64_t offset = slot - block * block_size;
  if (block < 0 || block >= num_blocks) {
    return;
  }
  int64_t scale_idx = scale_mode == 0 ? 0 : (scale_mode == 1 ? head : layer * num_kv_heads + head);
  int64_t linear_idx = (((layer * num_tokens + token) * num_kv_heads + head) * head_dim + dim);
  int64_t cache_idx = ((((layer * num_blocks + block) * num_kv_heads + head) * head_dim + dim) * block_size + offset);
  key_cache[cache_idx] = musa_aiter_float_to_e4m3fn(musa_aiter_load_f16_or_bf16(key_linear, linear_idx, src_dtype) / k_scale[scale_idx]);
  value_cache[cache_idx] = musa_aiter_float_to_e4m3fn(musa_aiter_load_f16_or_bf16(value_linear, linear_idx, src_dtype) / v_scale[scale_idx]);
}

extern "C" __global__ void musa_aiter_kv_paged_fp8_to_linear_multi_layer_e4m3_kernel(
    const unsigned char* key_cache,
    const unsigned char* value_cache,
    const int64_t* slot_mapping,
    const float* k_scale,
    const float* v_scale,
    void* out_k,
    void* out_v,
    int64_t num_layers,
    int64_t num_tokens,
    int64_t num_blocks,
    int64_t num_kv_heads,
    int64_t head_dim,
    int64_t block_size,
    int32_t out_dtype,
    int32_t scale_mode) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  int64_t elems = num_layers * num_tokens * num_kv_heads * head_dim;
  if (idx >= elems) {
    return;
  }
  int64_t dim = idx % head_dim;
  int64_t tmp = idx / head_dim;
  int64_t head = tmp % num_kv_heads;
  tmp = tmp / num_kv_heads;
  int64_t token = tmp % num_tokens;
  int64_t layer = tmp / num_tokens;
  int64_t out_idx = (((layer * num_tokens + token) * num_kv_heads + head) * head_dim + dim);
  int64_t slot = slot_mapping[token];
  if (slot < 0) {
    musa_aiter_store_f16_or_bf16(out_k, out_idx, out_dtype, 0.0f);
    musa_aiter_store_f16_or_bf16(out_v, out_idx, out_dtype, 0.0f);
    return;
  }
  int64_t block = slot / block_size;
  int64_t offset = slot - block * block_size;
  if (block < 0 || block >= num_blocks) {
    musa_aiter_store_f16_or_bf16(out_k, out_idx, out_dtype, 0.0f);
    musa_aiter_store_f16_or_bf16(out_v, out_idx, out_dtype, 0.0f);
    return;
  }
  int64_t cache_idx = ((((layer * num_blocks + block) * num_kv_heads + head) * head_dim + dim) * block_size + offset);
  int64_t scale_idx = scale_mode == 0 ? 0 : (scale_mode == 1 ? head : layer * num_kv_heads + head);
  musa_aiter_store_f16_or_bf16(out_k, out_idx, out_dtype, musa_aiter_e4m3fn_to_float(key_cache[cache_idx]) * k_scale[scale_idx]);
  musa_aiter_store_f16_or_bf16(out_v, out_idx, out_dtype, musa_aiter_e4m3fn_to_float(value_cache[cache_idx]) * v_scale[scale_idx]);
}

extern "C" __global__ void musa_aiter_kv_linear_to_paged_fp8_multi_layer_e5m2_kernel(
    const void* key_linear,
    const void* value_linear,
    const int64_t* slot_mapping,
    unsigned char* key_cache,
    unsigned char* value_cache,
    const float* k_scale,
    const float* v_scale,
    int64_t num_layers,
    int64_t num_tokens,
    int64_t num_blocks,
    int64_t num_kv_heads,
    int64_t head_dim,
    int64_t block_size,
    int32_t src_dtype,
    int32_t scale_mode) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  int64_t elems = num_layers * num_tokens * num_kv_heads * head_dim;
  if (idx >= elems) {
    return;
  }
  int64_t dim = idx % head_dim;
  int64_t tmp = idx / head_dim;
  int64_t head = tmp % num_kv_heads;
  tmp = tmp / num_kv_heads;
  int64_t token = tmp % num_tokens;
  int64_t layer = tmp / num_tokens;
  int64_t slot = slot_mapping[token];
  if (slot < 0) {
    return;
  }
  int64_t block = slot / block_size;
  int64_t offset = slot - block * block_size;
  if (block < 0 || block >= num_blocks) {
    return;
  }
  int64_t scale_idx = scale_mode == 0 ? 0 : (scale_mode == 1 ? head : layer * num_kv_heads + head);
  int64_t linear_idx = (((layer * num_tokens + token) * num_kv_heads + head) * head_dim + dim);
  int64_t cache_idx = ((((layer * num_blocks + block) * num_kv_heads + head) * head_dim + dim) * block_size + offset);
  key_cache[cache_idx] = musa_aiter_float_to_e5m2(musa_aiter_load_f16_or_bf16(key_linear, linear_idx, src_dtype) / k_scale[scale_idx]);
  value_cache[cache_idx] = musa_aiter_float_to_e5m2(musa_aiter_load_f16_or_bf16(value_linear, linear_idx, src_dtype) / v_scale[scale_idx]);
}

extern "C" __global__ void musa_aiter_kv_paged_fp8_to_linear_multi_layer_e5m2_kernel(
    const unsigned char* key_cache,
    const unsigned char* value_cache,
    const int64_t* slot_mapping,
    const float* k_scale,
    const float* v_scale,
    void* out_k,
    void* out_v,
    int64_t num_layers,
    int64_t num_tokens,
    int64_t num_blocks,
    int64_t num_kv_heads,
    int64_t head_dim,
    int64_t block_size,
    int32_t out_dtype,
    int32_t scale_mode) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  int64_t elems = num_layers * num_tokens * num_kv_heads * head_dim;
  if (idx >= elems) {
    return;
  }
  int64_t dim = idx % head_dim;
  int64_t tmp = idx / head_dim;
  int64_t head = tmp % num_kv_heads;
  tmp = tmp / num_kv_heads;
  int64_t token = tmp % num_tokens;
  int64_t layer = tmp / num_tokens;
  int64_t out_idx = (((layer * num_tokens + token) * num_kv_heads + head) * head_dim + dim);
  int64_t slot = slot_mapping[token];
  if (slot < 0) {
    musa_aiter_store_f16_or_bf16(out_k, out_idx, out_dtype, 0.0f);
    musa_aiter_store_f16_or_bf16(out_v, out_idx, out_dtype, 0.0f);
    return;
  }
  int64_t block = slot / block_size;
  int64_t offset = slot - block * block_size;
  if (block < 0 || block >= num_blocks) {
    musa_aiter_store_f16_or_bf16(out_k, out_idx, out_dtype, 0.0f);
    musa_aiter_store_f16_or_bf16(out_v, out_idx, out_dtype, 0.0f);
    return;
  }
  int64_t cache_idx = ((((layer * num_blocks + block) * num_kv_heads + head) * head_dim + dim) * block_size + offset);
  int64_t scale_idx = scale_mode == 0 ? 0 : (scale_mode == 1 ? head : layer * num_kv_heads + head);
  musa_aiter_store_f16_or_bf16(out_k, out_idx, out_dtype, musa_aiter_e5m2_to_float(key_cache[cache_idx]) * k_scale[scale_idx]);
  musa_aiter_store_f16_or_bf16(out_v, out_idx, out_dtype, musa_aiter_e5m2_to_float(value_cache[cache_idx]) * v_scale[scale_idx]);
}

extern "C" void musa_aiter_kv_linear_to_paged_fp8_multi_layer_e4m3(
    const void* key_linear,
    const void* value_linear,
    const int64_t* slot_mapping,
    unsigned char* key_cache,
    unsigned char* value_cache,
    const float* k_scale,
    const float* v_scale,
    int64_t num_layers,
    int64_t num_tokens,
    int64_t num_blocks,
    int64_t num_kv_heads,
    int64_t head_dim,
    int64_t block_size,
    int32_t src_dtype,
    gpuStream_t stream,
    int32_t scale_mode) {
#ifdef __MUSACC__
  int threads = 256;
  int64_t elems = num_layers * num_tokens * num_kv_heads * head_dim;
  int blocks = static_cast<int>((elems + threads - 1) / threads);
  musa_aiter_kv_linear_to_paged_fp8_multi_layer_e4m3_kernel<<<blocks, threads, 0, stream>>>(
      key_linear, value_linear, slot_mapping, key_cache, value_cache, k_scale, v_scale,
      num_layers, num_tokens, num_blocks, num_kv_heads, head_dim, block_size, src_dtype, scale_mode);
#endif
}

extern "C" void musa_aiter_kv_paged_fp8_to_linear_multi_layer_e4m3(
    const unsigned char* key_cache,
    const unsigned char* value_cache,
    const int64_t* slot_mapping,
    const float* k_scale,
    const float* v_scale,
    void* out_k,
    void* out_v,
    int64_t num_layers,
    int64_t num_tokens,
    int64_t num_blocks,
    int64_t num_kv_heads,
    int64_t head_dim,
    int64_t block_size,
    int32_t out_dtype,
    gpuStream_t stream,
    int32_t scale_mode) {
#ifdef __MUSACC__
  int threads = 256;
  int64_t elems = num_layers * num_tokens * num_kv_heads * head_dim;
  int blocks = static_cast<int>((elems + threads - 1) / threads);
  musa_aiter_kv_paged_fp8_to_linear_multi_layer_e4m3_kernel<<<blocks, threads, 0, stream>>>(
      key_cache, value_cache, slot_mapping, k_scale, v_scale, out_k, out_v,
      num_layers, num_tokens, num_blocks, num_kv_heads, head_dim, block_size, out_dtype, scale_mode);
#endif
}

extern "C" void musa_aiter_kv_linear_to_paged_fp8_multi_layer_e5m2(
    const void* key_linear,
    const void* value_linear,
    const int64_t* slot_mapping,
    unsigned char* key_cache,
    unsigned char* value_cache,
    const float* k_scale,
    const float* v_scale,
    int64_t num_layers,
    int64_t num_tokens,
    int64_t num_blocks,
    int64_t num_kv_heads,
    int64_t head_dim,
    int64_t block_size,
    int32_t src_dtype,
    gpuStream_t stream,
    int32_t scale_mode) {
#ifdef __MUSACC__
  int threads = 256;
  int64_t elems = num_layers * num_tokens * num_kv_heads * head_dim;
  int blocks = static_cast<int>((elems + threads - 1) / threads);
  musa_aiter_kv_linear_to_paged_fp8_multi_layer_e5m2_kernel<<<blocks, threads, 0, stream>>>(
      key_linear, value_linear, slot_mapping, key_cache, value_cache, k_scale, v_scale,
      num_layers, num_tokens, num_blocks, num_kv_heads, head_dim, block_size, src_dtype, scale_mode);
#endif
}

extern "C" void musa_aiter_kv_paged_fp8_to_linear_multi_layer_e5m2(
    const unsigned char* key_cache,
    const unsigned char* value_cache,
    const int64_t* slot_mapping,
    const float* k_scale,
    const float* v_scale,
    void* out_k,
    void* out_v,
    int64_t num_layers,
    int64_t num_tokens,
    int64_t num_blocks,
    int64_t num_kv_heads,
    int64_t head_dim,
    int64_t block_size,
    int32_t out_dtype,
    gpuStream_t stream,
    int32_t scale_mode) {
#ifdef __MUSACC__
  int threads = 256;
  int64_t elems = num_layers * num_tokens * num_kv_heads * head_dim;
  int blocks = static_cast<int>((elems + threads - 1) / threads);
  musa_aiter_kv_paged_fp8_to_linear_multi_layer_e5m2_kernel<<<blocks, threads, 0, stream>>>(
      key_cache, value_cache, slot_mapping, k_scale, v_scale, out_k, out_v,
      num_layers, num_tokens, num_blocks, num_kv_heads, head_dim, block_size, out_dtype, scale_mode);
#endif
}

extern "C" __global__ void musa_aiter_kv_linear_to_paged_fp8_vec4_e4m3_kernel(
    const void* key_linear,
    const void* value_linear,
    const int64_t* slot_mapping,
    unsigned char* key_cache,
    unsigned char* value_cache,
    const float* k_scale,
    const float* v_scale,
    int64_t num_tokens,
    int64_t num_blocks,
    int64_t num_kv_heads,
    int64_t head_dim,
    int64_t block_size,
    int32_t src_dtype,
    int32_t scale_mode) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  int64_t packs_per_head = head_dim / 4;
  int64_t elems = num_tokens * num_kv_heads * packs_per_head;
  if (idx >= elems) {
    return;
  }
  int64_t pack = idx % packs_per_head;
  int64_t dim0 = pack * 4;
  int64_t tmp = idx / packs_per_head;
  int64_t head = tmp % num_kv_heads;
  int64_t token = tmp / num_kv_heads;
  int64_t slot = slot_mapping[token];
  if (slot < 0) {
    return;
  }
  int64_t block = slot / block_size;
  int64_t offset = slot - block * block_size;
  if (block < 0 || block >= num_blocks) {
    return;
  }
  int64_t linear_base = (token * num_kv_heads + head) * head_dim + dim0;
  float ks = k_scale[scale_mode == 0 ? 0 : head];
  float vs = v_scale[scale_mode == 0 ? 0 : head];
  for (int i = 0; i < 4; ++i) {
    int64_t cache_idx = (((block * num_kv_heads + head) * head_dim + dim0 + i) * block_size + offset);
    key_cache[cache_idx] = musa_aiter_float_to_e4m3fn(musa_aiter_load_f16_or_bf16(key_linear, linear_base + i, src_dtype) / ks);
    value_cache[cache_idx] = musa_aiter_float_to_e4m3fn(musa_aiter_load_f16_or_bf16(value_linear, linear_base + i, src_dtype) / vs);
  }
}

extern "C" __global__ void musa_aiter_kv_linear_to_paged_fp8_vec4_e5m2_kernel(
    const void* key_linear,
    const void* value_linear,
    const int64_t* slot_mapping,
    unsigned char* key_cache,
    unsigned char* value_cache,
    const float* k_scale,
    const float* v_scale,
    int64_t num_tokens,
    int64_t num_blocks,
    int64_t num_kv_heads,
    int64_t head_dim,
    int64_t block_size,
    int32_t src_dtype,
    int32_t scale_mode) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  int64_t packs_per_head = head_dim / 4;
  int64_t elems = num_tokens * num_kv_heads * packs_per_head;
  if (idx >= elems) {
    return;
  }
  int64_t pack = idx % packs_per_head;
  int64_t dim0 = pack * 4;
  int64_t tmp = idx / packs_per_head;
  int64_t head = tmp % num_kv_heads;
  int64_t token = tmp / num_kv_heads;
  int64_t slot = slot_mapping[token];
  if (slot < 0) {
    return;
  }
  int64_t block = slot / block_size;
  int64_t offset = slot - block * block_size;
  if (block < 0 || block >= num_blocks) {
    return;
  }
  int64_t linear_base = (token * num_kv_heads + head) * head_dim + dim0;
  float ks = k_scale[scale_mode == 0 ? 0 : head];
  float vs = v_scale[scale_mode == 0 ? 0 : head];
  for (int i = 0; i < 4; ++i) {
    int64_t cache_idx = (((block * num_kv_heads + head) * head_dim + dim0 + i) * block_size + offset);
    key_cache[cache_idx] = musa_aiter_float_to_e5m2(musa_aiter_load_f16_or_bf16(key_linear, linear_base + i, src_dtype) / ks);
    value_cache[cache_idx] = musa_aiter_float_to_e5m2(musa_aiter_load_f16_or_bf16(value_linear, linear_base + i, src_dtype) / vs);
  }
}

extern "C" void musa_aiter_kv_linear_to_paged_fp8_vec4_e4m3(
    const void* key_linear,
    const void* value_linear,
    const int64_t* slot_mapping,
    unsigned char* key_cache,
    unsigned char* value_cache,
    const float* k_scale,
    const float* v_scale,
    int64_t num_tokens,
    int64_t num_blocks,
    int64_t num_kv_heads,
    int64_t head_dim,
    int64_t block_size,
    int32_t src_dtype,
    gpuStream_t stream,
    int32_t scale_mode) {
#ifdef __MUSACC__
  int threads = 256;
  int64_t elems = num_tokens * num_kv_heads * (head_dim / 4);
  int blocks = static_cast<int>((elems + threads - 1) / threads);
  musa_aiter_kv_linear_to_paged_fp8_vec4_e4m3_kernel<<<blocks, threads, 0, stream>>>(
      key_linear, value_linear, slot_mapping, key_cache, value_cache, k_scale, v_scale,
      num_tokens, num_blocks, num_kv_heads, head_dim, block_size, src_dtype, scale_mode);
#endif
}

extern "C" void musa_aiter_kv_linear_to_paged_fp8_vec4_e5m2(
    const void* key_linear,
    const void* value_linear,
    const int64_t* slot_mapping,
    unsigned char* key_cache,
    unsigned char* value_cache,
    const float* k_scale,
    const float* v_scale,
    int64_t num_tokens,
    int64_t num_blocks,
    int64_t num_kv_heads,
    int64_t head_dim,
    int64_t block_size,
    int32_t src_dtype,
    gpuStream_t stream,
    int32_t scale_mode) {
#ifdef __MUSACC__
  int threads = 256;
  int64_t elems = num_tokens * num_kv_heads * (head_dim / 4);
  int blocks = static_cast<int>((elems + threads - 1) / threads);
  musa_aiter_kv_linear_to_paged_fp8_vec4_e5m2_kernel<<<blocks, threads, 0, stream>>>(
      key_linear, value_linear, slot_mapping, key_cache, value_cache, k_scale, v_scale,
      num_tokens, num_blocks, num_kv_heads, head_dim, block_size, src_dtype, scale_mode);
#endif
}

extern "C" __global__ void musa_aiter_kv_paged_fp8_to_linear_hd64_bs16_e4m3_kernel(
    const unsigned char* key_cache,
    const unsigned char* value_cache,
    const int64_t* slot_mapping,
    const float* k_scale,
    const float* v_scale,
    void* out_k,
    void* out_v,
    int64_t num_tokens,
    int64_t num_blocks,
    int64_t num_kv_heads,
    int32_t out_dtype) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  int64_t elems = num_tokens * num_kv_heads * 16;
  if (idx >= elems) {
    return;
  }
  int64_t pack = idx & 15;
  int64_t dim0 = pack << 2;
  int64_t tmp = idx >> 4;
  int64_t head = tmp % num_kv_heads;
  int64_t token = tmp / num_kv_heads;
  int64_t out_base = (token * num_kv_heads + head) * 64 + dim0;
  int64_t slot = slot_mapping[token];
  if (slot < 0) {
    for (int i = 0; i < 4; ++i) {
      musa_aiter_store_f16_or_bf16(out_k, out_base + i, out_dtype, 0.0f);
      musa_aiter_store_f16_or_bf16(out_v, out_base + i, out_dtype, 0.0f);
    }
    return;
  }
  int64_t block = slot >> 4;
  int64_t offset = slot & 15;
  if (block < 0 || block >= num_blocks) {
    for (int i = 0; i < 4; ++i) {
      musa_aiter_store_f16_or_bf16(out_k, out_base + i, out_dtype, 0.0f);
      musa_aiter_store_f16_or_bf16(out_v, out_base + i, out_dtype, 0.0f);
    }
    return;
  }
  for (int i = 0; i < 4; ++i) {
    int64_t cache_idx = (((block * num_kv_heads + head) * 64 + dim0 + i) * 16 + offset);
    musa_aiter_store_f16_or_bf16(out_k, out_base + i, out_dtype, musa_aiter_e4m3fn_to_float(key_cache[cache_idx]) * k_scale[0]);
    musa_aiter_store_f16_or_bf16(out_v, out_base + i, out_dtype, musa_aiter_e4m3fn_to_float(value_cache[cache_idx]) * v_scale[0]);
  }
}

extern "C" __global__ void musa_aiter_kv_paged_fp8_to_linear_hd64_bs16_e5m2_kernel(
    const unsigned char* key_cache,
    const unsigned char* value_cache,
    const int64_t* slot_mapping,
    const float* k_scale,
    const float* v_scale,
    void* out_k,
    void* out_v,
    int64_t num_tokens,
    int64_t num_blocks,
    int64_t num_kv_heads,
    int32_t out_dtype) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  int64_t elems = num_tokens * num_kv_heads * 16;
  if (idx >= elems) {
    return;
  }
  int64_t pack = idx & 15;
  int64_t dim0 = pack << 2;
  int64_t tmp = idx >> 4;
  int64_t head = tmp % num_kv_heads;
  int64_t token = tmp / num_kv_heads;
  int64_t out_base = (token * num_kv_heads + head) * 64 + dim0;
  int64_t slot = slot_mapping[token];
  if (slot < 0) {
    for (int i = 0; i < 4; ++i) {
      musa_aiter_store_f16_or_bf16(out_k, out_base + i, out_dtype, 0.0f);
      musa_aiter_store_f16_or_bf16(out_v, out_base + i, out_dtype, 0.0f);
    }
    return;
  }
  int64_t block = slot >> 4;
  int64_t offset = slot & 15;
  if (block < 0 || block >= num_blocks) {
    for (int i = 0; i < 4; ++i) {
      musa_aiter_store_f16_or_bf16(out_k, out_base + i, out_dtype, 0.0f);
      musa_aiter_store_f16_or_bf16(out_v, out_base + i, out_dtype, 0.0f);
    }
    return;
  }
  for (int i = 0; i < 4; ++i) {
    int64_t cache_idx = (((block * num_kv_heads + head) * 64 + dim0 + i) * 16 + offset);
    musa_aiter_store_f16_or_bf16(out_k, out_base + i, out_dtype, musa_aiter_e5m2_to_float(key_cache[cache_idx]) * k_scale[0]);
    musa_aiter_store_f16_or_bf16(out_v, out_base + i, out_dtype, musa_aiter_e5m2_to_float(value_cache[cache_idx]) * v_scale[0]);
  }
}

extern "C" void musa_aiter_kv_paged_fp8_to_linear_hd64_bs16_e4m3(
    const unsigned char* key_cache,
    const unsigned char* value_cache,
    const int64_t* slot_mapping,
    const float* k_scale,
    const float* v_scale,
    void* out_k,
    void* out_v,
    int64_t num_tokens,
    int64_t num_blocks,
    int64_t num_kv_heads,
    int32_t out_dtype,
    gpuStream_t stream) {
#ifdef __MUSACC__
  int threads = 256;
  int64_t elems = num_tokens * num_kv_heads * 16;
  int blocks = static_cast<int>((elems + threads - 1) / threads);
  musa_aiter_kv_paged_fp8_to_linear_hd64_bs16_e4m3_kernel<<<blocks, threads, 0, stream>>>(
      key_cache, value_cache, slot_mapping, k_scale, v_scale, out_k, out_v,
      num_tokens, num_blocks, num_kv_heads, out_dtype);
#endif
}

extern "C" void musa_aiter_kv_paged_fp8_to_linear_hd64_bs16_e5m2(
    const unsigned char* key_cache,
    const unsigned char* value_cache,
    const int64_t* slot_mapping,
    const float* k_scale,
    const float* v_scale,
    void* out_k,
    void* out_v,
    int64_t num_tokens,
    int64_t num_blocks,
    int64_t num_kv_heads,
    int32_t out_dtype,
    gpuStream_t stream) {
#ifdef __MUSACC__
  int threads = 256;
  int64_t elems = num_tokens * num_kv_heads * 16;
  int blocks = static_cast<int>((elems + threads - 1) / threads);
  musa_aiter_kv_paged_fp8_to_linear_hd64_bs16_e5m2_kernel<<<blocks, threads, 0, stream>>>(
      key_cache, value_cache, slot_mapping, k_scale, v_scale, out_k, out_v,
      num_tokens, num_blocks, num_kv_heads, out_dtype);
#endif
}

extern "C" __global__ void musa_aiter_kv_paged_fp8_to_linear_hd128_bs32_e4m3_kernel(
    const unsigned char* key_cache,
    const unsigned char* value_cache,
    const int64_t* slot_mapping,
    const float* k_scale,
    const float* v_scale,
    void* out_k,
    void* out_v,
    int64_t num_tokens,
    int64_t num_blocks,
    int64_t num_kv_heads,
    int32_t out_dtype) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  int64_t elems = num_tokens * num_kv_heads * 32;
  if (idx >= elems) {
    return;
  }
  int64_t pack = idx & 31;
  int64_t dim0 = pack << 2;
  int64_t tmp = idx >> 5;
  int64_t head = tmp % num_kv_heads;
  int64_t token = tmp / num_kv_heads;
  int64_t out_base = (token * num_kv_heads + head) * 128 + dim0;
  int64_t slot = slot_mapping[token];
  if (slot < 0) {
    for (int i = 0; i < 4; ++i) {
      musa_aiter_store_f16_or_bf16(out_k, out_base + i, out_dtype, 0.0f);
      musa_aiter_store_f16_or_bf16(out_v, out_base + i, out_dtype, 0.0f);
    }
    return;
  }
  int64_t block = slot >> 5;
  int64_t offset = slot & 31;
  if (block < 0 || block >= num_blocks) {
    for (int i = 0; i < 4; ++i) {
      musa_aiter_store_f16_or_bf16(out_k, out_base + i, out_dtype, 0.0f);
      musa_aiter_store_f16_or_bf16(out_v, out_base + i, out_dtype, 0.0f);
    }
    return;
  }
  for (int i = 0; i < 4; ++i) {
    int64_t cache_idx = (((block * num_kv_heads + head) * 128 + dim0 + i) * 32 + offset);
    musa_aiter_store_f16_or_bf16(out_k, out_base + i, out_dtype, musa_aiter_e4m3fn_to_float(key_cache[cache_idx]) * k_scale[0]);
    musa_aiter_store_f16_or_bf16(out_v, out_base + i, out_dtype, musa_aiter_e4m3fn_to_float(value_cache[cache_idx]) * v_scale[0]);
  }
}

extern "C" __global__ void musa_aiter_kv_paged_fp8_to_linear_hd128_bs32_e5m2_kernel(
    const unsigned char* key_cache,
    const unsigned char* value_cache,
    const int64_t* slot_mapping,
    const float* k_scale,
    const float* v_scale,
    void* out_k,
    void* out_v,
    int64_t num_tokens,
    int64_t num_blocks,
    int64_t num_kv_heads,
    int32_t out_dtype) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  int64_t elems = num_tokens * num_kv_heads * 32;
  if (idx >= elems) {
    return;
  }
  int64_t pack = idx & 31;
  int64_t dim0 = pack << 2;
  int64_t tmp = idx >> 5;
  int64_t head = tmp % num_kv_heads;
  int64_t token = tmp / num_kv_heads;
  int64_t out_base = (token * num_kv_heads + head) * 128 + dim0;
  int64_t slot = slot_mapping[token];
  if (slot < 0) {
    for (int i = 0; i < 4; ++i) {
      musa_aiter_store_f16_or_bf16(out_k, out_base + i, out_dtype, 0.0f);
      musa_aiter_store_f16_or_bf16(out_v, out_base + i, out_dtype, 0.0f);
    }
    return;
  }
  int64_t block = slot >> 5;
  int64_t offset = slot & 31;
  if (block < 0 || block >= num_blocks) {
    for (int i = 0; i < 4; ++i) {
      musa_aiter_store_f16_or_bf16(out_k, out_base + i, out_dtype, 0.0f);
      musa_aiter_store_f16_or_bf16(out_v, out_base + i, out_dtype, 0.0f);
    }
    return;
  }
  for (int i = 0; i < 4; ++i) {
    int64_t cache_idx = (((block * num_kv_heads + head) * 128 + dim0 + i) * 32 + offset);
    musa_aiter_store_f16_or_bf16(out_k, out_base + i, out_dtype, musa_aiter_e5m2_to_float(key_cache[cache_idx]) * k_scale[0]);
    musa_aiter_store_f16_or_bf16(out_v, out_base + i, out_dtype, musa_aiter_e5m2_to_float(value_cache[cache_idx]) * v_scale[0]);
  }
}

extern "C" void musa_aiter_kv_paged_fp8_to_linear_hd128_bs32_e4m3(
    const unsigned char* key_cache,
    const unsigned char* value_cache,
    const int64_t* slot_mapping,
    const float* k_scale,
    const float* v_scale,
    void* out_k,
    void* out_v,
    int64_t num_tokens,
    int64_t num_blocks,
    int64_t num_kv_heads,
    int32_t out_dtype,
    gpuStream_t stream) {
#ifdef __MUSACC__
  int threads = 256;
  int64_t elems = num_tokens * num_kv_heads * 32;
  int blocks = static_cast<int>((elems + threads - 1) / threads);
  musa_aiter_kv_paged_fp8_to_linear_hd128_bs32_e4m3_kernel<<<blocks, threads, 0, stream>>>(
      key_cache, value_cache, slot_mapping, k_scale, v_scale, out_k, out_v,
      num_tokens, num_blocks, num_kv_heads, out_dtype);
#endif
}

extern "C" void musa_aiter_kv_paged_fp8_to_linear_hd128_bs32_e5m2(
    const unsigned char* key_cache,
    const unsigned char* value_cache,
    const int64_t* slot_mapping,
    const float* k_scale,
    const float* v_scale,
    void* out_k,
    void* out_v,
    int64_t num_tokens,
    int64_t num_blocks,
    int64_t num_kv_heads,
    int32_t out_dtype,
    gpuStream_t stream) {
#ifdef __MUSACC__
  int threads = 256;
  int64_t elems = num_tokens * num_kv_heads * 32;
  int blocks = static_cast<int>((elems + threads - 1) / threads);
  musa_aiter_kv_paged_fp8_to_linear_hd128_bs32_e5m2_kernel<<<blocks, threads, 0, stream>>>(
      key_cache, value_cache, slot_mapping, k_scale, v_scale, out_k, out_v,
      num_tokens, num_blocks, num_kv_heads, out_dtype);
#endif
}

extern "C" __global__ void musa_aiter_kv_paged_fp8_to_linear_hd128_bs16_e4m3_kernel(
    const unsigned char* key_cache,
    const unsigned char* value_cache,
    const int64_t* slot_mapping,
    const float* k_scale,
    const float* v_scale,
    void* out_k,
    void* out_v,
    int64_t num_tokens,
    int64_t num_blocks,
    int64_t num_kv_heads,
    int32_t out_dtype) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  int64_t elems = num_tokens * num_kv_heads * 32;
  if (idx >= elems) {
    return;
  }
  int64_t pack = idx & 31;
  int64_t dim0 = pack << 2;
  int64_t tmp = idx >> 5;
  int64_t head = tmp % num_kv_heads;
  int64_t token = tmp / num_kv_heads;
  int64_t out_base = (token * num_kv_heads + head) * 128 + dim0;
  int64_t slot = slot_mapping[token];
  if (slot < 0) {
    for (int i = 0; i < 4; ++i) {
      musa_aiter_store_f16_or_bf16(out_k, out_base + i, out_dtype, 0.0f);
      musa_aiter_store_f16_or_bf16(out_v, out_base + i, out_dtype, 0.0f);
    }
    return;
  }
  int64_t block = slot >> 4;
  int64_t offset = slot & 15;
  if (block < 0 || block >= num_blocks) {
    for (int i = 0; i < 4; ++i) {
      musa_aiter_store_f16_or_bf16(out_k, out_base + i, out_dtype, 0.0f);
      musa_aiter_store_f16_or_bf16(out_v, out_base + i, out_dtype, 0.0f);
    }
    return;
  }
  for (int i = 0; i < 4; ++i) {
    int64_t cache_idx = (((block * num_kv_heads + head) * 128 + dim0 + i) * 16 + offset);
    musa_aiter_store_f16_or_bf16(out_k, out_base + i, out_dtype, musa_aiter_e4m3fn_to_float(key_cache[cache_idx]) * k_scale[0]);
    musa_aiter_store_f16_or_bf16(out_v, out_base + i, out_dtype, musa_aiter_e4m3fn_to_float(value_cache[cache_idx]) * v_scale[0]);
  }
}

extern "C" __global__ void musa_aiter_kv_paged_fp8_to_linear_hd128_bs16_e5m2_kernel(
    const unsigned char* key_cache,
    const unsigned char* value_cache,
    const int64_t* slot_mapping,
    const float* k_scale,
    const float* v_scale,
    void* out_k,
    void* out_v,
    int64_t num_tokens,
    int64_t num_blocks,
    int64_t num_kv_heads,
    int32_t out_dtype) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  int64_t elems = num_tokens * num_kv_heads * 32;
  if (idx >= elems) {
    return;
  }
  int64_t pack = idx & 31;
  int64_t dim0 = pack << 2;
  int64_t tmp = idx >> 5;
  int64_t head = tmp % num_kv_heads;
  int64_t token = tmp / num_kv_heads;
  int64_t out_base = (token * num_kv_heads + head) * 128 + dim0;
  int64_t slot = slot_mapping[token];
  if (slot < 0) {
    for (int i = 0; i < 4; ++i) {
      musa_aiter_store_f16_or_bf16(out_k, out_base + i, out_dtype, 0.0f);
      musa_aiter_store_f16_or_bf16(out_v, out_base + i, out_dtype, 0.0f);
    }
    return;
  }
  int64_t block = slot >> 4;
  int64_t offset = slot & 15;
  if (block < 0 || block >= num_blocks) {
    for (int i = 0; i < 4; ++i) {
      musa_aiter_store_f16_or_bf16(out_k, out_base + i, out_dtype, 0.0f);
      musa_aiter_store_f16_or_bf16(out_v, out_base + i, out_dtype, 0.0f);
    }
    return;
  }
  for (int i = 0; i < 4; ++i) {
    int64_t cache_idx = (((block * num_kv_heads + head) * 128 + dim0 + i) * 16 + offset);
    musa_aiter_store_f16_or_bf16(out_k, out_base + i, out_dtype, musa_aiter_e5m2_to_float(key_cache[cache_idx]) * k_scale[0]);
    musa_aiter_store_f16_or_bf16(out_v, out_base + i, out_dtype, musa_aiter_e5m2_to_float(value_cache[cache_idx]) * v_scale[0]);
  }
}

extern "C" void musa_aiter_kv_paged_fp8_to_linear_hd128_bs16_e4m3(
    const unsigned char* key_cache,
    const unsigned char* value_cache,
    const int64_t* slot_mapping,
    const float* k_scale,
    const float* v_scale,
    void* out_k,
    void* out_v,
    int64_t num_tokens,
    int64_t num_blocks,
    int64_t num_kv_heads,
    int32_t out_dtype,
    gpuStream_t stream) {
#ifdef __MUSACC__
  int threads = 256;
  int64_t elems = num_tokens * num_kv_heads * 32;
  int blocks = static_cast<int>((elems + threads - 1) / threads);
  musa_aiter_kv_paged_fp8_to_linear_hd128_bs16_e4m3_kernel<<<blocks, threads, 0, stream>>>(
      key_cache, value_cache, slot_mapping, k_scale, v_scale, out_k, out_v,
      num_tokens, num_blocks, num_kv_heads, out_dtype);
#endif
}

extern "C" void musa_aiter_kv_paged_fp8_to_linear_hd128_bs16_e5m2(
    const unsigned char* key_cache,
    const unsigned char* value_cache,
    const int64_t* slot_mapping,
    const float* k_scale,
    const float* v_scale,
    void* out_k,
    void* out_v,
    int64_t num_tokens,
    int64_t num_blocks,
    int64_t num_kv_heads,
    int32_t out_dtype,
    gpuStream_t stream) {
#ifdef __MUSACC__
  int threads = 256;
  int64_t elems = num_tokens * num_kv_heads * 32;
  int blocks = static_cast<int>((elems + threads - 1) / threads);
  musa_aiter_kv_paged_fp8_to_linear_hd128_bs16_e5m2_kernel<<<blocks, threads, 0, stream>>>(
      key_cache, value_cache, slot_mapping, k_scale, v_scale, out_k, out_v,
      num_tokens, num_blocks, num_kv_heads, out_dtype);
#endif
}

extern "C" __global__ void musa_aiter_kv_paged_fp8_to_linear_vec4_e4m3_kernel(
    const unsigned char* key_cache,
    const unsigned char* value_cache,
    const int64_t* slot_mapping,
    const float* k_scale,
    const float* v_scale,
    void* out_k,
    void* out_v,
    int64_t num_tokens,
    int64_t num_blocks,
    int64_t num_kv_heads,
    int64_t head_dim,
    int64_t block_size,
    int32_t out_dtype,
    int32_t scale_mode) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  int64_t packs_per_head = head_dim / 4;
  int64_t elems = num_tokens * num_kv_heads * packs_per_head;
  if (idx >= elems) {
    return;
  }
  int64_t pack = idx % packs_per_head;
  int64_t dim0 = pack * 4;
  int64_t tmp = idx / packs_per_head;
  int64_t head = tmp % num_kv_heads;
  int64_t token = tmp / num_kv_heads;
  int64_t out_base = (token * num_kv_heads + head) * head_dim + dim0;
  int64_t slot = slot_mapping[token];
  if (slot < 0) {
    for (int i = 0; i < 4; ++i) {
      musa_aiter_store_f16_or_bf16(out_k, out_base + i, out_dtype, 0.0f);
      musa_aiter_store_f16_or_bf16(out_v, out_base + i, out_dtype, 0.0f);
    }
    return;
  }
  int64_t block = slot / block_size;
  int64_t offset = slot - block * block_size;
  if (block < 0 || block >= num_blocks) {
    for (int i = 0; i < 4; ++i) {
      musa_aiter_store_f16_or_bf16(out_k, out_base + i, out_dtype, 0.0f);
      musa_aiter_store_f16_or_bf16(out_v, out_base + i, out_dtype, 0.0f);
    }
    return;
  }
  float ks = k_scale[scale_mode == 0 ? 0 : head];
  float vs = v_scale[scale_mode == 0 ? 0 : head];
  for (int i = 0; i < 4; ++i) {
    int64_t cache_idx = (((block * num_kv_heads + head) * head_dim + dim0 + i) * block_size + offset);
    musa_aiter_store_f16_or_bf16(out_k, out_base + i, out_dtype, musa_aiter_e4m3fn_to_float(key_cache[cache_idx]) * ks);
    musa_aiter_store_f16_or_bf16(out_v, out_base + i, out_dtype, musa_aiter_e4m3fn_to_float(value_cache[cache_idx]) * vs);
  }
}

extern "C" __global__ void musa_aiter_kv_paged_fp8_to_linear_vec4_e5m2_kernel(
    const unsigned char* key_cache,
    const unsigned char* value_cache,
    const int64_t* slot_mapping,
    const float* k_scale,
    const float* v_scale,
    void* out_k,
    void* out_v,
    int64_t num_tokens,
    int64_t num_blocks,
    int64_t num_kv_heads,
    int64_t head_dim,
    int64_t block_size,
    int32_t out_dtype,
    int32_t scale_mode) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  int64_t packs_per_head = head_dim / 4;
  int64_t elems = num_tokens * num_kv_heads * packs_per_head;
  if (idx >= elems) {
    return;
  }
  int64_t pack = idx % packs_per_head;
  int64_t dim0 = pack * 4;
  int64_t tmp = idx / packs_per_head;
  int64_t head = tmp % num_kv_heads;
  int64_t token = tmp / num_kv_heads;
  int64_t out_base = (token * num_kv_heads + head) * head_dim + dim0;
  int64_t slot = slot_mapping[token];
  if (slot < 0) {
    for (int i = 0; i < 4; ++i) {
      musa_aiter_store_f16_or_bf16(out_k, out_base + i, out_dtype, 0.0f);
      musa_aiter_store_f16_or_bf16(out_v, out_base + i, out_dtype, 0.0f);
    }
    return;
  }
  int64_t block = slot / block_size;
  int64_t offset = slot - block * block_size;
  if (block < 0 || block >= num_blocks) {
    for (int i = 0; i < 4; ++i) {
      musa_aiter_store_f16_or_bf16(out_k, out_base + i, out_dtype, 0.0f);
      musa_aiter_store_f16_or_bf16(out_v, out_base + i, out_dtype, 0.0f);
    }
    return;
  }
  float ks = k_scale[scale_mode == 0 ? 0 : head];
  float vs = v_scale[scale_mode == 0 ? 0 : head];
  for (int i = 0; i < 4; ++i) {
    int64_t cache_idx = (((block * num_kv_heads + head) * head_dim + dim0 + i) * block_size + offset);
    musa_aiter_store_f16_or_bf16(out_k, out_base + i, out_dtype, musa_aiter_e5m2_to_float(key_cache[cache_idx]) * ks);
    musa_aiter_store_f16_or_bf16(out_v, out_base + i, out_dtype, musa_aiter_e5m2_to_float(value_cache[cache_idx]) * vs);
  }
}

extern "C" void musa_aiter_kv_paged_fp8_to_linear_vec4_e4m3(
    const unsigned char* key_cache,
    const unsigned char* value_cache,
    const int64_t* slot_mapping,
    const float* k_scale,
    const float* v_scale,
    void* out_k,
    void* out_v,
    int64_t num_tokens,
    int64_t num_blocks,
    int64_t num_kv_heads,
    int64_t head_dim,
    int64_t block_size,
    int32_t out_dtype,
    gpuStream_t stream,
    int32_t scale_mode) {
#ifdef __MUSACC__
  int threads = 256;
  int64_t elems = num_tokens * num_kv_heads * (head_dim / 4);
  int blocks = static_cast<int>((elems + threads - 1) / threads);
  musa_aiter_kv_paged_fp8_to_linear_vec4_e4m3_kernel<<<blocks, threads, 0, stream>>>(
      key_cache, value_cache, slot_mapping, k_scale, v_scale, out_k, out_v,
      num_tokens, num_blocks, num_kv_heads, head_dim, block_size, out_dtype, scale_mode);
#endif
}

extern "C" void musa_aiter_kv_paged_fp8_to_linear_vec4_e5m2(
    const unsigned char* key_cache,
    const unsigned char* value_cache,
    const int64_t* slot_mapping,
    const float* k_scale,
    const float* v_scale,
    void* out_k,
    void* out_v,
    int64_t num_tokens,
    int64_t num_blocks,
    int64_t num_kv_heads,
    int64_t head_dim,
    int64_t block_size,
    int32_t out_dtype,
    gpuStream_t stream,
    int32_t scale_mode) {
#ifdef __MUSACC__
  int threads = 256;
  int64_t elems = num_tokens * num_kv_heads * (head_dim / 4);
  int blocks = static_cast<int>((elems + threads - 1) / threads);
  musa_aiter_kv_paged_fp8_to_linear_vec4_e5m2_kernel<<<blocks, threads, 0, stream>>>(
      key_cache, value_cache, slot_mapping, k_scale, v_scale, out_k, out_v,
      num_tokens, num_blocks, num_kv_heads, head_dim, block_size, out_dtype, scale_mode);
#endif
}

extern "C" __global__ void musa_aiter_kv_copy_paged_blocks_fp8_vec8_kernel(
    const unsigned char* src_key_cache,
    const unsigned char* src_value_cache,
    unsigned char* dst_key_cache,
    unsigned char* dst_value_cache,
    const int64_t* block_mapping,
    int64_t num_pairs,
    int64_t num_layers,
    int64_t src_blocks,
    int64_t dst_blocks,
    int64_t block_bytes,
    int64_t vec_count) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  int64_t total = num_pairs * num_layers * vec_count;
  if (idx >= total) {
    return;
  }
  int64_t vec = idx % vec_count;
  int64_t tmp = idx / vec_count;
  int64_t layer = tmp % num_layers;
  int64_t pair = tmp / num_layers;
  int64_t src_block = block_mapping[pair * 2];
  int64_t dst_block = block_mapping[pair * 2 + 1];
  if (src_block < 0 || dst_block < 0 || src_block >= src_blocks || dst_block >= dst_blocks) {
    return;
  }
  int64_t src_offset = (layer * src_blocks + src_block) * block_bytes;
  int64_t dst_offset = (layer * dst_blocks + dst_block) * block_bytes;
  int64_t byte_offset = vec * 8;
  int64_t remaining = block_bytes - byte_offset;
  if (remaining >= 8) {
    *reinterpret_cast<uint64_t*>(dst_key_cache + dst_offset + byte_offset) = *reinterpret_cast<const uint64_t*>(src_key_cache + src_offset + byte_offset);
    *reinterpret_cast<uint64_t*>(dst_value_cache + dst_offset + byte_offset) = *reinterpret_cast<const uint64_t*>(src_value_cache + src_offset + byte_offset);
  } else {
    for (int64_t i = 0; i < remaining; ++i) {
      dst_key_cache[dst_offset + byte_offset + i] = src_key_cache[src_offset + byte_offset + i];
      dst_value_cache[dst_offset + byte_offset + i] = src_value_cache[src_offset + byte_offset + i];
    }
  }
}

extern "C" void musa_aiter_kv_copy_paged_blocks_fp8_vec8(
    const unsigned char* src_key_cache,
    const unsigned char* src_value_cache,
    unsigned char* dst_key_cache,
    unsigned char* dst_value_cache,
    const int64_t* block_mapping,
    int64_t num_pairs,
    int64_t num_layers,
    int64_t src_blocks,
    int64_t dst_blocks,
    int64_t num_kv_heads,
    int64_t head_dim,
    int64_t block_size,
    gpuStream_t stream) {
#ifdef __MUSACC__
  int64_t block_bytes = num_kv_heads * head_dim * block_size;
  int64_t vec_count = (block_bytes + 7) / 8;
  int threads = 256;
  int64_t total = num_pairs * num_layers * vec_count;
  int blocks = static_cast<int>((total + threads - 1) / threads);
  musa_aiter_kv_copy_paged_blocks_fp8_vec8_kernel<<<blocks, threads, 0, stream>>>(
      src_key_cache, src_value_cache, dst_key_cache, dst_value_cache, block_mapping,
      num_pairs, num_layers, src_blocks, dst_blocks, block_bytes, vec_count);
#endif
}

extern "C" int musa_aiter_fp8_kv_cache_abi_version() {
  return 1;
}
