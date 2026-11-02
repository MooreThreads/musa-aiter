#include <stdint.h>
#include <math.h>
#include "../common/musa_compat.h"

static __device__ float musa_aiter_u32_to_float_attn(uint32_t bits) {
  union {
    uint32_t u;
    float f;
  } v;
  v.u = bits;
  return v.f;
}

static __device__ uint32_t musa_aiter_float_to_u32_attn(float value) {
  union {
    float f;
    uint32_t u;
  } v;
  v.f = value;
  return v.u;
}

static __device__ float musa_aiter_half_to_float_attn(uint16_t h) {
  uint32_t sign = (static_cast<uint32_t>(h & 0x8000)) << 16;
  uint32_t exp = (h >> 10) & 0x1f;
  uint32_t mant = h & 0x03ff;
  if (exp == 0) {
    if (mant == 0) {
      return musa_aiter_u32_to_float_attn(sign);
    }
    while ((mant & 0x0400) == 0) {
      mant <<= 1;
      exp--;
    }
    exp++;
    mant &= 0x03ff;
  } else if (exp == 31) {
    return musa_aiter_u32_to_float_attn(sign | 0x7f800000u | (mant << 13));
  }
  exp = exp + (127 - 15);
  return musa_aiter_u32_to_float_attn(sign | (exp << 23) | (mant << 13));
}

static __device__ uint16_t musa_aiter_float_to_half_attn(float value) {
  uint32_t bits = musa_aiter_float_to_u32_attn(value);
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

static __device__ float musa_aiter_bfloat16_to_float_attn(uint16_t b) {
  return musa_aiter_u32_to_float_attn(static_cast<uint32_t>(b) << 16);
}

static __device__ uint16_t musa_aiter_float_to_bfloat16_attn(float value) {
  uint32_t bits = musa_aiter_float_to_u32_attn(value);
  uint32_t lsb = (bits >> 16) & 1;
  bits += 0x7fffu + lsb;
  return static_cast<uint16_t>(bits >> 16);
}

extern "C" __global__ void musa_aiter_dense_decode_attention_float_kernel(
    const float* q,
    const float* k,
    const float* v,
    float* out,
    int64_t batch,
    int64_t seq_len,
    int64_t num_q_heads,
    int64_t num_kv_heads,
    int64_t head_dim,
    float scale) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  int64_t elems = batch * num_q_heads * head_dim;
  if (idx >= elems) {
    return;
  }

  int64_t dim = idx % head_dim;
  int64_t tmp = idx / head_dim;
  int64_t head = tmp % num_q_heads;
  int64_t batch_id = tmp / num_q_heads;
  int64_t kv_head = head / (num_q_heads / num_kv_heads);

  float max_score = -3.4028234663852886e38f;
  for (int64_t pos = 0; pos < seq_len; ++pos) {
    float dot = 0.0f;
    for (int64_t d = 0; d < head_dim; ++d) {
      dot += q[(batch_id * num_q_heads + head) * head_dim + d] * k[((batch_id * seq_len + pos) * num_kv_heads + kv_head) * head_dim + d];
    }
    float score = dot * scale;
    if (score > max_score) {
      max_score = score;
    }
  }

  float denom = 0.0f;
  float weighted = 0.0f;
  for (int64_t pos = 0; pos < seq_len; ++pos) {
    float dot = 0.0f;
    for (int64_t d = 0; d < head_dim; ++d) {
      dot += q[(batch_id * num_q_heads + head) * head_dim + d] * k[((batch_id * seq_len + pos) * num_kv_heads + kv_head) * head_dim + d];
    }
    float weight = expf(dot * scale - max_score);
    denom += weight;
    weighted += weight * v[((batch_id * seq_len + pos) * num_kv_heads + kv_head) * head_dim + dim];
  }
  out[idx] = denom == 0.0f ? 0.0f : weighted / denom;
}

extern "C" void musa_aiter_dense_decode_attention_float(
    const float* q,
    const float* k,
    const float* v,
    float* out,
    int64_t batch,
    int64_t seq_len,
    int64_t num_q_heads,
    int64_t num_kv_heads,
    int64_t head_dim,
    float scale,
    gpuStream_t stream) {
#ifdef __MUSACC__
  int threads = 128;
  int64_t elems = batch * num_q_heads * head_dim;
  int blocks = static_cast<int>((elems + threads - 1) / threads);
  musa_aiter_dense_decode_attention_float_kernel<<<blocks, threads, 0, stream>>>(q, k, v, out, batch, seq_len, num_q_heads, num_kv_heads, head_dim, scale);
#endif
}

extern "C" __global__ void musa_aiter_dense_decode_attention_u16_kernel(
    const uint16_t* q,
    const uint16_t* k,
    const uint16_t* v,
    uint16_t* out,
    int64_t batch,
    int64_t seq_len,
    int64_t num_q_heads,
    int64_t num_kv_heads,
    int64_t head_dim,
    float scale,
    int32_t dtype_code) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  int64_t elems = batch * num_q_heads * head_dim;
  if (idx >= elems) {
    return;
  }

  int64_t dim = idx % head_dim;
  int64_t tmp = idx / head_dim;
  int64_t head = tmp % num_q_heads;
  int64_t batch_id = tmp / num_q_heads;
  int64_t kv_head = head / (num_q_heads / num_kv_heads);

  float max_score = -3.4028234663852886e38f;
  for (int64_t pos = 0; pos < seq_len; ++pos) {
    float dot = 0.0f;
    for (int64_t d = 0; d < head_dim; ++d) {
      float qv = dtype_code == 1 ? musa_aiter_half_to_float_attn(q[(batch_id * num_q_heads + head) * head_dim + d]) : musa_aiter_bfloat16_to_float_attn(q[(batch_id * num_q_heads + head) * head_dim + d]);
      float kv = dtype_code == 1 ? musa_aiter_half_to_float_attn(k[((batch_id * seq_len + pos) * num_kv_heads + kv_head) * head_dim + d]) : musa_aiter_bfloat16_to_float_attn(k[((batch_id * seq_len + pos) * num_kv_heads + kv_head) * head_dim + d]);
      dot += qv * kv;
    }
    float score = dot * scale;
    if (score > max_score) {
      max_score = score;
    }
  }

  float denom = 0.0f;
  float weighted = 0.0f;
  for (int64_t pos = 0; pos < seq_len; ++pos) {
    float dot = 0.0f;
    for (int64_t d = 0; d < head_dim; ++d) {
      float qv = dtype_code == 1 ? musa_aiter_half_to_float_attn(q[(batch_id * num_q_heads + head) * head_dim + d]) : musa_aiter_bfloat16_to_float_attn(q[(batch_id * num_q_heads + head) * head_dim + d]);
      float kv = dtype_code == 1 ? musa_aiter_half_to_float_attn(k[((batch_id * seq_len + pos) * num_kv_heads + kv_head) * head_dim + d]) : musa_aiter_bfloat16_to_float_attn(k[((batch_id * seq_len + pos) * num_kv_heads + kv_head) * head_dim + d]);
      dot += qv * kv;
    }
    float weight = expf(dot * scale - max_score);
    float vv = dtype_code == 1 ? musa_aiter_half_to_float_attn(v[((batch_id * seq_len + pos) * num_kv_heads + kv_head) * head_dim + dim]) : musa_aiter_bfloat16_to_float_attn(v[((batch_id * seq_len + pos) * num_kv_heads + kv_head) * head_dim + dim]);
    denom += weight;
    weighted += weight * vv;
  }
  float result = denom == 0.0f ? 0.0f : weighted / denom;
  out[idx] = dtype_code == 1 ? musa_aiter_float_to_half_attn(result) : musa_aiter_float_to_bfloat16_attn(result);
}

#define MUSA_AITER_DEFINE_DENSE_DECODE_U16_WRAPPER(SUFFIX, DTYPE_CODE) \
extern "C" void musa_aiter_dense_decode_attention_##SUFFIX( \
    const uint16_t* q, const uint16_t* k, const uint16_t* v, uint16_t* out, \
    int64_t batch, int64_t seq_len, int64_t num_q_heads, int64_t num_kv_heads, int64_t head_dim, float scale, gpuStream_t stream) { \
  int threads = 128; \
  int64_t elems = batch * num_q_heads * head_dim; \
  int blocks = static_cast<int>((elems + threads - 1) / threads); \
  musa_aiter_dense_decode_attention_u16_kernel<<<blocks, threads, 0, stream>>>(q, k, v, out, batch, seq_len, num_q_heads, num_kv_heads, head_dim, scale, DTYPE_CODE); \
}

MUSA_AITER_DEFINE_DENSE_DECODE_U16_WRAPPER(half, 1)
MUSA_AITER_DEFINE_DENSE_DECODE_U16_WRAPPER(bfloat16, 2)

extern "C" int musa_aiter_attention_abi_version() {
  return 1;
}
