#include <stdint.h>
#include <math.h>
#include "../common/musa_compat.h"

static __device__ float musa_aiter_u32_to_float_sparse_attn(uint32_t bits) {
  union {
    uint32_t u;
    float f;
  } v;
  v.u = bits;
  return v.f;
}

static __device__ uint32_t musa_aiter_float_to_u32_sparse_attn(float value) {
  union {
    float f;
    uint32_t u;
  } v;
  v.f = value;
  return v.u;
}

static __device__ float musa_aiter_half_to_float_sparse_attn(uint16_t h) {
  uint32_t sign = (static_cast<uint32_t>(h & 0x8000)) << 16;
  uint32_t exp = (h >> 10) & 0x1f;
  uint32_t mant = h & 0x03ff;
  if (exp == 0) {
    if (mant == 0) {
      return musa_aiter_u32_to_float_sparse_attn(sign);
    }
    while ((mant & 0x0400) == 0) {
      mant <<= 1;
      exp--;
    }
    exp++;
    mant &= 0x03ff;
  } else if (exp == 31) {
    return musa_aiter_u32_to_float_sparse_attn(sign | 0x7f800000u | (mant << 13));
  }
  exp = exp + (127 - 15);
  return musa_aiter_u32_to_float_sparse_attn(sign | (exp << 23) | (mant << 13));
}

static __device__ uint16_t musa_aiter_float_to_half_sparse_attn(float value) {
  uint32_t bits = musa_aiter_float_to_u32_sparse_attn(value);
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

static __device__ float musa_aiter_bfloat16_to_float_sparse_attn(uint16_t b) {
  return musa_aiter_u32_to_float_sparse_attn(static_cast<uint32_t>(b) << 16);
}

static __device__ uint16_t musa_aiter_float_to_bfloat16_sparse_attn(float value) {
  uint32_t bits = musa_aiter_float_to_u32_sparse_attn(value);
  uint32_t lsb = (bits >> 16) & 1;
  bits += 0x7fffu + lsb;
  return static_cast<uint16_t>(bits >> 16);
}

extern "C" __global__ void musa_aiter_paged_sparse_decode_attention_float_kernel(
    const float* q,
    const float* k,
    const float* v,
    const int64_t* kv_indptr,
    const int64_t* kv_indices,
    float* out,
    int64_t batch,
    int64_t index_count,
    int64_t num_tokens,
    int64_t num_q_heads,
    int64_t num_kv_heads,
    int64_t head_dim,
    float scale) {
  __shared__ float reduce[256];
  int64_t head_idx = static_cast<int64_t>(blockIdx.x);
  if (head_idx >= batch * num_q_heads) {
    return;
  }

  int64_t tid = threadIdx.x;
  int64_t batch_id = head_idx / num_q_heads;
  int64_t q_head = head_idx % num_q_heads;
  int64_t kv_head = q_head / (num_q_heads / num_kv_heads);
  int64_t start = kv_indptr[batch_id];
  int64_t end = kv_indptr[batch_id + 1];
  int64_t q_base = (batch_id * num_q_heads + q_head) * head_dim;
  int64_t out_base = q_base;

  if (start < 0 || end < start || start > index_count) {
    for (int64_t d = tid; d < head_dim; d += blockDim.x) {
      out[out_base + d] = 0.0f;
    }
    return;
  }
  if (end > index_count) {
    end = index_count;
  }

  float max_score = -3.4028234663852886e38f;
  for (int64_t pos = start; pos < end; ++pos) {
    int64_t token = kv_indices[pos];
    float partial = 0.0f;
    if (token >= 0 && token < num_tokens) {
      int64_t k_base = (token * num_kv_heads + kv_head) * head_dim;
      for (int64_t d = tid; d < head_dim; d += blockDim.x) {
        partial += q[q_base + d] * k[k_base + d];
      }
    }
    reduce[tid] = partial;
    __syncthreads();
    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
      if (tid < stride) {
        reduce[tid] += reduce[tid + stride];
      }
      __syncthreads();
    }
    if (tid == 0) {
      if (token >= 0 && token < num_tokens) {
        float score = reduce[0] * scale;
        if (score > max_score) {
          max_score = score;
        }
      }
      reduce[0] = max_score;
    }
    __syncthreads();
    max_score = reduce[0];
    __syncthreads();
  }

  float denom = 0.0f;
  float weighted = 0.0f;
  for (int64_t pos = start; pos < end; ++pos) {
    int64_t token = kv_indices[pos];
    float partial = 0.0f;
    if (token >= 0 && token < num_tokens) {
      int64_t k_base = (token * num_kv_heads + kv_head) * head_dim;
      for (int64_t d = tid; d < head_dim; d += blockDim.x) {
        partial += q[q_base + d] * k[k_base + d];
      }
    }
    reduce[tid] = partial;
    __syncthreads();
    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
      if (tid < stride) {
        reduce[tid] += reduce[tid + stride];
      }
      __syncthreads();
    }
    float weight = 0.0f;
    if (tid == 0) {
      if (token >= 0 && token < num_tokens) {
        weight = expf(reduce[0] * scale - max_score);
        denom += weight;
      }
      reduce[0] = weight;
      reduce[1] = denom;
    }
    __syncthreads();
    weight = reduce[0];
    denom = reduce[1];
    if (token >= 0 && token < num_tokens) {
      int64_t v_base = (token * num_kv_heads + kv_head) * head_dim;
      for (int64_t d = tid; d < head_dim; d += blockDim.x) {
        weighted += weight * v[v_base + d];
      }
    }
    __syncthreads();
  }

  for (int64_t d = tid; d < head_dim; d += blockDim.x) {
    out[out_base + d] = denom == 0.0f ? 0.0f : weighted / denom;
  }
}

extern "C" void musa_aiter_paged_sparse_decode_attention_float(
    const float* q,
    const float* k,
    const float* v,
    const int64_t* kv_indptr,
    const int64_t* kv_indices,
    float* out,
    int64_t batch,
    int64_t index_count,
    int64_t num_tokens,
    int64_t num_q_heads,
    int64_t num_kv_heads,
    int64_t head_dim,
    float scale,
    gpuStream_t stream) {
#ifdef __MUSACC__
  int threads = 256;
  int blocks = static_cast<int>(batch * num_q_heads);
  musa_aiter_paged_sparse_decode_attention_float_kernel<<<blocks, threads, 0, stream>>>(q, k, v, kv_indptr, kv_indices, out, batch, index_count, num_tokens, num_q_heads, num_kv_heads, head_dim, scale);
#endif
}

extern "C" __global__ void musa_aiter_paged_sparse_decode_attention_u16_kernel(
    const uint16_t* q,
    const uint16_t* k,
    const uint16_t* v,
    const int64_t* kv_indptr,
    const int64_t* kv_indices,
    uint16_t* out,
    int64_t batch,
    int64_t index_count,
    int64_t num_tokens,
    int64_t num_q_heads,
    int64_t num_kv_heads,
    int64_t head_dim,
    float scale,
    int32_t dtype_code) {
  __shared__ float reduce[256];
  int64_t head_idx = static_cast<int64_t>(blockIdx.x);
  if (head_idx >= batch * num_q_heads) {
    return;
  }

  int64_t tid = threadIdx.x;
  int64_t batch_id = head_idx / num_q_heads;
  int64_t q_head = head_idx % num_q_heads;
  int64_t kv_head = q_head / (num_q_heads / num_kv_heads);
  int64_t start = kv_indptr[batch_id];
  int64_t end = kv_indptr[batch_id + 1];
  int64_t q_base = (batch_id * num_q_heads + q_head) * head_dim;
  int64_t out_base = q_base;

  if (start < 0 || end < start || start > index_count) {
    for (int64_t d = tid; d < head_dim; d += blockDim.x) {
      out[out_base + d] = 0;
    }
    return;
  }
  if (end > index_count) {
    end = index_count;
  }

  float max_score = -3.4028234663852886e38f;
  for (int64_t pos = start; pos < end; ++pos) {
    int64_t token = kv_indices[pos];
    float partial = 0.0f;
    if (token >= 0 && token < num_tokens) {
      int64_t k_base = (token * num_kv_heads + kv_head) * head_dim;
      for (int64_t d = tid; d < head_dim; d += blockDim.x) {
        float qv = dtype_code == 1 ? musa_aiter_half_to_float_sparse_attn(q[q_base + d]) : musa_aiter_bfloat16_to_float_sparse_attn(q[q_base + d]);
        float kv = dtype_code == 1 ? musa_aiter_half_to_float_sparse_attn(k[k_base + d]) : musa_aiter_bfloat16_to_float_sparse_attn(k[k_base + d]);
        partial += qv * kv;
      }
    }
    reduce[tid] = partial;
    __syncthreads();
    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
      if (tid < stride) {
        reduce[tid] += reduce[tid + stride];
      }
      __syncthreads();
    }
    if (tid == 0) {
      if (token >= 0 && token < num_tokens) {
        float score = reduce[0] * scale;
        if (score > max_score) {
          max_score = score;
        }
      }
      reduce[0] = max_score;
    }
    __syncthreads();
    max_score = reduce[0];
    __syncthreads();
  }

  float denom = 0.0f;
  float weighted = 0.0f;
  for (int64_t pos = start; pos < end; ++pos) {
    int64_t token = kv_indices[pos];
    float partial = 0.0f;
    if (token >= 0 && token < num_tokens) {
      int64_t k_base = (token * num_kv_heads + kv_head) * head_dim;
      for (int64_t d = tid; d < head_dim; d += blockDim.x) {
        float qv = dtype_code == 1 ? musa_aiter_half_to_float_sparse_attn(q[q_base + d]) : musa_aiter_bfloat16_to_float_sparse_attn(q[q_base + d]);
        float kv = dtype_code == 1 ? musa_aiter_half_to_float_sparse_attn(k[k_base + d]) : musa_aiter_bfloat16_to_float_sparse_attn(k[k_base + d]);
        partial += qv * kv;
      }
    }
    reduce[tid] = partial;
    __syncthreads();
    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
      if (tid < stride) {
        reduce[tid] += reduce[tid + stride];
      }
      __syncthreads();
    }
    float weight = 0.0f;
    if (tid == 0) {
      if (token >= 0 && token < num_tokens) {
        weight = expf(reduce[0] * scale - max_score);
        denom += weight;
      }
      reduce[0] = weight;
      reduce[1] = denom;
    }
    __syncthreads();
    weight = reduce[0];
    denom = reduce[1];
    if (token >= 0 && token < num_tokens) {
      int64_t v_base = (token * num_kv_heads + kv_head) * head_dim;
      for (int64_t d = tid; d < head_dim; d += blockDim.x) {
        float vv = dtype_code == 1 ? musa_aiter_half_to_float_sparse_attn(v[v_base + d]) : musa_aiter_bfloat16_to_float_sparse_attn(v[v_base + d]);
        weighted += weight * vv;
      }
    }
    __syncthreads();
  }

  for (int64_t d = tid; d < head_dim; d += blockDim.x) {
    float result = denom == 0.0f ? 0.0f : weighted / denom;
    out[out_base + d] = dtype_code == 1 ? musa_aiter_float_to_half_sparse_attn(result) : musa_aiter_float_to_bfloat16_sparse_attn(result);
  }
}

#define MUSA_AITER_DEFINE_PAGED_SPARSE_U16_WRAPPER(SUFFIX, DTYPE_CODE) \
extern "C" void musa_aiter_paged_sparse_decode_attention_##SUFFIX( \
    const uint16_t* q, const uint16_t* k, const uint16_t* v, const int64_t* kv_indptr, const int64_t* kv_indices, uint16_t* out, \
    int64_t batch, int64_t index_count, int64_t num_tokens, int64_t num_q_heads, int64_t num_kv_heads, int64_t head_dim, float scale, gpuStream_t stream) { \
  int threads = 256; \
  int blocks = static_cast<int>(batch * num_q_heads); \
  musa_aiter_paged_sparse_decode_attention_u16_kernel<<<blocks, threads, 0, stream>>>(q, k, v, kv_indptr, kv_indices, out, batch, index_count, num_tokens, num_q_heads, num_kv_heads, head_dim, scale, DTYPE_CODE); \
}

MUSA_AITER_DEFINE_PAGED_SPARSE_U16_WRAPPER(half, 1)
MUSA_AITER_DEFINE_PAGED_SPARSE_U16_WRAPPER(bfloat16, 2)

extern "C" __global__ void musa_aiter_paged_sparse_decode_attention_bfloat16_tiled_stage1_kernel(
    const uint16_t* q,
    const uint16_t* k,
    const uint16_t* v,
    const int64_t* kv_indptr,
    const int64_t* kv_indices,
    float* partial_max,
    float* partial_sum,
    uint16_t* partial_out,
    int64_t batch,
    int64_t index_count,
    int64_t num_tokens,
    int64_t num_q_heads,
    int64_t num_kv_heads,
    int64_t head_dim,
    int64_t tile_size,
    int64_t num_tiles,
    float scale) {
  __shared__ float reduce[256];
  __shared__ float q_shared[256];
  int64_t tile_block = static_cast<int64_t>(blockIdx.x);
  int64_t tile_id = tile_block % num_tiles;
  int64_t head_idx = tile_block / num_tiles;
  if (head_idx >= batch * num_q_heads) {
    return;
  }

  int64_t tid = threadIdx.x;
  int64_t batch_id = head_idx / num_q_heads;
  int64_t q_head = head_idx % num_q_heads;
  int64_t kv_head = q_head / (num_q_heads / num_kv_heads);
  int64_t start = kv_indptr[batch_id];
  int64_t end = kv_indptr[batch_id + 1];
  int64_t q_base = (batch_id * num_q_heads + q_head) * head_dim;
  int64_t tile_base = (head_idx * num_tiles + tile_id);
  int64_t out_base = tile_base * head_dim;

  for (int64_t d = tid; d < head_dim && d < 256; d += blockDim.x) {
    q_shared[d] = musa_aiter_bfloat16_to_float_sparse_attn(q[q_base + d]);
  }
  __syncthreads();

  for (int64_t d = tid; d < head_dim; d += blockDim.x) {
    partial_out[out_base + d] = 0.0f;
  }
  if (tid == 0) {
    partial_max[tile_base] = -3.4028234663852886e38f;
    partial_sum[tile_base] = 0.0f;
  }
  __syncthreads();

  if (start < 0 || end < start || start > index_count) {
    return;
  }
  if (end > index_count) {
    end = index_count;
  }
  int64_t tile_start = start + tile_id * tile_size;
  int64_t tile_end = tile_start + tile_size;
  if (tile_start >= end) {
    return;
  }
  if (tile_end > end) {
    tile_end = end;
  }

  float local_max = -3.4028234663852886e38f;
  float denom = 0.0f;
  float weighted[8];
  int64_t value_lanes = (head_dim + static_cast<int64_t>(blockDim.x) - 1) / static_cast<int64_t>(blockDim.x);
  if (value_lanes > 8) {
    value_lanes = 8;
  }
  for (int64_t lane = 0; lane < value_lanes; ++lane) {
    weighted[lane] = 0.0f;
  }
  for (int64_t pos = tile_start; pos < tile_end; ++pos) {
    int64_t token = kv_indices[pos];
    float partial = 0.0f;
    if (token >= 0 && token < num_tokens) {
      int64_t k_base = (token * num_kv_heads + kv_head) * head_dim;
      for (int64_t d = tid; d < head_dim; d += blockDim.x) {
        partial += (head_dim <= static_cast<int64_t>(blockDim.x) ? q_shared[d] : musa_aiter_bfloat16_to_float_sparse_attn(q[q_base + d])) * musa_aiter_bfloat16_to_float_sparse_attn(k[k_base + d]);
      }
    }
    reduce[tid] = partial;
    __syncthreads();
    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
      if (tid < stride) {
        reduce[tid] += reduce[tid + stride];
      }
      __syncthreads();
    }
    if (tid == 0) {
      float old_scale = 1.0f;
      float weight = 0.0f;
      if (token >= 0 && token < num_tokens) {
        float score = reduce[0] * scale;
        float next_max = score > local_max ? score : local_max;
        old_scale = expf(local_max - next_max);
        weight = expf(score - next_max);
        denom = denom * old_scale + weight;
        local_max = next_max;
      }
      reduce[0] = old_scale;
      reduce[1] = weight;
      reduce[2] = denom;
      reduce[3] = local_max;
    }
    __syncthreads();
    float old_scale = reduce[0];
    float weight = reduce[1];
    denom = reduce[2];
    local_max = reduce[3];
    for (int64_t lane = 0; lane < value_lanes; ++lane) {
      weighted[lane] *= old_scale;
    }
    if (token >= 0 && token < num_tokens) {
      int64_t v_base = (token * num_kv_heads + kv_head) * head_dim;
      for (int64_t lane = 0; lane < value_lanes; ++lane) {
        int64_t d = static_cast<int64_t>(tid) + lane * static_cast<int64_t>(blockDim.x);
        if (d < head_dim) {
          weighted[lane] += weight * musa_aiter_bfloat16_to_float_sparse_attn(v[v_base + d]);
        }
      }
    }
    __syncthreads();
  }

  if (tid == 0) {
    partial_max[tile_base] = local_max;
    partial_sum[tile_base] = denom;
  }
  for (int64_t lane = 0; lane < value_lanes; ++lane) {
    int64_t d = static_cast<int64_t>(tid) + lane * static_cast<int64_t>(blockDim.x);
    if (d < head_dim) {
      partial_out[out_base + d] = musa_aiter_float_to_bfloat16_sparse_attn(weighted[lane]);
    }
  }
}

extern "C" __global__ void musa_aiter_paged_sparse_decode_attention_bfloat16_tiled_stage2_kernel(
    const float* partial_max,
    const float* partial_sum,
    const uint16_t* partial_out,
    uint16_t* out,
    int64_t batch,
    int64_t num_q_heads,
    int64_t head_dim,
    int64_t num_tiles) {
  __shared__ float tile_rescale[256];
  int64_t head_idx = static_cast<int64_t>(blockIdx.x);
  if (head_idx >= batch * num_q_heads) {
    return;
  }
  int64_t tid = threadIdx.x;
  int64_t base = head_idx * num_tiles;
  float global_max = -3.4028234663852886e38f;
  for (int64_t tile = 0; tile < num_tiles; ++tile) {
    float value = partial_max[base + tile];
    if (value > global_max) {
      global_max = value;
    }
  }

  bool cache_rescale = num_tiles <= static_cast<int64_t>(blockDim.x);
  if (cache_rescale) {
    if (tid < num_tiles) {
      float tile_sum = partial_sum[base + tid];
      tile_rescale[tid] = tile_sum == 0.0f ? 0.0f : expf(partial_max[base + tid] - global_max);
    }
    __syncthreads();
  }

  float denom = 0.0f;
  for (int64_t tile = 0; tile < num_tiles; ++tile) {
    float tile_sum = partial_sum[base + tile];
    if (tile_sum != 0.0f) {
      float rescale = cache_rescale ? tile_rescale[tile] : expf(partial_max[base + tile] - global_max);
      denom += rescale * tile_sum;
    }
  }

  int64_t out_base = head_idx * head_dim;
  for (int64_t d = tid; d < head_dim; d += blockDim.x) {
    float weighted = 0.0f;
    for (int64_t tile = 0; tile < num_tiles; ++tile) {
      float tile_sum = partial_sum[base + tile];
      if (tile_sum != 0.0f) {
        float rescale = cache_rescale ? tile_rescale[tile] : expf(partial_max[base + tile] - global_max);
        weighted += rescale * musa_aiter_bfloat16_to_float_sparse_attn(partial_out[(base + tile) * head_dim + d]);
      }
    }
    float result = denom == 0.0f ? 0.0f : weighted / denom;
    out[out_base + d] = musa_aiter_float_to_bfloat16_sparse_attn(result);
  }
}

extern "C" void musa_aiter_paged_sparse_decode_attention_bfloat16_tiled(
    const uint16_t* q,
    const uint16_t* k,
    const uint16_t* v,
    const int64_t* kv_indptr,
    const int64_t* kv_indices,
    uint16_t* out,
    float* partial_max,
    float* partial_sum,
    uint16_t* partial_out,
    int64_t batch,
    int64_t index_count,
    int64_t num_tokens,
    int64_t num_q_heads,
    int64_t num_kv_heads,
    int64_t head_dim,
    int64_t tile_size,
    int64_t num_tiles,
    float scale,
    gpuStream_t stream) {
#ifdef __MUSACC__
  int threads = head_dim <= 128 ? 32 : 256;
  int stage1_blocks = static_cast<int>(batch * num_q_heads * num_tiles);
  int stage2_blocks = static_cast<int>(batch * num_q_heads);
  musa_aiter_paged_sparse_decode_attention_bfloat16_tiled_stage1_kernel<<<stage1_blocks, threads, 0, stream>>>(q, k, v, kv_indptr, kv_indices, partial_max, partial_sum, partial_out, batch, index_count, num_tokens, num_q_heads, num_kv_heads, head_dim, tile_size, num_tiles, scale);
  musa_aiter_paged_sparse_decode_attention_bfloat16_tiled_stage2_kernel<<<stage2_blocks, threads, 0, stream>>>(partial_max, partial_sum, partial_out, out, batch, num_q_heads, head_dim, num_tiles);
#endif
}

extern "C" int musa_aiter_paged_sparse_attention_abi_version() {
  return 2;
}
