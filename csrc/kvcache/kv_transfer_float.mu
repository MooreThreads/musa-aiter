#include <stdint.h>
#include "../common/musa_compat.h"

struct __attribute__((aligned(16))) MusaAiterU16x8 {
  unsigned long long lo;
  unsigned long long hi;
};

struct __attribute__((aligned(32))) MusaAiterU16x16 {
  unsigned long long x0;
  unsigned long long x1;
  unsigned long long x2;
  unsigned long long x3;
};

extern "C" __global__ void musa_aiter_kv_paged_to_linear_float_kernel(
    const float* key_cache,
    const float* value_cache,
    const int64_t* slot_mapping,
    float* out_k,
    float* out_v,
    int64_t num_tokens,
    int64_t num_blocks,
    int64_t num_kv_heads,
    int64_t head_dim,
    int64_t block_size) {
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
    out_k[idx] = 0.0f;
    out_v[idx] = 0.0f;
    return;
  }

  int64_t block = slot / block_size;
  int64_t offset = slot - block * block_size;
  if (block < 0 || block >= num_blocks) {
    out_k[idx] = 0.0f;
    out_v[idx] = 0.0f;
    return;
  }

  int64_t cache_idx = (((block * num_kv_heads + head) * head_dim + dim) * block_size + offset);
  out_k[idx] = key_cache[cache_idx];
  out_v[idx] = value_cache[cache_idx];
}

extern "C" void musa_aiter_kv_paged_to_linear_float(
    const float* key_cache,
    const float* value_cache,
    const int64_t* slot_mapping,
    float* out_k,
    float* out_v,
    int64_t num_tokens,
    int64_t num_blocks,
    int64_t num_kv_heads,
    int64_t head_dim,
    int64_t block_size,
    gpuStream_t stream) {
#ifdef __MUSACC__
  int threads = 256;
  int64_t elems = num_tokens * num_kv_heads * head_dim;
  int blocks = static_cast<int>((elems + threads - 1) / threads);
  musa_aiter_kv_paged_to_linear_float_kernel<<<blocks, threads, 0, stream>>>(
      key_cache,
      value_cache,
      slot_mapping,
      out_k,
      out_v,
      num_tokens,
      num_blocks,
      num_kv_heads,
      head_dim,
      block_size);
#endif
}

extern "C" __global__ void musa_aiter_kv_triton_indexed_to_dense_float_kernel(
    const float* key_cache,
    const float* value_cache,
    const int64_t* kv_indptr,
    const int64_t* kv_indices,
    float* out_k,
    float* out_v,
    int64_t batch,
    int64_t dense_len,
    int64_t index_count,
    int64_t num_tokens,
    int64_t num_kv_heads,
    int64_t head_dim) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  int64_t elems = batch * dense_len * num_kv_heads * head_dim;
  if (idx >= elems) {
    return;
  }

  int64_t dim = idx % head_dim;
  int64_t tmp = idx / head_dim;
  int64_t head = tmp % num_kv_heads;
  tmp = tmp / num_kv_heads;
  int64_t seq_pos = tmp % dense_len;
  int64_t batch_id = tmp / dense_len;

  int64_t start = kv_indptr[batch_id];
  int64_t end = kv_indptr[batch_id + 1];
  int64_t offset = start + seq_pos;
  if (start < 0 || end < start || offset >= end || offset < 0 || offset >= index_count) {
    out_k[idx] = 0.0f;
    out_v[idx] = 0.0f;
    return;
  }

  int64_t token = kv_indices[offset];
  if (token < 0 || token >= num_tokens) {
    out_k[idx] = 0.0f;
    out_v[idx] = 0.0f;
    return;
  }

  int64_t cache_idx = (token * num_kv_heads + head) * head_dim + dim;
  out_k[idx] = key_cache[cache_idx];
  out_v[idx] = value_cache[cache_idx];
}

extern "C" void musa_aiter_kv_triton_indexed_to_dense_float(
    const float* key_cache,
    const float* value_cache,
    const int64_t* kv_indptr,
    const int64_t* kv_indices,
    float* out_k,
    float* out_v,
    int64_t batch,
    int64_t dense_len,
    int64_t index_count,
    int64_t num_tokens,
    int64_t num_kv_heads,
    int64_t head_dim,
    gpuStream_t stream) {
#ifdef __MUSACC__
  int threads = 256;
  int64_t elems = batch * dense_len * num_kv_heads * head_dim;
  int blocks = static_cast<int>((elems + threads - 1) / threads);
  musa_aiter_kv_triton_indexed_to_dense_float_kernel<<<blocks, threads, 0, stream>>>(
      key_cache,
      value_cache,
      kv_indptr,
      kv_indices,
      out_k,
      out_v,
      batch,
      dense_len,
      index_count,
      num_tokens,
      num_kv_heads,
      head_dim);
#endif
}

extern "C" __global__ void musa_aiter_kv_linear_to_paged_float_kernel(
    const float* key_linear,
    const float* value_linear,
    const int64_t* slot_mapping,
    float* key_cache,
    float* value_cache,
    int64_t num_tokens,
    int64_t num_blocks,
    int64_t num_kv_heads,
    int64_t head_dim,
    int64_t block_size) {
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
  key_cache[cache_idx] = key_linear[idx];
  value_cache[cache_idx] = value_linear[idx];
}

extern "C" void musa_aiter_kv_linear_to_paged_float(
    const float* key_linear,
    const float* value_linear,
    const int64_t* slot_mapping,
    float* key_cache,
    float* value_cache,
    int64_t num_tokens,
    int64_t num_blocks,
    int64_t num_kv_heads,
    int64_t head_dim,
    int64_t block_size,
    gpuStream_t stream) {
#ifdef __MUSACC__
  int threads = 256;
  int64_t elems = num_tokens * num_kv_heads * head_dim;
  int blocks = static_cast<int>((elems + threads - 1) / threads);
  musa_aiter_kv_linear_to_paged_float_kernel<<<blocks, threads, 0, stream>>>(
      key_linear,
      value_linear,
      slot_mapping,
      key_cache,
      value_cache,
      num_tokens,
      num_blocks,
      num_kv_heads,
      head_dim,
      block_size);
#endif
}

extern "C" __global__ void musa_aiter_kv_paged_to_linear_multi_layer_float_kernel(
    const float* key_cache,
    const float* value_cache,
    const int64_t* slot_mapping,
    float* out_k,
    float* out_v,
    int64_t num_layers,
    int64_t num_tokens,
    int64_t num_blocks,
    int64_t num_kv_heads,
    int64_t head_dim,
    int64_t block_size) {
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
    out_k[idx] = 0.0f;
    out_v[idx] = 0.0f;
    return;
  }

  int64_t block = slot / block_size;
  int64_t offset = slot - block * block_size;
  if (block < 0 || block >= num_blocks) {
    out_k[idx] = 0.0f;
    out_v[idx] = 0.0f;
    return;
  }

  int64_t cache_idx = ((((layer * num_blocks + block) * num_kv_heads + head) * head_dim + dim) * block_size + offset);
  out_k[idx] = key_cache[cache_idx];
  out_v[idx] = value_cache[cache_idx];
}

extern "C" void musa_aiter_kv_paged_to_linear_multi_layer_float(
    const float* key_cache,
    const float* value_cache,
    const int64_t* slot_mapping,
    float* out_k,
    float* out_v,
    int64_t num_layers,
    int64_t num_tokens,
    int64_t num_blocks,
    int64_t num_kv_heads,
    int64_t head_dim,
    int64_t block_size,
    gpuStream_t stream) {
#ifdef __MUSACC__
  int threads = 256;
  int64_t elems = num_layers * num_tokens * num_kv_heads * head_dim;
  int blocks = static_cast<int>((elems + threads - 1) / threads);
  musa_aiter_kv_paged_to_linear_multi_layer_float_kernel<<<blocks, threads, 0, stream>>>(
      key_cache,
      value_cache,
      slot_mapping,
      out_k,
      out_v,
      num_layers,
      num_tokens,
      num_blocks,
      num_kv_heads,
      head_dim,
      block_size);
#endif
}

extern "C" __global__ void musa_aiter_kv_linear_to_paged_multi_layer_float_kernel(
    const float* key_linear,
    const float* value_linear,
    const int64_t* slot_mapping,
    float* key_cache,
    float* value_cache,
    int64_t num_layers,
    int64_t num_tokens,
    int64_t num_blocks,
    int64_t num_kv_heads,
    int64_t head_dim,
    int64_t block_size) {
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

  int64_t cache_idx = ((((layer * num_blocks + block) * num_kv_heads + head) * head_dim + dim) * block_size + offset);
  key_cache[cache_idx] = key_linear[idx];
  value_cache[cache_idx] = value_linear[idx];
}

extern "C" void musa_aiter_kv_linear_to_paged_multi_layer_float(
    const float* key_linear,
    const float* value_linear,
    const int64_t* slot_mapping,
    float* key_cache,
    float* value_cache,
    int64_t num_layers,
    int64_t num_tokens,
    int64_t num_blocks,
    int64_t num_kv_heads,
    int64_t head_dim,
    int64_t block_size,
    gpuStream_t stream) {
#ifdef __MUSACC__
  int threads = 256;
  int64_t elems = num_layers * num_tokens * num_kv_heads * head_dim;
  int blocks = static_cast<int>((elems + threads - 1) / threads);
  musa_aiter_kv_linear_to_paged_multi_layer_float_kernel<<<blocks, threads, 0, stream>>>(
      key_linear,
      value_linear,
      slot_mapping,
      key_cache,
      value_cache,
      num_layers,
      num_tokens,
      num_blocks,
      num_kv_heads,
      head_dim,
      block_size);
#endif
}

extern "C" __global__ void musa_aiter_kv_paged_to_merged_linear_float_kernel(
    const float* key_cache,
    const float* value_cache,
    const int64_t* slot_mapping,
    float* out_merged,
    int64_t num_tokens,
    int64_t num_blocks,
    int64_t num_kv_heads,
    int64_t head_dim,
    int64_t block_size) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  int64_t elems = num_tokens * 2 * num_kv_heads * head_dim;
  if (idx >= elems) {
    return;
  }

  int64_t dim = idx % head_dim;
  int64_t tmp = idx / head_dim;
  int64_t head = tmp % num_kv_heads;
  tmp = tmp / num_kv_heads;
  int64_t kv_axis = tmp % 2;
  int64_t token = tmp / 2;
  int64_t slot = slot_mapping[token];

  if (slot < 0) {
    out_merged[idx] = 0.0f;
    return;
  }

  int64_t block = slot / block_size;
  int64_t offset = slot - block * block_size;
  if (block < 0 || block >= num_blocks) {
    out_merged[idx] = 0.0f;
    return;
  }

  int64_t cache_idx = (((block * num_kv_heads + head) * head_dim + dim) * block_size + offset);
  out_merged[idx] = kv_axis == 0 ? key_cache[cache_idx] : value_cache[cache_idx];
}

extern "C" void musa_aiter_kv_paged_to_merged_linear_float(
    const float* key_cache,
    const float* value_cache,
    const int64_t* slot_mapping,
    float* out_merged,
    int64_t num_tokens,
    int64_t num_blocks,
    int64_t num_kv_heads,
    int64_t head_dim,
    int64_t block_size,
    gpuStream_t stream) {
#ifdef __MUSACC__
  int threads = 256;
  int64_t elems = num_tokens * 2 * num_kv_heads * head_dim;
  int blocks = static_cast<int>((elems + threads - 1) / threads);
  musa_aiter_kv_paged_to_merged_linear_float_kernel<<<blocks, threads, 0, stream>>>(
      key_cache,
      value_cache,
      slot_mapping,
      out_merged,
      num_tokens,
      num_blocks,
      num_kv_heads,
      head_dim,
      block_size);
#endif
}

extern "C" __global__ void musa_aiter_kv_merged_linear_to_paged_float_kernel(
    const float* merged_linear,
    const int64_t* slot_mapping,
    float* key_cache,
    float* value_cache,
    int64_t num_tokens,
    int64_t num_blocks,
    int64_t num_kv_heads,
    int64_t head_dim,
    int64_t block_size) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  int64_t elems = num_tokens * 2 * num_kv_heads * head_dim;
  if (idx >= elems) {
    return;
  }

  int64_t dim = idx % head_dim;
  int64_t tmp = idx / head_dim;
  int64_t head = tmp % num_kv_heads;
  tmp = tmp / num_kv_heads;
  int64_t kv_axis = tmp % 2;
  int64_t token = tmp / 2;
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
  if (kv_axis == 0) {
    key_cache[cache_idx] = merged_linear[idx];
  } else {
    value_cache[cache_idx] = merged_linear[idx];
  }
}

extern "C" void musa_aiter_kv_merged_linear_to_paged_float(
    const float* merged_linear,
    const int64_t* slot_mapping,
    float* key_cache,
    float* value_cache,
    int64_t num_tokens,
    int64_t num_blocks,
    int64_t num_kv_heads,
    int64_t head_dim,
    int64_t block_size,
    gpuStream_t stream) {
#ifdef __MUSACC__
  int threads = 256;
  int64_t elems = num_tokens * 2 * num_kv_heads * head_dim;
  int blocks = static_cast<int>((elems + threads - 1) / threads);
  musa_aiter_kv_merged_linear_to_paged_float_kernel<<<blocks, threads, 0, stream>>>(
      merged_linear,
      slot_mapping,
      key_cache,
      value_cache,
      num_tokens,
      num_blocks,
      num_kv_heads,
      head_dim,
      block_size);
#endif
}

extern "C" __global__ void musa_aiter_kv_paged_to_merged_linear_multi_layer_float_kernel(
    const float* key_cache,
    const float* value_cache,
    const int64_t* slot_mapping,
    float* out_merged,
    int64_t num_layers,
    int64_t num_tokens,
    int64_t num_blocks,
    int64_t num_kv_heads,
    int64_t head_dim,
    int64_t block_size) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  int64_t elems = num_layers * num_tokens * 2 * num_kv_heads * head_dim;
  if (idx >= elems) {
    return;
  }

  int64_t dim = idx % head_dim;
  int64_t tmp = idx / head_dim;
  int64_t head = tmp % num_kv_heads;
  tmp = tmp / num_kv_heads;
  int64_t kv_axis = tmp % 2;
  tmp = tmp / 2;
  int64_t token = tmp % num_tokens;
  int64_t layer = tmp / num_tokens;
  int64_t slot = slot_mapping[token];

  if (slot < 0) {
    out_merged[idx] = 0.0f;
    return;
  }

  int64_t block = slot / block_size;
  int64_t offset = slot - block * block_size;
  if (block < 0 || block >= num_blocks) {
    out_merged[idx] = 0.0f;
    return;
  }

  int64_t cache_idx = ((((layer * num_blocks + block) * num_kv_heads + head) * head_dim + dim) * block_size + offset);
  out_merged[idx] = kv_axis == 0 ? key_cache[cache_idx] : value_cache[cache_idx];
}

extern "C" void musa_aiter_kv_paged_to_merged_linear_multi_layer_float(
    const float* key_cache,
    const float* value_cache,
    const int64_t* slot_mapping,
    float* out_merged,
    int64_t num_layers,
    int64_t num_tokens,
    int64_t num_blocks,
    int64_t num_kv_heads,
    int64_t head_dim,
    int64_t block_size,
    gpuStream_t stream) {
#ifdef __MUSACC__
  int threads = 256;
  int64_t elems = num_layers * num_tokens * 2 * num_kv_heads * head_dim;
  int blocks = static_cast<int>((elems + threads - 1) / threads);
  musa_aiter_kv_paged_to_merged_linear_multi_layer_float_kernel<<<blocks, threads, 0, stream>>>(
      key_cache,
      value_cache,
      slot_mapping,
      out_merged,
      num_layers,
      num_tokens,
      num_blocks,
      num_kv_heads,
      head_dim,
      block_size);
#endif
}

extern "C" __global__ void musa_aiter_kv_merged_linear_to_paged_multi_layer_float_kernel(
    const float* merged_linear,
    const int64_t* slot_mapping,
    float* key_cache,
    float* value_cache,
    int64_t num_layers,
    int64_t num_tokens,
    int64_t num_blocks,
    int64_t num_kv_heads,
    int64_t head_dim,
    int64_t block_size) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  int64_t elems = num_layers * num_tokens * 2 * num_kv_heads * head_dim;
  if (idx >= elems) {
    return;
  }

  int64_t dim = idx % head_dim;
  int64_t tmp = idx / head_dim;
  int64_t head = tmp % num_kv_heads;
  tmp = tmp / num_kv_heads;
  int64_t kv_axis = tmp % 2;
  tmp = tmp / 2;
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

  int64_t cache_idx = ((((layer * num_blocks + block) * num_kv_heads + head) * head_dim + dim) * block_size + offset);
  if (kv_axis == 0) {
    key_cache[cache_idx] = merged_linear[idx];
  } else {
    value_cache[cache_idx] = merged_linear[idx];
  }
}

extern "C" void musa_aiter_kv_merged_linear_to_paged_multi_layer_float(
    const float* merged_linear,
    const int64_t* slot_mapping,
    float* key_cache,
    float* value_cache,
    int64_t num_layers,
    int64_t num_tokens,
    int64_t num_blocks,
    int64_t num_kv_heads,
    int64_t head_dim,
    int64_t block_size,
    gpuStream_t stream) {
#ifdef __MUSACC__
  int threads = 256;
  int64_t elems = num_layers * num_tokens * 2 * num_kv_heads * head_dim;
  int blocks = static_cast<int>((elems + threads - 1) / threads);
  musa_aiter_kv_merged_linear_to_paged_multi_layer_float_kernel<<<blocks, threads, 0, stream>>>(
      merged_linear,
      slot_mapping,
      key_cache,
      value_cache,
      num_layers,
      num_tokens,
      num_blocks,
      num_kv_heads,
      head_dim,
      block_size);
#endif
}

extern "C" __global__ void musa_aiter_kv_paged_to_mla_linear_float_kernel(
    const float* kv_cache,
    const float* k_pe_cache,
    const int64_t* slot_mapping,
    float* out_kv,
    float* out_k_pe,
    int64_t num_tokens,
    int64_t num_blocks,
    int64_t block_size,
    int64_t kv_hidden_dim,
    int64_t pe_hidden_dim) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  int64_t row_width = kv_hidden_dim + pe_hidden_dim;
  int64_t elems = num_tokens * row_width;
  if (idx >= elems) {
    return;
  }

  int64_t inner = idx % row_width;
  int64_t token = idx / row_width;
  int64_t slot = slot_mapping[token];
  bool is_kv = inner < kv_hidden_dim;
  int64_t dim = is_kv ? inner : inner - kv_hidden_dim;

  if (slot < 0) {
    if (is_kv) {
      out_kv[token * kv_hidden_dim + dim] = 0.0f;
    } else {
      out_k_pe[token * pe_hidden_dim + dim] = 0.0f;
    }
    return;
  }

  int64_t block = slot / block_size;
  int64_t offset = slot - block * block_size;
  if (block < 0 || block >= num_blocks) {
    if (is_kv) {
      out_kv[token * kv_hidden_dim + dim] = 0.0f;
    } else {
      out_k_pe[token * pe_hidden_dim + dim] = 0.0f;
    }
    return;
  }

  if (is_kv) {
    int64_t cache_idx = (block * block_size + offset) * kv_hidden_dim + dim;
    out_kv[token * kv_hidden_dim + dim] = kv_cache[cache_idx];
  } else {
    int64_t cache_idx = (block * block_size + offset) * pe_hidden_dim + dim;
    out_k_pe[token * pe_hidden_dim + dim] = k_pe_cache[cache_idx];
  }
}

extern "C" void musa_aiter_kv_paged_to_mla_linear_float(
    const float* kv_cache,
    const float* k_pe_cache,
    const int64_t* slot_mapping,
    float* out_kv,
    float* out_k_pe,
    int64_t num_tokens,
    int64_t num_blocks,
    int64_t block_size,
    int64_t kv_hidden_dim,
    int64_t pe_hidden_dim,
    gpuStream_t stream) {
#ifdef __MUSACC__
  int threads = 256;
  int64_t elems = num_tokens * (kv_hidden_dim + pe_hidden_dim);
  int blocks = static_cast<int>((elems + threads - 1) / threads);
  musa_aiter_kv_paged_to_mla_linear_float_kernel<<<blocks, threads, 0, stream>>>(
      kv_cache,
      k_pe_cache,
      slot_mapping,
      out_kv,
      out_k_pe,
      num_tokens,
      num_blocks,
      block_size,
      kv_hidden_dim,
      pe_hidden_dim);
#endif
}

extern "C" __global__ void musa_aiter_kv_mla_linear_to_paged_float_kernel(
    const float* kv_linear,
    const float* k_pe_linear,
    const int64_t* slot_mapping,
    float* kv_cache,
    float* k_pe_cache,
    int64_t num_tokens,
    int64_t num_blocks,
    int64_t block_size,
    int64_t kv_hidden_dim,
    int64_t pe_hidden_dim) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  int64_t row_width = kv_hidden_dim + pe_hidden_dim;
  int64_t elems = num_tokens * row_width;
  if (idx >= elems) {
    return;
  }

  int64_t inner = idx % row_width;
  int64_t token = idx / row_width;
  int64_t slot = slot_mapping[token];
  if (slot < 0) {
    return;
  }

  int64_t block = slot / block_size;
  int64_t offset = slot - block * block_size;
  if (block < 0 || block >= num_blocks) {
    return;
  }

  if (inner < kv_hidden_dim) {
    int64_t cache_idx = (block * block_size + offset) * kv_hidden_dim + inner;
    kv_cache[cache_idx] = kv_linear[token * kv_hidden_dim + inner];
  } else {
    int64_t dim = inner - kv_hidden_dim;
    int64_t cache_idx = (block * block_size + offset) * pe_hidden_dim + dim;
    k_pe_cache[cache_idx] = k_pe_linear[token * pe_hidden_dim + dim];
  }
}

extern "C" void musa_aiter_kv_mla_linear_to_paged_float(
    const float* kv_linear,
    const float* k_pe_linear,
    const int64_t* slot_mapping,
    float* kv_cache,
    float* k_pe_cache,
    int64_t num_tokens,
    int64_t num_blocks,
    int64_t block_size,
    int64_t kv_hidden_dim,
    int64_t pe_hidden_dim,
    gpuStream_t stream) {
#ifdef __MUSACC__
  int threads = 256;
  int64_t elems = num_tokens * (kv_hidden_dim + pe_hidden_dim);
  int blocks = static_cast<int>((elems + threads - 1) / threads);
  musa_aiter_kv_mla_linear_to_paged_float_kernel<<<blocks, threads, 0, stream>>>(
      kv_linear,
      k_pe_linear,
      slot_mapping,
      kv_cache,
      k_pe_cache,
      num_tokens,
      num_blocks,
      block_size,
      kv_hidden_dim,
      pe_hidden_dim);
#endif
}

#ifdef MUSA_AITER_ENABLE_CACHEGEN_NATIVE
extern "C" __global__ void musa_aiter_kv_cachegen_encode_float_kernel(
    const float* input,
    unsigned char* encoded,
    float min_val,
    float scale,
    int64_t numel,
    int64_t num_bins) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= numel) {
    return;
  }
  float value = (input[idx] - min_val) * scale;
  int64_t bin = static_cast<int64_t>(value + 0.5f);
  if (bin < 0) {
    bin = 0;
  }
  if (bin >= num_bins) {
    bin = num_bins - 1;
  }
  encoded[idx] = static_cast<unsigned char>(bin);
}

extern "C" void musa_aiter_kv_cachegen_encode_float(
    const float* input,
    unsigned char* encoded,
    float min_val,
    float scale,
    int64_t numel,
    int64_t num_bins,
    gpuStream_t stream) {
#ifdef __MUSACC__
  int threads = 256;
  if (numel <= 0) {
    return;
  }
  int blocks = static_cast<int>((numel + threads - 1) / threads);
  musa_aiter_kv_cachegen_encode_float_kernel<<<blocks, threads, 0, stream>>>(
      input,
      encoded,
      min_val,
      scale,
      numel,
      num_bins);
#endif
}

extern "C" __global__ void musa_aiter_kv_cachegen_decode_float_kernel(
    const unsigned char* encoded,
    const float* codebook,
    float* output,
    int64_t numel) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= numel) {
    return;
  }
  output[idx] = codebook[static_cast<int64_t>(encoded[idx])];
}

extern "C" void musa_aiter_kv_cachegen_decode_float(
    const unsigned char* encoded,
    const float* codebook,
    float* output,
    int64_t numel,
    gpuStream_t stream) {
#ifdef __MUSACC__
  int threads = 256;
  if (numel <= 0) {
    return;
  }
  int blocks = static_cast<int>((numel + threads - 1) / threads);
  musa_aiter_kv_cachegen_decode_float_kernel<<<blocks, threads, 0, stream>>>(
      encoded,
      codebook,
      output,
      numel);
#endif
}
#endif

extern "C" __global__ void musa_aiter_kv_copy_paged_blocks_float_kernel(
    const float* src_key_cache,
    const float* src_value_cache,
    float* dst_key_cache,
    float* dst_value_cache,
    const int64_t* block_mapping,
    int64_t num_pairs,
    int64_t num_layers,
    int64_t src_blocks,
    int64_t dst_blocks,
    int64_t num_kv_heads,
    int64_t head_dim,
    int64_t block_size) {
  int64_t elems_per_pair = num_layers * num_kv_heads * head_dim * block_size;
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  int64_t elems = num_pairs * elems_per_pair;
  if (idx >= elems) {
    return;
  }

  int64_t inner = idx % elems_per_pair;
  int64_t pair = idx / elems_per_pair;
  int64_t offset = inner % block_size;
  inner = inner / block_size;
  int64_t dim = inner % head_dim;
  inner = inner / head_dim;
  int64_t head = inner % num_kv_heads;
  int64_t layer = inner / num_kv_heads;

  int64_t src_block = block_mapping[pair * 2];
  int64_t dst_block = block_mapping[pair * 2 + 1];
  if (src_block < 0 || dst_block < 0 || src_block >= src_blocks || dst_block >= dst_blocks) {
    return;
  }

  int64_t src_idx = ((((layer * src_blocks + src_block) * num_kv_heads + head) * head_dim + dim) * block_size + offset);
  int64_t dst_idx = ((((layer * dst_blocks + dst_block) * num_kv_heads + head) * head_dim + dim) * block_size + offset);
  dst_key_cache[dst_idx] = src_key_cache[src_idx];
  dst_value_cache[dst_idx] = src_value_cache[src_idx];
}

extern "C" void musa_aiter_kv_copy_paged_blocks_float(
    const float* src_key_cache,
    const float* src_value_cache,
    float* dst_key_cache,
    float* dst_value_cache,
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
  int threads = 256;
  int64_t elems = num_pairs * num_layers * num_kv_heads * head_dim * block_size;
  if (elems <= 0) {
    return;
  }
  int blocks = static_cast<int>((elems + threads - 1) / threads);
  musa_aiter_kv_copy_paged_blocks_float_kernel<<<blocks, threads, 0, stream>>>(
      src_key_cache,
      src_value_cache,
      dst_key_cache,
      dst_value_cache,
      block_mapping,
      num_pairs,
      num_layers,
      src_blocks,
      dst_blocks,
      num_kv_heads,
      head_dim,
      block_size);
#endif
}

extern "C" __global__ void musa_aiter_kv_load_and_reshape_flash_float_kernel(
    const float* key_cache,
    const float* value_cache,
    const int64_t* block_tables,
    const int64_t* seq_lens,
    float* out_k,
    float* out_v,
    int64_t batch,
    int64_t dense_len,
    int64_t max_blocks,
    int64_t num_blocks,
    int64_t num_kv_heads,
    int64_t head_dim,
    int64_t block_size) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  int64_t elems = batch * dense_len * num_kv_heads * head_dim;
  if (idx >= elems) {
    return;
  }

  int64_t dim = idx % head_dim;
  int64_t tmp = idx / head_dim;
  int64_t head = tmp % num_kv_heads;
  tmp = tmp / num_kv_heads;
  int64_t seq_pos = tmp % dense_len;
  int64_t batch_id = tmp / dense_len;

  int64_t seq_len = seq_lens[batch_id];
  if (seq_pos >= seq_len) {
    out_k[idx] = 0.0f;
    out_v[idx] = 0.0f;
    return;
  }

  int64_t logical_block = seq_pos / block_size;
  int64_t offset = seq_pos - logical_block * block_size;
  if (logical_block < 0 || logical_block >= max_blocks) {
    out_k[idx] = 0.0f;
    out_v[idx] = 0.0f;
    return;
  }

  int64_t block_id = block_tables[batch_id * max_blocks + logical_block];
  if (block_id < 0 || block_id >= num_blocks) {
    out_k[idx] = 0.0f;
    out_v[idx] = 0.0f;
    return;
  }

  int64_t cache_idx = (((block_id * num_kv_heads + head) * head_dim + dim) * block_size + offset);
  out_k[idx] = key_cache[cache_idx];
  out_v[idx] = value_cache[cache_idx];
}

extern "C" void musa_aiter_kv_load_and_reshape_flash_float(
    const float* key_cache,
    const float* value_cache,
    const int64_t* block_tables,
    const int64_t* seq_lens,
    float* out_k,
    float* out_v,
    int64_t batch,
    int64_t dense_len,
    int64_t max_blocks,
    int64_t num_blocks,
    int64_t num_kv_heads,
    int64_t head_dim,
    int64_t block_size,
    gpuStream_t stream) {
#ifdef __MUSACC__
  int threads = 256;
  int64_t elems = batch * dense_len * num_kv_heads * head_dim;
  if (elems <= 0) {
    return;
  }
  int blocks = static_cast<int>((elems + threads - 1) / threads);
  musa_aiter_kv_load_and_reshape_flash_float_kernel<<<blocks, threads, 0, stream>>>(
      key_cache,
      value_cache,
      block_tables,
      seq_lens,
      out_k,
      out_v,
      batch,
      dense_len,
      max_blocks,
      num_blocks,
      num_kv_heads,
      head_dim,
      block_size);
#endif
}

extern "C" __global__ void musa_aiter_kv_reshape_and_cache_back_flash_float_kernel(
    const float* dense_k,
    const float* dense_v,
    const int64_t* block_tables,
    const int64_t* seq_lens,
    float* key_cache,
    float* value_cache,
    int64_t batch,
    int64_t dense_len,
    int64_t max_blocks,
    int64_t num_blocks,
    int64_t num_kv_heads,
    int64_t head_dim,
    int64_t block_size) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  int64_t elems = batch * dense_len * num_kv_heads * head_dim;
  if (idx >= elems) {
    return;
  }

  int64_t dim = idx % head_dim;
  int64_t tmp = idx / head_dim;
  int64_t head = tmp % num_kv_heads;
  tmp = tmp / num_kv_heads;
  int64_t seq_pos = tmp % dense_len;
  int64_t batch_id = tmp / dense_len;

  int64_t seq_len = seq_lens[batch_id];
  if (seq_pos >= seq_len) {
    return;
  }

  int64_t logical_block = seq_pos / block_size;
  int64_t offset = seq_pos - logical_block * block_size;
  if (logical_block < 0 || logical_block >= max_blocks) {
    return;
  }

  int64_t block_id = block_tables[batch_id * max_blocks + logical_block];
  if (block_id < 0 || block_id >= num_blocks) {
    return;
  }

  int64_t dense_idx = (((batch_id * dense_len + seq_pos) * num_kv_heads + head) * head_dim + dim);
  int64_t cache_idx = (((block_id * num_kv_heads + head) * head_dim + dim) * block_size + offset);
  key_cache[cache_idx] = dense_k[dense_idx];
  value_cache[cache_idx] = dense_v[dense_idx];
}

extern "C" void musa_aiter_kv_reshape_and_cache_back_flash_float(
    const float* dense_k,
    const float* dense_v,
    const int64_t* block_tables,
    const int64_t* seq_lens,
    float* key_cache,
    float* value_cache,
    int64_t batch,
    int64_t dense_len,
    int64_t max_blocks,
    int64_t num_blocks,
    int64_t num_kv_heads,
    int64_t head_dim,
    int64_t block_size,
    gpuStream_t stream) {
#ifdef __MUSACC__
  int threads = 256;
  int64_t elems = batch * dense_len * num_kv_heads * head_dim;
  if (elems <= 0) {
    return;
  }
  int blocks = static_cast<int>((elems + threads - 1) / threads);
  musa_aiter_kv_reshape_and_cache_back_flash_float_kernel<<<blocks, threads, 0, stream>>>(
      dense_k,
      dense_v,
      block_tables,
      seq_lens,
      key_cache,
      value_cache,
      batch,
      dense_len,
      max_blocks,
      num_blocks,
      num_kv_heads,
      head_dim,
      block_size);
#endif
}

extern "C" __global__ void musa_aiter_kv_paged_to_linear_u16_kernel(
    const uint16_t* key_cache,
    const uint16_t* value_cache,
    const int64_t* slot_mapping,
    uint16_t* out_k,
    uint16_t* out_v,
    int64_t num_tokens,
    int64_t num_blocks,
    int64_t num_kv_heads,
    int64_t head_dim,
    int64_t block_size) {
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
    out_k[idx] = 0;
    out_v[idx] = 0;
    return;
  }

  int64_t block = slot / block_size;
  int64_t offset = slot - block * block_size;
  if (block < 0 || block >= num_blocks) {
    out_k[idx] = 0;
    out_v[idx] = 0;
    return;
  }

  int64_t cache_idx = (((block * num_kv_heads + head) * head_dim + dim) * block_size + offset);
  out_k[idx] = key_cache[cache_idx];
  out_v[idx] = value_cache[cache_idx];
}

extern "C" __global__ void musa_aiter_kv_triton_indexed_to_dense_u16_kernel(
    const uint16_t* key_cache,
    const uint16_t* value_cache,
    const int64_t* kv_indptr,
    const int64_t* kv_indices,
    uint16_t* out_k,
    uint16_t* out_v,
    int64_t batch,
    int64_t dense_len,
    int64_t index_count,
    int64_t num_tokens,
    int64_t num_kv_heads,
    int64_t head_dim) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  int64_t elems = batch * dense_len * num_kv_heads * head_dim;
  if (idx >= elems) {
    return;
  }

  int64_t dim = idx % head_dim;
  int64_t tmp = idx / head_dim;
  int64_t head = tmp % num_kv_heads;
  tmp = tmp / num_kv_heads;
  int64_t seq_pos = tmp % dense_len;
  int64_t batch_id = tmp / dense_len;

  int64_t start = kv_indptr[batch_id];
  int64_t end = kv_indptr[batch_id + 1];
  int64_t offset = start + seq_pos;
  if (start < 0 || end < start || offset >= end || offset < 0 || offset >= index_count) {
    out_k[idx] = 0;
    out_v[idx] = 0;
    return;
  }

  int64_t token = kv_indices[offset];
  if (token < 0 || token >= num_tokens) {
    out_k[idx] = 0;
    out_v[idx] = 0;
    return;
  }

  int64_t cache_idx = (token * num_kv_heads + head) * head_dim + dim;
  out_k[idx] = key_cache[cache_idx];
  out_v[idx] = value_cache[cache_idx];
}

extern "C" __global__ void musa_aiter_kv_linear_to_paged_u16_kernel(
    const uint16_t* key_linear,
    const uint16_t* value_linear,
    const int64_t* slot_mapping,
    uint16_t* key_cache,
    uint16_t* value_cache,
    int64_t num_tokens,
    int64_t num_blocks,
    int64_t num_kv_heads,
    int64_t head_dim,
    int64_t block_size) {
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
  key_cache[cache_idx] = key_linear[idx];
  value_cache[cache_idx] = value_linear[idx];
}

extern "C" __global__ void musa_aiter_kv_paged_to_linear_multi_layer_u16_kernel(
    const uint16_t* key_cache,
    const uint16_t* value_cache,
    const int64_t* slot_mapping,
    uint16_t* out_k,
    uint16_t* out_v,
    int64_t num_layers,
    int64_t num_tokens,
    int64_t num_blocks,
    int64_t num_kv_heads,
    int64_t head_dim,
    int64_t block_size) {
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
    out_k[idx] = 0;
    out_v[idx] = 0;
    return;
  }

  int64_t block = slot / block_size;
  int64_t offset = slot - block * block_size;
  if (block < 0 || block >= num_blocks) {
    out_k[idx] = 0;
    out_v[idx] = 0;
    return;
  }

  int64_t cache_idx = ((((layer * num_blocks + block) * num_kv_heads + head) * head_dim + dim) * block_size + offset);
  out_k[idx] = key_cache[cache_idx];
  out_v[idx] = value_cache[cache_idx];
}

extern "C" __global__ void musa_aiter_kv_linear_to_paged_multi_layer_u16_kernel(
    const uint16_t* key_linear,
    const uint16_t* value_linear,
    const int64_t* slot_mapping,
    uint16_t* key_cache,
    uint16_t* value_cache,
    int64_t num_layers,
    int64_t num_tokens,
    int64_t num_blocks,
    int64_t num_kv_heads,
    int64_t head_dim,
    int64_t block_size) {
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

  int64_t cache_idx = ((((layer * num_blocks + block) * num_kv_heads + head) * head_dim + dim) * block_size + offset);
  key_cache[cache_idx] = key_linear[idx];
  value_cache[cache_idx] = value_linear[idx];
}

extern "C" __global__ void musa_aiter_kv_copy_paged_blocks_u16_kernel(
    const uint16_t* src_key_cache,
    const uint16_t* src_value_cache,
    uint16_t* dst_key_cache,
    uint16_t* dst_value_cache,
    const int64_t* block_mapping,
    int64_t num_pairs,
    int64_t num_layers,
    int64_t src_blocks,
    int64_t dst_blocks,
    int64_t num_kv_heads,
    int64_t head_dim,
    int64_t block_size) {
  int64_t elems_per_pair = num_layers * num_kv_heads * head_dim * block_size;
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  int64_t elems = num_pairs * elems_per_pair;
  if (idx >= elems) {
    return;
  }

  int64_t inner = idx % elems_per_pair;
  int64_t pair = idx / elems_per_pair;
  int64_t offset = inner % block_size;
  inner = inner / block_size;
  int64_t dim = inner % head_dim;
  inner = inner / head_dim;
  int64_t head = inner % num_kv_heads;
  int64_t layer = inner / num_kv_heads;

  int64_t src_block = block_mapping[pair * 2];
  int64_t dst_block = block_mapping[pair * 2 + 1];
  if (src_block < 0 || dst_block < 0 || src_block >= src_blocks || dst_block >= dst_blocks) {
    return;
  }

  int64_t src_idx = ((((layer * src_blocks + src_block) * num_kv_heads + head) * head_dim + dim) * block_size + offset);
  int64_t dst_idx = ((((layer * dst_blocks + dst_block) * num_kv_heads + head) * head_dim + dim) * block_size + offset);
  dst_key_cache[dst_idx] = src_key_cache[src_idx];
  dst_value_cache[dst_idx] = src_value_cache[src_idx];
}

extern "C" __global__ void musa_aiter_kv_load_and_reshape_flash_u16_kernel(
    const uint16_t* key_cache,
    const uint16_t* value_cache,
    const int64_t* block_tables,
    const int64_t* seq_lens,
    uint16_t* out_k,
    uint16_t* out_v,
    int64_t batch,
    int64_t dense_len,
    int64_t max_blocks,
    int64_t num_blocks,
    int64_t num_kv_heads,
    int64_t head_dim,
    int64_t block_size) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  int64_t elems = batch * dense_len * num_kv_heads * head_dim;
  if (idx >= elems) {
    return;
  }

  int64_t dim = idx % head_dim;
  int64_t tmp = idx / head_dim;
  int64_t head = tmp % num_kv_heads;
  tmp = tmp / num_kv_heads;
  int64_t seq_pos = tmp % dense_len;
  int64_t batch_id = tmp / dense_len;

  int64_t seq_len = seq_lens[batch_id];
  if (seq_pos >= seq_len) {
    out_k[idx] = 0;
    out_v[idx] = 0;
    return;
  }

  int64_t logical_block = seq_pos / block_size;
  int64_t offset = seq_pos - logical_block * block_size;
  if (logical_block < 0 || logical_block >= max_blocks) {
    out_k[idx] = 0;
    out_v[idx] = 0;
    return;
  }

  int64_t block_id = block_tables[batch_id * max_blocks + logical_block];
  if (block_id < 0 || block_id >= num_blocks) {
    out_k[idx] = 0;
    out_v[idx] = 0;
    return;
  }

  int64_t cache_idx = (((block_id * num_kv_heads + head) * head_dim + dim) * block_size + offset);
  out_k[idx] = key_cache[cache_idx];
  out_v[idx] = value_cache[cache_idx];
}

extern "C" __global__ void musa_aiter_kv_reshape_and_cache_back_flash_u16_kernel(
    const uint16_t* dense_k,
    const uint16_t* dense_v,
    const int64_t* block_tables,
    const int64_t* seq_lens,
    uint16_t* key_cache,
    uint16_t* value_cache,
    int64_t batch,
    int64_t dense_len,
    int64_t max_blocks,
    int64_t num_blocks,
    int64_t num_kv_heads,
    int64_t head_dim,
    int64_t block_size) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  int64_t elems = batch * dense_len * num_kv_heads * head_dim;
  if (idx >= elems) {
    return;
  }

  int64_t dim = idx % head_dim;
  int64_t tmp = idx / head_dim;
  int64_t head = tmp % num_kv_heads;
  tmp = tmp / num_kv_heads;
  int64_t seq_pos = tmp % dense_len;
  int64_t batch_id = tmp / dense_len;

  int64_t seq_len = seq_lens[batch_id];
  if (seq_pos >= seq_len) {
    return;
  }

  int64_t logical_block = seq_pos / block_size;
  int64_t offset = seq_pos - logical_block * block_size;
  if (logical_block < 0 || logical_block >= max_blocks) {
    return;
  }

  int64_t block_id = block_tables[batch_id * max_blocks + logical_block];
  if (block_id < 0 || block_id >= num_blocks) {
    return;
  }

  int64_t dense_idx = (((batch_id * dense_len + seq_pos) * num_kv_heads + head) * head_dim + dim);
  int64_t cache_idx = (((block_id * num_kv_heads + head) * head_dim + dim) * block_size + offset);
  key_cache[cache_idx] = dense_k[dense_idx];
  value_cache[cache_idx] = dense_v[dense_idx];
}

#define MUSA_AITER_DEFINE_U16_TRANSFER_WRAPPERS(SUFFIX) \
extern "C" void musa_aiter_kv_paged_to_linear_##SUFFIX( \
    const uint16_t* key_cache, const uint16_t* value_cache, const int64_t* slot_mapping, uint16_t* out_k, uint16_t* out_v, \
    int64_t num_tokens, int64_t num_blocks, int64_t num_kv_heads, int64_t head_dim, int64_t block_size, gpuStream_t stream) { \
  int threads = 256; \
  int64_t elems = num_tokens * num_kv_heads * head_dim; \
  int blocks = static_cast<int>((elems + threads - 1) / threads); \
  musa_aiter_kv_paged_to_linear_u16_kernel<<<blocks, threads, 0, stream>>>(key_cache, value_cache, slot_mapping, out_k, out_v, num_tokens, num_blocks, num_kv_heads, head_dim, block_size); \
} \
extern "C" void musa_aiter_kv_linear_to_paged_##SUFFIX( \
    const uint16_t* key_linear, const uint16_t* value_linear, const int64_t* slot_mapping, uint16_t* key_cache, uint16_t* value_cache, \
    int64_t num_tokens, int64_t num_blocks, int64_t num_kv_heads, int64_t head_dim, int64_t block_size, gpuStream_t stream) { \
  int threads = 256; \
  int64_t elems = num_tokens * num_kv_heads * head_dim; \
  int blocks = static_cast<int>((elems + threads - 1) / threads); \
  musa_aiter_kv_linear_to_paged_u16_kernel<<<blocks, threads, 0, stream>>>(key_linear, value_linear, slot_mapping, key_cache, value_cache, num_tokens, num_blocks, num_kv_heads, head_dim, block_size); \
} \
extern "C" void musa_aiter_kv_paged_to_linear_multi_layer_##SUFFIX( \
    const uint16_t* key_cache, const uint16_t* value_cache, const int64_t* slot_mapping, uint16_t* out_k, uint16_t* out_v, \
    int64_t num_layers, int64_t num_tokens, int64_t num_blocks, int64_t num_kv_heads, int64_t head_dim, int64_t block_size, gpuStream_t stream) { \
  int threads = 256; \
  int64_t elems = num_layers * num_tokens * num_kv_heads * head_dim; \
  int blocks = static_cast<int>((elems + threads - 1) / threads); \
  musa_aiter_kv_paged_to_linear_multi_layer_u16_kernel<<<blocks, threads, 0, stream>>>(key_cache, value_cache, slot_mapping, out_k, out_v, num_layers, num_tokens, num_blocks, num_kv_heads, head_dim, block_size); \
} \
extern "C" void musa_aiter_kv_linear_to_paged_multi_layer_##SUFFIX( \
    const uint16_t* key_linear, const uint16_t* value_linear, const int64_t* slot_mapping, uint16_t* key_cache, uint16_t* value_cache, \
    int64_t num_layers, int64_t num_tokens, int64_t num_blocks, int64_t num_kv_heads, int64_t head_dim, int64_t block_size, gpuStream_t stream) { \
  int threads = 256; \
  int64_t elems = num_layers * num_tokens * num_kv_heads * head_dim; \
  int blocks = static_cast<int>((elems + threads - 1) / threads); \
  musa_aiter_kv_linear_to_paged_multi_layer_u16_kernel<<<blocks, threads, 0, stream>>>(key_linear, value_linear, slot_mapping, key_cache, value_cache, num_layers, num_tokens, num_blocks, num_kv_heads, head_dim, block_size); \
} \
extern "C" void musa_aiter_kv_copy_paged_blocks_##SUFFIX( \
    const uint16_t* src_key_cache, const uint16_t* src_value_cache, uint16_t* dst_key_cache, uint16_t* dst_value_cache, const int64_t* block_mapping, \
    int64_t num_pairs, int64_t num_layers, int64_t src_blocks, int64_t dst_blocks, int64_t num_kv_heads, int64_t head_dim, int64_t block_size, gpuStream_t stream) { \
  int threads = 256; \
  int64_t elems = num_pairs * num_layers * num_kv_heads * head_dim * block_size; \
  int blocks = static_cast<int>((elems + threads - 1) / threads); \
  musa_aiter_kv_copy_paged_blocks_u16_kernel<<<blocks, threads, 0, stream>>>(src_key_cache, src_value_cache, dst_key_cache, dst_value_cache, block_mapping, num_pairs, num_layers, src_blocks, dst_blocks, num_kv_heads, head_dim, block_size); \
} \
extern "C" void musa_aiter_kv_load_and_reshape_flash_##SUFFIX( \
    const uint16_t* key_cache, const uint16_t* value_cache, const int64_t* block_tables, const int64_t* seq_lens, uint16_t* out_k, uint16_t* out_v, \
    int64_t batch, int64_t dense_len, int64_t max_blocks, int64_t num_blocks, int64_t num_kv_heads, int64_t head_dim, int64_t block_size, gpuStream_t stream) { \
  int threads = 256; \
  int64_t elems = batch * dense_len * num_kv_heads * head_dim; \
  int blocks = static_cast<int>((elems + threads - 1) / threads); \
  musa_aiter_kv_load_and_reshape_flash_u16_kernel<<<blocks, threads, 0, stream>>>(key_cache, value_cache, block_tables, seq_lens, out_k, out_v, batch, dense_len, max_blocks, num_blocks, num_kv_heads, head_dim, block_size); \
} \
extern "C" void musa_aiter_kv_reshape_and_cache_back_flash_##SUFFIX( \
    const uint16_t* dense_k, const uint16_t* dense_v, const int64_t* block_tables, const int64_t* seq_lens, uint16_t* key_cache, uint16_t* value_cache, \
    int64_t batch, int64_t dense_len, int64_t max_blocks, int64_t num_blocks, int64_t num_kv_heads, int64_t head_dim, int64_t block_size, gpuStream_t stream) { \
  int threads = 256; \
  int64_t elems = batch * dense_len * num_kv_heads * head_dim; \
  int blocks = static_cast<int>((elems + threads - 1) / threads); \
  musa_aiter_kv_reshape_and_cache_back_flash_u16_kernel<<<blocks, threads, 0, stream>>>(dense_k, dense_v, block_tables, seq_lens, key_cache, value_cache, batch, dense_len, max_blocks, num_blocks, num_kv_heads, head_dim, block_size); \
} \
extern "C" void musa_aiter_kv_triton_indexed_to_dense_##SUFFIX( \
    const uint16_t* key_cache, const uint16_t* value_cache, const int64_t* kv_indptr, const int64_t* kv_indices, uint16_t* out_k, uint16_t* out_v, \
    int64_t batch, int64_t dense_len, int64_t index_count, int64_t num_tokens, int64_t num_kv_heads, int64_t head_dim, gpuStream_t stream) { \
  int threads = 256; \
  int64_t elems = batch * dense_len * num_kv_heads * head_dim; \
  int blocks = static_cast<int>((elems + threads - 1) / threads); \
  musa_aiter_kv_triton_indexed_to_dense_u16_kernel<<<blocks, threads, 0, stream>>>(key_cache, value_cache, kv_indptr, kv_indices, out_k, out_v, batch, dense_len, index_count, num_tokens, num_kv_heads, head_dim); \
}

MUSA_AITER_DEFINE_U16_TRANSFER_WRAPPERS(half)
MUSA_AITER_DEFINE_U16_TRANSFER_WRAPPERS(bfloat16)

extern "C" __global__ void lmcache_kv_buffer_to_paged_kernel(
    const float* kv_buffer,
    float* paged_kv_cache,
    const int64_t* slot_mapping,
    int64_t num_layers,
    int64_t num_tokens,
    int64_t num_blocks,
    int64_t block_size,
    int64_t num_kv_heads,
    int64_t head_dim) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  int64_t elems = 2 * num_layers * num_tokens * num_kv_heads * head_dim;
  if (idx >= elems) {
    return;
  }

  int64_t dim = idx % head_dim;
  int64_t tmp = idx / head_dim;
  int64_t head = tmp % num_kv_heads;
  tmp = tmp / num_kv_heads;
  int64_t token = tmp % num_tokens;
  tmp = tmp / num_tokens;
  int64_t layer = tmp % num_layers;
  int64_t kv_axis = tmp / num_layers;
  int64_t slot = slot_mapping[token];
  if (slot < 0) {
    return;
  }
  int64_t block = slot / block_size;
  int64_t offset = slot - block * block_size;
  if (block < 0 || block >= num_blocks) {
    return;
  }

  int64_t cache_idx = (((((layer * 2 + kv_axis) * num_blocks + block) * block_size + offset) * num_kv_heads + head) * head_dim + dim);
  paged_kv_cache[cache_idx] = kv_buffer[idx];
}

extern "C" void lmcache_kv_buffer_to_paged(
    const float* kv_buffer,
    float* paged_kv_cache,
    const int64_t* slot_mapping,
    int64_t num_layers,
    int64_t num_tokens,
    int64_t num_blocks,
    int64_t block_size,
    int64_t num_kv_heads,
    int64_t head_dim,
    gpuStream_t stream) {
#ifdef __MUSACC__
  int threads = 256;
  int64_t elems = 2 * num_layers * num_tokens * num_kv_heads * head_dim;
  if (elems <= 0) {
    return;
  }
  int blocks = static_cast<int>((elems + threads - 1) / threads);
  lmcache_kv_buffer_to_paged_kernel<<<blocks, threads, 0, stream>>>(
      kv_buffer, paged_kv_cache, slot_mapping, num_layers, num_tokens, num_blocks, block_size, num_kv_heads, head_dim);
#endif
}

extern "C" __global__ void lmcache_paged_to_kv_buffer_kernel(
    const float* paged_kv_cache,
    float* kv_buffer,
    const int64_t* slot_mapping,
    int64_t num_layers,
    int64_t num_tokens,
    int64_t num_blocks,
    int64_t block_size,
    int64_t num_kv_heads,
    int64_t head_dim) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  int64_t elems = 2 * num_layers * num_tokens * num_kv_heads * head_dim;
  if (idx >= elems) {
    return;
  }

  int64_t dim = idx % head_dim;
  int64_t tmp = idx / head_dim;
  int64_t head = tmp % num_kv_heads;
  tmp = tmp / num_kv_heads;
  int64_t token = tmp % num_tokens;
  tmp = tmp / num_tokens;
  int64_t layer = tmp % num_layers;
  int64_t kv_axis = tmp / num_layers;
  int64_t slot = slot_mapping[token];
  if (slot < 0) {
    kv_buffer[idx] = 0.0f;
    return;
  }
  int64_t block = slot / block_size;
  int64_t offset = slot - block * block_size;
  if (block < 0 || block >= num_blocks) {
    kv_buffer[idx] = 0.0f;
    return;
  }

  int64_t cache_idx = (((((layer * 2 + kv_axis) * num_blocks + block) * block_size + offset) * num_kv_heads + head) * head_dim + dim);
  kv_buffer[idx] = paged_kv_cache[cache_idx];
}

extern "C" void lmcache_paged_to_kv_buffer(
    const float* paged_kv_cache,
    float* kv_buffer,
    const int64_t* slot_mapping,
    int64_t num_layers,
    int64_t num_tokens,
    int64_t num_blocks,
    int64_t block_size,
    int64_t num_kv_heads,
    int64_t head_dim,
    gpuStream_t stream) {
#ifdef __MUSACC__
  int threads = 256;
  int64_t elems = 2 * num_layers * num_tokens * num_kv_heads * head_dim;
  if (elems <= 0) {
    return;
  }
  int blocks = static_cast<int>((elems + threads - 1) / threads);
  lmcache_paged_to_kv_buffer_kernel<<<blocks, threads, 0, stream>>>(
      paged_kv_cache, kv_buffer, slot_mapping, num_layers, num_tokens, num_blocks, block_size, num_kv_heads, head_dim);
#endif
}

extern "C" __global__ void lmcache_kv_buffer_to_paged_mla_kernel(
    const float* kv_buffer,
    float* paged_kv_cache,
    const int64_t* slot_mapping,
    int64_t num_layers,
    int64_t num_tokens,
    int64_t num_blocks,
    int64_t block_size,
    int64_t hidden_size) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  int64_t elems = num_layers * num_tokens * hidden_size;
  if (idx >= elems) {
    return;
  }

  int64_t dim = idx % hidden_size;
  int64_t tmp = idx / hidden_size;
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

  int64_t cache_idx = (((layer * num_blocks + block) * block_size + offset) * hidden_size + dim);
  paged_kv_cache[cache_idx] = kv_buffer[idx];
}

extern "C" void lmcache_kv_buffer_to_paged_mla(
    const float* kv_buffer,
    float* paged_kv_cache,
    const int64_t* slot_mapping,
    int64_t num_layers,
    int64_t num_tokens,
    int64_t num_blocks,
    int64_t block_size,
    int64_t hidden_size,
    gpuStream_t stream) {
#ifdef __MUSACC__
  int threads = 256;
  int64_t elems = num_layers * num_tokens * hidden_size;
  if (elems <= 0) {
    return;
  }
  int blocks = static_cast<int>((elems + threads - 1) / threads);
  lmcache_kv_buffer_to_paged_mla_kernel<<<blocks, threads, 0, stream>>>(
      kv_buffer, paged_kv_cache, slot_mapping, num_layers, num_tokens, num_blocks, block_size, hidden_size);
#endif
}

extern "C" __global__ void lmcache_paged_to_kv_buffer_mla_kernel(
    const float* paged_kv_cache,
    float* kv_buffer,
    const int64_t* slot_mapping,
    int64_t num_layers,
    int64_t num_tokens,
    int64_t num_blocks,
    int64_t block_size,
    int64_t hidden_size) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  int64_t elems = num_layers * num_tokens * hidden_size;
  if (idx >= elems) {
    return;
  }

  int64_t dim = idx % hidden_size;
  int64_t tmp = idx / hidden_size;
  int64_t token = tmp % num_tokens;
  int64_t layer = tmp / num_tokens;
  int64_t slot = slot_mapping[token];
  if (slot < 0) {
    kv_buffer[idx] = 0.0f;
    return;
  }
  int64_t block = slot / block_size;
  int64_t offset = slot - block * block_size;
  if (block < 0 || block >= num_blocks) {
    kv_buffer[idx] = 0.0f;
    return;
  }

  int64_t cache_idx = (((layer * num_blocks + block) * block_size + offset) * hidden_size + dim);
  kv_buffer[idx] = paged_kv_cache[cache_idx];
}

extern "C" void lmcache_paged_to_kv_buffer_mla(
    const float* paged_kv_cache,
    float* kv_buffer,
    const int64_t* slot_mapping,
    int64_t num_layers,
    int64_t num_tokens,
    int64_t num_blocks,
    int64_t block_size,
    int64_t hidden_size,
    gpuStream_t stream) {
#ifdef __MUSACC__
  int threads = 256;
  int64_t elems = num_layers * num_tokens * hidden_size;
  if (elems <= 0) {
    return;
  }
  int blocks = static_cast<int>((elems + threads - 1) / threads);
  lmcache_paged_to_kv_buffer_mla_kernel<<<blocks, threads, 0, stream>>>(
      paged_kv_cache, kv_buffer, slot_mapping, num_layers, num_tokens, num_blocks, block_size, hidden_size);
#endif
}

extern "C" __global__ void lmcache_kv_buffer_to_paged_u16_kernel(
    const uint16_t* kv_buffer,
    uint16_t* paged_kv_cache,
    const int64_t* slot_mapping,
    int64_t num_layers,
    int64_t num_tokens,
    int64_t num_blocks,
    int64_t block_size,
    int64_t num_kv_heads,
    int64_t head_dim) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  int64_t elems = 2 * num_layers * num_tokens * num_kv_heads * head_dim;
  if (idx >= elems) return;
  int64_t dim = idx % head_dim;
  int64_t tmp = idx / head_dim;
  int64_t head = tmp % num_kv_heads;
  tmp /= num_kv_heads;
  int64_t token = tmp % num_tokens;
  tmp /= num_tokens;
  int64_t layer = tmp % num_layers;
  int64_t kv_axis = tmp / num_layers;
  int64_t slot = slot_mapping[token];
  if (slot < 0) return;
  int64_t block = slot / block_size;
  int64_t offset = slot - block * block_size;
  if (block < 0 || block >= num_blocks) return;
  int64_t cache_idx = (((((layer * 2 + kv_axis) * num_blocks + block) * block_size + offset) * num_kv_heads + head) * head_dim + dim);
  paged_kv_cache[cache_idx] = kv_buffer[idx];
}

extern "C" void lmcache_kv_buffer_to_paged_u16(
    const uint16_t* kv_buffer,
    uint16_t* paged_kv_cache,
    const int64_t* slot_mapping,
    int64_t num_layers,
    int64_t num_tokens,
    int64_t num_blocks,
    int64_t block_size,
    int64_t num_kv_heads,
    int64_t head_dim,
    gpuStream_t stream) {
#ifdef __MUSACC__
  int threads = 256;
  int64_t elems = 2 * num_layers * num_tokens * num_kv_heads * head_dim;
  if (elems <= 0) return;
  int blocks = static_cast<int>((elems + threads - 1) / threads);
  lmcache_kv_buffer_to_paged_u16_kernel<<<blocks, threads, 0, stream>>>(
      kv_buffer, paged_kv_cache, slot_mapping, num_layers, num_tokens, num_blocks, block_size, num_kv_heads, head_dim);
#endif
}

extern "C" __global__ void lmcache_paged_to_kv_buffer_u16_kernel(
    const uint16_t* paged_kv_cache,
    uint16_t* kv_buffer,
    const int64_t* slot_mapping,
    int64_t num_layers,
    int64_t num_tokens,
    int64_t num_blocks,
    int64_t block_size,
    int64_t num_kv_heads,
    int64_t head_dim) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  int64_t elems = 2 * num_layers * num_tokens * num_kv_heads * head_dim;
  if (idx >= elems) return;
  int64_t dim = idx % head_dim;
  int64_t tmp = idx / head_dim;
  int64_t head = tmp % num_kv_heads;
  tmp /= num_kv_heads;
  int64_t token = tmp % num_tokens;
  tmp /= num_tokens;
  int64_t layer = tmp % num_layers;
  int64_t kv_axis = tmp / num_layers;
  int64_t slot = slot_mapping[token];
  if (slot < 0) {
    kv_buffer[idx] = static_cast<uint16_t>(0);
    return;
  }
  int64_t block = slot / block_size;
  int64_t offset = slot - block * block_size;
  if (block < 0 || block >= num_blocks) {
    kv_buffer[idx] = static_cast<uint16_t>(0);
    return;
  }
  int64_t cache_idx = (((((layer * 2 + kv_axis) * num_blocks + block) * block_size + offset) * num_kv_heads + head) * head_dim + dim);
  kv_buffer[idx] = paged_kv_cache[cache_idx];
}

extern "C" void lmcache_paged_to_kv_buffer_u16(
    const uint16_t* paged_kv_cache,
    uint16_t* kv_buffer,
    const int64_t* slot_mapping,
    int64_t num_layers,
    int64_t num_tokens,
    int64_t num_blocks,
    int64_t block_size,
    int64_t num_kv_heads,
    int64_t head_dim,
    gpuStream_t stream) {
#ifdef __MUSACC__
  int threads = 256;
  int64_t elems = 2 * num_layers * num_tokens * num_kv_heads * head_dim;
  if (elems <= 0) return;
  int blocks = static_cast<int>((elems + threads - 1) / threads);
  lmcache_paged_to_kv_buffer_u16_kernel<<<blocks, threads, 0, stream>>>(
      paged_kv_cache, kv_buffer, slot_mapping, num_layers, num_tokens, num_blocks, block_size, num_kv_heads, head_dim);
#endif
}

extern "C" __global__ void lmcache_kv_buffer_to_paged_mla_u16_kernel(
    const uint16_t* kv_buffer,
    uint16_t* paged_kv_cache,
    const int64_t* slot_mapping,
    int64_t num_layers,
    int64_t num_tokens,
    int64_t num_blocks,
    int64_t block_size,
    int64_t hidden_size) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  int64_t elems = num_layers * num_tokens * hidden_size;
  if (idx >= elems) return;
  int64_t dim = idx % hidden_size;
  int64_t tmp = idx / hidden_size;
  int64_t token = tmp % num_tokens;
  int64_t layer = tmp / num_tokens;
  int64_t slot = slot_mapping[token];
  if (slot < 0) return;
  int64_t block = slot / block_size;
  int64_t offset = slot - block * block_size;
  if (block < 0 || block >= num_blocks) return;
  int64_t cache_idx = (((layer * num_blocks + block) * block_size + offset) * hidden_size + dim);
  paged_kv_cache[cache_idx] = kv_buffer[idx];
}

extern "C" void lmcache_kv_buffer_to_paged_mla_u16(
    const uint16_t* kv_buffer,
    uint16_t* paged_kv_cache,
    const int64_t* slot_mapping,
    int64_t num_layers,
    int64_t num_tokens,
    int64_t num_blocks,
    int64_t block_size,
    int64_t hidden_size,
    gpuStream_t stream) {
#ifdef __MUSACC__
  int threads = 256;
  int64_t elems = num_layers * num_tokens * hidden_size;
  if (elems <= 0) return;
  int blocks = static_cast<int>((elems + threads - 1) / threads);
  lmcache_kv_buffer_to_paged_mla_u16_kernel<<<blocks, threads, 0, stream>>>(
      kv_buffer, paged_kv_cache, slot_mapping, num_layers, num_tokens, num_blocks, block_size, hidden_size);
#endif
}

extern "C" __global__ void lmcache_paged_to_kv_buffer_mla_u16_kernel(
    const uint16_t* paged_kv_cache,
    uint16_t* kv_buffer,
    const int64_t* slot_mapping,
    int64_t num_layers,
    int64_t num_tokens,
    int64_t num_blocks,
    int64_t block_size,
    int64_t hidden_size) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  int64_t elems = num_layers * num_tokens * hidden_size;
  if (idx >= elems) return;
  int64_t dim = idx % hidden_size;
  int64_t tmp = idx / hidden_size;
  int64_t token = tmp % num_tokens;
  int64_t layer = tmp / num_tokens;
  int64_t slot = slot_mapping[token];
  if (slot < 0) {
    kv_buffer[idx] = static_cast<uint16_t>(0);
    return;
  }
  int64_t block = slot / block_size;
  int64_t offset = slot - block * block_size;
  if (block < 0 || block >= num_blocks) {
    kv_buffer[idx] = static_cast<uint16_t>(0);
    return;
  }
  int64_t cache_idx = (((layer * num_blocks + block) * block_size + offset) * hidden_size + dim);
  kv_buffer[idx] = paged_kv_cache[cache_idx];
}

extern "C" void lmcache_paged_to_kv_buffer_mla_u16(
    const uint16_t* paged_kv_cache,
    uint16_t* kv_buffer,
    const int64_t* slot_mapping,
    int64_t num_layers,
    int64_t num_tokens,
    int64_t num_blocks,
    int64_t block_size,
    int64_t hidden_size,
    gpuStream_t stream) {
#ifdef __MUSACC__
  int threads = 256;
  int64_t elems = num_layers * num_tokens * hidden_size;
  if (elems <= 0) return;
  int blocks = static_cast<int>((elems + threads - 1) / threads);
  lmcache_paged_to_kv_buffer_mla_u16_kernel<<<blocks, threads, 0, stream>>>(
      paged_kv_cache, kv_buffer, slot_mapping, num_layers, num_tokens, num_blocks, block_size, hidden_size);
#endif
}

struct LmcacheTwoXnlCachePointers {
  uint16_t* values[128];
};

extern "C" __global__ void lmcache_kv_buffer_to_paged_two_x_nl_coalesced_u16_kernel(
    const uint16_t* memory_tensor,
    LmcacheTwoXnlCachePointers caches,
    const int64_t* slot_mapping,
    int64_t num_layers,
    int64_t memory_tokens,
    int64_t num_tokens,
    int64_t skip_prefix_n_tokens,
    int64_t num_blocks,
    int64_t block_size,
    int64_t hidden) {
  const int64_t cache_count = 2 * num_layers;
  const int64_t cache_index = static_cast<int64_t>(blockIdx.x) % cache_count;
  int64_t token = static_cast<int64_t>(blockIdx.x) / cache_count;
  const int64_t token_groups = gridDim.x / cache_count;
  const int64_t kv_axis = cache_index / num_layers;
  const int64_t layer = cache_index % num_layers;
  uint16_t* cache = caches.values[cache_index];
  for (int64_t row = 0; row < 8; ++row, token += token_groups) {
    if (token >= num_tokens) return;
    const int64_t slot = slot_mapping[token];
    const int64_t src_token = token + skip_prefix_n_tokens;
    if (slot < 0 || src_token >= memory_tokens) continue;
    const int64_t block = slot / block_size;
    if (block < 0 || block >= num_blocks) continue;
    const int64_t src_base =
        ((kv_axis * num_layers + layer) * memory_tokens + src_token) * hidden;
    const int64_t dst_base = slot * hidden;
    const unsigned long long* src =
        reinterpret_cast<const unsigned long long*>(memory_tensor + src_base);
    unsigned long long* dst =
        reinterpret_cast<unsigned long long*>(cache + dst_base);
    for (int64_t i = threadIdx.x; i < hidden / 4; i += blockDim.x) dst[i] = src[i];
  }
}

extern "C" void lmcache_kv_buffer_to_paged_two_x_nl_coalesced_u16(
    const uint16_t* memory_tensor,
    const uint64_t* cache_ptrs,
    const int64_t* slot_mapping,
    int64_t num_layers,
    int64_t memory_tokens,
    int64_t num_tokens,
    int64_t skip_prefix_n_tokens,
    int64_t num_blocks,
    int64_t block_size,
    int64_t num_heads,
    int64_t head_size,
    gpuStream_t stream) {
#ifdef __MUSACC__
  if (num_tokens <= 0 || num_layers <= 0 || 2 * num_layers > 128) return;
  LmcacheTwoXnlCachePointers caches{};
  for (int64_t i = 0; i < 2 * num_layers; ++i)
    caches.values[i] = reinterpret_cast<uint16_t*>(cache_ptrs[i]);
  const int blocks = static_cast<int>(((num_tokens + 7) / 8) * 2 * num_layers);
  lmcache_kv_buffer_to_paged_two_x_nl_coalesced_u16_kernel<<<blocks, 256, 0, stream>>>(
      memory_tensor, caches, slot_mapping, num_layers, memory_tokens, num_tokens,
      skip_prefix_n_tokens, num_blocks, block_size, num_heads * head_size);
#endif
}

extern "C" __global__ void lmcache_paged_to_kv_buffer_two_x_nl_coalesced_u16_kernel(
    uint16_t* memory_tensor,
    LmcacheTwoXnlCachePointers caches,
    const int64_t* slot_mapping,
    int64_t num_layers,
    int64_t memory_tokens,
    int64_t num_tokens,
    int64_t num_blocks,
    int64_t block_size,
    int64_t hidden) {
  const int64_t cache_count = 2 * num_layers;
  const int64_t cache_index = static_cast<int64_t>(blockIdx.x) % cache_count;
  int64_t token = static_cast<int64_t>(blockIdx.x) / cache_count;
  const int64_t token_groups = gridDim.x / cache_count;
  const int64_t kv_axis = cache_index / num_layers;
  const int64_t layer = cache_index % num_layers;
  const uint16_t* cache = caches.values[cache_index];
  for (int64_t row = 0; row < 8; ++row, token += token_groups) {
    if (token >= num_tokens) return;
    const int64_t dst_base =
        ((kv_axis * num_layers + layer) * memory_tokens + token) * hidden;
    const int64_t slot = slot_mapping[token];
    if (slot < 0) {
      for (int64_t i = threadIdx.x; i < hidden; i += blockDim.x)
        memory_tensor[dst_base + i] = 0;
      continue;
    }
    const int64_t block = slot / block_size;
    if (block < 0 || block >= num_blocks) continue;
    const unsigned long long* src =
        reinterpret_cast<const unsigned long long*>(cache + slot * hidden);
    unsigned long long* dst =
        reinterpret_cast<unsigned long long*>(memory_tensor + dst_base);
    for (int64_t i = threadIdx.x; i < hidden / 4; i += blockDim.x) dst[i] = src[i];
  }
}

extern "C" void lmcache_paged_to_kv_buffer_two_x_nl_coalesced_u16(
    uint16_t* memory_tensor,
    const uint64_t* cache_ptrs,
    const int64_t* slot_mapping,
    int64_t num_layers,
    int64_t memory_tokens,
    int64_t num_tokens,
    int64_t num_blocks,
    int64_t block_size,
    int64_t num_heads,
    int64_t head_size,
    gpuStream_t stream) {
#ifdef __MUSACC__
  if (num_tokens <= 0 || num_layers <= 0 || 2 * num_layers > 128) return;
  LmcacheTwoXnlCachePointers caches{};
  for (int64_t i = 0; i < 2 * num_layers; ++i)
    caches.values[i] = reinterpret_cast<uint16_t*>(cache_ptrs[i]);
  const int blocks = static_cast<int>(((num_tokens + 7) / 8) * 2 * num_layers);
  lmcache_paged_to_kv_buffer_two_x_nl_coalesced_u16_kernel<<<blocks, 256, 0, stream>>>(
      memory_tensor, caches, slot_mapping, num_layers, memory_tokens, num_tokens,
      num_blocks, block_size, num_heads * head_size);
#endif
}

extern "C" __global__ void lmcache_kv_buffer_to_paged_two_x_nl_u16_kernel(
    const uint16_t* memory_tensor,
    uint16_t* cache,
    const int64_t* slot_mapping,
    int64_t kv_axis,
    int64_t layer_index,
    int64_t num_layers,
    int64_t memory_tokens,
    int64_t num_tokens,
    int64_t skip_prefix_n_tokens,
    int64_t num_blocks,
    int64_t block_size,
    int64_t num_heads,
    int64_t head_size) {
  const int64_t rows_per_block = 8;
  int64_t token = static_cast<int64_t>(blockIdx.x);
  const int64_t hidden = num_heads * head_size;
  for (int64_t row_offset = 0; row_offset < rows_per_block;
       ++row_offset, token += gridDim.x) {
    if (token >= num_tokens) return;
    const int64_t slot = slot_mapping[token];
    if (slot < 0) continue;
    const int64_t block = slot / block_size;
    const int64_t offset = slot - block * block_size;
    const int64_t src_token = token + skip_prefix_n_tokens;
    if (block < 0 || block >= num_blocks || src_token < 0 ||
        src_token >= memory_tokens) {
      continue;
    }
    const int64_t src_base =
        ((kv_axis * num_layers + layer_index) * memory_tokens + src_token) *
        hidden;
    const int64_t dst_base = (block * block_size + offset) * hidden;
    const int64_t vec_elems = hidden / 4;
    const unsigned long long* src_vec =
        reinterpret_cast<const unsigned long long*>(memory_tensor + src_base);
    unsigned long long* dst_vec =
        reinterpret_cast<unsigned long long*>(cache + dst_base);
    for (int64_t i = threadIdx.x; i < vec_elems; i += blockDim.x) {
      dst_vec[i] = src_vec[i];
    }
    for (int64_t i = vec_elems * 4 + threadIdx.x; i < hidden;
         i += blockDim.x) {
      cache[dst_base + i] = memory_tensor[src_base + i];
    }
  }
}

extern "C" void lmcache_kv_buffer_to_paged_two_x_nl_u16(
    const uint16_t* memory_tensor,
    uint16_t* cache,
    const int64_t* slot_mapping,
    int64_t kv_axis,
    int64_t layer_index,
    int64_t num_layers,
    int64_t memory_tokens,
    int64_t num_tokens,
    int64_t skip_prefix_n_tokens,
    int64_t num_blocks,
    int64_t block_size,
    int64_t num_heads,
    int64_t head_size,
    gpuStream_t stream) {
#ifdef __MUSACC__
  if (num_tokens <= 0) return;
  const int64_t rows_per_block = 8;
  const int blocks =
      static_cast<int>((num_tokens + rows_per_block - 1) / rows_per_block);
  lmcache_kv_buffer_to_paged_two_x_nl_u16_kernel<<<blocks, 256, 0, stream>>>(
      memory_tensor, cache, slot_mapping, kv_axis, layer_index, num_layers,
      memory_tokens, num_tokens, skip_prefix_n_tokens, num_blocks, block_size,
      num_heads, head_size);
#endif
}

extern "C" __global__ void lmcache_paged_to_kv_buffer_two_x_nl_u16_kernel(
    uint16_t* memory_tensor,
    const uint16_t* cache,
    const int64_t* slot_mapping,
    int64_t kv_axis,
    int64_t layer_index,
    int64_t num_layers,
    int64_t memory_tokens,
    int64_t num_tokens,
    int64_t skip_prefix_n_tokens,
    int64_t num_blocks,
    int64_t block_size,
    int64_t num_heads,
    int64_t head_size) {
  const int64_t rows_per_block = 8;
  int64_t token = static_cast<int64_t>(blockIdx.x);
  const int64_t hidden = num_heads * head_size;
  for (int64_t row_offset = 0; row_offset < rows_per_block;
       ++row_offset, token += gridDim.x) {
    if (token >= num_tokens) return;
    const int64_t dst_token = token + skip_prefix_n_tokens;
    if (dst_token < 0 || dst_token >= memory_tokens) continue;
    const int64_t dst_base =
        ((kv_axis * num_layers + layer_index) * memory_tokens + dst_token) *
        hidden;
    const int64_t slot = slot_mapping[token];
    const int64_t block = slot >= 0 ? slot / block_size : -1;
    const int64_t offset = slot >= 0 ? slot - block * block_size : -1;
    if (slot < 0 || block < 0 || block >= num_blocks) {
      for (int64_t i = threadIdx.x; i < hidden; i += blockDim.x) {
        memory_tensor[dst_base + i] = static_cast<uint16_t>(0);
      }
      continue;
    }
    const int64_t src_base = (block * block_size + offset) * hidden;
    const int64_t vec_elems = hidden / 4;
    const unsigned long long* src_vec =
        reinterpret_cast<const unsigned long long*>(cache + src_base);
    unsigned long long* dst_vec =
        reinterpret_cast<unsigned long long*>(memory_tensor + dst_base);
    for (int64_t i = threadIdx.x; i < vec_elems; i += blockDim.x) {
      dst_vec[i] = src_vec[i];
    }
    for (int64_t i = vec_elems * 4 + threadIdx.x; i < hidden;
         i += blockDim.x) {
      memory_tensor[dst_base + i] = cache[src_base + i];
    }
  }
}

extern "C" void lmcache_paged_to_kv_buffer_two_x_nl_u16(
    uint16_t* memory_tensor,
    const uint16_t* cache,
    const int64_t* slot_mapping,
    int64_t kv_axis,
    int64_t layer_index,
    int64_t num_layers,
    int64_t memory_tokens,
    int64_t num_tokens,
    int64_t skip_prefix_n_tokens,
    int64_t num_blocks,
    int64_t block_size,
    int64_t num_heads,
    int64_t head_size,
    gpuStream_t stream) {
#ifdef __MUSACC__
  if (num_tokens <= 0) return;
  const int64_t rows_per_block = 8;
  const int blocks =
      static_cast<int>((num_tokens + rows_per_block - 1) / rows_per_block);
  lmcache_paged_to_kv_buffer_two_x_nl_u16_kernel<<<blocks, 256, 0, stream>>>(
      memory_tensor, cache, slot_mapping, kv_axis, layer_index, num_layers,
      memory_tokens, num_tokens, skip_prefix_n_tokens, num_blocks, block_size,
      num_heads, head_size);
#endif
}

extern "C" __global__ void lmcache_kv_buffer_to_paged_list_u16_kernel(
    const uint16_t* memory_tensor,
    uint16_t* kvcache,
    const int64_t* slot_mapping,
    int64_t layer_index,
    int64_t num_layers,
    int64_t memory_tokens,
    int64_t num_tokens,
    int64_t skip_prefix_n_tokens,
    int64_t num_blocks,
    int64_t block_size,
    int64_t num_heads,
    int64_t head_size) {
  const int64_t rows_per_block = 8;
  int64_t row = static_cast<int64_t>(blockIdx.x);
  for (int64_t row_offset = 0; row_offset < rows_per_block; ++row_offset, row += gridDim.x) {
    if (row >= 2 * num_tokens) return;
    int64_t kv_axis = row / num_tokens;
    int64_t token = row - kv_axis * num_tokens;
    int64_t slot = slot_mapping[token];
    if (slot < 0) continue;
    int64_t block = slot / block_size;
    int64_t offset = slot - block * block_size;
    if (block < 0 || block >= num_blocks) continue;
    int64_t hidden = num_heads * head_size;
    int64_t src_token = token + skip_prefix_n_tokens;
    if (src_token < 0 || src_token >= memory_tokens) continue;
    int64_t src_base = ((kv_axis * num_layers + layer_index) * memory_tokens + src_token) * hidden;
    int64_t dst_base = ((kv_axis * num_blocks + block) * block_size + offset) * hidden;
    int64_t vec_elems = hidden / 4;
    const unsigned long long* src_vec = reinterpret_cast<const unsigned long long*>(memory_tensor + src_base);
    unsigned long long* dst_vec = reinterpret_cast<unsigned long long*>(kvcache + dst_base);
    for (int64_t i = threadIdx.x; i < vec_elems; i += blockDim.x) {
      dst_vec[i] = src_vec[i];
    }
    for (int64_t i = vec_elems * 4 + threadIdx.x; i < hidden; i += blockDim.x) {
      kvcache[dst_base + i] = memory_tensor[src_base + i];
    }
  }
}

extern "C" void lmcache_kv_buffer_to_paged_list_u16(
    const uint16_t* memory_tensor,
    uint16_t* kvcache,
    const int64_t* slot_mapping,
    int64_t layer_index,
    int64_t num_layers,
    int64_t memory_tokens,
    int64_t num_tokens,
    int64_t skip_prefix_n_tokens,
    int64_t num_blocks,
    int64_t block_size,
    int64_t num_heads,
    int64_t head_size,
    gpuStream_t stream) {
#ifdef __MUSACC__
  if (num_tokens <= 0) return;
  const int64_t rows_per_block = 8;
  int blocks = static_cast<int>((2 * num_tokens + rows_per_block - 1) / rows_per_block);
  lmcache_kv_buffer_to_paged_list_u16_kernel<<<blocks, 256, 0, stream>>>(
      memory_tensor, kvcache, slot_mapping, layer_index, num_layers, memory_tokens, num_tokens, skip_prefix_n_tokens, num_blocks, block_size, num_heads, head_size);
#endif
}

extern "C" __global__ void lmcache_paged_to_kv_buffer_list_u16_kernel(
    uint16_t* memory_tensor,
    const uint16_t* kvcache,
    const int64_t* slot_mapping,
    int64_t layer_index,
    int64_t num_layers,
    int64_t memory_tokens,
    int64_t num_tokens,
    int64_t skip_prefix_n_tokens,
    int64_t num_blocks,
    int64_t block_size,
    int64_t num_heads,
    int64_t head_size) {
  const int64_t rows_per_block = 8;
  int64_t row = static_cast<int64_t>(blockIdx.x);
  for (int64_t row_offset = 0; row_offset < rows_per_block; ++row_offset, row += gridDim.x) {
    if (row >= 2 * num_tokens) return;
    int64_t kv_axis = row / num_tokens;
    int64_t token = row - kv_axis * num_tokens;
    int64_t slot = slot_mapping[token];
    int64_t hidden = num_heads * head_size;
    int64_t dst_token = token + skip_prefix_n_tokens;
    if (dst_token < 0 || dst_token >= memory_tokens) continue;
    int64_t dst_base = ((kv_axis * num_layers + layer_index) * memory_tokens + dst_token) * hidden;
    if (slot < 0) {
      for (int64_t i = threadIdx.x; i < hidden; i += blockDim.x) {
        memory_tensor[dst_base + i] = static_cast<uint16_t>(0);
      }
      continue;
    }
    int64_t block = slot / block_size;
    int64_t offset = slot - block * block_size;
    if (block < 0 || block >= num_blocks) continue;
    int64_t src_base = ((kv_axis * num_blocks + block) * block_size + offset) * hidden;
    int64_t vec_elems = hidden / 4;
    const unsigned long long* src_vec = reinterpret_cast<const unsigned long long*>(kvcache + src_base);
    unsigned long long* dst_vec = reinterpret_cast<unsigned long long*>(memory_tensor + dst_base);
    for (int64_t i = threadIdx.x; i < vec_elems; i += blockDim.x) {
      dst_vec[i] = src_vec[i];
    }
    for (int64_t i = vec_elems * 4 + threadIdx.x; i < hidden; i += blockDim.x) {
      memory_tensor[dst_base + i] = kvcache[src_base + i];
    }
  }
}

extern "C" void lmcache_paged_to_kv_buffer_list_u16(
    uint16_t* memory_tensor,
    const uint16_t* kvcache,
    const int64_t* slot_mapping,
    int64_t layer_index,
    int64_t num_layers,
    int64_t memory_tokens,
    int64_t num_tokens,
    int64_t skip_prefix_n_tokens,
    int64_t num_blocks,
    int64_t block_size,
    int64_t num_heads,
    int64_t head_size,
    gpuStream_t stream) {
#ifdef __MUSACC__
  if (num_tokens <= 0) return;
  const int64_t rows_per_block = 8;
  int blocks = static_cast<int>((2 * num_tokens + rows_per_block - 1) / rows_per_block);
  lmcache_paged_to_kv_buffer_list_u16_kernel<<<blocks, 256, 0, stream>>>(
      memory_tensor, kvcache, slot_mapping, layer_index, num_layers, memory_tokens, num_tokens, skip_prefix_n_tokens, num_blocks, block_size, num_heads, head_size);
#endif
}

extern "C" __global__ void lmcache_kv_buffer_to_paged_list2_u16_kernel(
    const uint16_t* memory_tensor,
    uint16_t* kvcache0,
    uint16_t* kvcache1,
    const int64_t* slot_mapping,
    int64_t memory_tokens,
    int64_t num_tokens,
    int64_t skip_prefix_n_tokens,
    int64_t num_blocks,
    int64_t block_size,
    int64_t num_heads,
    int64_t head_size) {
  const int64_t num_layers = 2;
  const int64_t rows_per_kv = num_layers * num_tokens;
  const int64_t row = static_cast<int64_t>(blockIdx.x);
  if (row >= 2 * rows_per_kv) return;
  int64_t kv_axis = 0;
  int64_t local_row = row;
  if (local_row >= rows_per_kv) {
    kv_axis = 1;
    local_row -= rows_per_kv;
  }
  int64_t layer = 0;
  int64_t token = local_row;
  if (token >= num_tokens) {
    layer = 1;
    token -= num_tokens;
  }
  const int64_t slot = slot_mapping[token];
  if (slot < 0) return;
  int64_t block;
  int64_t offset;
  if (block_size == 16) {
    block = slot >> 4;
    offset = slot & 15;
  } else {
    block = slot / block_size;
    offset = slot - block * block_size;
  }
  if (block < 0 || block >= num_blocks) return;
  const int64_t hidden = num_heads * head_size;
  const int64_t src_token = token + skip_prefix_n_tokens;
  if (src_token < 0 || src_token >= memory_tokens) return;
  const int64_t src_base = ((kv_axis * 2 + layer) * memory_tokens + src_token) * hidden;
  const int64_t dst_base = ((kv_axis * num_blocks + block) * block_size + offset) * hidden;
  uint16_t* kvcache = layer == 0 ? kvcache0 : kvcache1;
  const int64_t vec_elems = hidden / 16;
  const MusaAiterU16x16* src_vec = reinterpret_cast<const MusaAiterU16x16*>(memory_tensor + src_base);
  MusaAiterU16x16* dst_vec = reinterpret_cast<MusaAiterU16x16*>(kvcache + dst_base);
  for (int64_t i = threadIdx.x; i < vec_elems; i += blockDim.x) {
    dst_vec[i] = src_vec[i];
  }
}

extern "C" void lmcache_kv_buffer_to_paged_list2_u16(
    const uint16_t* memory_tensor,
    uint16_t* kvcache0,
    uint16_t* kvcache1,
    const int64_t* slot_mapping,
    int64_t memory_tokens,
    int64_t num_tokens,
    int64_t skip_prefix_n_tokens,
    int64_t num_blocks,
    int64_t block_size,
    int64_t num_heads,
    int64_t head_size,
    gpuStream_t stream) {
#ifdef __MUSACC__
  if (num_tokens <= 0) return;
  int blocks = static_cast<int>(4 * num_tokens);
  lmcache_kv_buffer_to_paged_list2_u16_kernel<<<blocks, 128, 0, stream>>>(
      memory_tensor, kvcache0, kvcache1, slot_mapping, memory_tokens, num_tokens, skip_prefix_n_tokens, num_blocks, block_size, num_heads, head_size);
#endif
}

extern "C" __global__ void lmcache_paged_to_kv_buffer_list2_u16_kernel(
    uint16_t* memory_tensor,
    const uint16_t* kvcache0,
    const uint16_t* kvcache1,
    const int64_t* slot_mapping,
    int64_t memory_tokens,
    int64_t num_tokens,
    int64_t skip_prefix_n_tokens,
    int64_t num_blocks,
    int64_t block_size,
    int64_t num_heads,
    int64_t head_size) {
  const int64_t num_layers = 2;
  const int64_t rows_per_kv = num_layers * num_tokens;
  const int64_t row = static_cast<int64_t>(blockIdx.x);
  if (row >= 2 * rows_per_kv) return;
  int64_t kv_axis = 0;
  int64_t local_row = row;
  if (local_row >= rows_per_kv) {
    kv_axis = 1;
    local_row -= rows_per_kv;
  }
  int64_t layer = 0;
  int64_t token = local_row;
  if (token >= num_tokens) {
    layer = 1;
    token -= num_tokens;
  }
  const int64_t slot = slot_mapping[token];
  const int64_t hidden = num_heads * head_size;
  const int64_t dst_token = token + skip_prefix_n_tokens;
  if (dst_token < 0 || dst_token >= memory_tokens) return;
  const int64_t dst_base = ((kv_axis * 2 + layer) * memory_tokens + dst_token) * hidden;
  if (slot < 0) {
    for (int64_t i = threadIdx.x; i < hidden; i += blockDim.x) {
      memory_tensor[dst_base + i] = static_cast<uint16_t>(0);
    }
    return;
  }
  int64_t block;
  int64_t offset;
  if (block_size == 16) {
    block = slot >> 4;
    offset = slot & 15;
  } else {
    block = slot / block_size;
    offset = slot - block * block_size;
  }
  if (block < 0 || block >= num_blocks) return;
  const uint16_t* kvcache = layer == 0 ? kvcache0 : kvcache1;
  const int64_t src_base = ((kv_axis * num_blocks + block) * block_size + offset) * hidden;
  const int64_t vec_elems = hidden / 16;
  const MusaAiterU16x16* src_vec = reinterpret_cast<const MusaAiterU16x16*>(kvcache + src_base);
  MusaAiterU16x16* dst_vec = reinterpret_cast<MusaAiterU16x16*>(memory_tensor + dst_base);
  for (int64_t i = threadIdx.x; i < vec_elems; i += blockDim.x) {
    dst_vec[i] = src_vec[i];
  }
}

extern "C" void lmcache_paged_to_kv_buffer_list2_u16(
    uint16_t* memory_tensor,
    const uint16_t* kvcache0,
    const uint16_t* kvcache1,
    const int64_t* slot_mapping,
    int64_t memory_tokens,
    int64_t num_tokens,
    int64_t skip_prefix_n_tokens,
    int64_t num_blocks,
    int64_t block_size,
    int64_t num_heads,
    int64_t head_size,
    gpuStream_t stream) {
#ifdef __MUSACC__
  if (num_tokens <= 0) return;
  int blocks = static_cast<int>(4 * num_tokens);
  lmcache_paged_to_kv_buffer_list2_u16_kernel<<<blocks, 128, 0, stream>>>(
      memory_tensor, kvcache0, kvcache1, slot_mapping, memory_tokens, num_tokens, skip_prefix_n_tokens, num_blocks, block_size, num_heads, head_size);
#endif
}

extern "C" __global__ void lmcache_kv_buffer_to_paged_list2_u16_coalesced_kernel(
    const uint16_t* memory_tensor,
    uint16_t* kvcache0,
    uint16_t* kvcache1,
    const int64_t* slot_mapping,
    int64_t memory_tokens,
    int64_t num_tokens,
    int64_t skip_prefix_n_tokens,
    int64_t num_blocks,
    int64_t block_size,
    int64_t num_heads,
    int64_t head_size) {
  const int64_t lane = threadIdx.x;
  int64_t kv_axis = 0;
  int64_t layer = threadIdx.y;
  if (layer >= 2) {
    kv_axis = 1;
    layer -= 2;
  }
  const int64_t token = static_cast<int64_t>(blockIdx.x);
  if (token >= num_tokens) return;
  const int64_t slot = slot_mapping[token];
  if (slot < 0) return;
  int64_t block;
  int64_t offset;
  if (block_size == 16) {
    block = slot >> 4;
    offset = slot & 15;
  } else {
    block = slot / block_size;
    offset = slot - block * block_size;
  }
  if (block < 0 || block >= num_blocks) return;
  const int64_t hidden = num_heads * head_size;
  const int64_t src_token = token + skip_prefix_n_tokens;
  if (src_token < 0 || src_token >= memory_tokens) return;
  const int64_t src_base = ((kv_axis * 2 + layer) * memory_tokens + src_token) * hidden;
  const int64_t dst_base = ((kv_axis * num_blocks + block) * block_size + offset) * hidden;
  uint16_t* kvcache = layer == 0 ? kvcache0 : kvcache1;
  const int64_t vec_elems = hidden / 16;
  const MusaAiterU16x16* src_vec = reinterpret_cast<const MusaAiterU16x16*>(memory_tensor + src_base);
  MusaAiterU16x16* dst_vec = reinterpret_cast<MusaAiterU16x16*>(kvcache + dst_base);
  for (int64_t i = lane; i < vec_elems; i += blockDim.x) {
    dst_vec[i] = src_vec[i];
  }
}

extern "C" void lmcache_kv_buffer_to_paged_list2_u16_coalesced(
    const uint16_t* memory_tensor,
    uint16_t* kvcache0,
    uint16_t* kvcache1,
    const int64_t* slot_mapping,
    int64_t memory_tokens,
    int64_t num_tokens,
    int64_t skip_prefix_n_tokens,
    int64_t num_blocks,
    int64_t block_size,
    int64_t num_heads,
    int64_t head_size,
    gpuStream_t stream) {
#ifdef __MUSACC__
  if (num_tokens <= 0) return;
  int blocks = static_cast<int>(num_tokens);
  dim3 threads(64, 4, 1);
  lmcache_kv_buffer_to_paged_list2_u16_coalesced_kernel<<<blocks, threads, 0, stream>>>(
      memory_tensor, kvcache0, kvcache1, slot_mapping, memory_tokens, num_tokens, skip_prefix_n_tokens, num_blocks, block_size, num_heads, head_size);
#endif
}

extern "C" __global__ void lmcache_paged_to_kv_buffer_list2_u16_coalesced_kernel(
    uint16_t* memory_tensor,
    const uint16_t* kvcache0,
    const uint16_t* kvcache1,
    const int64_t* slot_mapping,
    int64_t memory_tokens,
    int64_t num_tokens,
    int64_t skip_prefix_n_tokens,
    int64_t num_blocks,
    int64_t block_size,
    int64_t num_heads,
    int64_t head_size) {
  const int64_t lane = threadIdx.x;
  int64_t kv_axis = 0;
  int64_t layer = threadIdx.y;
  if (layer >= 2) {
    kv_axis = 1;
    layer -= 2;
  }
  const int64_t token = static_cast<int64_t>(blockIdx.x);
  if (token >= num_tokens) return;
  const int64_t slot = slot_mapping[token];
  const int64_t hidden = num_heads * head_size;
  const int64_t dst_token = token + skip_prefix_n_tokens;
  if (dst_token < 0 || dst_token >= memory_tokens) return;
  const int64_t dst_base = ((kv_axis * 2 + layer) * memory_tokens + dst_token) * hidden;
  if (slot < 0) {
    for (int64_t i = lane; i < hidden; i += blockDim.x) {
      memory_tensor[dst_base + i] = static_cast<uint16_t>(0);
    }
    return;
  }
  int64_t block;
  int64_t offset;
  if (block_size == 16) {
    block = slot >> 4;
    offset = slot & 15;
  } else {
    block = slot / block_size;
    offset = slot - block * block_size;
  }
  if (block < 0 || block >= num_blocks) return;
  const uint16_t* kvcache = layer == 0 ? kvcache0 : kvcache1;
  const int64_t src_base = ((kv_axis * num_blocks + block) * block_size + offset) * hidden;
  const int64_t vec_elems = hidden / 16;
  const MusaAiterU16x16* src_vec = reinterpret_cast<const MusaAiterU16x16*>(kvcache + src_base);
  MusaAiterU16x16* dst_vec = reinterpret_cast<MusaAiterU16x16*>(memory_tensor + dst_base);
  for (int64_t i = lane; i < vec_elems; i += blockDim.x) {
    dst_vec[i] = src_vec[i];
  }
}

extern "C" void lmcache_paged_to_kv_buffer_list2_u16_coalesced(
    uint16_t* memory_tensor,
    const uint16_t* kvcache0,
    const uint16_t* kvcache1,
    const int64_t* slot_mapping,
    int64_t memory_tokens,
    int64_t num_tokens,
    int64_t skip_prefix_n_tokens,
    int64_t num_blocks,
    int64_t block_size,
    int64_t num_heads,
    int64_t head_size,
    gpuStream_t stream) {
#ifdef __MUSACC__
  if (num_tokens <= 0) return;
  int blocks = static_cast<int>(num_tokens);
  dim3 threads(64, 4, 1);
  lmcache_paged_to_kv_buffer_list2_u16_coalesced_kernel<<<blocks, threads, 0, stream>>>(
      memory_tensor, kvcache0, kvcache1, slot_mapping, memory_tokens, num_tokens, skip_prefix_n_tokens, num_blocks, block_size, num_heads, head_size);
#endif
}

extern "C" __global__ void lmcache_paged_to_kv_buffer_list2_u16_torchmusa_style_kernel(
    uint16_t* memory_tensor,
    const uint16_t* kvcache0,
    const uint16_t* kvcache1,
    const int64_t* slot_mapping,
    int64_t memory_tokens,
    int64_t num_tokens,
    int64_t skip_prefix_n_tokens,
    int64_t num_blocks,
    int64_t block_size,
    int64_t num_heads,
    int64_t head_size) {
  const int64_t hidden = num_heads * head_size;
  const int64_t vec_per_token = hidden / 8;
  const int64_t vec_h = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t token = static_cast<int64_t>(blockIdx.y) * blockDim.y + threadIdx.y;
  const int64_t group = static_cast<int64_t>(blockIdx.z);
  if (vec_h >= vec_per_token || token >= num_tokens) return;

  const int64_t kv_axis = group >> 1;
  const int64_t layer = group & 1;
  const int64_t slot = slot_mapping[token];
  const int64_t total_slots = num_blocks * block_size;
  const int64_t dst_token = token + skip_prefix_n_tokens;
  if (slot < 0 || slot >= total_slots || dst_token < 0 || dst_token >= memory_tokens) return;

  const uint16_t* kvcache = layer == 0 ? kvcache0 : kvcache1;
  const MusaAiterU16x8* src_vec = reinterpret_cast<const MusaAiterU16x8*>(kvcache);
  MusaAiterU16x8* dst_vec = reinterpret_cast<MusaAiterU16x8*>(memory_tensor);
  const int64_t src_index = (kv_axis * total_slots + slot) * vec_per_token + vec_h;
  const int64_t dst_index = ((kv_axis * 2 + layer) * memory_tokens + dst_token) * vec_per_token + vec_h;
  dst_vec[dst_index] = src_vec[src_index];
}

extern "C" void lmcache_paged_to_kv_buffer_list2_u16_torchmusa_style(
    uint16_t* memory_tensor,
    const uint16_t* kvcache0,
    const uint16_t* kvcache1,
    const int64_t* slot_mapping,
    int64_t memory_tokens,
    int64_t num_tokens,
    int64_t skip_prefix_n_tokens,
    int64_t num_blocks,
    int64_t block_size,
    int64_t num_heads,
    int64_t head_size,
    gpuStream_t stream) {
#ifdef __MUSACC__
  if (num_tokens <= 0) return;
  const int64_t hidden = num_heads * head_size;
  const int64_t vec_per_token = hidden / 8;
  dim3 threads(16, 64, 1);
  dim3 blocks(static_cast<unsigned int>((vec_per_token + threads.x - 1) / threads.x),
              static_cast<unsigned int>((num_tokens + threads.y - 1) / threads.y),
              4);
  lmcache_paged_to_kv_buffer_list2_u16_torchmusa_style_kernel<<<blocks, threads, 0, stream>>>(
      memory_tensor, kvcache0, kvcache1, slot_mapping, memory_tokens, num_tokens, skip_prefix_n_tokens, num_blocks, block_size, num_heads, head_size);
#endif
}

extern "C" __global__ void lmcache_paged_to_kv_buffer_list2_u16_contiguous_bulk_kernel(
    uint16_t* memory_tensor,
    const uint16_t* kvcache0,
    const uint16_t* kvcache1,
    const int64_t* slot_mapping,
    int64_t memory_tokens,
    int64_t num_tokens,
    int64_t skip_prefix_n_tokens,
    int64_t num_blocks,
    int64_t block_size,
    int64_t num_heads,
    int64_t head_size) {
  const int64_t vec_per_token = (num_heads * head_size) / 16;
  const int64_t total_slots = num_blocks * block_size;
  const int64_t group_vecs = num_tokens * vec_per_token;
  const int64_t total_vecs = group_vecs * 4;
  const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= total_vecs) return;

  const int64_t group = idx / group_vecs;
  const int64_t vec_in_group = idx - group * group_vecs;
  const int64_t token = vec_in_group / vec_per_token;
  const int64_t vec_h = vec_in_group - token * vec_per_token;
  const int64_t kv_axis = group >> 1;
  const int64_t layer = group & 1;
  const int64_t base_slot = slot_mapping[0];
  const int64_t slot = base_slot + token;
  const int64_t dst_token = token + skip_prefix_n_tokens;
  if (slot < 0 || slot >= total_slots || dst_token < 0 || dst_token >= memory_tokens) return;

  const uint16_t* kvcache = layer == 0 ? kvcache0 : kvcache1;
  const MusaAiterU16x16* src_vec = reinterpret_cast<const MusaAiterU16x16*>(kvcache);
  MusaAiterU16x16* dst_vec = reinterpret_cast<MusaAiterU16x16*>(memory_tensor);
  const int64_t src_index = (kv_axis * total_slots + slot) * vec_per_token + vec_h;
  const int64_t dst_index = ((kv_axis * 2 + layer) * memory_tokens + dst_token) * vec_per_token + vec_h;
  dst_vec[dst_index] = src_vec[src_index];
}

extern "C" void lmcache_paged_to_kv_buffer_list2_u16_contiguous_bulk(
    uint16_t* memory_tensor,
    const uint16_t* kvcache0,
    const uint16_t* kvcache1,
    const int64_t* slot_mapping,
    int64_t memory_tokens,
    int64_t num_tokens,
    int64_t skip_prefix_n_tokens,
    int64_t num_blocks,
    int64_t block_size,
    int64_t num_heads,
    int64_t head_size,
    gpuStream_t stream) {
#ifdef __MUSACC__
  if (num_tokens <= 0) return;
  const int64_t vec_per_token = (num_heads * head_size) / 16;
  const int64_t total_vecs = 4 * num_tokens * vec_per_token;
  int threads = 256;
  int blocks = static_cast<int>((total_vecs + threads - 1) / threads);
  lmcache_paged_to_kv_buffer_list2_u16_contiguous_bulk_kernel<<<blocks, threads, 0, stream>>>(
      memory_tensor, kvcache0, kvcache1, slot_mapping, memory_tokens, num_tokens, skip_prefix_n_tokens, num_blocks, block_size, num_heads, head_size);
#endif
}

extern "C" __global__ void lmcache_paged_to_kv_buffer_list2_u16_contiguous_kernel(
    uint16_t* memory_tensor,
    const uint16_t* kvcache0,
    const uint16_t* kvcache1,
    const int64_t* slot_mapping,
    int64_t memory_tokens,
    int64_t num_tokens,
    int64_t skip_prefix_n_tokens,
    int64_t num_blocks,
    int64_t block_size,
    int64_t num_heads,
    int64_t head_size) {
  const int64_t lane = threadIdx.x;
  int64_t kv_axis = 0;
  int64_t layer = threadIdx.y;
  if (layer >= 2) {
    kv_axis = 1;
    layer -= 2;
  }
  const int64_t token = static_cast<int64_t>(blockIdx.x);
  if (token >= num_tokens) return;
  const int64_t base_slot = slot_mapping[0];
  const int64_t slot = base_slot + token;
  const int64_t hidden = num_heads * head_size;
  const int64_t dst_token = token + skip_prefix_n_tokens;
  if (dst_token < 0 || dst_token >= memory_tokens) return;
  const int64_t dst_base = ((kv_axis * 2 + layer) * memory_tokens + dst_token) * hidden;
  if (slot < 0) {
    for (int64_t i = lane; i < hidden; i += blockDim.x) {
      memory_tensor[dst_base + i] = static_cast<uint16_t>(0);
    }
    return;
  }
  int64_t block;
  int64_t offset;
  if (block_size == 16) {
    block = slot >> 4;
    offset = slot & 15;
  } else {
    block = slot / block_size;
    offset = slot - block * block_size;
  }
  if (block < 0 || block >= num_blocks) return;
  const uint16_t* kvcache = layer == 0 ? kvcache0 : kvcache1;
  const int64_t src_base = ((kv_axis * num_blocks + block) * block_size + offset) * hidden;
  const int64_t vec_elems = hidden / 16;
  const MusaAiterU16x16* src_vec = reinterpret_cast<const MusaAiterU16x16*>(kvcache + src_base);
  MusaAiterU16x16* dst_vec = reinterpret_cast<MusaAiterU16x16*>(memory_tensor + dst_base);
  for (int64_t i = lane; i < vec_elems; i += blockDim.x) {
    dst_vec[i] = src_vec[i];
  }
}

extern "C" void lmcache_paged_to_kv_buffer_list2_u16_contiguous(
    uint16_t* memory_tensor,
    const uint16_t* kvcache0,
    const uint16_t* kvcache1,
    const int64_t* slot_mapping,
    int64_t memory_tokens,
    int64_t num_tokens,
    int64_t skip_prefix_n_tokens,
    int64_t num_blocks,
    int64_t block_size,
    int64_t num_heads,
    int64_t head_size,
    gpuStream_t stream) {
#ifdef __MUSACC__
  if (num_tokens <= 0) return;
  int blocks = static_cast<int>(num_tokens);
  dim3 threads(64, 4, 1);
  lmcache_paged_to_kv_buffer_list2_u16_contiguous_kernel<<<blocks, threads, 0, stream>>>(
      memory_tensor, kvcache0, kvcache1, slot_mapping, memory_tokens, num_tokens, skip_prefix_n_tokens, num_blocks, block_size, num_heads, head_size);
#endif
}

extern "C" __global__ void lmcache_paged_to_kv_buffer_list2_u16_sorted_kernel(
    uint16_t* memory_tensor,
    const uint16_t* kvcache0,
    const uint16_t* kvcache1,
    const int64_t* slot_mapping,
    const uint8_t* sorted_indices_bytes,
    int64_t memory_tokens,
    int64_t num_tokens,
    int64_t skip_prefix_n_tokens,
    int64_t num_blocks,
    int64_t block_size,
    int64_t num_heads,
    int64_t head_size) {
  const int64_t lane = threadIdx.x;
  int64_t kv_axis = 0;
  int64_t layer = threadIdx.y;
  if (layer >= 2) {
    kv_axis = 1;
    layer -= 2;
  }
  const int64_t sorted_pos = static_cast<int64_t>(blockIdx.x);
  if (sorted_pos >= num_tokens) return;
  const uint8_t* token_bytes = sorted_indices_bytes + sorted_pos + sorted_pos + sorted_pos + sorted_pos;
  int32_t token_i32 = static_cast<int32_t>(token_bytes[0]) |
                      (static_cast<int32_t>(token_bytes[1]) << 8) |
                      (static_cast<int32_t>(token_bytes[2]) << 16) |
                      (static_cast<int32_t>(token_bytes[3]) << 24);
  const int64_t token = static_cast<int64_t>(token_i32);
  if (token < 0 || token >= num_tokens) return;
  const int64_t slot = slot_mapping[token];
  const int64_t hidden = num_heads * head_size;
  const int64_t dst_token = token + skip_prefix_n_tokens;
  if (dst_token < 0 || dst_token >= memory_tokens) return;
  const int64_t dst_base = ((kv_axis * 2 + layer) * memory_tokens + dst_token) * hidden;
  if (slot < 0) {
    for (int64_t i = lane; i < hidden; i += blockDim.x) {
      memory_tensor[dst_base + i] = static_cast<uint16_t>(0);
    }
    return;
  }
  int64_t block;
  int64_t offset;
  if (block_size == 16) {
    block = slot >> 4;
    offset = slot & 15;
  } else {
    block = slot / block_size;
    offset = slot - block * block_size;
  }
  if (block < 0 || block >= num_blocks) return;
  const uint16_t* kvcache = layer == 0 ? kvcache0 : kvcache1;
  const int64_t src_base = ((kv_axis * num_blocks + block) * block_size + offset) * hidden;
  const int64_t vec_elems = hidden / 16;
  const MusaAiterU16x16* src_vec = reinterpret_cast<const MusaAiterU16x16*>(kvcache + src_base);
  MusaAiterU16x16* dst_vec = reinterpret_cast<MusaAiterU16x16*>(memory_tensor + dst_base);
  for (int64_t i = lane; i < vec_elems; i += blockDim.x) {
    dst_vec[i] = src_vec[i];
  }
}

extern "C" void lmcache_paged_to_kv_buffer_list2_u16_sorted(
    uint16_t* memory_tensor,
    const uint16_t* kvcache0,
    const uint16_t* kvcache1,
    const int64_t* slot_mapping,
    const uint8_t* sorted_indices_bytes,
    int64_t memory_tokens,
    int64_t num_tokens,
    int64_t skip_prefix_n_tokens,
    int64_t num_blocks,
    int64_t block_size,
    int64_t num_heads,
    int64_t head_size,
    gpuStream_t stream) {
#ifdef __MUSACC__
  if (num_tokens <= 0) return;
  int blocks = static_cast<int>(num_tokens);
  dim3 threads(64, 4, 1);
  lmcache_paged_to_kv_buffer_list2_u16_sorted_kernel<<<blocks, threads, 0, stream>>>(
      memory_tensor, kvcache0, kvcache1, slot_mapping, sorted_indices_bytes, memory_tokens, num_tokens, skip_prefix_n_tokens, num_blocks, block_size, num_heads, head_size);
#endif
}

extern "C" int native_lmcache_kv_transfer_abi_version() {
  return 1;
}

extern "C" int musa_aiter_kv_transfer_abi_version() {
  return 1;
}

// ---------------------------------------------------------------------------
// DSV4 flat byte-row gather/scatter (kuae-ssd req3musaiter-dsv4 REQ-1/3).
// Rows are opaque uint8 byte blocks (item_bytes arbitrary, 1728~37440
// observed); no KV/float semantics.
//
// Perf pattern follows the proven lmcache u16_sorted kernel (7x vs torch):
//  - 32-byte vectorized struct loads/stores (contiguous per-thread chunks)
//  - one block handles several rows via threadIdx.y
//  - block shape (64, rows_per_block): each x-lane owns a contiguous run
//    of 32B segments, giving fully coalesced wide transactions
//  - 16B/1B tails for arbitrary item_bytes (non-aligned fp8 scale rows)
// ---------------------------------------------------------------------------

struct __attribute__((aligned(32))) MusaAiterU8x32 {
  uint4 x0;
  uint4 x1;
};

static inline __device__ void musa_aiter_dsv4_copy_row(
    const uint8_t* __restrict__ s,
    uint8_t* __restrict__ d,
    int64_t item_bytes) {
  const int64_t lane = threadIdx.x;
  const int64_t nlanes = blockDim.x;
  if ((item_bytes & 31) == 0) {
    // 32B-aligned row width: contiguous 32B vectors per lane.
    const int64_t n32 = item_bytes >> 5;
    const MusaAiterU8x32* sv = reinterpret_cast<const MusaAiterU8x32*>(s);
    MusaAiterU8x32* dv = reinterpret_cast<MusaAiterU8x32*>(d);
    for (int64_t i = lane; i < n32; i += nlanes) {
      dv[i] = sv[i];
    }
    return;
  }
  if ((item_bytes & 15) == 0) {
    // 16B-aligned width: contiguous 16B vectors per lane.
    const int64_t n16 = item_bytes >> 4;
    const uint4* sv = reinterpret_cast<const uint4*>(s);
    uint4* dv = reinterpret_cast<uint4*>(d);
    for (int64_t i = lane; i < n16; i += nlanes) {
      dv[i] = sv[i];
    }
    return;
  }
  // Arbitrary width: vector body + byte tail.
  int64_t n16 = item_bytes >> 4;
  const uint4* sv = reinterpret_cast<const uint4*>(s);
  uint4* dv = reinterpret_cast<uint4*>(d);
  for (int64_t i = lane; i < n16; i += nlanes) {
    dv[i] = sv[i];
  }
  int64_t done = n16 << 4;
  if (lane == 0) {
    for (int64_t b = done; b < item_bytes; ++b) {
      d[b] = s[b];
    }
  }
}

__global__ void musa_aiter_kv_copy_paged_rows_u8_kernel(
    const uint8_t* __restrict__ src,
    uint8_t* __restrict__ dst,
    const int64_t* __restrict__ src_rows,
    const int64_t* __restrict__ dst_rows,
    int64_t num_rows,
    int64_t item_bytes,
    int64_t rows_per_block,
    int64_t num_src_rows,
    int64_t num_dst_rows) {
  int64_t i = static_cast<int64_t>(blockIdx.x) * rows_per_block + threadIdx.y;
  if (i >= num_rows) return;
  int64_t sr = src_rows[i];
  int64_t dr = dst_rows[i];
  if (sr < 0 || sr >= num_src_rows || dr < 0 || dr >= num_dst_rows) return;  // OOB guard (host skips sync checks on GPU indices)
  const uint8_t* s = src + sr * item_bytes;
  uint8_t* d = dst + dr * item_bytes;
  musa_aiter_dsv4_copy_row(s, d, item_bytes);
}

__global__ void musa_aiter_kv_copy_paged_rows_layer_table_u8_kernel(
    const uint8_t* const* __restrict__ src_ptrs,
    uint8_t* const* __restrict__ dst_ptrs,
    const int64_t* __restrict__ src_rows,
    const int64_t* __restrict__ dst_rows,
    int64_t num_rows,
    int64_t item_bytes,
    int64_t num_layers,
    int64_t rows_per_block,
    int64_t num_src_rows,
    int64_t num_dst_rows) {
  // Flat grid.x = row_groups * num_layers; decode (row_group, layer).
  int64_t bi = static_cast<int64_t>(blockIdx.x);
  int64_t row_groups = (num_rows + rows_per_block - 1) / rows_per_block;
  int64_t layer = bi / row_groups;
  int64_t row_group = bi % row_groups;
  int64_t i = row_group * rows_per_block + threadIdx.y;
  if (i >= num_rows) return;
  int64_t sr = src_rows[i];
  int64_t dr = dst_rows[i];
  if (sr < 0 || sr >= num_src_rows || dr < 0 || dr >= num_dst_rows) return;
  const uint8_t* s = src_ptrs[layer] + sr * item_bytes;
  uint8_t* d = dst_ptrs[layer] + dr * item_bytes;
  musa_aiter_dsv4_copy_row(s, d, item_bytes);
}

// REQ-3: multiple pools sharing one index set; layer-major grid.z spans
// pool*layer so a single launch covers every pool's every layer.
__global__ void musa_aiter_kv_copy_paged_rows_multi_pool_u8_kernel(
    const uint8_t* const* __restrict__ src_ptrs,   // flattened [pool][layer]
    uint8_t* const* __restrict__ dst_ptrs,
    const int64_t* __restrict__ pool_layer_offsets, // [num_pools+1], prefix sum
    const int64_t* __restrict__ rows,               // shared indices
    int64_t num_rows,
    const int64_t* __restrict__ pool_item_bytes,    // [num_pools]
    int64_t total_layers) {
  int64_t pool = blockIdx.z;
  int64_t layer_in_pool = blockIdx.y / num_rows;  // reuse y for row index below
  // grid = (chunks, num_rows * max_layers_per_pool? ) -- simpler: decode from y
  int64_t y = blockIdx.y;
  // y encodes (layer, row) as layer*num_rows + row
  int64_t row = y % num_rows;
  int64_t layer = y / num_rows;
  (void)layer_in_pool;
  int64_t chunk = (blockIdx.x * blockDim.x + threadIdx.x) * 16;
  int64_t item_bytes = pool_item_bytes[pool];
  if (chunk >= item_bytes) return;
  int64_t base = pool_layer_offsets[pool];
  if (base + layer >= pool_layer_offsets[pool + 1]) return;  // padding layer
  const uint8_t* s = src_ptrs[base + layer] + rows[row] * item_bytes;
  uint8_t* d = dst_ptrs[base + layer] + rows[row] * item_bytes;
  int64_t remain = item_bytes - chunk;
  if (remain >= 16) {
    *reinterpret_cast<uint4*>(d + chunk) = *reinterpret_cast<const uint4*>(s + chunk);
  } else {
    for (int64_t b = 0; b < remain; ++b) {
      d[chunk + b] = s[chunk + b];
    }
  }
}

extern "C" void musa_aiter_kv_copy_paged_rows_u8(
    const uint8_t* src,
    uint8_t* dst,
    const int64_t* src_rows,
    const int64_t* dst_rows,
    int64_t num_rows,
    int64_t item_bytes,
    int64_t num_src_rows,
    int64_t num_dst_rows,
    gpuStream_t stream) {
#ifdef __MUSACC__
  if (num_rows <= 0 || item_bytes <= 0) return;
  // lmcache-style block: (64 x-lanes, rows_per_block rows).
  int64_t rows_per_block = 4;
  if (item_bytes > 16384) {
    rows_per_block = 2;
  }
  dim3 threads(64, static_cast<unsigned>(rows_per_block), 1);
  int64_t row_groups = (num_rows + rows_per_block - 1) / rows_per_block;
  dim3 grid(static_cast<unsigned>(row_groups), 1, 1);
  musa_aiter_kv_copy_paged_rows_u8_kernel<<<grid, threads, 0, stream>>>(
      src, dst, src_rows, dst_rows, num_rows, item_bytes, rows_per_block,
      num_src_rows, num_dst_rows);
#endif
}

extern "C" void musa_aiter_kv_copy_paged_rows_layer_table_u8(
    const uint8_t* const* src_ptrs,
    uint8_t* const* dst_ptrs,
    const int64_t* src_rows,
    const int64_t* dst_rows,
    int64_t num_rows,
    int64_t item_bytes,
    int64_t num_layers,
    int64_t num_src_rows,
    int64_t num_dst_rows,
    gpuStream_t stream) {
#ifdef __MUSACC__
  if (num_rows <= 0 || item_bytes <= 0 || num_layers <= 0) return;
  int64_t rows_per_block = 4;
  if (item_bytes > 16384) {
    rows_per_block = 2;
  }
  dim3 threads(64, static_cast<unsigned>(rows_per_block), 1);
  int64_t row_groups = (num_rows + rows_per_block - 1) / rows_per_block;
  // Flat 1-D grid: row_groups * num_layers blocks (better SM occupancy on MUSA
  // than tall grid.y).
  dim3 grid(static_cast<unsigned>(row_groups * num_layers), 1, 1);
  musa_aiter_kv_copy_paged_rows_layer_table_u8_kernel<<<grid, threads, 0, stream>>>(
      src_ptrs, dst_ptrs, src_rows, dst_rows, num_rows, item_bytes, num_layers, rows_per_block,
      num_src_rows, num_dst_rows);
#endif
}

// P1'': contiguous-layers fast path. When the layer list comes from
// unbind(0) of one [L, P, IB] tensor (the sglang DSV4 pool layout), each
// layer's base is base + layer * P * IB: no pointer-table indirection.
__global__ void musa_aiter_kv_copy_paged_rows_contig_layers_u8_kernel(
    const uint8_t* __restrict__ src_base,
    uint8_t* __restrict__ dst_base,
    const int64_t* __restrict__ src_rows,
    const int64_t* __restrict__ dst_rows,
    int64_t num_rows,
    int64_t item_bytes,
    int64_t num_layers,
    int64_t num_src_rows,
    int64_t num_dst_rows) {
  // Flat grid.x = row_groups * num_layers; decode (row_group, layer).
  int64_t bi = static_cast<int64_t>(blockIdx.x);
  int64_t row_groups = (num_rows + (int64_t)blockDim.y - 1) / (int64_t)blockDim.y;
  int64_t layer = bi / row_groups;
  int64_t row_group = bi - layer * row_groups;
  int64_t i = row_group * (int64_t)blockDim.y + threadIdx.y;
  if (i >= num_rows) return;
  int64_t sr = src_rows[i];
  int64_t dr = dst_rows[i];
  if (sr < 0 || sr >= num_src_rows || dr < 0 || dr >= num_dst_rows) return;
  const uint8_t* s = src_base + layer * num_src_rows * item_bytes + sr * item_bytes;
  uint8_t* d = dst_base + layer * num_dst_rows * item_bytes + dr * item_bytes;
  musa_aiter_dsv4_copy_row(s, d, item_bytes);
}

extern "C" void musa_aiter_kv_copy_paged_rows_contig_layers_u8(
    const uint8_t* src_base,
    uint8_t* dst_base,
    const int64_t* src_rows,
    const int64_t* dst_rows,
    int64_t num_rows,
    int64_t item_bytes,
    int64_t num_layers,
    int64_t num_src_rows,
    int64_t num_dst_rows,
    gpuStream_t stream) {
#ifdef __MUSACC__
  if (num_rows <= 0 || item_bytes <= 0 || num_layers <= 0) return;
  int64_t rows_per_block = 4;
  if (item_bytes > 16384) {
    rows_per_block = 2;
  }
  dim3 threads(64, static_cast<unsigned>(rows_per_block), 1);
  int64_t row_groups = (num_rows + rows_per_block - 1) / rows_per_block;
  dim3 grid(static_cast<unsigned>(row_groups * num_layers), 1, 1);
  musa_aiter_kv_copy_paged_rows_contig_layers_u8_kernel<<<grid, threads, 0, stream>>>(
      src_base, dst_base, src_rows, dst_rows, num_rows, item_bytes, num_layers,
      num_src_rows, num_dst_rows);
#endif
}

extern "C" int musa_aiter_dsv4_kv_copy_abi_version() {
  return 1;
}

// ---------------------------------------------------------------------------
// Stride-aware byte-block transfer for LMCache DSV4 MLA / blocked-scale
// kernel groups (20261013). One launch covers a whole kernel-group transfer
// batch: all layers, all blocks, value+scale planes, arbitrary inter-block
// padding. Operates on raw bytes (uint32 words) so fp8/uint8 and fp32
// payloads share the same path.
//
// Staging layout: [NL, staging_tokens, row_bytes] contiguous.
//   mode 0 (NL_X_NB_BS_HS): row_bytes = hs * elem_size; the paged block row
//     for token t sits at block*stride + t*row_bytes.
//   mode 1 (NL_X_NB_BSV_BSS): row_bytes = value_bytes + scale_bytes; each
//     staging token row is [value_bytes | scale_bytes], while the paged page
//     stores all tokens' values first then all tokens' scales:
//     values at block*stride + t*value_bytes,
//     scales at block*stride + bs*value_bytes + t*scale_bytes.
// ---------------------------------------------------------------------------

struct LmcacheStridedLayerPointers {
  uint8_t* values[128];
};

extern "C" __global__ void lmcache_kv_strided_block_transfer_kernel(
    uint8_t* staging,
    LmcacheStridedLayerPointers layers,
    const int64_t* __restrict__ block_ids,
    int64_t direction,
    int64_t num_layers,
    int64_t idx_start,
    int64_t n_valid,
    int64_t offset_in_object,
    int64_t staging_tokens,
    int64_t block_size,
    int64_t row_bytes,
    int64_t row_words,
    int64_t paged_block_stride,
    int64_t mode,
    int64_t value_bytes,
    int64_t scale_bytes) {
  const int64_t total = num_layers * n_valid * block_size * row_words;
  for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       i < total;
       i += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    const int64_t w = i % row_words;
    int64_t tmp = i / row_words;
    const int64_t t = tmp % block_size;
    tmp /= block_size;
    const int64_t p = tmp % n_valid;
    const int64_t layer = tmp / n_valid;
    const int64_t engine_block = block_ids[idx_start + p];
    uint8_t* staged =
        staging +
        ((layer * staging_tokens) + offset_in_object + p * block_size + t) *
            row_bytes +
        w * 4;
    int64_t paged_off;
    if (mode == 0) {
      paged_off = engine_block * paged_block_stride + t * row_bytes + w * 4;
    } else {
      const int64_t value_words = value_bytes >> 2;
      if (w < value_words) {
        paged_off = engine_block * paged_block_stride + t * value_bytes +
                    w * 4;
      } else {
        paged_off = engine_block * paged_block_stride +
                    block_size * value_bytes + t * scale_bytes +
                    (w - value_words) * 4;
      }
    }
    uint8_t* paged = layers.values[layer] + paged_off;
    if (direction == 0) {
      *reinterpret_cast<uint32_t*>(paged) =
          *reinterpret_cast<const uint32_t*>(staged);
    } else {
      *reinterpret_cast<uint32_t*>(staged) =
          *reinterpret_cast<const uint32_t*>(paged);
    }
  }
}

extern "C" void lmcache_kv_strided_block_transfer(
    uint8_t* staging,
    const uint64_t* paged_ptrs,
    const int64_t* block_ids,
    int64_t direction,
    int64_t num_layers,
    int64_t idx_start,
    int64_t n_valid,
    int64_t offset_in_object,
    int64_t staging_tokens,
    int64_t block_size,
    int64_t row_bytes,
    int64_t paged_block_stride,
    int64_t mode,
    int64_t value_bytes,
    int64_t scale_bytes,
    gpuStream_t stream) {
#ifdef __MUSACC__
  if (n_valid <= 0 || num_layers <= 0 || num_layers > 128 ||
      block_size <= 0 || staging_tokens <= 0 || row_bytes <= 0 ||
      (row_bytes & 3) != 0) {
    return;
  }
  if (mode != 0 &&
      (value_bytes <= 0 || scale_bytes <= 0 || (value_bytes & 3) != 0 ||
       (scale_bytes & 3) != 0)) {
    return;
  }
  if (paged_block_stride <= 0) {
    return;
  }
  LmcacheStridedLayerPointers layers{};
  for (int64_t i = 0; i < num_layers; ++i)
    layers.values[i] = reinterpret_cast<uint8_t*>(paged_ptrs[i]);
  const int64_t row_words = row_bytes >> 2;
  const int64_t total = num_layers * n_valid * block_size * row_words;
  const int threads = 256;
  int blocks = static_cast<int>((total + threads - 1) / threads);
  if (blocks < 1) blocks = 1;
  if (blocks > 65535) blocks = 65535;
  lmcache_kv_strided_block_transfer_kernel<<<blocks, threads, 0, stream>>>(
      staging, layers, block_ids, direction, num_layers, idx_start, n_valid,
      offset_in_object, staging_tokens, block_size, row_bytes, row_words,
      paged_block_stride, mode, value_bytes, scale_bytes);
#endif
}
