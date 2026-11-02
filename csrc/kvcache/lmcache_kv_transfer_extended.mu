#include <stdint.h>

#include "../common/musa_compat.h"

namespace {

template <typename T>
__global__ void lmcache_list_to_paged_kernel(
    const T* memory_tensor,
    T* kvcache,
    const int64_t* slots,
    int64_t layer_index,
    int64_t num_layers,
    int64_t memory_tokens,
    int64_t num_tokens,
    int64_t skip_prefix,
    int64_t num_blocks,
    int64_t block_size,
    int64_t num_heads,
    int64_t head_size) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  int64_t elems = 2 * num_tokens * num_heads * head_size;
  if (idx >= elems) {
    return;
  }
  int64_t dim = idx % head_size;
  int64_t tmp = idx / head_size;
  int64_t head = tmp % num_heads;
  tmp /= num_heads;
  int64_t token = tmp % num_tokens;
  int64_t kv_axis = tmp / num_tokens;
  int64_t slot = slots[token];
  if (slot < 0) {
    return;
  }
  int64_t block = slot / block_size;
  int64_t offset = slot - block * block_size;
  if (block < 0 || block >= num_blocks) {
    return;
  }
  int64_t memory_idx = (((kv_axis * num_layers + layer_index) * memory_tokens
                         + skip_prefix + token) * num_heads + head) * head_size + dim;
  int64_t cache_idx = ((((kv_axis * num_blocks + block) * block_size + offset)
                        * num_heads + head) * head_size + dim);
  kvcache[cache_idx] = memory_tensor[memory_idx];
}

template <typename T>
__global__ void lmcache_paged_to_list_kernel(
    const T* kvcache,
    T* memory_tensor,
    const int64_t* slots,
    int64_t layer_index,
    int64_t num_layers,
    int64_t memory_tokens,
    int64_t num_tokens,
    int64_t skip_prefix,
    int64_t num_blocks,
    int64_t block_size,
    int64_t num_heads,
    int64_t head_size) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  int64_t elems = 2 * num_tokens * num_heads * head_size;
  if (idx >= elems) {
    return;
  }
  int64_t dim = idx % head_size;
  int64_t tmp = idx / head_size;
  int64_t head = tmp % num_heads;
  tmp /= num_heads;
  int64_t token = tmp % num_tokens;
  int64_t kv_axis = tmp / num_tokens;
  int64_t memory_idx = (((kv_axis * num_layers + layer_index) * memory_tokens
                         + skip_prefix + token) * num_heads + head) * head_size + dim;
  int64_t slot = slots[token];
  if (slot < 0) {
    memory_tensor[memory_idx] = static_cast<T>(0);
    return;
  }
  int64_t block = slot / block_size;
  int64_t offset = slot - block * block_size;
  if (block < 0 || block >= num_blocks) {
    memory_tensor[memory_idx] = static_cast<T>(0);
    return;
  }
  int64_t cache_idx = ((((kv_axis * num_blocks + block) * block_size + offset)
                        * num_heads + head) * head_size + dim);
  memory_tensor[memory_idx] = kvcache[cache_idx];
}

template <typename T>
__global__ void lmcache_two_x_nl_to_paged_kernel(
    const T* memory_tensor,
    T* kvcache,
    const int64_t* slots,
    int64_t kv_axis,
    int64_t layer_index,
    int64_t num_layers,
    int64_t memory_tokens,
    int64_t num_tokens,
    int64_t skip_prefix,
    int64_t num_blocks,
    int64_t block_size,
    int64_t num_heads,
    int64_t head_size) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  int64_t elems = num_tokens * num_heads * head_size;
  if (idx >= elems) {
    return;
  }
  int64_t dim = idx % head_size;
  int64_t tmp = idx / head_size;
  int64_t head = tmp % num_heads;
  int64_t token = tmp / num_heads;
  int64_t slot = slots[token];
  if (slot < 0) {
    return;
  }
  int64_t block = slot / block_size;
  int64_t offset = slot - block * block_size;
  if (block < 0 || block >= num_blocks) {
    return;
  }
  int64_t memory_idx = (((kv_axis * num_layers + layer_index) * memory_tokens
                         + skip_prefix + token) * num_heads + head) * head_size + dim;
  int64_t cache_idx = (((block * block_size + offset) * num_heads + head)
                       * head_size + dim);
  kvcache[cache_idx] = memory_tensor[memory_idx];
}

template <typename T>
__global__ void lmcache_two_x_nl_to_buffer_kernel(
    const T* kvcache,
    T* memory_tensor,
    const int64_t* slots,
    int64_t kv_axis,
    int64_t layer_index,
    int64_t num_layers,
    int64_t memory_tokens,
    int64_t num_tokens,
    int64_t skip_prefix,
    int64_t num_blocks,
    int64_t block_size,
    int64_t num_heads,
    int64_t head_size) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  int64_t elems = num_tokens * num_heads * head_size;
  if (idx >= elems) {
    return;
  }
  int64_t dim = idx % head_size;
  int64_t tmp = idx / head_size;
  int64_t head = tmp % num_heads;
  int64_t token = tmp / num_heads;
  int64_t memory_idx = (((kv_axis * num_layers + layer_index) * memory_tokens
                         + skip_prefix + token) * num_heads + head) * head_size + dim;
  int64_t slot = slots[token];
  if (slot < 0) {
    memory_tensor[memory_idx] = static_cast<T>(0);
    return;
  }
  int64_t block = slot / block_size;
  int64_t offset = slot - block * block_size;
  if (block < 0 || block >= num_blocks) {
    memory_tensor[memory_idx] = static_cast<T>(0);
    return;
  }
  int64_t cache_idx = (((block * block_size + offset) * num_heads + head)
                       * head_size + dim);
  memory_tensor[memory_idx] = kvcache[cache_idx];
}

template <typename T>
__global__ void lmcache_mla_list_to_paged_kernel(
    const T* memory_tensor,
    T* kvcache,
    const int64_t* slots,
    int64_t layer_index,
    int64_t memory_tokens,
    int64_t num_tokens,
    int64_t skip_prefix,
    int64_t num_blocks,
    int64_t block_size,
    int64_t head_size) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  int64_t elems = num_tokens * head_size;
  if (idx >= elems) {
    return;
  }
  int64_t dim = idx % head_size;
  int64_t token = idx / head_size;
  int64_t slot = slots[token];
  if (slot < 0) {
    return;
  }
  int64_t block = slot / block_size;
  int64_t offset = slot - block * block_size;
  if (block < 0 || block >= num_blocks) {
    return;
  }
  int64_t memory_idx = ((layer_index * memory_tokens + skip_prefix + token)
                        * head_size + dim);
  int64_t cache_idx = ((block * block_size + offset) * head_size + dim);
  kvcache[cache_idx] = memory_tensor[memory_idx];
}

template <typename T>
__global__ void lmcache_mla_paged_to_list_kernel(
    const T* kvcache,
    T* memory_tensor,
    const int64_t* slots,
    int64_t layer_index,
    int64_t memory_tokens,
    int64_t num_tokens,
    int64_t num_blocks,
    int64_t block_size,
    int64_t head_size) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  int64_t elems = num_tokens * head_size;
  if (idx >= elems) {
    return;
  }
  int64_t dim = idx % head_size;
  int64_t token = idx / head_size;
  int64_t memory_idx = ((layer_index * memory_tokens + token) * head_size + dim);
  int64_t slot = slots[token];
  if (slot < 0) {
    memory_tensor[memory_idx] = static_cast<T>(0);
    return;
  }
  int64_t block = slot / block_size;
  int64_t offset = slot - block * block_size;
  if (block < 0 || block >= num_blocks) {
    memory_tensor[memory_idx] = static_cast<T>(0);
    return;
  }
  int64_t cache_idx = ((block * block_size + offset) * head_size + dim);
  memory_tensor[memory_idx] = kvcache[cache_idx];
}

template <typename T>
__global__ void kv_paged_to_merged_kernel(
    const T* key_cache,
    const T* value_cache,
    const int64_t* slots,
    T* out,
    int64_t num_tokens,
    int64_t num_blocks,
    int64_t num_heads,
    int64_t head_size,
    int64_t block_size) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  int64_t elems = num_tokens * 2 * num_heads * head_size;
  if (idx >= elems) {
    return;
  }
  int64_t dim = idx % head_size;
  int64_t tmp = idx / head_size;
  int64_t head = tmp % num_heads;
  tmp /= num_heads;
  int64_t kv_axis = tmp % 2;
  int64_t token = tmp / 2;
  int64_t slot = slots[token];
  if (slot < 0) {
    out[idx] = static_cast<T>(0);
    return;
  }
  int64_t block = slot / block_size;
  int64_t offset = slot - block * block_size;
  if (block < 0 || block >= num_blocks) {
    out[idx] = static_cast<T>(0);
    return;
  }
  int64_t cache_idx = (((block * num_heads + head) * head_size + dim)
                       * block_size + offset);
  out[idx] = kv_axis == 0 ? key_cache[cache_idx] : value_cache[cache_idx];
}

template <typename T>
__global__ void kv_merged_to_paged_kernel(
    const T* merged,
    const int64_t* slots,
    T* key_cache,
    T* value_cache,
    int64_t num_tokens,
    int64_t num_blocks,
    int64_t num_heads,
    int64_t head_size,
    int64_t block_size) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  int64_t elems = num_tokens * 2 * num_heads * head_size;
  if (idx >= elems) {
    return;
  }
  int64_t dim = idx % head_size;
  int64_t tmp = idx / head_size;
  int64_t head = tmp % num_heads;
  tmp /= num_heads;
  int64_t kv_axis = tmp % 2;
  int64_t token = tmp / 2;
  int64_t slot = slots[token];
  if (slot < 0) {
    return;
  }
  int64_t block = slot / block_size;
  int64_t offset = slot - block * block_size;
  if (block < 0 || block >= num_blocks) {
    return;
  }
  int64_t cache_idx = (((block * num_heads + head) * head_size + dim)
                       * block_size + offset);
  if (kv_axis == 0) {
    key_cache[cache_idx] = merged[idx];
  } else {
    value_cache[cache_idx] = merged[idx];
  }
}

template <typename T>
__global__ void kv_paged_to_merged_multi_layer_kernel(
    const T* key_cache,
    const T* value_cache,
    const int64_t* slots,
    T* out,
    int64_t num_layers,
    int64_t num_tokens,
    int64_t num_blocks,
    int64_t num_heads,
    int64_t head_size,
    int64_t block_size) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  int64_t elems = num_layers * num_tokens * 2 * num_heads * head_size;
  if (idx >= elems) {
    return;
  }
  int64_t dim = idx % head_size;
  int64_t tmp = idx / head_size;
  int64_t head = tmp % num_heads;
  tmp /= num_heads;
  int64_t kv_axis = tmp % 2;
  tmp /= 2;
  int64_t token = tmp % num_tokens;
  int64_t layer = tmp / num_tokens;
  int64_t slot = slots[token];
  if (slot < 0) {
    out[idx] = static_cast<T>(0);
    return;
  }
  int64_t block = slot / block_size;
  int64_t offset = slot - block * block_size;
  if (block < 0 || block >= num_blocks) {
    out[idx] = static_cast<T>(0);
    return;
  }
  int64_t cache_idx = ((((layer * num_blocks + block) * num_heads + head)
                        * head_size + dim) * block_size + offset);
  out[idx] = kv_axis == 0 ? key_cache[cache_idx] : value_cache[cache_idx];
}

template <typename T>
__global__ void kv_merged_to_paged_multi_layer_kernel(
    const T* merged,
    const int64_t* slots,
    T* key_cache,
    T* value_cache,
    int64_t num_layers,
    int64_t num_tokens,
    int64_t num_blocks,
    int64_t num_heads,
    int64_t head_size,
    int64_t block_size) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  int64_t elems = num_layers * num_tokens * 2 * num_heads * head_size;
  if (idx >= elems) {
    return;
  }
  int64_t dim = idx % head_size;
  int64_t tmp = idx / head_size;
  int64_t head = tmp % num_heads;
  tmp /= num_heads;
  int64_t kv_axis = tmp % 2;
  tmp /= 2;
  int64_t token = tmp % num_tokens;
  int64_t layer = tmp / num_tokens;
  int64_t slot = slots[token];
  if (slot < 0) {
    return;
  }
  int64_t block = slot / block_size;
  int64_t offset = slot - block * block_size;
  if (block < 0 || block >= num_blocks) {
    return;
  }
  int64_t cache_idx = ((((layer * num_blocks + block) * num_heads + head)
                        * head_size + dim) * block_size + offset);
  if (kv_axis == 0) {
    key_cache[cache_idx] = merged[idx];
  } else {
    value_cache[cache_idx] = merged[idx];
  }
}

template <typename T>
__global__ void kv_paged_to_mla_kernel(
    const T* kv_cache,
    const T* k_pe_cache,
    const int64_t* slots,
    T* out_kv,
    T* out_pe,
    int64_t num_tokens,
    int64_t num_blocks,
    int64_t block_size,
    int64_t kv_dim,
    int64_t pe_dim) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  int64_t row_width = kv_dim + pe_dim;
  int64_t elems = num_tokens * row_width;
  if (idx >= elems) {
    return;
  }
  int64_t inner = idx % row_width;
  int64_t token = idx / row_width;
  int64_t slot = slots[token];
  bool is_kv = inner < kv_dim;
  int64_t dim = is_kv ? inner : inner - kv_dim;
  if (slot < 0) {
    if (is_kv) out_kv[token * kv_dim + dim] = static_cast<T>(0);
    else out_pe[token * pe_dim + dim] = static_cast<T>(0);
    return;
  }
  int64_t block = slot / block_size;
  int64_t offset = slot - block * block_size;
  if (block < 0 || block >= num_blocks) {
    if (is_kv) out_kv[token * kv_dim + dim] = static_cast<T>(0);
    else out_pe[token * pe_dim + dim] = static_cast<T>(0);
    return;
  }
  if (is_kv) {
    out_kv[token * kv_dim + dim] = kv_cache[(block * block_size + offset) * kv_dim + dim];
  } else {
    out_pe[token * pe_dim + dim] = k_pe_cache[(block * block_size + offset) * pe_dim + dim];
  }
}

template <typename T>
__global__ void kv_mla_to_paged_kernel(
    const T* kv_linear,
    const T* pe_linear,
    const int64_t* slots,
    T* kv_cache,
    T* pe_cache,
    int64_t num_tokens,
    int64_t num_blocks,
    int64_t block_size,
    int64_t kv_dim,
    int64_t pe_dim) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  int64_t row_width = kv_dim + pe_dim;
  int64_t elems = num_tokens * row_width;
  if (idx >= elems) {
    return;
  }
  int64_t inner = idx % row_width;
  int64_t token = idx / row_width;
  int64_t slot = slots[token];
  if (slot < 0) return;
  int64_t block = slot / block_size;
  int64_t offset = slot - block * block_size;
  if (block < 0 || block >= num_blocks) return;
  if (inner < kv_dim) {
    kv_cache[(block * block_size + offset) * kv_dim + inner] =
        kv_linear[token * kv_dim + inner];
  } else {
    int64_t dim = inner - kv_dim;
    pe_cache[(block * block_size + offset) * pe_dim + dim] =
        pe_linear[token * pe_dim + dim];
  }
}

}  // namespace

#ifdef __MUSACC__
#define MUSA_AITER_LAUNCH_1D(kernel, elems, stream, ...) \
  do { \
    if ((elems) > 0) { \
      int blocks = static_cast<int>(((elems) + 255) / 256); \
      (kernel)<<<blocks, 256, 0, (stream)>>>(__VA_ARGS__); \
    } \
  } while (0)
#else
#define MUSA_AITER_LAUNCH_1D(kernel, elems, stream, ...) \
  do { (void)(elems); (void)(stream); } while (0)
#endif

extern "C" void lmcache_kv_buffer_to_paged_list_float(
    const float* memory_tensor, float* kvcache, const int64_t* slots,
    int64_t layer_index, int64_t num_layers, int64_t memory_tokens,
    int64_t num_tokens, int64_t skip_prefix, int64_t num_blocks,
    int64_t block_size, int64_t num_heads, int64_t head_size, gpuStream_t stream) {
  MUSA_AITER_LAUNCH_1D(lmcache_list_to_paged_kernel<float>,
    2 * num_tokens * num_heads * head_size, stream, memory_tensor, kvcache,
    slots, layer_index, num_layers, memory_tokens, num_tokens, skip_prefix,
    num_blocks, block_size, num_heads, head_size);
}

extern "C" void lmcache_paged_to_kv_buffer_list_float(
    float* memory_tensor, const float* kvcache, const int64_t* slots,
    int64_t layer_index, int64_t num_layers, int64_t memory_tokens,
    int64_t num_tokens, int64_t skip_prefix, int64_t num_blocks,
    int64_t block_size, int64_t num_heads, int64_t head_size, gpuStream_t stream) {
  MUSA_AITER_LAUNCH_1D(lmcache_paged_to_list_kernel<float>,
    2 * num_tokens * num_heads * head_size, stream, kvcache, memory_tensor,
    slots, layer_index, num_layers, memory_tokens, num_tokens, skip_prefix,
    num_blocks, block_size, num_heads, head_size);
}

extern "C" void lmcache_kv_buffer_to_paged_two_x_nl_float(
    const float* memory_tensor, float* kvcache, const int64_t* slots,
    int64_t kv_axis, int64_t layer_index, int64_t num_layers,
    int64_t memory_tokens, int64_t num_tokens, int64_t skip_prefix,
    int64_t num_blocks, int64_t block_size, int64_t num_heads, int64_t head_size,
    gpuStream_t stream) {
  MUSA_AITER_LAUNCH_1D(lmcache_two_x_nl_to_paged_kernel<float>,
    num_tokens * num_heads * head_size, stream, memory_tensor, kvcache, slots,
    kv_axis, layer_index, num_layers, memory_tokens, num_tokens, skip_prefix,
    num_blocks, block_size, num_heads, head_size);
}

extern "C" void lmcache_paged_to_kv_buffer_two_x_nl_float(
    float* memory_tensor, const float* kvcache, const int64_t* slots,
    int64_t kv_axis, int64_t layer_index, int64_t num_layers,
    int64_t memory_tokens, int64_t num_tokens, int64_t skip_prefix,
    int64_t num_blocks, int64_t block_size, int64_t num_heads,
    int64_t head_size, gpuStream_t stream) {
  MUSA_AITER_LAUNCH_1D(lmcache_two_x_nl_to_buffer_kernel<float>,
    num_tokens * num_heads * head_size, stream, kvcache, memory_tensor, slots,
    kv_axis, layer_index, num_layers, memory_tokens, num_tokens, skip_prefix,
    num_blocks, block_size, num_heads, head_size);
}

extern "C" void lmcache_mla_buffer_to_paged_list_float(
    const float* memory_tensor, float* kvcache, const int64_t* slots,
    int64_t layer_index, int64_t memory_tokens, int64_t num_tokens, int64_t skip_prefix,
    int64_t num_blocks, int64_t block_size, int64_t head_size, gpuStream_t stream) {
  MUSA_AITER_LAUNCH_1D(lmcache_mla_list_to_paged_kernel<float>,
    num_tokens * head_size, stream, memory_tensor, kvcache, slots, layer_index,
    memory_tokens, num_tokens, skip_prefix, num_blocks, block_size, head_size);
}

extern "C" void lmcache_mla_paged_to_buffer_list_float(
    float* memory_tensor, const float* kvcache, const int64_t* slots,
    int64_t layer_index, int64_t memory_tokens, int64_t num_tokens, int64_t num_blocks,
    int64_t block_size, int64_t head_size, gpuStream_t stream) {
  MUSA_AITER_LAUNCH_1D(lmcache_mla_paged_to_list_kernel<float>,
    num_tokens * head_size, stream, kvcache, memory_tensor, slots, layer_index,
    memory_tokens, num_tokens, num_blocks, block_size, head_size);
}

#define MUSA_AITER_DEFINE_MLA_LIST_WRAPPERS(suffix) \
extern "C" void lmcache_mla_buffer_to_paged_list_##suffix( \
    const uint16_t* memory_tensor, uint16_t* kvcache, const int64_t* slots, \
    int64_t layer_index, int64_t memory_tokens, int64_t num_tokens, int64_t skip_prefix, \
    int64_t num_blocks, int64_t block_size, int64_t head_size, gpuStream_t stream) { \
  MUSA_AITER_LAUNCH_1D(lmcache_mla_list_to_paged_kernel<uint16_t>, \
    num_tokens * head_size, stream, memory_tensor, kvcache, slots, layer_index, \
    memory_tokens, num_tokens, skip_prefix, num_blocks, block_size, head_size); \
} \
extern "C" void lmcache_mla_paged_to_buffer_list_##suffix( \
    uint16_t* memory_tensor, const uint16_t* kvcache, const int64_t* slots, \
    int64_t layer_index, int64_t memory_tokens, int64_t num_tokens, int64_t num_blocks, \
    int64_t block_size, int64_t head_size, gpuStream_t stream) { \
  MUSA_AITER_LAUNCH_1D(lmcache_mla_paged_to_list_kernel<uint16_t>, \
    num_tokens * head_size, stream, kvcache, memory_tensor, slots, layer_index, \
    memory_tokens, num_tokens, num_blocks, block_size, head_size); \
}

#define MUSA_AITER_DEFINE_KV_LINEAR_WRAPPERS(suffix) \
extern "C" void musa_aiter_kv_paged_to_merged_linear_##suffix( \
    const uint16_t* key_cache, const uint16_t* value_cache, const int64_t* slots, \
    uint16_t* out, int64_t num_tokens, int64_t num_blocks, int64_t num_heads, \
    int64_t head_size, int64_t block_size, gpuStream_t stream) { \
  MUSA_AITER_LAUNCH_1D(kv_paged_to_merged_kernel<uint16_t>, \
    num_tokens * 2 * num_heads * head_size, stream, key_cache, value_cache, slots, \
    out, num_tokens, num_blocks, num_heads, head_size, block_size); \
} \
extern "C" void musa_aiter_kv_merged_linear_to_paged_##suffix( \
    const uint16_t* merged, const int64_t* slots, uint16_t* key_cache, \
    uint16_t* value_cache, int64_t num_tokens, int64_t num_blocks, int64_t num_heads, \
    int64_t head_size, int64_t block_size, gpuStream_t stream) { \
  MUSA_AITER_LAUNCH_1D(kv_merged_to_paged_kernel<uint16_t>, \
    num_tokens * 2 * num_heads * head_size, stream, merged, slots, key_cache, \
    value_cache, num_tokens, num_blocks, num_heads, head_size, block_size); \
} \
extern "C" void musa_aiter_kv_paged_to_merged_linear_multi_layer_##suffix( \
    const uint16_t* key_cache, const uint16_t* value_cache, const int64_t* slots, \
    uint16_t* out, int64_t num_layers, int64_t num_tokens, int64_t num_blocks, \
    int64_t num_heads, int64_t head_size, int64_t block_size, gpuStream_t stream) { \
  MUSA_AITER_LAUNCH_1D(kv_paged_to_merged_multi_layer_kernel<uint16_t>, \
    num_layers * num_tokens * 2 * num_heads * head_size, stream, key_cache, \
    value_cache, slots, out, num_layers, num_tokens, num_blocks, num_heads, \
    head_size, block_size); \
} \
extern "C" void musa_aiter_kv_merged_linear_to_paged_multi_layer_##suffix( \
    const uint16_t* merged, const int64_t* slots, uint16_t* key_cache, \
    uint16_t* value_cache, int64_t num_layers, int64_t num_tokens, int64_t num_blocks, \
    int64_t num_heads, int64_t head_size, int64_t block_size, gpuStream_t stream) { \
  MUSA_AITER_LAUNCH_1D(kv_merged_to_paged_multi_layer_kernel<uint16_t>, \
    num_layers * num_tokens * 2 * num_heads * head_size, stream, merged, slots, \
    key_cache, value_cache, num_layers, num_tokens, num_blocks, num_heads, \
    head_size, block_size); \
} \
extern "C" void musa_aiter_kv_paged_to_mla_linear_##suffix( \
    const uint16_t* kv_cache, const uint16_t* pe_cache, const int64_t* slots, \
    uint16_t* out_kv, uint16_t* out_pe, int64_t num_tokens, int64_t num_blocks, \
    int64_t block_size, int64_t kv_dim, int64_t pe_dim, gpuStream_t stream) { \
  MUSA_AITER_LAUNCH_1D(kv_paged_to_mla_kernel<uint16_t>, \
    num_tokens * (kv_dim + pe_dim), stream, kv_cache, pe_cache, slots, out_kv, \
    out_pe, num_tokens, num_blocks, block_size, kv_dim, pe_dim); \
} \
extern "C" void musa_aiter_kv_mla_linear_to_paged_##suffix( \
    const uint16_t* kv_linear, const uint16_t* pe_linear, const int64_t* slots, \
    uint16_t* kv_cache, uint16_t* pe_cache, int64_t num_tokens, int64_t num_blocks, \
    int64_t block_size, int64_t kv_dim, int64_t pe_dim, gpuStream_t stream) { \
  MUSA_AITER_LAUNCH_1D(kv_mla_to_paged_kernel<uint16_t>, \
    num_tokens * (kv_dim + pe_dim), stream, kv_linear, pe_linear, slots, kv_cache, \
    pe_cache, num_tokens, num_blocks, block_size, kv_dim, pe_dim); \
}

MUSA_AITER_DEFINE_MLA_LIST_WRAPPERS(half)
MUSA_AITER_DEFINE_MLA_LIST_WRAPPERS(bfloat16)
MUSA_AITER_DEFINE_MLA_LIST_WRAPPERS(u16)
MUSA_AITER_DEFINE_KV_LINEAR_WRAPPERS(half)
MUSA_AITER_DEFINE_KV_LINEAR_WRAPPERS(bfloat16)

#undef MUSA_AITER_DEFINE_MLA_LIST_WRAPPERS
#undef MUSA_AITER_DEFINE_KV_LINEAR_WRAPPERS
#undef MUSA_AITER_LAUNCH_1D
