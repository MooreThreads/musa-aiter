#include <stdint.h>
#include "../common/musa_compat.h"

extern "C" __global__ void musa_aiter_hicache_transfer_kernel(
    const uint8_t* src_k,
    uint8_t* dst_k,
    const uint8_t* src_v,
    uint8_t* dst_v,
    const int64_t* src_indices,
    const int64_t* dst_indices,
    int64_t num_items,
    int64_t start_layer,
    int64_t num_layers,
    int64_t item_size_bytes,
    int64_t src_page_stride_bytes,
    int64_t src_layer_stride_bytes,
    int64_t dst_page_stride_bytes,
    int64_t dst_layer_stride_bytes) {
  const int64_t word_index =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t item_words = item_size_bytes / static_cast<int64_t>(sizeof(uint64_t));
  const int64_t total_words = num_items * num_layers * item_words;
  if (word_index >= total_words) {
    return;
  }

  const int64_t byte_word = word_index % item_words;
  const int64_t item_and_layer = word_index / item_words;
  const int64_t item = item_and_layer % num_items;
  const int64_t layer = item_and_layer / num_items;
  const int64_t src_page = src_indices[item];
  const int64_t dst_page = dst_indices[item];
  if (src_page < 0 || dst_page < 0) {
    return;
  }

  const int64_t src_offset =
      src_page * src_page_stride_bytes +
      (start_layer + layer) * src_layer_stride_bytes +
      byte_word * static_cast<int64_t>(sizeof(uint64_t));
  const int64_t dst_offset =
      dst_page * dst_page_stride_bytes +
      (start_layer + layer) * dst_layer_stride_bytes +
      byte_word * static_cast<int64_t>(sizeof(uint64_t));
  reinterpret_cast<uint64_t*>(dst_k + dst_offset)[0] =
      reinterpret_cast<const uint64_t*>(src_k + src_offset)[0];
  if (src_v != nullptr && dst_v != nullptr) {
    reinterpret_cast<uint64_t*>(dst_v + dst_offset)[0] =
        reinterpret_cast<const uint64_t*>(src_v + src_offset)[0];
  }
}

extern "C" int musa_aiter_hicache_transfer_abi_version() {
  return 1;
}

extern "C" void musa_aiter_hicache_transfer_bytes(
    const void* src_k,
    void* dst_k,
    const void* src_v,
    void* dst_v,
    const int64_t* src_indices,
    const int64_t* dst_indices,
    int64_t num_items,
    int64_t start_layer,
    int64_t num_layers,
    int64_t item_size_bytes,
    int64_t src_page_stride_bytes,
    int64_t src_layer_stride_bytes,
    int64_t dst_page_stride_bytes,
    int64_t dst_layer_stride_bytes,
    gpuStream_t stream) {
#ifdef __MUSACC__
  const int64_t item_words =
      item_size_bytes / static_cast<int64_t>(sizeof(uint64_t));
  const int64_t total_words = num_items * num_layers * item_words;
  const int threads = 256;
  const int blocks = static_cast<int>((total_words + threads - 1) / threads);
  if (blocks <= 0) {
    return;
  }
  musa_aiter_hicache_transfer_kernel<<<blocks, threads, 0, stream>>>(
      static_cast<const uint8_t*>(src_k),
      static_cast<uint8_t*>(dst_k),
      static_cast<const uint8_t*>(src_v),
      static_cast<uint8_t*>(dst_v),
      src_indices,
      dst_indices,
      num_items,
      start_layer,
      num_layers,
      item_size_bytes,
      src_page_stride_bytes,
      src_layer_stride_bytes,
      dst_page_stride_bytes,
      dst_layer_stride_bytes);
#else
  (void)src_k;
  (void)dst_k;
  (void)src_v;
  (void)dst_v;
  (void)src_indices;
  (void)dst_indices;
  (void)num_items;
  (void)start_layer;
  (void)num_layers;
  (void)item_size_bytes;
  (void)src_page_stride_bytes;
  (void)src_layer_stride_bytes;
  (void)dst_page_stride_bytes;
  (void)dst_layer_stride_bytes;
  (void)stream;
#endif
}
