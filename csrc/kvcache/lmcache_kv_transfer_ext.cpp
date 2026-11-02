#include <torch/extension.h>
#include <ATen/ATen.h>
#include <c10/core/DeviceGuard.h>
#include <stdexcept>
#include <vector>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <musa_runtime_api.h>
#include <torch_musa/csrc/core/MUSAStream.h>

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
    musaStream_t stream);

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
    musaStream_t stream);

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
    musaStream_t stream);

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
    musaStream_t stream);

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
    musaStream_t stream);

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
    musaStream_t stream);

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
    musaStream_t stream);

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
    musaStream_t stream);

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
    musaStream_t stream);

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
    musaStream_t stream);

namespace {

void check_common(
    const at::Tensor& memory_tensor,
    const std::vector<at::Tensor>& kvcaches,
    const at::Tensor& slot_mapping,
    int64_t block_size,
    int64_t num_heads,
    int64_t head_size) {
  TORCH_CHECK(kvcaches.size() == 2, "LMCache C++ extension currently expects exactly 2 layers");
  TORCH_CHECK(memory_tensor.is_musa(), "memory_tensor must be a MUSA tensor");
  TORCH_CHECK(slot_mapping.is_musa(), "slot_mapping must be a MUSA tensor");
  TORCH_CHECK(memory_tensor.is_contiguous(), "memory_tensor must be contiguous");
  TORCH_CHECK(slot_mapping.is_contiguous(), "slot_mapping must be contiguous");
  TORCH_CHECK(slot_mapping.scalar_type() == at::ScalarType::Long, "slot_mapping must be int64");
  TORCH_CHECK(memory_tensor.scalar_type() == at::ScalarType::Half || memory_tensor.scalar_type() == at::ScalarType::BFloat16,
              "memory_tensor must be float16 or bfloat16");
  TORCH_CHECK(memory_tensor.dim() == 4, "memory_tensor must have shape [2, NL, T, hidden]");
  TORCH_CHECK(memory_tensor.size(0) == 2, "memory_tensor first dim must be 2");
  TORCH_CHECK(memory_tensor.size(1) == 2, "memory_tensor NL must be 2 for list2 extension path");
  TORCH_CHECK(memory_tensor.size(3) == num_heads * head_size, "hidden dim mismatch");
  for (const auto& kvcache : kvcaches) {
    TORCH_CHECK(kvcache.is_musa(), "kvcache must be a MUSA tensor");
    TORCH_CHECK(kvcache.is_contiguous(), "kvcache must be contiguous");
    TORCH_CHECK(kvcache.scalar_type() == memory_tensor.scalar_type(), "kvcache dtype must match memory_tensor dtype");
    TORCH_CHECK(kvcache.dim() == 5, "kvcache must have shape [2, NB, BS, NH, HS]");
    TORCH_CHECK(kvcache.size(0) == 2, "kvcache first dim must be 2");
    TORCH_CHECK(kvcache.size(2) == block_size, "block_size mismatch");
    TORCH_CHECK(kvcache.size(3) == num_heads, "num_heads mismatch");
    TORCH_CHECK(kvcache.size(4) == head_size, "head_size mismatch");
  }
}

void check_two_x_nl(
    const at::Tensor& memory_tensor,
    const std::vector<std::vector<at::Tensor>>& kvcaches,
    const at::Tensor& slot_mapping,
    int64_t block_size,
    int64_t num_heads,
    int64_t head_size) {
  TORCH_CHECK(kvcaches.size() == 2, "split cache must contain K and V groups");
  TORCH_CHECK(memory_tensor.is_musa() && memory_tensor.is_contiguous(), "memory_tensor must be contiguous MUSA");
  TORCH_CHECK(memory_tensor.dim() == 4 && memory_tensor.size(0) == 2, "memory_tensor must be [2, NL, T, hidden]");
  const int64_t num_layers = memory_tensor.size(1);
  const int64_t memory_tokens = memory_tensor.size(2);
  const int64_t hidden = num_heads * head_size;
  TORCH_CHECK(num_layers > 0 && num_layers <= 64, "split cache layer count must be in [1, 64]");
  TORCH_CHECK(memory_tokens >= 0, "memory token count must be non-negative");
  TORCH_CHECK(block_size > 0 && num_heads > 0 && head_size > 0, "block_size, num_heads, and head_size must be positive");
  TORCH_CHECK(hidden % 4 == 0, "split cache hidden size must be divisible by 4");
  TORCH_CHECK(memory_tensor.scalar_type() == at::ScalarType::Half || memory_tensor.scalar_type() == at::ScalarType::BFloat16, "memory_tensor must be float16 or bfloat16");
  TORCH_CHECK(memory_tensor.size(3) == hidden, "hidden dim mismatch");
  TORCH_CHECK(slot_mapping.is_musa() && slot_mapping.is_contiguous() && slot_mapping.scalar_type() == at::ScalarType::Long, "slot_mapping must be contiguous MUSA int64");
  TORCH_CHECK(slot_mapping.dim() == 1, "slot_mapping must be one-dimensional");
  TORCH_CHECK(slot_mapping.device() == memory_tensor.device(), "slot_mapping and memory_tensor must share a device");
  int64_t num_blocks = -1;
  for (const auto& group : kvcaches) {
    TORCH_CHECK(static_cast<int64_t>(group.size()) == num_layers, "split cache layer count mismatch");
    for (const auto& cache : group) {
      TORCH_CHECK(cache.is_musa() && cache.is_contiguous(), "split cache must be contiguous MUSA");
      TORCH_CHECK(cache.device() == memory_tensor.device(), "split cache and memory_tensor must share a device");
      TORCH_CHECK(cache.scalar_type() == memory_tensor.scalar_type(), "split cache dtype mismatch");
      TORCH_CHECK(cache.dim() == 4 && cache.size(1) == block_size && cache.size(2) == num_heads && cache.size(3) == head_size, "split cache must be [NB, BS, NH, HS]");
      TORCH_CHECK(cache.size(0) > 0, "split cache num_blocks must be positive");
      if (num_blocks < 0) num_blocks = cache.size(0);
      TORCH_CHECK(cache.size(0) == num_blocks, "split cache num_blocks mismatch");
    }
  }
}

std::vector<uint64_t> two_x_nl_pointers(
    const std::vector<std::vector<at::Tensor>>& kvcaches) {
  std::vector<uint64_t> pointers;
  pointers.reserve(kvcaches[0].size() + kvcaches[1].size());
  for (const auto& group : kvcaches)
    for (const auto& cache : group)
      pointers.push_back(reinterpret_cast<uint64_t>(cache.data_ptr()));
  return pointers;
}

musaStream_t current_stream_for(const at::Tensor& tensor) {
  c10::DeviceGuard guard(tensor.device());
  return c10::musa::getCurrentMUSAStream(tensor.device().index()).stream();
}

bool timing_enabled() {
  const char* cpp_value = std::getenv("MUSA_AITER_LMCACHE_CPP_TIMING");
  if (cpp_value != nullptr) return cpp_value[0] == '1';
  const char* value = std::getenv("MUSA_AITER_LMCACHE_TIMING");
  return value != nullptr && value[0] == '1';
}

bool aten_transfer_enabled() {
  const char* value = std::getenv("MUSA_AITER_LMCACHE_ATEN_TRANSFER");
  return value != nullptr && value[0] == '1';
}

bool coalesced_transfer_enabled() {
  const char* value = std::getenv("MUSA_AITER_LMCACHE_COALESCED_TRANSFER");
  return value != nullptr && value[0] == '1';
}

bool coalesced_to_gpu_enabled() {
  const char* value = std::getenv("MUSA_AITER_LMCACHE_COALESCED_TO_GPU");
  if (value != nullptr) return value[0] == '1';
  return coalesced_transfer_enabled();
}

bool coalesced_from_gpu_enabled() {
  const char* value = std::getenv("MUSA_AITER_LMCACHE_COALESCED_FROM_GPU");
  if (value != nullptr) return value[0] == '1';
  return coalesced_transfer_enabled();
}

bool sorted_gather_enabled() {
  const char* value = std::getenv("MUSA_AITER_LMCACHE_SORTED_GATHER");
  return value != nullptr && value[0] == '1';
}

bool contiguous_gather_enabled() {
  const char* value = std::getenv("MUSA_AITER_LMCACHE_CONTIGUOUS_GATHER");
  return value != nullptr && value[0] == '1';
}

bool contiguous_bulk_enabled() {
  const char* value = std::getenv("MUSA_AITER_LMCACHE_CONTIGUOUS_BULK");
  return value != nullptr && value[0] == '1';
}

bool torchmusa_style_enabled() {
  const char* value = std::getenv("MUSA_AITER_LMCACHE_TORCHMUSA_STYLE");
  return value != nullptr && value[0] == '1';
}

void aten_kv_buffer_to_paged_list2(
    const at::Tensor& memory_tensor,
    const std::vector<at::Tensor>& kvcaches,
    const at::Tensor& slot_mapping,
    int64_t skip_prefix_n_tokens,
    int64_t block_size,
    int64_t hidden) {
  const int64_t num_tokens = slot_mapping.numel();
  const int64_t total_slots = kvcaches[0].size(1) * block_size;
  const std::vector<int64_t> flat_shape{total_slots, hidden};
  for (int64_t layer = 0; layer < 2; ++layer) {
    for (int64_t kv_axis = 0; kv_axis < 2; ++kv_axis) {
      at::Tensor dst = kvcaches[layer].select(0, kv_axis).reshape(flat_shape);
      at::Tensor src = memory_tensor.select(0, kv_axis).select(0, layer).narrow(0, skip_prefix_n_tokens, num_tokens);
      dst.index_copy_(0, slot_mapping, src);
    }
  }
}

void aten_paged_to_kv_buffer_list2(
    const std::vector<at::Tensor>& kvcaches,
    const at::Tensor& memory_tensor,
    const at::Tensor& slot_mapping,
    int64_t block_size,
    int64_t hidden) {
  const int64_t num_tokens = slot_mapping.numel();
  const int64_t total_slots = kvcaches[0].size(1) * block_size;
  const std::vector<int64_t> flat_shape{total_slots, hidden};
  for (int64_t layer = 0; layer < 2; ++layer) {
    for (int64_t kv_axis = 0; kv_axis < 2; ++kv_axis) {
      at::Tensor src = kvcaches[layer].select(0, kv_axis).reshape(flat_shape);
      at::Tensor dst = memory_tensor.select(0, kv_axis).select(0, layer).narrow(0, 0, num_tokens);
      dst.copy_(src.index_select(0, slot_mapping));
    }
  }
}

const char* transfer_impl_name(const char* op) {
  if (aten_transfer_enabled()) return "aten";
  if (std::strcmp(op, "from_gpu_paged_to_buffer") == 0) {
    if (torchmusa_style_enabled()) return "torchmusa-style";
    if (contiguous_bulk_enabled()) return "contiguous-bulk";
    if (contiguous_gather_enabled()) return "contiguous-gather";
    if (sorted_gather_enabled()) return "sorted-gather";
    return coalesced_from_gpu_enabled() ? "coalesced" : "mu";
  }
  if (std::strcmp(op, "to_gpu_buffer_to_paged") == 0) {
    return coalesced_to_gpu_enabled() ? "coalesced" : "mu";
  }
  return coalesced_transfer_enabled() ? "coalesced" : "mu";
}

struct TimingScope {
  const char* op;
  int64_t num_tokens;
  int64_t hidden;
  size_t bytes;
  musaStream_t stream;
  bool enabled;
  std::chrono::steady_clock::time_point host_start;
  musaEvent_t device_start{};
  musaEvent_t device_stop{};

  TimingScope(const char* op_name, int64_t tokens, int64_t hidden_dim, size_t bytes_copied, musaStream_t musa_stream)
      : op(op_name), num_tokens(tokens), hidden(hidden_dim), bytes(bytes_copied), stream(musa_stream), enabled(timing_enabled()) {
    if (!enabled) return;
    host_start = std::chrono::steady_clock::now();
    musaEventCreate(&device_start);
    musaEventCreate(&device_stop);
    musaEventRecord(device_start, stream);
  }

  void finish_after_launch() {
    if (!enabled) return;
    const auto after_launch = std::chrono::steady_clock::now();
    musaEventRecord(device_stop, stream);
    musaEventSynchronize(device_stop);
    float device_ms = 0.0f;
    musaEventElapsedTime(&device_ms, device_start, device_stop);
    const auto after_sync = std::chrono::steady_clock::now();
    const double host_launch_us = std::chrono::duration<double, std::micro>(after_launch - host_start).count();
    const double host_total_us = std::chrono::duration<double, std::micro>(after_sync - host_start).count();
    const double gb = static_cast<double>(bytes) / 1.0e9;
    const double bandwidth = device_ms > 0.0f ? gb / (static_cast<double>(device_ms) / 1000.0) : 0.0;
    const char* transfer_impl = transfer_impl_name(op);
    std::fprintf(stderr,
                 "[musa_aiter_lmcache_timing] op=%s impl=%s tokens=%ld hidden=%ld bytes=%zu host_launch_us=%.3f device_ms=%.6f host_total_us=%.3f bandwidth_gbps=%.3f\n",
                 op,
                 transfer_impl,
                 static_cast<long>(num_tokens),
                 static_cast<long>(hidden),
                 bytes,
                 host_launch_us,
                 device_ms,
                 host_total_us,
                 bandwidth);
    musaEventDestroy(device_start);
    musaEventDestroy(device_stop);
  }
};

}  // namespace

bool kv_buffer_to_paged_list2(
    const at::Tensor& memory_tensor,
    const std::vector<at::Tensor>& kvcaches,
    const at::Tensor& slot_mapping,
    int64_t skip_prefix_n_tokens,
    int64_t block_size,
    int64_t num_heads,
    int64_t head_size) {
  check_common(memory_tensor, kvcaches, slot_mapping, block_size, num_heads, head_size);
  c10::DeviceGuard guard(memory_tensor.device());
  musaStream_t stream = current_stream_for(memory_tensor);
  const int64_t hidden = num_heads * head_size;
  const size_t bytes = static_cast<size_t>(2) * static_cast<size_t>(kvcaches.size()) * static_cast<size_t>(slot_mapping.numel()) * static_cast<size_t>(hidden) * static_cast<size_t>(memory_tensor.element_size());
  TimingScope timing("to_gpu_buffer_to_paged", slot_mapping.numel(), hidden, bytes, stream);
  if (aten_transfer_enabled()) {
    aten_kv_buffer_to_paged_list2(memory_tensor, kvcaches, slot_mapping, skip_prefix_n_tokens, block_size, hidden);
  } else if (coalesced_to_gpu_enabled()) {
    lmcache_kv_buffer_to_paged_list2_u16_coalesced(
        reinterpret_cast<const uint16_t*>(memory_tensor.data_ptr()),
        reinterpret_cast<uint16_t*>(kvcaches[0].data_ptr()),
        reinterpret_cast<uint16_t*>(kvcaches[1].data_ptr()),
        reinterpret_cast<const int64_t*>(slot_mapping.data_ptr()),
        memory_tensor.size(2),
        slot_mapping.numel(),
        skip_prefix_n_tokens,
        kvcaches[0].size(1),
        block_size,
        num_heads,
        head_size,
        stream);
  } else {
    lmcache_kv_buffer_to_paged_list2_u16(
        reinterpret_cast<const uint16_t*>(memory_tensor.data_ptr()),
        reinterpret_cast<uint16_t*>(kvcaches[0].data_ptr()),
        reinterpret_cast<uint16_t*>(kvcaches[1].data_ptr()),
        reinterpret_cast<const int64_t*>(slot_mapping.data_ptr()),
        memory_tensor.size(2),
        slot_mapping.numel(),
        skip_prefix_n_tokens,
        kvcaches[0].size(1),
        block_size,
        num_heads,
        head_size,
        stream);
  }
  timing.finish_after_launch();
  return true;
}

bool paged_to_kv_buffer_list2(
    const std::vector<at::Tensor>& kvcaches,
    const at::Tensor& memory_tensor,
    const at::Tensor& slot_mapping,
    int64_t block_size,
    int64_t num_heads,
    int64_t head_size) {
  check_common(memory_tensor, kvcaches, slot_mapping, block_size, num_heads, head_size);
  c10::DeviceGuard guard(memory_tensor.device());
  musaStream_t stream = current_stream_for(memory_tensor);
  const int64_t hidden = num_heads * head_size;
  const size_t bytes = static_cast<size_t>(2) * static_cast<size_t>(kvcaches.size()) * static_cast<size_t>(slot_mapping.numel()) * static_cast<size_t>(hidden) * static_cast<size_t>(memory_tensor.element_size());
  TimingScope timing("from_gpu_paged_to_buffer", slot_mapping.numel(), hidden, bytes, stream);
  if (aten_transfer_enabled()) {
    aten_paged_to_kv_buffer_list2(kvcaches, memory_tensor, slot_mapping, block_size, hidden);
  } else if (torchmusa_style_enabled()) {
    lmcache_paged_to_kv_buffer_list2_u16_torchmusa_style(
        reinterpret_cast<uint16_t*>(memory_tensor.data_ptr()),
        reinterpret_cast<const uint16_t*>(kvcaches[0].data_ptr()),
        reinterpret_cast<const uint16_t*>(kvcaches[1].data_ptr()),
        reinterpret_cast<const int64_t*>(slot_mapping.data_ptr()),
        memory_tensor.size(2),
        slot_mapping.numel(),
        0,
        kvcaches[0].size(1),
        block_size,
        num_heads,
        head_size,
        stream);
  } else if (contiguous_bulk_enabled()) {
    lmcache_paged_to_kv_buffer_list2_u16_contiguous_bulk(
        reinterpret_cast<uint16_t*>(memory_tensor.data_ptr()),
        reinterpret_cast<const uint16_t*>(kvcaches[0].data_ptr()),
        reinterpret_cast<const uint16_t*>(kvcaches[1].data_ptr()),
        reinterpret_cast<const int64_t*>(slot_mapping.data_ptr()),
        memory_tensor.size(2),
        slot_mapping.numel(),
        0,
        kvcaches[0].size(1),
        block_size,
        num_heads,
        head_size,
        stream);
  } else if (contiguous_gather_enabled()) {
    lmcache_paged_to_kv_buffer_list2_u16_contiguous(
        reinterpret_cast<uint16_t*>(memory_tensor.data_ptr()),
        reinterpret_cast<const uint16_t*>(kvcaches[0].data_ptr()),
        reinterpret_cast<const uint16_t*>(kvcaches[1].data_ptr()),
        reinterpret_cast<const int64_t*>(slot_mapping.data_ptr()),
        memory_tensor.size(2),
        slot_mapping.numel(),
        0,
        kvcaches[0].size(1),
        block_size,
        num_heads,
        head_size,
        stream);
  } else if (sorted_gather_enabled()) {
    auto sorted = slot_mapping.sort(0, false);
    at::Tensor sorted_indices = std::get<1>(sorted).to(at::ScalarType::Int);
    lmcache_paged_to_kv_buffer_list2_u16_sorted(
        reinterpret_cast<uint16_t*>(memory_tensor.data_ptr()),
        reinterpret_cast<const uint16_t*>(kvcaches[0].data_ptr()),
        reinterpret_cast<const uint16_t*>(kvcaches[1].data_ptr()),
        reinterpret_cast<const int64_t*>(slot_mapping.data_ptr()),
        reinterpret_cast<const uint8_t*>(sorted_indices.data_ptr()),
        memory_tensor.size(2),
        slot_mapping.numel(),
        0,
        kvcaches[0].size(1),
        block_size,
        num_heads,
        head_size,
        stream);
  } else if (coalesced_from_gpu_enabled()) {
    lmcache_paged_to_kv_buffer_list2_u16_coalesced(
        reinterpret_cast<uint16_t*>(memory_tensor.data_ptr()),
        reinterpret_cast<const uint16_t*>(kvcaches[0].data_ptr()),
        reinterpret_cast<const uint16_t*>(kvcaches[1].data_ptr()),
        reinterpret_cast<const int64_t*>(slot_mapping.data_ptr()),
        memory_tensor.size(2),
        slot_mapping.numel(),
        0,
        kvcaches[0].size(1),
        block_size,
        num_heads,
        head_size,
        stream);
  } else {
    lmcache_paged_to_kv_buffer_list2_u16(
        reinterpret_cast<uint16_t*>(memory_tensor.data_ptr()),
        reinterpret_cast<const uint16_t*>(kvcaches[0].data_ptr()),
        reinterpret_cast<const uint16_t*>(kvcaches[1].data_ptr()),
        reinterpret_cast<const int64_t*>(slot_mapping.data_ptr()),
        memory_tensor.size(2),
        slot_mapping.numel(),
        0,
        kvcaches[0].size(1),
        block_size,
        num_heads,
        head_size,
        stream);
  }
  timing.finish_after_launch();
  return true;
}

bool kv_buffer_to_paged_two_x_nl(
    const at::Tensor& memory_tensor,
    const std::vector<std::vector<at::Tensor>>& kvcaches,
    const at::Tensor& slot_mapping,
    int64_t skip_prefix_n_tokens,
    int64_t block_size,
    int64_t num_heads,
    int64_t head_size) {
  check_two_x_nl(memory_tensor, kvcaches, slot_mapping, block_size, num_heads, head_size);
  TORCH_CHECK(skip_prefix_n_tokens >= 0, "skip_prefix_n_tokens must be non-negative");
  TORCH_CHECK(skip_prefix_n_tokens + slot_mapping.numel() <= memory_tensor.size(2), "transfer range exceeds memory_tensor token extent");
  c10::DeviceGuard guard(memory_tensor.device());
  const auto pointers = two_x_nl_pointers(kvcaches);
  lmcache_kv_buffer_to_paged_two_x_nl_coalesced_u16(
      reinterpret_cast<const uint16_t*>(memory_tensor.data_ptr()), pointers.data(),
      reinterpret_cast<const int64_t*>(slot_mapping.data_ptr()), memory_tensor.size(1),
      memory_tensor.size(2), slot_mapping.numel(), skip_prefix_n_tokens,
      kvcaches[0][0].size(0), block_size, num_heads, head_size,
      current_stream_for(memory_tensor));
  return true;
}

bool paged_to_kv_buffer_two_x_nl(
    const std::vector<std::vector<at::Tensor>>& kvcaches,
    const at::Tensor& memory_tensor,
    const at::Tensor& slot_mapping,
    int64_t block_size,
    int64_t num_heads,
    int64_t head_size) {
  check_two_x_nl(memory_tensor, kvcaches, slot_mapping, block_size, num_heads, head_size);
  c10::DeviceGuard guard(memory_tensor.device());
  const auto pointers = two_x_nl_pointers(kvcaches);
  lmcache_paged_to_kv_buffer_two_x_nl_coalesced_u16(
      reinterpret_cast<uint16_t*>(memory_tensor.data_ptr()), pointers.data(),
      reinterpret_cast<const int64_t*>(slot_mapping.data_ptr()), memory_tensor.size(1),
      memory_tensor.size(2), slot_mapping.numel(), kvcaches[0][0].size(0),
      block_size, num_heads, head_size, current_stream_for(memory_tensor));
  return true;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("kv_buffer_to_paged_list2", &kv_buffer_to_paged_list2, "LMCache non-MLA buffer to paged list2");
  m.def("paged_to_kv_buffer_list2", &paged_to_kv_buffer_list2, "LMCache non-MLA paged list2 to buffer");
  m.def("kv_buffer_to_paged_two_x_nl", &kv_buffer_to_paged_two_x_nl, "LMCache split K/V buffer to paged caches");
  m.def("paged_to_kv_buffer_two_x_nl", &paged_to_kv_buffer_two_x_nl, "LMCache split K/V paged caches to buffer");
}
