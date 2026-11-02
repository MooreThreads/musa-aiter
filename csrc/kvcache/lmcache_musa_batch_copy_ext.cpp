#include <torch/extension.h>

#include <ATen/ATen.h>
#include <ATen/core/CachingHostAllocator.h>
#include <c10/core/DeviceGuard.h>
#include <c10/util/Exception.h>
#include <musa_runtime_api.h>
#include <torch_musa/csrc/core/MUSACachingAllocator.h>
#include <torch_musa/csrc/core/MUSAStream.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <unordered_set>
#include <utility>
#include <vector>

namespace py = pybind11;

namespace {

constexpr int64_t kAbiVersion = 1;
constexpr int kMinimumBatchCopyVersion = 5010;

struct RuntimeCapability {
  int runtime_version = 0;
  int driver_version = 0;
  bool supported = false;
};

struct PreparedCopy {
  std::vector<void *> destination_pointers;
  std::vector<void *> source_pointers;
  std::vector<size_t> sizes;
  std::vector<at::Tensor> host_tensors;
  std::vector<at::Tensor> musa_tensors;
  int64_t musa_device_index = -1;
  bool destination_is_musa = false;
};

int64_t abi_version() { return kAbiVersion; }

RuntimeCapability probe_runtime_capability() {
  RuntimeCapability capability;
#if defined(REAL_MUSA_VERSION) && REAL_MUSA_VERSION >= 5010
  const bool runtime_ok =
      musaRuntimeGetVersion(&capability.runtime_version) == musaSuccess;
  const bool driver_ok =
      musaDriverGetVersion(&capability.driver_version) == musaSuccess;
  capability.supported =
      runtime_ok && driver_ok &&
      capability.runtime_version >= kMinimumBatchCopyVersion &&
      capability.driver_version >= kMinimumBatchCopyVersion;
#endif
  return capability;
}

const RuntimeCapability &runtime_capability() {
  static const RuntimeCapability capability = probe_runtime_capability();
  return capability;
}

int runtime_version() { return runtime_capability().runtime_version; }

int driver_version() { return runtime_capability().driver_version; }

bool batch_memcpy_supported() { return runtime_capability().supported; }

size_t checked_offset(int64_t value, const char *name, size_t index) {
  TORCH_CHECK(value >= 0, name, "[", index, "] must be non-negative");
  return static_cast<size_t>(value);
}

size_t tensor_nbytes(const at::Tensor &tensor) { return tensor.nbytes(); }

PreparedCopy prepare_copy(const std::vector<at::Tensor> &destinations,
                          const std::vector<at::Tensor> &sources,
                          const std::vector<int64_t> &destination_offsets,
                          const std::vector<int64_t> &source_offsets,
                          const std::vector<int64_t> &copy_sizes) {
  const size_t count = copy_sizes.size();
  TORCH_CHECK(destinations.size() == count && sources.size() == count,
              "destinations, sources, and sizes must have the same length");
  TORCH_CHECK(destination_offsets.size() == count &&
                  source_offsets.size() == count,
              "offset vectors must have the same length as sizes");

  PreparedCopy prepared;
  if (count == 0) {
    return prepared;
  }

  std::vector<std::pair<uintptr_t, uintptr_t>> destination_ranges;
  prepared.destination_pointers.reserve(count);
  prepared.source_pointers.reserve(count);
  prepared.sizes.reserve(count);
  destination_ranges.reserve(count);
  prepared.host_tensors.reserve(count);
  prepared.musa_tensors.reserve(count);

  bool direction_initialized = false;
  for (size_t index = 0; index < count; ++index) {
    const auto &destination = destinations[index];
    const auto &source = sources[index];
    TORCH_CHECK(destination.is_contiguous() && source.is_contiguous(),
                "batch-copy tensors must be contiguous");
    const bool destination_is_musa = destination.is_musa();
    const bool source_is_musa = source.is_musa();
    TORCH_CHECK(destination_is_musa != source_is_musa,
                "each entry must have exactly one MUSA endpoint");
    if (!direction_initialized) {
      prepared.destination_is_musa = destination_is_musa;
      direction_initialized = true;
    }
    TORCH_CHECK(destination_is_musa == prepared.destination_is_musa,
                "one batch cannot mix h2d and d2h copies");

    const auto &musa_tensor = destination_is_musa ? destination : source;
    const auto &host_tensor = destination_is_musa ? source : destination;
    TORCH_CHECK(host_tensor.device().is_cpu(),
                "the non-MUSA endpoint must be a CPU tensor");
    if (prepared.musa_device_index < 0) {
      prepared.musa_device_index = musa_tensor.device().index();
    }
    TORCH_CHECK(musa_tensor.device().index() == prepared.musa_device_index,
                "all entries must use the same MUSA device");

    const size_t destination_offset = checked_offset(
        destination_offsets[index], "destination_offsets", index);
    const size_t source_offset =
        checked_offset(source_offsets[index], "source_offsets", index);
    TORCH_CHECK(copy_sizes[index] > 0, "copy_sizes[", index,
                "] must be positive");
    const size_t copy_size = static_cast<size_t>(copy_sizes[index]);
    const size_t destination_capacity = tensor_nbytes(destination);
    const size_t source_capacity = tensor_nbytes(source);
    TORCH_CHECK(destination_offset <= destination_capacity &&
                    copy_size <= destination_capacity - destination_offset,
                "destination range at index ", index,
                " exceeds tensor capacity");
    TORCH_CHECK(source_offset <= source_capacity &&
                    copy_size <= source_capacity - source_offset,
                "source range at index ", index, " exceeds tensor capacity");

    auto *destination_pointer =
        static_cast<char *>(destination.data_ptr()) + destination_offset;
    auto *source_pointer =
        static_cast<char *>(source.data_ptr()) + source_offset;
    const auto destination_start =
        reinterpret_cast<uintptr_t>(destination_pointer);
    TORCH_CHECK(destination_start <=
                    std::numeric_limits<uintptr_t>::max() - copy_size,
                "destination address range overflowed");
    destination_ranges.emplace_back(destination_start,
                                    destination_start + copy_size);

    prepared.destination_pointers.push_back(destination_pointer);
    prepared.source_pointers.push_back(source_pointer);
    prepared.sizes.push_back(copy_size);
    prepared.host_tensors.push_back(host_tensor);
    prepared.musa_tensors.push_back(musa_tensor);
  }

  std::sort(destination_ranges.begin(), destination_ranges.end());
  for (size_t index = 1; index < destination_ranges.size(); ++index) {
    const auto previous_end = destination_ranges[index - 1].second;
    const auto current_start = destination_ranges[index].first;
    TORCH_CHECK(!(current_start < previous_end),
                "overlapping destination ranges are invalid in an unordered "
                "batch");
  }
  return prepared;
}

void record_allocator_usage(const PreparedCopy &prepared,
                            const c10::musa::MUSAStream &stream) {
  std::unordered_set<void *> recorded_host_allocations;
  for (const auto &host_tensor : prepared.host_tensors) {
    void *pointer = host_tensor.data_ptr();
    if (recorded_host_allocations.insert(pointer).second) {
      void *context = host_tensor.storage().data_ptr().get_context();
      at::getHostAllocator(at::kMUSA)->record_event(pointer, context, stream);
    }
  }

  std::unordered_set<void *> recorded_musa_allocations;
  for (const auto &musa_tensor : prepared.musa_tensors) {
    const auto &data_pointer = musa_tensor.storage().data_ptr();
    if (recorded_musa_allocations.insert(data_pointer.get()).second) {
      c10::musa::MUSACachingAllocator::recordStream(data_pointer, stream);
    }
  }
}

bool execute_batch_copy(const std::vector<at::Tensor> &destinations,
                        const std::vector<at::Tensor> &sources,
                        const std::vector<int64_t> &destination_offsets,
                        const std::vector<int64_t> &source_offsets,
                        const std::vector<int64_t> &copy_sizes,
                        bool prefer_overlap_with_compute) {
#if defined(REAL_MUSA_VERSION) && REAL_MUSA_VERSION >= 5010
  PreparedCopy prepared = prepare_copy(
      destinations, sources, destination_offsets, source_offsets, copy_sizes);
  if (prepared.sizes.empty()) {
    return true;
  }
  TORCH_CHECK(batch_memcpy_supported(),
              "MUSA runtime and driver do not support musaMemcpyBatchAsync");

  c10::DeviceGuard device_guard(prepared.destination_is_musa
                                    ? destinations[0].device()
                                    : sources[0].device());
  const auto stream =
      c10::musa::getCurrentMUSAStream(prepared.musa_device_index);
  musaMemcpyAttributes attributes{};
  attributes.srcAccessOrder = musaMemcpySrcAccessOrderStream;
  attributes.flags =
      prefer_overlap_with_compute ? musaMemcpyFlagPreferOverlapWithCompute : 0;
  size_t first_attribute_index = 0;
  size_t fail_index = prepared.sizes.size();
  const musaError_t status = musaMemcpyBatchAsync(
      prepared.destination_pointers.data(), prepared.source_pointers.data(),
      prepared.sizes.data(), prepared.sizes.size(), &attributes,
      &first_attribute_index, 1, &fail_index, stream.stream());
  TORCH_CHECK(status == musaSuccess,
              "musaMemcpyBatchAsync failed at batch index ", fail_index, ": ",
              musaGetErrorString(status));
  record_allocator_usage(prepared, stream);
  return true;
#else
  TORCH_CHECK(false, "musaMemcpyBatchAsync requires MUSA SDK 5.1 or newer");
  return false;
#endif
}

bool execute_copy_loop(const std::vector<at::Tensor> &destinations,
                       const std::vector<at::Tensor> &sources,
                       const std::vector<int64_t> &destination_offsets,
                       const std::vector<int64_t> &source_offsets,
                       const std::vector<int64_t> &copy_sizes) {
  PreparedCopy prepared = prepare_copy(
      destinations, sources, destination_offsets, source_offsets, copy_sizes);
  if (prepared.sizes.empty()) {
    return true;
  }

  c10::DeviceGuard device_guard(prepared.destination_is_musa
                                    ? destinations[0].device()
                                    : sources[0].device());
  const auto stream =
      c10::musa::getCurrentMUSAStream(prepared.musa_device_index);
  const musaMemcpyKind kind = prepared.destination_is_musa
                                  ? musaMemcpyHostToDevice
                                  : musaMemcpyDeviceToHost;
  for (size_t index = 0; index < prepared.sizes.size(); ++index) {
    const musaError_t status = musaMemcpyAsync(
        prepared.destination_pointers[index], prepared.source_pointers[index],
        prepared.sizes[index], kind, stream.stream());
    TORCH_CHECK(status == musaSuccess, "musaMemcpyAsync failed at batch index ",
                index, ": ", musaGetErrorString(status));
  }
  record_allocator_usage(prepared, stream);
  return true;
}

} // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("abi_version", &abi_version);
  module.def("runtime_version", &runtime_version);
  module.def("driver_version", &driver_version);
  module.def("batch_memcpy_supported", &batch_memcpy_supported);
  module.def("execute_batch_copy", &execute_batch_copy,
             py::call_guard<py::gil_scoped_release>());
  module.def("execute_copy_loop", &execute_copy_loop,
             py::call_guard<py::gil_scoped_release>());
}
