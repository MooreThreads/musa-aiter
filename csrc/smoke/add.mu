#include <stdint.h>
#include "../common/musa_compat.h"

extern "C" __global__ void musa_aiter_add_kernel(
    const float* a,
    const float* b,
    float* out,
    int64_t n) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < n) {
    out[idx] = a[idx] + b[idx];
  }
}

extern "C" void musa_aiter_add_float(
    const float* a,
    const float* b,
    float* out,
    int64_t n,
    gpuStream_t stream) {
#ifdef __MUSACC__
  int threads = 256;
  int blocks = static_cast<int>((n + threads - 1) / threads);
  musa_aiter_add_kernel<<<blocks, threads, 0, stream>>>(a, b, out, n);
#endif
}

extern "C" int musa_aiter_smoke_abi_version() {
  return 1;
}
