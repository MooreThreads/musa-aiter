#include <stdint.h>
#include "../common/musa_compat.h"

/// Single-block parallel reduction sum (Phase P2 smoke / teaching kernel).
/// Correct for arbitrary *n*, but deliberately uses one CTA — not a production perf path.

static constexpr int kBlock = 256;

extern "C" __global__ void musa_aiter_reduce_sum_kernel(const float* x, float* out, int64_t n) {
  __shared__ float partial[kBlock];
  float thread_sum = 0.f;
  for (int64_t idx = static_cast<int64_t>(threadIdx.x); idx < n; idx += kBlock) {
    thread_sum += x[idx];
  }
  partial[threadIdx.x] = thread_sum;
  __syncthreads();
  for (int stride = kBlock / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      partial[threadIdx.x] += partial[threadIdx.x + stride];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    *out = partial[0];
  }
}

extern "C" void musa_aiter_reduce_sum_float(const float* x, float* out, int64_t n, gpuStream_t stream) {
#ifdef __MUSACC__
  musa_aiter_reduce_sum_kernel<<<1, kBlock, 0, stream>>>(x, out, n);
#endif
}

extern "C" int musa_aiter_smoke_reduce_abi_version() {
  return 1;
}
