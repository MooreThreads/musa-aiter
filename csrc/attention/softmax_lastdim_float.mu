#include <math.h>
#include <stdint.h>
#include "../common/musa_compat.h"

/// Last-dimension softmax (float32) — one CTA per row.
/// Matches the stable attention pattern used in fused attention (Flash-style): row max,
/// subtract, exp, normalize. Teaching / smoke kernel; not a fused attention tile.

static constexpr int kBlock = 256;

extern "C" __global__ void musa_aiter_softmax_lastdim_kernel(
    const float* __restrict__ in,
    float* __restrict__ out,
    int outer,
    int dim) {
  const int row = blockIdx.x;
  if (row >= outer)
    return;
  const int64_t offset = static_cast<int64_t>(row) * static_cast<int64_t>(dim);
  __shared__ float partial[kBlock];

  float local_max = -INFINITY;
  for (int j = threadIdx.x; j < dim; j += kBlock) {
    float v = in[offset + j];
    local_max = fmaxf(local_max, v);
  }
  partial[threadIdx.x] = local_max;
  __syncthreads();
  for (int stride = kBlock / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      partial[threadIdx.x] = fmaxf(partial[threadIdx.x], partial[threadIdx.x + stride]);
    }
    __syncthreads();
  }
  const float row_max = partial[0];
  __syncthreads();

  float local_sum = 0.f;
  for (int j = threadIdx.x; j < dim; j += kBlock) {
    local_sum += expf(in[offset + j] - row_max);
  }
  partial[threadIdx.x] = local_sum;
  __syncthreads();
  for (int stride = kBlock / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      partial[threadIdx.x] += partial[threadIdx.x + stride];
    }
    __syncthreads();
  }
  const float inv_sum = partial[0] > 0.f ? (1.f / partial[0]) : 0.f;
  __syncthreads();

  for (int j = threadIdx.x; j < dim; j += kBlock) {
    out[offset + j] = expf(in[offset + j] - row_max) * inv_sum;
  }
}

extern "C" void musa_aiter_softmax_lastdim_float(
    const float* in,
    float* out,
    int outer,
    int dim,
    gpuStream_t stream) {
#ifdef __MUSACC__
  if (outer <= 0 || dim <= 0)
    return;
  musa_aiter_softmax_lastdim_kernel<<<outer, kBlock, 0, stream>>>(in, out, outer, dim);
#endif
}

extern "C" int musa_aiter_smoke_softmax_abi_version() {
  return 1;
}
