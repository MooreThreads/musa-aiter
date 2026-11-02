#include <math.h>
#include <stdint.h>
#include "../common/musa_compat.h"

/// RMSNorm over the last dimension (float32): one CTA per row.
/// Matches the fused-norm placement used ahead of attention in MLA/decoder stacks.

static constexpr int kBlock = 256;

extern "C" __global__ void musa_aiter_rmsnorm_kernel(
    const float* __restrict__ x,
    const float* __restrict__ weight,
    float* __restrict__ out,
    int outer,
    int dim,
    float eps) {
  const int row = blockIdx.x;
  if (row >= outer)
    return;
  const int64_t offset = static_cast<int64_t>(row) * static_cast<int64_t>(dim);
  __shared__ float partial[kBlock];

  float local_sum_sq = 0.f;
  for (int j = threadIdx.x; j < dim; j += kBlock) {
    float v = x[offset + j];
    local_sum_sq += v * v;
  }
  partial[threadIdx.x] = local_sum_sq;
  __syncthreads();
  for (int stride = kBlock / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      partial[threadIdx.x] += partial[threadIdx.x + stride];
    }
    __syncthreads();
  }
  const float mean_sq = partial[0] / static_cast<float>(dim);
  const float rsigma = rsqrtf(mean_sq + eps);

  for (int j = threadIdx.x; j < dim; j += kBlock) {
    const int ij = offset + j;
    out[ij] = x[ij] * rsigma * weight[j];
  }
}

extern "C" void musa_aiter_rmsnorm_float(
    const float* x,
    const float* weight,
    float* out,
    int outer,
    int dim,
    float eps,
    gpuStream_t stream) {
#ifdef __MUSACC__
  if (outer <= 0 || dim <= 0)
    return;
  musa_aiter_rmsnorm_kernel<<<outer, kBlock, 0, stream>>>(x, weight, out, outer, dim, eps);
#endif
}

extern "C" int musa_aiter_smoke_rmsnorm_abi_version() {
  return 1;
}
