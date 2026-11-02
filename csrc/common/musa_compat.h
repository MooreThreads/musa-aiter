#pragma once

#ifdef __MUSACC__
#include <musa_runtime.h>
using gpuStream_t = musaStream_t;
using gpuError_t = musaError_t;
#define gpuGetLastError musaGetLastError
#define gpuDeviceSynchronize musaDeviceSynchronize
#else
// Phase-0 generic declarations keep host-only parsing tools happy. Real MUSA
// builds should define __MUSACC__ through mcc.
using gpuStream_t = void*;
using gpuError_t = int;
#endif
