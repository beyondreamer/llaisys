// ============================================================================
// src/ops/swiglu/nvidia/swiglu_nvidia.cu — SwiGLU 算子的 CUDA 实现（A4）
// 公式：out = up * SiLU(gate) = up * gate / (1 + e^(-gate))
// ============================================================================
#include "swiglu_nvidia.hpp"

#include "../../nvidia_common.cuh"
#include "../../../utils.hpp"

#include <cuda_runtime.h>

namespace llaisys::ops::nvidia {

template <typename T>
__global__ void swiglu_kernel(T *out, const T *gate, const T *up, size_t n) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float g = to_f32(gate[i]);
        float u = to_f32(up[i]);
        out[i] = from_f32<T>(u * (g / (1.0f + __expf(-g))));
    }
}

template <typename T>
void launch_swiglu(T *out, const T *gate, const T *up, size_t n) {
    swiglu_kernel<T><<<grid_1d(n, 256), 256>>>(out, gate, up, n);
}


void swiglu(std::byte *out, const std::byte *gate, const std::byte *up,
            llaisysDataType_t type, size_t numel) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return launch_swiglu(reinterpret_cast<float *>(out), reinterpret_cast<const float *>(gate),
                             reinterpret_cast<const float *>(up), numel);
    case LLAISYS_DTYPE_F16:
        return launch_swiglu(reinterpret_cast<__half *>(out), reinterpret_cast<const __half *>(gate),
                             reinterpret_cast<const __half *>(up), numel);
    case LLAISYS_DTYPE_BF16:
        return launch_swiglu(reinterpret_cast<__nv_bfloat16 *>(out),
                             reinterpret_cast<const __nv_bfloat16 *>(gate),
                             reinterpret_cast<const __nv_bfloat16 *>(up), numel);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::nvidia
