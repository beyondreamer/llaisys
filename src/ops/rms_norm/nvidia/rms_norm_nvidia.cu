// ============================================================================
// src/ops/rms_norm/nvidia/rms_norm_nvidia.cu — RMSNorm 算子的 CUDA 实现（A4）
// 公式：out[i][j] = x[i][j] / sqrt(mean_j(x[i][j]^2) + eps) * w[j]
// ============================================================================
#include "rms_norm_nvidia.hpp"

#include "../../nvidia_common.cuh"
#include "../../../utils.hpp"

#include <cuda_runtime.h>

namespace llaisys::ops::nvidia {

// 每行一个线程：先算行内平方均值（float），再逐元素归一化并乘权重。
// 行宽 d 可能很大（4096），串行循环即可（正确性优先，优化留给后续）。
template <typename T>
__global__ void rms_norm_kernel(T *out, const T *x, const T *w, size_t rows, size_t d,
                                float eps) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rows) return;

    float sum_sq = 0.0f;
    for (size_t j = 0; j < d; ++j) {
        float v = to_f32(x[i * d + j]);
        sum_sq += v * v;
    }
    float rms = rsqrtf(sum_sq / (float)d + eps);
    for (size_t j = 0; j < d; ++j) {
        out[i * d + j] = from_f32<T>(to_f32(x[i * d + j]) * rms * to_f32(w[j]));
    }
}

template <typename T>
void launch_rms_norm(T *out, const T *x, const T *w, size_t rows, size_t d, float eps) {
    rms_norm_kernel<T><<<grid_1d(rows, 256), 256>>>(out, x, w, rows, d, eps);
}


void rms_norm(std::byte *out, const std::byte *in, const std::byte *weight,
              llaisysDataType_t type, size_t rows, size_t d, float eps) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return launch_rms_norm(reinterpret_cast<float *>(out), reinterpret_cast<const float *>(in),
                               reinterpret_cast<const float *>(weight), rows, d, eps);
    case LLAISYS_DTYPE_F16:
        return launch_rms_norm(reinterpret_cast<__half *>(out), reinterpret_cast<const __half *>(in),
                               reinterpret_cast<const __half *>(weight), rows, d, eps);
    case LLAISYS_DTYPE_BF16:
        return launch_rms_norm(reinterpret_cast<__nv_bfloat16 *>(out),
                               reinterpret_cast<const __nv_bfloat16 *>(in),
                               reinterpret_cast<const __nv_bfloat16 *>(weight), rows, d, eps);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::nvidia
