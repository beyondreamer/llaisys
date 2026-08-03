// ============================================================================
// src/ops/linear/nvidia/linear_nvidia.cu — 全连接算子的 CUDA 实现（A4）
// 公式：out[i][j] = Σ_p in[i][p] * weight[j][p] + bias[j]
//       （weight 未转置，形状 [n, k]，按行取 —— 与 CPU 层同一约定）
// ============================================================================
#include "linear_nvidia.hpp"

#include "../../nvidia_common.cuh"
#include "../../../utils.hpp"

#include <cuda_runtime.h>

namespace llaisys::ops::nvidia {

// 朴素 GEMM 内核：每个线程算 out 的一个元素（i 行 j 列），内层 p 循环累加。
// f16/bf16 用 float 累加后一次舍入，与 CPU 实现数值一致。
// 性能优化（共享内存分块/cuBLAS）留给后续，先保证正确性。
template <typename T>
__global__ void linear_kernel(T *out, const T *x, const T *w, const T *bias,
                              size_t m, size_t n, size_t k) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= m * n) return;
    size_t i = idx / n;
    size_t j = idx % n;

    float acc = 0.0f;
    for (size_t p = 0; p < k; ++p) {
        acc += to_f32(x[i * k + p]) * to_f32(w[j * k + p]);
    }
    if (bias != nullptr) {
        acc += to_f32(bias[j]);
    }
    out[idx] = from_f32<T>(acc);
}

template <typename T>
void launch_linear(T *out, const T *x, const T *w, const T *bias, size_t m, size_t n, size_t k) {
    linear_kernel<T><<<grid_1d(m * n, 256), 256>>>(out, x, w, bias, m, n, k);
}


void linear(std::byte *out, const std::byte *in, const std::byte *weight, const std::byte *bias,
            llaisysDataType_t type, size_t m, size_t n, size_t k) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return launch_linear(reinterpret_cast<float *>(out), reinterpret_cast<const float *>(in),
                             reinterpret_cast<const float *>(weight),
                             reinterpret_cast<const float *>(bias), m, n, k);
    case LLAISYS_DTYPE_F16:
        return launch_linear(reinterpret_cast<__half *>(out), reinterpret_cast<const __half *>(in),
                             reinterpret_cast<const __half *>(weight),
                             reinterpret_cast<const __half *>(bias), m, n, k);
    case LLAISYS_DTYPE_BF16:
        return launch_linear(reinterpret_cast<__nv_bfloat16 *>(out),
                             reinterpret_cast<const __nv_bfloat16 *>(in),
                             reinterpret_cast<const __nv_bfloat16 *>(weight),
                             reinterpret_cast<const __nv_bfloat16 *>(bias), m, n, k);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::nvidia
