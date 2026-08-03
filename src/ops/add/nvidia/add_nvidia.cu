// ============================================================================
// src/ops/add/nvidia/add_nvidia.cu — add 算子的 CUDA 实现（A4）
// 公式：c[i] = a[i] + b[i]（逐元素，float 中间精度）
// ============================================================================
#include "add_nvidia.hpp"

#include "../../nvidia_common.cuh"
#include "../../../utils.hpp"

#include <cuda_runtime.h>

namespace llaisys::ops::nvidia {

// 逐元素加法内核：每个线程处理一个元素，f16/bf16 转 float 相加再转回。
template <typename T>
__global__ void add_kernel(T *c, const T *a, const T *b, size_t n) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        c[i] = from_f32<T>(to_f32(a[i]) + to_f32(b[i]));
    }
}

// 按 dtype 分发到模板实例并启动内核（block = 256 线程）。
template <typename T>
void launch_add(T *c, const T *a, const T *b, size_t n) {
    add_kernel<T><<<grid_1d(n, 256), 256>>>(c, a, b, n);
}


void add(std::byte *out, const std::byte *a, const std::byte *b,
         llaisysDataType_t type, size_t numel) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return launch_add(reinterpret_cast<float *>(out), reinterpret_cast<const float *>(a),
                          reinterpret_cast<const float *>(b), numel);
    case LLAISYS_DTYPE_F16:
        return launch_add(reinterpret_cast<__half *>(out), reinterpret_cast<const __half *>(a),
                          reinterpret_cast<const __half *>(b), numel);
    case LLAISYS_DTYPE_BF16:
        return launch_add(reinterpret_cast<__nv_bfloat16 *>(out),
                          reinterpret_cast<const __nv_bfloat16 *>(a),
                          reinterpret_cast<const __nv_bfloat16 *>(b), numel);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::nvidia
