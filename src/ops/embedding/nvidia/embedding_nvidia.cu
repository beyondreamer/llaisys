// ============================================================================
// src/ops/embedding/nvidia/embedding_nvidia.cu — embedding 算子的 CUDA 实现（A4）
// 公式：out[i, j] = weight[index[i], j]（index 是 i64）
// ============================================================================
#include "embedding_nvidia.hpp"

#include "../../nvidia_common.cuh"
#include "../../../utils.hpp"

#include <cuda_runtime.h>

namespace llaisys::ops::nvidia {

// 查表内核：每个线程负责 out 的一个元素（i 行 j 列），
// 从 weight 的第 index[i] 行取数据（行主序，直接按字节赋值即可，无需类型转换）。
template <typename T>
__global__ void embedding_kernel(T *out, const int64_t *index, const T *weight,
                                 size_t nidx, size_t d) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < nidx * d) {
        size_t row = i / d;
        size_t col = i % d;
        out[i] = weight[(size_t)index[row] * d + col];
    }
}

template <typename T>
void launch_embedding(T *out, const int64_t *index, const T *weight, size_t nidx, size_t d) {
    embedding_kernel<T><<<grid_1d(nidx * d, 256), 256>>>(out, index, weight, nidx, d);
}


void embedding(std::byte *out, const std::byte *index, const std::byte *weight,
               llaisysDataType_t type, size_t nidx, size_t d) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return launch_embedding(reinterpret_cast<float *>(out),
                                reinterpret_cast<const int64_t *>(index),
                                reinterpret_cast<const float *>(weight), nidx, d);
    case LLAISYS_DTYPE_F16:
        return launch_embedding(reinterpret_cast<__half *>(out),
                                reinterpret_cast<const int64_t *>(index),
                                reinterpret_cast<const __half *>(weight), nidx, d);
    case LLAISYS_DTYPE_BF16:
        return launch_embedding(reinterpret_cast<__nv_bfloat16 *>(out),
                                reinterpret_cast<const int64_t *>(index),
                                reinterpret_cast<const __nv_bfloat16 *>(weight), nidx, d);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::nvidia
