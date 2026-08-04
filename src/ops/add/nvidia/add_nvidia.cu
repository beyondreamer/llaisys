// ============================================================================
// src/ops/add/nvidia/add_nvidia.cu — add 算子的 CUDA 实现（优化版）
// 公式：c[i] = a[i] + b[i]（逐元素，float 中间精度）
// ============================================================================
#include "add_nvidia.hpp"

#include "../../nvidia_common.cuh"
#include "../../../utils.hpp"

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

namespace llaisys::ops::nvidia {

// 【优化点（相对原版）】
//   原版：每线程 1 个标量元素，bf16/f16 标量读写。
//   新版：f32 保持标量；bf16/f16 用 half2 向量化，一次处理 2 元素，
//         用 __hadd2 做 SIMD 加法，吞吐翻倍；奇数尾巴用标量 kernel 处理。
//         add 是典型 memory-bound 算子，向量化主要省访存指令数与解码开销。

constexpr int VEC_BLOCK = 256;

// ---- f32 标量路径（f32 没有更宽的 packed 类型）----
__global__ void add_f32_kernel(float *c, const float *a, const float *b, size_t n) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}

// ---- bf16 向量化路径：一次处理 2 元素 ----
__global__ void add_bf16_vec_kernel(__nv_bfloat16 *c, const __nv_bfloat16 *a,
                                    const __nv_bfloat16 *b, size_t npairs) {
    size_t k = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (k < npairs) {
        size_t i = k * 2;
        __nv_bfloat162 a2 = *reinterpret_cast<const __nv_bfloat162 *>(a + i);
        __nv_bfloat162 b2 = *reinterpret_cast<const __nv_bfloat162 *>(b + i);
        *reinterpret_cast<__nv_bfloat162 *>(c + i) = __hadd2(a2, b2);
    }
}

// ---- f16 向量化路径：一次处理 2 元素 ----
__global__ void add_f16_vec_kernel(__half *c, const __half *a, const __half *b, size_t npairs) {
    size_t k = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (k < npairs) {
        size_t i = k * 2;
        __half2 a2 = *reinterpret_cast<const __half2 *>(a + i);
        __half2 b2 = *reinterpret_cast<const __half2 *>(b + i);
        *reinterpret_cast<__half2 *>(c + i) = __hadd2(a2, b2);
    }
}

// ---- 标量尾巴：处理奇数长度最后一个元素（转 f32 相加再转回，保持精度对齐）----
__global__ void add_bf16_tail_kernel(__nv_bfloat16 *c, const __nv_bfloat16 *a,
                                     const __nv_bfloat16 *b) {
    *c = from_f32<__nv_bfloat16>(to_f32(*a) + to_f32(*b));
}
__global__ void add_f16_tail_kernel(__half *c, const __half *a, const __half *b) {
    *c = from_f32<__half>(to_f32(*a) + to_f32(*b));
}

void add(std::byte *out, const std::byte *a, const std::byte *b,
         llaisysDataType_t type, size_t numel) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        add_f32_kernel<<<grid_1d(numel, VEC_BLOCK), VEC_BLOCK>>>(
            reinterpret_cast<float *>(out), reinterpret_cast<const float *>(a),
            reinterpret_cast<const float *>(b), numel);
        return;
    case LLAISYS_DTYPE_F16: {
        size_t npairs = numel / 2;
        add_f16_vec_kernel<<<grid_1d(npairs, VEC_BLOCK), VEC_BLOCK>>>(
            reinterpret_cast<__half *>(out), reinterpret_cast<const __half *>(a),
            reinterpret_cast<const __half *>(b), npairs);
        if (numel % 2 == 1) {
            size_t last = numel - 1;
            add_f16_tail_kernel<<<1, 1>>>(reinterpret_cast<__half *>(out) + last,
                                          reinterpret_cast<const __half *>(a) + last,
                                          reinterpret_cast<const __half *>(b) + last);
        }
        return;
    }
    case LLAISYS_DTYPE_BF16: {
        size_t npairs = numel / 2;
        add_bf16_vec_kernel<<<grid_1d(npairs, VEC_BLOCK), VEC_BLOCK>>>(
            reinterpret_cast<__nv_bfloat16 *>(out), reinterpret_cast<const __nv_bfloat16 *>(a),
            reinterpret_cast<const __nv_bfloat16 *>(b), npairs);
        if (numel % 2 == 1) {
            size_t last = numel - 1;
            add_bf16_tail_kernel<<<1, 1>>>(
                reinterpret_cast<__nv_bfloat16 *>(out) + last,
                reinterpret_cast<const __nv_bfloat16 *>(a) + last,
                reinterpret_cast<const __nv_bfloat16 *>(b) + last);
        }
        return;
    }
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::nvidia
