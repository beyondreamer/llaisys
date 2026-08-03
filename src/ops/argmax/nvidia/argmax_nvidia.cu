// ============================================================================
// src/ops/argmax/nvidia/argmax_nvidia.cu — argmax 算子的 CUDA 实现（A4）
// 语义：一维输入求最大值 + 下标（max_idx 为 i64，max_val 与输入同 dtype）
// 策略：两阶段归约
//   stage1：每个 block 归约自己那 256 个元素的局部最大值/下标，写入全局数组
//   stage2：单个 block 在全局数组上再归约一次，得到最终结果
// ============================================================================
#include "argmax_nvidia.hpp"

#include "../../nvidia_common.cuh"
#include "../../../utils.hpp"

#include <cuda_runtime.h>
#include <iostream>
#include <stdexcept>
#include <math_constants.h>

namespace llaisys::ops::nvidia {

// 检查 CUDA 调用是否成功（host 端，如 cudaMalloc 失败）
#define CHECK_CUDA(call)                                                                     \
    do {                                                                                     \
        cudaError_t err_ = (call);                                                           \
        if (err_ != cudaSuccess) {                                                           \
            std::cerr << "[ERROR] CUDA: " << cudaGetErrorString(err_) << " at " << __FILE__  \
                      << ":" << __LINE__ << std::endl;                                       \
            throw std::runtime_error("CUDA error");                                          \
        }                                                                                    \
    } while (0)

constexpr int BLOCK = 256;

// 阶段一：每个 block 处理连续 256 个元素，shared memory 树形归约。
template <typename T>
__global__ void argmax_stage1_kernel(const T *vals, size_t n, int *blk_idx, float *blk_val) {
    __shared__ float sval[BLOCK];
    __shared__ int sidx[BLOCK];

    size_t i = (size_t)blockIdx.x * BLOCK + threadIdx.x;
    if (i < n) {
        sval[threadIdx.x] = to_f32(vals[i]);
        sidx[threadIdx.x] = (int)i; // 输入规模 < 2^31，int 足够
    } else {
        sval[threadIdx.x] = -CUDART_INF_F;
        sidx[threadIdx.x] = -1;
    }
    __syncthreads();

    // 树形归约：严格大于才替换，保证并列时取第一个（与 CPU 行为一致）
    for (int s = BLOCK / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            if (sval[threadIdx.x + s] > sval[threadIdx.x]) {
                sval[threadIdx.x] = sval[threadIdx.x + s];
                sidx[threadIdx.x] = sidx[threadIdx.x + s];
            }
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        blk_val[blockIdx.x] = sval[0];
        blk_idx[blockIdx.x] = sidx[0];
    }
}

// 阶段二：一个 block 归约阶段一的输出（nblocks 一般很小，线程跨步扫描）。
template <typename T>
__global__ void argmax_stage2_kernel(const int *blk_idx, const float *blk_val, int nblocks,
                                     int64_t *out_idx, T *out_val) {
    __shared__ float sval[BLOCK];
    __shared__ int sidx[BLOCK];

    // 每个线程扫描多个 block 结果
    float my_val = -CUDART_INF_F;
    int my_idx = -1;
    for (int i = threadIdx.x; i < nblocks; i += BLOCK) {
        if (blk_val[i] > my_val) {
            my_val = blk_val[i];
            my_idx = blk_idx[i];
        }
    }
    sval[threadIdx.x] = my_val;
    sidx[threadIdx.x] = my_idx;
    __syncthreads();

    for (int s = BLOCK / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            if (sval[threadIdx.x + s] > sval[threadIdx.x]) {
                sval[threadIdx.x] = sval[threadIdx.x + s];
                sidx[threadIdx.x] = sidx[threadIdx.x + s];
            }
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        *out_idx = (int64_t)sidx[0];
        *out_val = from_f32<T>(sval[0]);
    }
}

template <typename T>
void launch_argmax(int64_t *max_idx, T *max_val, const T *vals, size_t n) {
    // 阶段一：n 个元素 -> nblocks 个局部结果（放设备临时缓冲）
    int nblocks = (int)((n + BLOCK - 1) / BLOCK);
    int *blk_idx = nullptr;
    float *blk_val = nullptr;
    CHECK_CUDA(cudaMalloc(&blk_idx, nblocks * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&blk_val, nblocks * sizeof(float)));

    argmax_stage1_kernel<T><<<nblocks, BLOCK>>>(vals, n, blk_idx, blk_val);
    argmax_stage2_kernel<T><<<1, BLOCK>>>(blk_idx, blk_val, nblocks, max_idx, max_val);
    CHECK_CUDA(cudaGetLastError()); // 捕获内核启动错误

    CHECK_CUDA(cudaFree(blk_idx));
    CHECK_CUDA(cudaFree(blk_val));
}


void argmax(std::byte *max_idx, std::byte *max_val, const std::byte *vals,
            llaisysDataType_t type, size_t n) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return launch_argmax(reinterpret_cast<int64_t *>(max_idx),
                             reinterpret_cast<float *>(max_val),
                             reinterpret_cast<const float *>(vals), n);
    case LLAISYS_DTYPE_F16:
        return launch_argmax(reinterpret_cast<int64_t *>(max_idx),
                             reinterpret_cast<__half *>(max_val),
                             reinterpret_cast<const __half *>(vals), n);
    case LLAISYS_DTYPE_BF16:
        return launch_argmax(reinterpret_cast<int64_t *>(max_idx),
                             reinterpret_cast<__nv_bfloat16 *>(max_val),
                             reinterpret_cast<const __nv_bfloat16 *>(vals), n);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::nvidia
