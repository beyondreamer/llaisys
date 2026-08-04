// ============================================================================
// src/ops/argmax/nvidia/argmax_nvidia.cu — argmax 算子的 CUDA 实现（优化版）
// 语义：一维输入求最大值 + 下标（max_idx 为 i64，max_val 与输入同 dtype）
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

// 块大小：每 block 256 线程。vocab=151936，nblocks ≈ 594。
// 阶段一并行归约仍需一个临时 buffer 存「每 block 局部 max」。
constexpr int BLOCK = 256;

// 【优化点（相对原版）】
//   原版：launch_argmax 里每次调用都 cudaMalloc + cudaFree 两个临时 buffer。
//         在 test_infer 里 argmax 每 step 调一次（128 步 → 128 次 cudaMalloc/Free），
//         cudaMalloc 会触发设备同步 + 驱动路径开销，是隐形大头。
//   新版：用一个全局的「持久化临时 buffer」，首次调用 cudaMalloc，之后复用。
//         通过 cudaDeviceSynchronize 不必要——buffer 大小只增不减（用静态变量记录当前容量）。
//         避免每次推理步都 malloc/free。
//
// 并行策略不变（仍两阶段）：vocab=151936 用单 block 全归约反而不如「多 block 分块 + 二次归约」快。

// 阶段一：每个 block 处理连续 BLOCK 个元素，shared memory 树形归约。
// 与原版相同（这部分已经是 warp-cooperative 的合理实现）。
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
// 与原版相同。
template <typename T>
__global__ void argmax_stage2_kernel(const int *blk_idx, const float *blk_val, int nblocks,
                                     int64_t *out_idx, T *out_val) {
    __shared__ float sval[BLOCK];
    __shared__ int sidx[BLOCK];

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

// 持久化临时 buffer（进程级单例）：容量只增不减，避免反复 malloc/free。
// 用静态局部变量 + lambdas 保证线程安全由 CUDA runtime 隐式保证（launch 是异步的，
// 但同一 stream 内先后两次 launch 顺序执行，buffer 复用安全）。
namespace {
struct ArgmaxScratch {
    int *blk_idx = nullptr;
    float *blk_val = nullptr;
    size_t capacity = 0; // 当前 buffer 容量（元素个数）
};
} // namespace

template <typename T>
void launch_argmax(int64_t *max_idx, T *max_val, const T *vals, size_t n) {
    int nblocks = (int)((n + BLOCK - 1) / BLOCK);

    // 取/扩容持久化 buffer（只在容量不够时才 realloc）
    static ArgmaxScratch scratch; // C++11 局部静态，线程安全初始化
    if (scratch.capacity < (size_t)nblocks) {
        if (scratch.blk_idx != nullptr) {
            cudaFree(scratch.blk_idx);
            cudaFree(scratch.blk_val);
        }
        CHECK_CUDA(cudaMalloc(&scratch.blk_idx, nblocks * sizeof(int)));
        CHECK_CUDA(cudaMalloc(&scratch.blk_val, nblocks * sizeof(float)));
        scratch.capacity = nblocks;
    }

    argmax_stage1_kernel<T><<<nblocks, BLOCK>>>(vals, n, scratch.blk_idx, scratch.blk_val);
    argmax_stage2_kernel<T><<<1, BLOCK>>>(scratch.blk_idx, scratch.blk_val, nblocks, max_idx,
                                          max_val);
    CHECK_CUDA(cudaGetLastError()); // 捕获内核启动错误
    // 注意：不 free，留给下一次推理复用
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
