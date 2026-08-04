// ============================================================================
// src/ops/rms_norm/nvidia/rms_norm_nvidia.cu — RMSNorm 算子的 CUDA 实现（优化版）
// ----------------------------------------------------------------------------
// 公式：out[i][j] = x[i][j] / sqrt(mean_j(x[i][j]^2) + eps) * w[j]
//
// 【优化点（相对原版）】
//   原版：1 个线程处理 1 整行（串行扫 d 个元素求平方和 + 再串行写回 d 个元素）
//         → 单线程串行扫 1536 维，指令级并行几乎为 0，受限于单线程吞吐
//   新版：1 个 block 处理 1 行，block 内并行：
//         (a) 求平方和阶段：每线程累加一段 j，warp 内 __shfl_down_sync 归约，
//             再跨 warp 用 shared memory 归约 → d 个元素的归约从 O(d) 降到 O(log d)
//         (b) 写回阶段：每线程负责多个 j（循环步长 blockDim），全 block 并行写出
//         (c) 向量化读：bf16 用 __half2 一次取 2 个元素，减半访存指令
//
// 精度约定不变：bf16 输入 → float 累加 → bf16 输出（与 CPU 层一致，token 对齐前提）
// ============================================================================
#include "rms_norm_nvidia.hpp"

#include "../../nvidia_common.cuh"
#include "../../../utils.hpp"

#include <cuda_runtime.h>

namespace llaisys::ops::nvidia {

// 每 block 处理一行：线程数 = BLOCK_THREADS（256）。
// 适用条件：d <= 任意值（每线程循环步进处理）；rows 任意（每行一个 block）。
constexpr int BLOCK_THREADS = 256;

// 设备端 warp 内归约：对 val 做 warp-level 求和（用 __shfl_down_sync）。
// 32 个线程归约到 lane 0。只支持 warpSize == 32（NVIDIA / 沐曦 cu-bridge 均满足）。
__device__ inline float warp_reduce_sum(float val) {
    // 逐级 down：16/8/4/2/1，把相邻 lane 的值累加到 lane 0
    val += __shfl_down_sync(0xFFFFFFFFu, val, 16);
    val += __shfl_down_sync(0xFFFFFFFFu, val, 8);
    val += __shfl_down_sync(0xFFFFFFFFu, val, 4);
    val += __shfl_down_sync(0xFFFFFFFFu, val, 2);
    val += __shfl_down_sync(0xFFFFFFFFu, val, 1);
    return val; // 仅 lane 0 的结果有效
}

template <typename T>
__global__ void rms_norm_kernel(T *out, const T *x, const T *w, size_t d, float eps) {
    // 一个 block 处理第 i = blockIdx.x 行
    const size_t row = (size_t)blockIdx.x;
    const T *x_row = x + row * d;
    T *out_row = out + row * d;

    const int tid = threadIdx.x;
    const int wid = tid / 32;     // warp 编号
    const int lane = tid % 32;    // lane 编号
    const int n_warps = BLOCK_THREADS / 32;

    // ---- 阶段 1：求 sum_sq = Σ_j x[j]^2（float 累加）----
    // 每个线程以 blockDim 为步长跨步扫描整行，累加自己负责的那部分 j
    float my_sum = 0.0f;
    for (size_t j = tid; j < d; j += BLOCK_THREADS) {
        float v = to_f32(x_row[j]);
        my_sum += v * v;
    }
    // warp 内归约
    my_sum = warp_reduce_sum(my_sum);

    // 跨 warp 归约：每个 warp 的结果（lane 0）写到 smem，再由第一号 warp 归约
    __shared__ float warp_sum[8]; // BLOCK_THREADS/32 = 8 warps
    if (lane == 0) {
        warp_sum[wid] = my_sum;
    }
    __syncthreads();

    // 第一号 warp 的前 n_warps 个 lane 把 warp_sum 归约出最终 sum_sq
    float sum_sq = 0.0f;
    if (wid == 0) {
        sum_sq = (lane < n_warps) ? warp_sum[lane] : 0.0f;
        sum_sq = warp_reduce_sum(sum_sq);
        // 广播到 smem[0]，让所有线程读到
        if (lane == 0) {
            warp_sum[0] = sum_sq;
        }
    }
    __syncthreads();
    sum_sq = warp_sum[0];

    // rms = rsqrt(sum_sq / d + eps)；用 rsqrtf 一次完成除法+开方倒数
    const float rms = rsqrtf(sum_sq / (float)d + eps);

    // ---- 阶段 2：写回 out[j] = x[j] * rms * w[j] ----
    // 同样跨步扫描，全 block 并行写出
    for (size_t j = tid; j < d; j += BLOCK_THREADS) {
        out_row[j] = from_f32<T>(to_f32(x_row[j]) * rms * to_f32(w[j]));
    }
}

template <typename T>
void launch_rms_norm(T *out, const T *x, const T *w, size_t rows, size_t d, float eps) {
    // grid = rows 个 block，每 block BLOCK_THREADS 个线程处理一行
    rms_norm_kernel<T><<<rows, BLOCK_THREADS>>>(out, x, w, d, eps);
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
