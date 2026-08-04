// ============================================================================
// src/ops/swiglu/nvidia/swiglu_nvidia.cu — SwiGLU 算子的 CUDA 实现（优化版）
// 公式：out = up * SiLU(gate) = up * gate / (1 + e^(-gate))
// ============================================================================
#include "swiglu_nvidia.hpp"

#include "../../nvidia_common.cuh"
#include "../../../utils.hpp"

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

namespace llaisys::ops::nvidia {

// 【优化点（相对原版）】
//   原版：block=256，每线程 1 个标量元素，bf16/f16 标量读写。
//   新版：调大 block 到 512 提高占用率；f32/bf16/f16 统一走标量路径，
//         用 __expf（快速 transcendental，比 expf 快）。
//
// 说明：曾尝试用 half2 向量化（__hadd2/h2exp）做 2 元素/线程，但 h2exp 在
//       CUDA 12.8 头文件中对 __half2 与 __nv_bfloat162 的重载存在歧义，跨平台
//       （沐曦 cu-bridge）更不可靠。SwiGLU 在 MLP 中虽然 di=8960 很大，但它是
//       逐元素 memory-bound 算子，标量 + 大 block 的带宽利用率已接近峰值，
//       向量化的边际收益不足以抵消可移植性风险，故回退标量路径。
//       主要瓶颈在 GEMM（linear）而非激活，这里求稳。

constexpr int VEC_BLOCK = 512;

// 统一标量 kernel：每线程 1 元素，f32 中间精度，__expf 快速近似
template <typename T>
__global__ void swiglu_scalar_kernel(T *out, const T *gate, const T *up, size_t n) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float g = to_f32(gate[i]);
        float u = to_f32(up[i]);
        // SiLU(g) = g / (1 + e^(-g))，用 __expf 快速近似（精度足够，与 CPU 的 expf 在
        // bf16 输出精度下结果一致）
        out[i] = from_f32<T>(u * (g / (1.0f + __expf(-g))));
    }
}

void swiglu(std::byte *out, const std::byte *gate, const std::byte *up,
            llaisysDataType_t type, size_t numel) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        swiglu_scalar_kernel<float><<<grid_1d(numel, VEC_BLOCK), VEC_BLOCK>>>(
            reinterpret_cast<float *>(out), reinterpret_cast<const float *>(gate),
            reinterpret_cast<const float *>(up), numel);
        return;
    case LLAISYS_DTYPE_F16:
        swiglu_scalar_kernel<__half><<<grid_1d(numel, VEC_BLOCK), VEC_BLOCK>>>(
            reinterpret_cast<__half *>(out), reinterpret_cast<const __half *>(gate),
            reinterpret_cast<const __half *>(up), numel);
        return;
    case LLAISYS_DTYPE_BF16:
        swiglu_scalar_kernel<__nv_bfloat16><<<grid_1d(numel, VEC_BLOCK), VEC_BLOCK>>>(
            reinterpret_cast<__nv_bfloat16 *>(out), reinterpret_cast<const __nv_bfloat16 *>(gate),
            reinterpret_cast<const __nv_bfloat16 *>(up), numel);
        return;
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::nvidia
