// ============================================================================
// src/ops/nvidia_common.cuh — 所有 CUDA 算子共享的设备端工具（A4）
// ----------------------------------------------------------------------------
// 提供：
//   to_f32()   元素值 -> float（f16/bf16 转精度，供中间计算使用）
//   from_f32() float -> 元素值（f16/bf16 舍入回目标精度）
//   grid_1d()  计算一维 grid 大小
// 与 CPU 层 utils::cast 的语义一致：中间计算一律 float，最后再转回目标类型，
// 保证 CPU/CUDA 数值行为一致（A3 的 token 对齐在 A4 也要保持）。
//
// 注意：bf16 相关设备函数需要 sm_80+，构建时用 -arch=sm_80（xmake/nvidia.lua）。
// ============================================================================
#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cstddef>
#include <cstdint>

namespace llaisys::ops::nvidia {

// 元素 -> float 转换（设备端）
__device__ inline float to_f32(float v) { return v; }
__device__ inline float to_f32(__half v) { return __half2float(v); }
__device__ inline float to_f32(__nv_bfloat16 v) { return __bfloat162float(v); }

// float -> 元素转换（设备端）
template <typename T>
__device__ inline T from_f32(float v);

template <>
__device__ inline float from_f32<float>(float v) {
    return v;
}
template <>
__device__ inline __half from_f32<__half>(float v) {
    return __float2half(v);
}
template <>
__device__ inline __nv_bfloat16 from_f32<__nv_bfloat16>(float v) {
    return __float2bfloat16(v);
}

// 一维 grid 大小：向上取整到 block 的整数倍
inline size_t grid_1d(size_t n, size_t block) {
    return (n + block - 1) / block;
}

} // namespace llaisys::ops::nvidia
