// ============================================================================
// rope/nvidia/rope_nvidia.hpp — NVIDIA/国产兼容 GPU 实现层声明（A4）
// ----------------------------------------------------------------------------
// 与 cpu/ 层的约定完全一致：入口函数收裸指针 + dtype + 维度，
// 实际 CUDA 内核在对应的 {op}_nvidia.cu 里。
// 本头文件不包含任何 CUDA 头，因此 CPU-only 构建也能安全 include。
// ============================================================================
#pragma once
#include "llaisys.h"

#include <cstddef>

namespace llaisys::ops::nvidia {
void rope(std::byte *out, const std::byte *in, const std::byte *pos_ids,
           llaisysDataType_t type, size_t seq, size_t nh, size_t d, float theta);
} // namespace llaisys::ops::nvidia
