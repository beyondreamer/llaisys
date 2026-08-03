// ============================================================================
// rms_norm/cpu/rms_norm_cpu.hpp — CPU 实现层声明
// ----------------------------------------------------------------------------
// CPU 实现的入口函数声明。约定：参数一律是裸指针 + dtype + 维度，
// 由 op.cpp 传入 tensor->data()，由实现内部 reinterpret_cast 成具体类型。
// 这样设备无关层（op.cpp）不关心具体类型，CPU 层不关心 Tensor 对象。
//
// 算子：RMS 归一化（Qwen2 用的不是 LayerNorm 而是 RMSNorm）
// ============================================================================
#pragma once
#include "llaisys.h"

#include <cstddef>

namespace llaisys::ops::cpu {
void rms_norm(std::byte *out, const std::byte *in, const std::byte *weight,
              llaisysDataType_t type, size_t rows, size_t d, float eps);
}
