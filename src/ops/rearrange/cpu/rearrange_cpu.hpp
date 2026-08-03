// ============================================================================
// rearrange/cpu/rearrange_cpu.hpp — CPU 实现层声明
// ----------------------------------------------------------------------------
// CPU 实现的入口函数声明。约定：参数一律是裸指针 + dtype + 维度，
// 由 op.cpp 传入 tensor->data()，由实现内部 reinterpret_cast 成具体类型。
// 这样设备无关层（op.cpp）不关心具体类型，CPU 层不关心 Tensor 对象。
//
// 算子：视图重排：[seq, nh·dh] ⟷ [seq, nh, dh]（A3 推理必需，README 未列出的隐藏算子）
// ============================================================================
#pragma once
#include "llaisys.h"

#include <cstddef>

namespace llaisys::ops::cpu {
void rearrange(std::byte *out, const std::byte *in, size_t bytes);
}
