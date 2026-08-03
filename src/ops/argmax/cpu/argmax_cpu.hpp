// ============================================================================
// argmax/cpu/argmax_cpu.hpp — CPU 实现层声明
// ----------------------------------------------------------------------------
// CPU 实现的入口函数声明。约定：参数一律是裸指针 + dtype + 维度，
// 由 op.cpp 传入 tensor->data()，由实现内部 reinterpret_cast 成具体类型。
// 这样设备无关层（op.cpp）不关心具体类型，CPU 层不关心 Tensor 对象。
//
// 算子：1D 张量求最大值及其下标
// ============================================================================
#pragma once
#include "llaisys.h"

#include <cstddef>

namespace llaisys::ops::cpu {
void argmax(std::byte *max_idx, std::byte *max_val, const std::byte *vals,
            llaisysDataType_t type, size_t n);
}
