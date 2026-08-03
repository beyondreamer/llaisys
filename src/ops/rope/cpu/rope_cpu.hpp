// ============================================================================
// rope/cpu/rope_cpu.hpp — CPU 实现层声明
// ----------------------------------------------------------------------------
// CPU 实现的入口函数声明。约定：参数一律是裸指针 + dtype + 维度，
// 由 op.cpp 传入 tensor->data()，由实现内部 reinterpret_cast 成具体类型。
// 这样设备无关层（op.cpp）不关心具体类型，CPU 层不关心 Tensor 对象。
//
// 算子：旋转位置编码（Rotary Position Embedding）
// ============================================================================
#pragma once
#include "llaisys.h"

#include <cstddef>

namespace llaisys::ops::cpu {
void rope(std::byte *out, const std::byte *in, const std::byte *pos_ids,
          llaisysDataType_t type, size_t seq, size_t nh, size_t d, float theta);
}
