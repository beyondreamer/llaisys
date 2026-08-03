// ============================================================================
// self_attention/cpu/self_attention_cpu.hpp — CPU 实现层声明
// ----------------------------------------------------------------------------
// CPU 实现的入口函数声明。约定：参数一律是裸指针 + dtype + 维度，
// 由 op.cpp 传入 tensor->data()，由实现内部 reinterpret_cast 成具体类型。
// 这样设备无关层（op.cpp）不关心具体类型，CPU 层不关心 Tensor 对象。
//
// 算子：带因果掩码的 Grouped Query Attention
// ============================================================================
#pragma once
#include "llaisys.h"

#include <cstddef>

namespace llaisys::ops::cpu {
void self_attention(std::byte *attn_val, const std::byte *q, const std::byte *k,
                    const std::byte *v, llaisysDataType_t type, size_t qlen,
                    size_t kvlen, size_t nh, size_t nkvh, size_t d, float scale);
}
