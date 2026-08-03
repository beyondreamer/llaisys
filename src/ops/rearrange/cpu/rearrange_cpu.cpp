// ============================================================================
// rearrange/cpu/rearrange_cpu.cpp — CPU 实现层
// ----------------------------------------------------------------------------
// 模式：模板函数 rearrange_<T>() 做实际计算（T = float/fp16_t/bf16_t），
//       外层函数按 llaisysDataType_t switch 分发到对应模板实例。
// 精度约定：f16/bf16 的中间计算一律 cast 到 float，算完再 cast 回 T，
//           保证与 PyTorch 的数值对齐（A3 逐 token 一致的前提）。
//
// 算子：视图重排：[seq, nh·dh] ⟷ [seq, nh, dh]（A3 推理必需，README 未列出的隐藏算子）
// 公式：纯内存布局视角转换，连续张量下等价于 view
// ============================================================================
#include "rearrange_cpu.hpp"

#include <cstring>

namespace llaisys::ops::cpu {
void rearrange(std::byte *out, const std::byte *in, size_t bytes) {
    std::memcpy(out, in, bytes);
}
} // namespace llaisys::ops::cpu
