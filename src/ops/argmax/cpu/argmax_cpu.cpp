// ============================================================================
// argmax/cpu/argmax_cpu.cpp — CPU 实现层
// ----------------------------------------------------------------------------
// 模式：模板函数 argmax_<T>() 做实际计算（T = float/fp16_t/bf16_t），
//       外层函数按 llaisysDataType_t switch 分发到对应模板实例。
// 精度约定：f16/bf16 的中间计算一律 cast 到 float，算完再 cast 回 T，
//           保证与 PyTorch 的数值对齐（A3 逐 token 一致的前提）。
//
// 算子：1D 张量求最大值及其下标
// 公式：max_val = max(vals[i]); max_idx = argmax_i vals[i]
// ============================================================================
#include "argmax_cpu.hpp"

#include "../../../utils.hpp"

namespace llaisys::ops::cpu {
namespace {
template <typename T>
void argmax_(int64_t *max_idx, T *max_val, const T *vals, size_t n) {
    size_t best = 0;
    float best_val = llaisys::utils::cast<float>(vals[0]);
    for (size_t i = 1; i < n; ++i) {
        float v = llaisys::utils::cast<float>(vals[i]);
        if (v > best_val) {
            best_val = v;
            best = i;
        }
    }
    *max_idx = (int64_t)best;
    *max_val = vals[best];
}
} // namespace

void argmax(std::byte *max_idx, std::byte *max_val, const std::byte *vals,
            llaisysDataType_t type, size_t n) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return argmax_(reinterpret_cast<int64_t *>(max_idx),
                       reinterpret_cast<float *>(max_val),
                       reinterpret_cast<const float *>(vals), n);
    case LLAISYS_DTYPE_F16:
        return argmax_(reinterpret_cast<int64_t *>(max_idx),
                       reinterpret_cast<llaisys::fp16_t *>(max_val),
                       reinterpret_cast<const llaisys::fp16_t *>(vals), n);
    case LLAISYS_DTYPE_BF16:
        return argmax_(reinterpret_cast<int64_t *>(max_idx),
                       reinterpret_cast<llaisys::bf16_t *>(max_val),
                       reinterpret_cast<const llaisys::bf16_t *>(vals), n);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::cpu
