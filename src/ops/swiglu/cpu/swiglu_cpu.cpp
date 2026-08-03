// ============================================================================
// swiglu/cpu/swiglu_cpu.cpp — CPU 实现层
// ----------------------------------------------------------------------------
// 模式：模板函数 swiglu_<T>() 做实际计算（T = float/fp16_t/bf16_t），
//       外层函数按 llaisysDataType_t switch 分发到对应模板实例。
// 精度约定：f16/bf16 的中间计算一律 cast 到 float，算完再 cast 回 T，
//           保证与 PyTorch 的数值对齐（A3 逐 token 一致的前提）。
//
// 算子：SwiGLU 激活（Qwen2 MLP 用）
// 公式：out = up * SiLU(gate) = up * gate / (1 + e^(−gate))
// ============================================================================
#include "swiglu_cpu.hpp"

#include "../../../utils.hpp"

#include <cmath>

namespace llaisys::ops::cpu {
namespace {
// SwiGLU: out = up * SiLU(gate) = up * gate / (1 + exp(-gate)).
template <typename T>
void swiglu_(T *out, const T *gate, const T *up, size_t numel) {
    for (size_t i = 0; i < numel; ++i) {
        float g = llaisys::utils::cast<float>(gate[i]);
        float u = llaisys::utils::cast<float>(up[i]);
        out[i] = llaisys::utils::cast<T>(u * (g / (1.0f + std::exp(-g))));
    }
}
} // namespace

void swiglu(std::byte *out, const std::byte *gate, const std::byte *up,
            llaisysDataType_t type, size_t numel) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return swiglu_(reinterpret_cast<float *>(out), reinterpret_cast<const float *>(gate),
                       reinterpret_cast<const float *>(up), numel);
    case LLAISYS_DTYPE_F16:
        return swiglu_(reinterpret_cast<llaisys::fp16_t *>(out),
                       reinterpret_cast<const llaisys::fp16_t *>(gate),
                       reinterpret_cast<const llaisys::fp16_t *>(up), numel);
    case LLAISYS_DTYPE_BF16:
        return swiglu_(reinterpret_cast<llaisys::bf16_t *>(out),
                       reinterpret_cast<const llaisys::bf16_t *>(gate),
                       reinterpret_cast<const llaisys::bf16_t *>(up), numel);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::cpu
