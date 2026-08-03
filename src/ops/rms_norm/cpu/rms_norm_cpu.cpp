// ============================================================================
// rms_norm/cpu/rms_norm_cpu.cpp — CPU 实现层
// ----------------------------------------------------------------------------
// 模式：模板函数 rms_norm_<T>() 做实际计算（T = float/fp16_t/bf16_t），
//       外层函数按 llaisysDataType_t switch 分发到对应模板实例。
// 精度约定：f16/bf16 的中间计算一律 cast 到 float，算完再 cast 回 T，
//           保证与 PyTorch 的数值对齐（A3 逐 token 一致的前提）。
//
// 算子：RMS 归一化（Qwen2 用的不是 LayerNorm 而是 RMSNorm）
// 公式：out[i][j] = x[i][j] / sqrt(mean_j(x[i][j]²) + eps) * w[j]
// ============================================================================
#include "rms_norm_cpu.hpp"

#include "../../../utils.hpp"

#include <cmath>

namespace llaisys::ops::cpu {
namespace {
// y_i = x_i / sqrt(mean(x^2) + eps) * w_i, row-wise over the last dim.
template <typename T>
void rms_norm_(T *out, const T *x, const T *w, size_t rows, size_t d, float eps) {
    for (size_t i = 0; i < rows; ++i) {
        float sum_sq = 0.0f;
        for (size_t j = 0; j < d; ++j) {
            float v = llaisys::utils::cast<float>(x[i * d + j]);
            sum_sq += v * v;
        }
        float rms = 1.0f / std::sqrt(sum_sq / (float)d + eps);
        for (size_t j = 0; j < d; ++j) {
            out[i * d + j] = llaisys::utils::cast<T>(
                llaisys::utils::cast<float>(x[i * d + j]) * rms *
                llaisys::utils::cast<float>(w[j]));
        }
    }
}
} // namespace

void rms_norm(std::byte *out, const std::byte *in, const std::byte *weight,
              llaisysDataType_t type, size_t rows, size_t d, float eps) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return rms_norm_(reinterpret_cast<float *>(out), reinterpret_cast<const float *>(in),
                         reinterpret_cast<const float *>(weight), rows, d, eps);
    case LLAISYS_DTYPE_F16:
        return rms_norm_(reinterpret_cast<llaisys::fp16_t *>(out),
                         reinterpret_cast<const llaisys::fp16_t *>(in),
                         reinterpret_cast<const llaisys::fp16_t *>(weight), rows, d, eps);
    case LLAISYS_DTYPE_BF16:
        return rms_norm_(reinterpret_cast<llaisys::bf16_t *>(out),
                         reinterpret_cast<const llaisys::bf16_t *>(in),
                         reinterpret_cast<const llaisys::bf16_t *>(weight), rows, d, eps);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::cpu
