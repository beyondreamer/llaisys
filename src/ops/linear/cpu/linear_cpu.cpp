// ============================================================================
// linear/cpu/linear_cpu.cpp — CPU 实现层
// ----------------------------------------------------------------------------
// 模式：模板函数 linear_<T>() 做实际计算（T = float/fp16_t/bf16_t），
//       外层函数按 llaisysDataType_t switch 分发到对应模板实例。
// 精度约定：f16/bf16 的中间计算一律 cast 到 float，算完再 cast 回 T，
//           保证与 PyTorch 的数值对齐（A3 逐 token 一致的前提）。
//
// 算子：全连接层：Y = X·Wᵀ + b
// 公式：out[i][j] = Σ_p in[i][p] * weight[j][p] + bias[j]
// ============================================================================
#include "linear_cpu.hpp"

#include "../../../utils.hpp"

namespace llaisys::ops::cpu {
namespace {
// Y = X W^T + b. Note: weight is NOT transposed: W has shape [n, k], so
// out[i][j] = sum_p x[i][p] * w[j][p] (+ bias[j]).
template <typename T>
void linear_(T *out, const T *x, const T *w, const T *bias, size_t m, size_t n, size_t k) {
    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < n; ++j) {
            float acc = 0.0f;
            for (size_t p = 0; p < k; ++p) {
                acc += llaisys::utils::cast<float>(x[i * k + p]) *
                       llaisys::utils::cast<float>(w[j * k + p]);
            }
            if (bias != nullptr) {
                acc += llaisys::utils::cast<float>(bias[j]);
            }
            out[i * n + j] = llaisys::utils::cast<T>(acc);
        }
    }
}
} // namespace

void linear(std::byte *out, const std::byte *in, const std::byte *weight, const std::byte *bias,
            llaisysDataType_t type, size_t m, size_t n, size_t k) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return linear_(reinterpret_cast<float *>(out), reinterpret_cast<const float *>(in),
                       reinterpret_cast<const float *>(weight),
                       reinterpret_cast<const float *>(bias), m, n, k);
    case LLAISYS_DTYPE_F16:
        return linear_(reinterpret_cast<llaisys::fp16_t *>(out),
                       reinterpret_cast<const llaisys::fp16_t *>(in),
                       reinterpret_cast<const llaisys::fp16_t *>(weight),
                       reinterpret_cast<const llaisys::fp16_t *>(bias), m, n, k);
    case LLAISYS_DTYPE_BF16:
        return linear_(reinterpret_cast<llaisys::bf16_t *>(out),
                       reinterpret_cast<const llaisys::bf16_t *>(in),
                       reinterpret_cast<const llaisys::bf16_t *>(weight),
                       reinterpret_cast<const llaisys::bf16_t *>(bias), m, n, k);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::cpu
