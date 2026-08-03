// ============================================================================
// rope/cpu/rope_cpu.cpp — CPU 实现层
// ----------------------------------------------------------------------------
// 模式：模板函数 rope_<T>() 做实际计算（T = float/fp16_t/bf16_t），
//       外层函数按 llaisysDataType_t switch 分发到对应模板实例。
// 精度约定：f16/bf16 的中间计算一律 cast 到 float，算完再 cast 回 T，
//           保证与 PyTorch 的数值对齐（A3 逐 token 一致的前提）。
//
// 算子：旋转位置编码（Rotary Position Embedding）
// 公式：φ = pos / θ^(2j/d)；out[j]=a·cosφ−b·sinφ；out[j+d/2]=b·cosφ+a·sinφ（a=x[:d/2], b=x[d/2:]）
// ============================================================================
#include "rope_cpu.hpp"

#include "../../../utils.hpp"

#include <cmath>

namespace llaisys::ops::cpu {
namespace {
// Rotary position embedding: split each [seq, nh, d] vector into a = x[:d/2],
// b = x[d/2:], rotate by angle phi = pos / theta^(2j/d):
//   out[j]       = a*cos(phi) - b*sin(phi)
//   out[j + d/2] = b*cos(phi) + a*sin(phi)
template <typename T>
void rope_(T *out, const T *x, const int64_t *pos, size_t seq, size_t nh,
           size_t d, float theta) {
    const size_t half = d / 2;
    for (size_t s = 0; s < seq; ++s) {
        const float p = (float)pos[s];
        for (size_t h = 0; h < nh; ++h) {
            const T *in_h = x + (s * nh + h) * d;
            T *out_h = out + (s * nh + h) * d;
            for (size_t j = 0; j < half; ++j) {
                float phi = p / std::pow(theta, 2.0f * (float)j / (float)d);
                float c = std::cos(phi);
                float sn = std::sin(phi);
                float a = llaisys::utils::cast<float>(in_h[j]);
                float b = llaisys::utils::cast<float>(in_h[j + half]);
                out_h[j] = llaisys::utils::cast<T>(a * c - b * sn);
                out_h[j + half] = llaisys::utils::cast<T>(b * c + a * sn);
            }
        }
    }
}
} // namespace

void rope(std::byte *out, const std::byte *in, const std::byte *pos_ids,
          llaisysDataType_t type, size_t seq, size_t nh, size_t d, float theta) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return rope_(reinterpret_cast<float *>(out), reinterpret_cast<const float *>(in),
                     reinterpret_cast<const int64_t *>(pos_ids), seq, nh, d, theta);
    case LLAISYS_DTYPE_F16:
        return rope_(reinterpret_cast<llaisys::fp16_t *>(out),
                     reinterpret_cast<const llaisys::fp16_t *>(in),
                     reinterpret_cast<const int64_t *>(pos_ids), seq, nh, d, theta);
    case LLAISYS_DTYPE_BF16:
        return rope_(reinterpret_cast<llaisys::bf16_t *>(out),
                     reinterpret_cast<const llaisys::bf16_t *>(in),
                     reinterpret_cast<const int64_t *>(pos_ids), seq, nh, d, theta);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::cpu
