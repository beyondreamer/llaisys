// ============================================================================
// self_attention/cpu/self_attention_cpu.cpp — CPU 实现层
// ----------------------------------------------------------------------------
// 模式：模板函数 self_attention_<T>() 做实际计算（T = float/fp16_t/bf16_t），
//       外层函数按 llaisysDataType_t switch 分发到对应模板实例。
// 精度约定：f16/bf16 的中间计算一律 cast 到 float，算完再 cast 回 T，
//           保证与 PyTorch 的数值对齐（A3 逐 token 一致的前提）。
//
// 算子：带因果掩码的 Grouped Query Attention
// 公式：attn = softmax(Q·Kᵀ·scale + mask) · V
// ============================================================================
#include "self_attention_cpu.hpp"

#include "../../../utils.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

namespace llaisys::ops::cpu {
namespace {
// Y = causal_softmax(Q K^T * scale) V with grouped query attention (GQA).
// q: [qlen, nh, d], k/v: [kvlen, nkvh, d], attn_val: [qlen, nh, d].
// Query head h attends to KV head h / groups (repeat_interleave semantics).
// Causal mask: query i may attend to keys j <= i + (kvlen - qlen).
template <typename T>
void self_attention_(T *attn, const T *q, const T *k, const T *v, size_t qlen,
                     size_t kvlen, size_t nh, size_t nkvh, size_t d, float scale) {
    const size_t groups = nh / nkvh;
    const size_t offset = kvlen - qlen; // diagonal of the causal mask
    std::vector<float> scores(std::max(qlen, kvlen));

    for (size_t i = 0; i < qlen; ++i) {
        for (size_t h = 0; h < nh; ++h) {
            const size_t kvh = h / groups;
            const T *q_h = q + (i * nh + h) * d;
            const T *k_h = k + kvh * d;
            const T *v_h = v + kvh * d;
            const size_t causal_kvlen = std::min(kvlen, i + offset + 1);

            // scores (float32) with max-subtraction for numerical stability
            float mx = -std::numeric_limits<float>::infinity();
            for (size_t j = 0; j < causal_kvlen; ++j) {
                float s = 0.0f;
                for (size_t p = 0; p < d; ++p) {
                    s += llaisys::utils::cast<float>(q_h[p]) *
                         llaisys::utils::cast<float>(k_h[j * nkvh * d + p]);
                }
                s *= scale;
                scores[j] = s;
                mx = std::max(mx, s);
            }

            float sum = 0.0f;
            for (size_t j = 0; j < causal_kvlen; ++j) {
                scores[j] = std::exp(scores[j] - mx);
                sum += scores[j];
            }

            T *attn_h = attn + (i * nh + h) * d;
            for (size_t p = 0; p < d; ++p) {
                float acc = 0.0f;
                for (size_t j = 0; j < causal_kvlen; ++j) {
                    acc += scores[j] *
                           llaisys::utils::cast<float>(v_h[j * nkvh * d + p]);
                }
                attn_h[p] = llaisys::utils::cast<T>(acc / sum);
            }
        }
    }
}
} // namespace

void self_attention(std::byte *attn_val, const std::byte *q, const std::byte *k,
                    const std::byte *v, llaisysDataType_t type, size_t qlen,
                    size_t kvlen, size_t nh, size_t nkvh, size_t d, float scale) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return self_attention_(reinterpret_cast<float *>(attn_val),
                               reinterpret_cast<const float *>(q),
                               reinterpret_cast<const float *>(k),
                               reinterpret_cast<const float *>(v), qlen, kvlen, nh, nkvh, d, scale);
    case LLAISYS_DTYPE_F16:
        return self_attention_(reinterpret_cast<llaisys::fp16_t *>(attn_val),
                               reinterpret_cast<const llaisys::fp16_t *>(q),
                               reinterpret_cast<const llaisys::fp16_t *>(k),
                               reinterpret_cast<const llaisys::fp16_t *>(v), qlen, kvlen, nh, nkvh, d, scale);
    case LLAISYS_DTYPE_BF16:
        return self_attention_(reinterpret_cast<llaisys::bf16_t *>(attn_val),
                               reinterpret_cast<const llaisys::bf16_t *>(q),
                               reinterpret_cast<const llaisys::bf16_t *>(k),
                               reinterpret_cast<const llaisys::bf16_t *>(v), qlen, kvlen, nh, nkvh, d, scale);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::cpu
