// ============================================================================
// embedding/cpu/embedding_cpu.cpp — CPU 实现层
// ----------------------------------------------------------------------------
// 模式：模板函数 embedding_<T>() 做实际计算（T = float/fp16_t/bf16_t），
//       外层函数按 llaisysDataType_t switch 分发到对应模板实例。
// 精度约定：f16/bf16 的中间计算一律 cast 到 float，算完再 cast 回 T，
//           保证与 PyTorch 的数值对齐（A3 逐 token 一致的前提）。
//
// 算子：查表：out[i,:] = weight[index[i],:]
// 公式：out[i, j] = weight[index[i], j]
// ============================================================================
#include "embedding_cpu.hpp"

#include "../../../utils.hpp"

#include <cstring>

namespace llaisys::ops::cpu {
namespace {
template <typename T>
void embedding_(T *out, const int64_t *index, const T *weight, size_t nidx, size_t d) {
    for (size_t i = 0; i < nidx; ++i) {
        std::memcpy(out + i * d, weight + (size_t)index[i] * d, d * sizeof(T));
    }
}
} // namespace

void embedding(std::byte *out, const std::byte *index, const std::byte *weight,
               llaisysDataType_t type, size_t nidx, size_t d) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return embedding_(reinterpret_cast<float *>(out),
                          reinterpret_cast<const int64_t *>(index),
                          reinterpret_cast<const float *>(weight), nidx, d);
    case LLAISYS_DTYPE_F16:
        return embedding_(reinterpret_cast<llaisys::fp16_t *>(out),
                          reinterpret_cast<const int64_t *>(index),
                          reinterpret_cast<const llaisys::fp16_t *>(weight), nidx, d);
    case LLAISYS_DTYPE_BF16:
        return embedding_(reinterpret_cast<llaisys::bf16_t *>(out),
                          reinterpret_cast<const int64_t *>(index),
                          reinterpret_cast<const llaisys::bf16_t *>(weight), nidx, d);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::cpu
