// ============================================================================
// src/ops/rope/nvidia/rope_nvidia.cu — 旋转位置编码的 CUDA 实现（A4）
// 公式：φ = pos / θ^(2j/d)
//       out[j] = a·cosφ − b·sinφ；out[j+d/2] = b·cosφ + a·sinφ
//       （a = x[:d/2]，b = x[d/2:]，d 必须为偶数）
// ============================================================================
#include "rope_nvidia.hpp"

#include "../../nvidia_common.cuh"
#include "../../../utils.hpp"

#include <cuda_runtime.h>

namespace llaisys::ops::nvidia {

// 每个线程负责一个 (seq, head, j) 的旋转对：线程总数 = seq * nh * (d/2)。
template <typename T>
__global__ void rope_kernel(T *out, const T *x, const int64_t *pos, size_t seq, size_t nh,
                            size_t d, float theta) {
    size_t half = d / 2;
    size_t total = seq * nh * half;
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    size_t j = idx % half;
    size_t rest = idx / half;
    size_t h = rest % nh;
    size_t s = rest / nh;

    float phi = (float)pos[s] / powf(theta, 2.0f * (float)j / (float)d);
    float c = cosf(phi);
    float sn = sinf(phi);

    const T *in_h = x + (s * nh + h) * d;
    T *out_h = out + (s * nh + h) * d;
    float a = to_f32(in_h[j]);
    float b = to_f32(in_h[j + half]);
    out_h[j] = from_f32<T>(a * c - b * sn);
    out_h[j + half] = from_f32<T>(b * c + a * sn);
}

template <typename T>
void launch_rope(T *out, const T *x, const int64_t *pos, size_t seq, size_t nh, size_t d,
                 float theta) {
    rope_kernel<T><<<grid_1d(seq * nh * (d / 2), 256), 256>>>(out, x, pos, seq, nh, d, theta);
}


void rope(std::byte *out, const std::byte *in, const std::byte *pos_ids,
          llaisysDataType_t type, size_t seq, size_t nh, size_t d, float theta) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return launch_rope(reinterpret_cast<float *>(out), reinterpret_cast<const float *>(in),
                           reinterpret_cast<const int64_t *>(pos_ids), seq, nh, d, theta);
    case LLAISYS_DTYPE_F16:
        return launch_rope(reinterpret_cast<__half *>(out), reinterpret_cast<const __half *>(in),
                           reinterpret_cast<const int64_t *>(pos_ids), seq, nh, d, theta);
    case LLAISYS_DTYPE_BF16:
        return launch_rope(reinterpret_cast<__nv_bfloat16 *>(out),
                           reinterpret_cast<const __nv_bfloat16 *>(in),
                           reinterpret_cast<const int64_t *>(pos_ids), seq, nh, d, theta);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::nvidia
