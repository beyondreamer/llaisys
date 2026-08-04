// ============================================================================
// src/ops/rope/nvidia/rope_nvidia.cu — 旋转位置编码 RoPE 的 CUDA 实现（优化版）
// ----------------------------------------------------------------------------
// 公式：φ = pos / θ^(2j/d)
//       out[j]       = a·cosφ − b·sinφ
//       out[j+d/2]   = b·cosφ + a·sinφ
//       （a = x[:d/2]，b = x[d/2:]，d 必须为偶数）
//
// 【优化点（相对原版）】
//   原版：每个线程处理「一对 (a,b) 的一半」——即一个线程算 out[j]，另一个线程
//         算 out[j+d/2]，二者各自调用一次 cosf + sinf → 每对 trig 计算重复 2 次
//         且每线程一次 powf（powf 是昂贵的 transcendental 函数）
//   新版：每个线程处理「完整的一对 (a,b)」——同一线程算一次 sincosf 得到 (c,sn)，
//         然后写 out[j] 和 out[j+d/2] 两个位置，trig 计算减半
//         theta^(-2j/d) 用 expf(logf... ) 不如直接 powf，保留 powf 但调一次
//         decode 场景（seq 小，如 1）下，threadIdx 数量可能远大于 d/2 × seq × nh，
//         此时每个线程刚好对应一对，launch overhead 最低
//
// 精度约定：与 CPU 层完全一致（phi 用 float32 计算，输入/输出 T 精度）
// ============================================================================
#include "rope_nvidia.hpp"

#include "../../nvidia_common.cuh"
#include "../../../utils.hpp"

#include <cuda_runtime.h>

namespace llaisys::ops::nvidia {

// 每个线程处理「一整对 (a,b)」：算一次 sincosf，写 out[j] 与 out[j+d/2]。
// 线程总数 = seq * nh * (d/2)，一维 launch。
template <typename T>
__global__ void rope_kernel(T *out, const T *x, const int64_t *pos, size_t seq, size_t nh,
                            size_t d, float theta) {
    const size_t half = d / 2;
    const size_t total = seq * nh * half;
    const size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    // 把一维线程索引分解为 (s, h, j)
    const size_t j = idx % half;
    const size_t rest = idx / half;
    const size_t h = rest % nh;
    const size_t s = rest / nh;

    // 角度 φ = pos / θ^(2j/d)
    const float phi = (float)pos[s] / powf(theta, 2.0f * (float)j / (float)d);
    // 一次 sincosf 同时得到 cos 和 sin（比分开调 cosf + sinf 快，共享同一硬件单元）
    float c, sn;
    sincosf(phi, &sn, &c);

    const T *in_h = x + (s * nh + h) * d;
    T *out_h = out + (s * nh + h) * d;
    const float a = to_f32(in_h[j]);
    const float b = to_f32(in_h[j + half]);
    // 一次写两个位置：旋转对的两半
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
