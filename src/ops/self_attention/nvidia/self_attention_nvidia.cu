// ============================================================================
// src/ops/self_attention/nvidia/self_attention_nvidia.cu — GQA 自注意力的 CUDA 实现（A4）
// 公式：attn = causal_softmax(Q·Kᵀ·scale) · V
// 要点（与 CPU 层完全一致）：
//   * GQA：query 头 h 用 kv 头 h/groups（groups = nh/nkvh）
//   * causal：query i 只能 attend 到 key j <= i + (kvlen - qlen)
//   * softmax：float32 + max-subtract 数值稳定
//
// 并行策略：一个 block 处理一个 (query i, head h) 对；
//   block 内所有线程先协作算 scores[j]（float32，放 shared memory），
//   再做跨线程的 max/sum 归约，最后每个线程负责输出维度 p 的加权和。
// ============================================================================
#include "self_attention_nvidia.hpp"

#include "../../nvidia_common.cuh"
#include "../../../utils.hpp"

#include <cuda_runtime.h>
#include <math_constants.h>

namespace llaisys::ops::nvidia {

constexpr int BLOCK = 128; // 每个 (i, h) 的线程数（head_dim 通常 <= 128）

template <typename T>
__global__ void self_attention_kernel(T *attn, const T *q, const T *k, const T *v,
                                      size_t qlen, size_t kvlen, size_t nh, size_t nkvh,
                                      size_t d, float scale) {
    // 动态 shared memory：存 float32 scores[j]，j < causal_kvlen <= kvlen
    extern __shared__ float scores[];

    const size_t ih = blockIdx.x;      // 第 i 个 query 的第 h 个 head
    const size_t i = ih / nh;
    const size_t h = ih % nh;
    const size_t groups = nh / nkvh;
    const size_t kvh = h / groups;     // GQA：query 头 h 用 kv 头 h/groups
    const size_t offset = kvlen - qlen; // causal 掩码对角线
    // 可移植写法：等价于 min(kvlen, i+offset+1)（部分 CUDA 兼容工具链不支持裸 min）
    const size_t causal_kvlen = (i + offset + 1 < kvlen) ? (i + offset + 1) : kvlen;

    const T *q_h = q + (i * nh + h) * d;
    const T *k_h = k + kvh * d;        // 第 j 个 key 行 = k + (j*nkvh + kvh)*d
    const T *v_h = v + kvh * d;

    // 1) 计算 scores[j] = q·k_j * scale（float32），同时找 max
    float my_max = -CUDART_INF_F;
    for (size_t j = threadIdx.x; j < causal_kvlen; j += BLOCK) {
        float s = 0.0f;
        for (size_t p = 0; p < d; ++p) {
            s += to_f32(q_h[p]) * to_f32(k_h[j * nkvh * d + p]);
        }
        s *= scale;
        scores[j] = s;
        my_max = fmaxf(my_max, s);
    }

    // 2) 跨线程归约 max（shared memory 树形归约）
    __shared__ float smax[BLOCK];
    smax[threadIdx.x] = my_max;
    __syncthreads();
    for (int s = BLOCK / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            smax[threadIdx.x] = fmaxf(smax[threadIdx.x], smax[threadIdx.x + s]);
        }
        __syncthreads();
    }
    const float max_score = smax[0];
    __syncthreads();

    // 3) exp(scores - max) 并求 sum
    float my_sum = 0.0f;
    for (size_t j = threadIdx.x; j < causal_kvlen; j += BLOCK) {
        scores[j] = __expf(scores[j] - max_score);
        my_sum += scores[j];
    }
    __shared__ float ssum[BLOCK];
    ssum[threadIdx.x] = my_sum;
    __syncthreads();
    for (int s = BLOCK / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            ssum[threadIdx.x] += ssum[threadIdx.x + s];
        }
        __syncthreads();
    }
    const float sum = ssum[0];
    __syncthreads();

    // 4) 每个线程负责输出维度 p：加权和 / sum
    T *attn_h = attn + (i * nh + h) * d;
    for (size_t p = threadIdx.x; p < d; p += BLOCK) {
        float acc = 0.0f;
        for (size_t j = 0; j < causal_kvlen; ++j) {
            acc += scores[j] * to_f32(v_h[j * nkvh * d + p]);
        }
        attn_h[p] = from_f32<T>(acc / sum);
    }
}

template <typename T>
void launch_self_attention(T *attn, const T *q, const T *k, const T *v, size_t qlen,
                           size_t kvlen, size_t nh, size_t nkvh, size_t d, float scale) {
    // 每个 (i, h) 一个 block；scores 用动态 shared memory（kvlen 个 float）
    size_t blocks = qlen * nh;
    size_t smem = kvlen * sizeof(float);
    self_attention_kernel<T><<<blocks, BLOCK, smem>>>(attn, q, k, v, qlen, kvlen, nh, nkvh, d,
                                                      scale);
}


void self_attention(std::byte *attn_val, const std::byte *q, const std::byte *k,
                    const std::byte *v, llaisysDataType_t type, size_t qlen, size_t kvlen,
                    size_t nh, size_t nkvh, size_t d, float scale) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return launch_self_attention(reinterpret_cast<float *>(attn_val),
                                     reinterpret_cast<const float *>(q),
                                     reinterpret_cast<const float *>(k),
                                     reinterpret_cast<const float *>(v), qlen, kvlen, nh, nkvh,
                                     d, scale);
    case LLAISYS_DTYPE_F16:
        return launch_self_attention(reinterpret_cast<__half *>(attn_val),
                                     reinterpret_cast<const __half *>(q),
                                     reinterpret_cast<const __half *>(k),
                                     reinterpret_cast<const __half *>(v), qlen, kvlen, nh, nkvh,
                                     d, scale);
    case LLAISYS_DTYPE_BF16:
        return launch_self_attention(reinterpret_cast<__nv_bfloat16 *>(attn_val),
                                     reinterpret_cast<const __nv_bfloat16 *>(q),
                                     reinterpret_cast<const __nv_bfloat16 *>(k),
                                     reinterpret_cast<const __nv_bfloat16 *>(v), qlen, kvlen,
                                     nh, nkvh, d, scale);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::nvidia
