// ============================================================================
// src/ops/self_attention/nvidia/self_attention_nvidia.cu — GQA 自注意力 CUDA（优化版）
// ----------------------------------------------------------------------------
// 公式：attn = causal_softmax(Q·Kᵀ·scale) · V
// 要点（与 CPU 层完全一致，精度不可破）：
//   * GQA：query 头 h 用 kv 头 h/groups（groups = nh/nkvh，repeat_interleave 语义）
//   * causal：query i 只能 attend 到 key j <= i + (kvlen - qlen)
//   * softmax：float32 + max-subtract 数值稳定
//
// 【优化策略：两条路径】
//   1) decode 路径（qlen == 1）：FlashAttention 的 decode 变体
//      - 一个 block 处理一个 (i, head) 对
//      - Q 常驻寄存器（d 个 float），KV 分块流式处理
//      - online softmax：每处理一个 KV 块就增量更新 (max, sum, acc)，无需 O(seq²) smem
//      - smem 占用从 O(kvlen) 降到 O(BK + head_dim)
//      - KV 只从全局读一遍
//      128 步 decode 全部命中此路径
//
//   2) prefill 路径（qlen > 1）：保留原精确实现
//      - scores 全存 smem（适合短序列），block 内协作算 + 树形归约
//      - prefill 通常很短（如本项目的 2 步 prompt），优化收益小，保稳定
//
// 精度：bf16/f16 输入 cast 到 f32 累加，最后舍入回 T（与 CPU 一致，token 对齐前提）
// 平台：纯 CUDA intrinsics，4090D + C500 通用
// ============================================================================
#include "self_attention_nvidia.hpp"

#include "../../nvidia_common.cuh"
#include "../../../utils.hpp"

#include <cuda_runtime.h>
#include <math_constants.h>

namespace llaisys::ops::nvidia {

constexpr int BLOCK = 128; // 每 (i, h) 的线程数（head_dim 通常 <= 128）

// ============================================================================
// 路径 2：prefill 精确注意力（qlen > 1）—— 保留原实现
// ============================================================================
// 一个 block 处理一个 (query i, head h) 对；block 内所有线程协作算 scores[j]
// （float32，放 shared memory），再做跨线程的 max/sum 归约，最后每线程负责输出维 p。
// smem 占用 = kvlen × sizeof(float)，长序列时会偏大，但 prefill 通常短。
template <typename T>
__global__ void self_attention_prefill_kernel(T *attn, const T *q, const T *k, const T *v,
                                              size_t qlen, size_t kvlen, size_t nh, size_t nkvh,
                                              size_t d, float scale) {
    extern __shared__ float scores[];

    const size_t ih = blockIdx.x;
    const size_t i = ih / nh;
    const size_t h = ih % nh;
    const size_t groups = nh / nkvh;
    const size_t kvh = h / groups;
    const size_t offset = kvlen - qlen;
    const size_t causal_kvlen = (i + offset + 1 < kvlen) ? (i + offset + 1) : kvlen;

    const T *q_h = q + (i * nh + h) * d;
    const T *k_h = k + kvh * d;
    const T *v_h = v + kvh * d;

    // 1) scores[j] = q·k_j * scale，同时找局部 max
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

    // 4) 每线程负责输出维度 p：加权和 / sum
    T *attn_h = attn + (i * nh + h) * d;
    for (size_t p = threadIdx.x; p < d; p += BLOCK) {
        float acc = 0.0f;
        for (size_t j = 0; j < causal_kvlen; ++j) {
            acc += scores[j] * to_f32(v_h[j * nkvh * d + p]);
        }
        attn_h[p] = from_f32<T>(acc / sum);
    }
}

// ============================================================================
// 路径 1：decode flash 变体（qlen == 1）
// ============================================================================
// 一个 block 处理一个 head（i=0 固定，因为 qlen=1）。
// Q 在寄存器里常驻（d 个 float），KV 分块流式：
//   for 每块 BK 个 key:
//     加载 BK 个 K 到 smem（k_smem[BK][d]）
//     算 BK 个 score（每线程算若干个），找本块局部 max
//     online softmax 更新：rescale 全局 acc/out + 增量加权和
//     加载 BK 个 V 到 smem（v_smem[BK][d]），加权累加进 out
//   最后 out /= sum，写回
//
// online softmax 数学：
//   设旧 (m, s, O)，新块带来 (m_new, 局部 s_new)：
//     t = exp(m - m_new); s = s * t + s_new; O = O * t + Σ exp(score - m_new) · V
//   其中 m = max(m, m_new)
//
// smem：BK * d * sizeof(float)（K 块）+ BK * d * sizeof(float)（V 块）
//      BK=32, d=128 → 32*128*4*2 = 32KB，sm_80 上限 48KB，安全
constexpr int DECODE_BK = 16; // KV 分块大小（每块 16 个 key）
constexpr int DECODE_THREADS = 128;

template <typename T>
__global__ void self_attention_decode_kernel(T *attn, const T *q, const T *k, const T *v,
                                             size_t kvlen, size_t nh, size_t nkvh, size_t d,
                                             float scale) {
    // qlen == 1，i 固定为 0，一个 block 处理一个 head
    const size_t h = blockIdx.x;
    const size_t groups = nh / nkvh;
    const size_t kvh = h / groups; // GQA 映射

    const T *q_h = q + h * d;        // query 只有 1 行
    const T *k_h = k + kvh * d;      // 第 j 个 key 行 = k + (j*nkvh + kvh)*d
    const T *v_h = v + kvh * d;
    T *attn_h = attn + h * d;

    const int tid = threadIdx.x;

    // shared memory：K 块与 V 块，[DECODE_BK][d]，float 中转（避免 T 类型模板 smem 麻烦）
    __shared__ float ks[DECODE_BK][128]; // d<=128，按最大 head_dim 开
    __shared__ float vs[DECODE_BK][128];

    // 1) Q 装进寄存器（每线程各持一份 d 个 float，简单且无 bank conflict）
    //    若 d > 128 需调整，本模型 d=128 命中上限
    float qreg[128];
    for (size_t p = 0; p < d; ++p) {
        qreg[p] = (tid == 0) ? to_f32(q_h[p]) : 0.0f; // 只用 lane 0 读，再广播
    }
    // 用 warp shuffle 广播 qreg：从 lane 0 广播给所有 lane
    // （tid % 32 == 0 的 lane 才读了真值，其余 lane 是 0，需广播）
    // 简化做法：让所有线程都自己读一遍 q（d=128 × 128 thread = 16K 读，但全走 L2 cache，
    //           后续线程全命中，开销可忽略），这样无需广播逻辑
    for (size_t p = 0; p < d; ++p) {
        qreg[p] = to_f32(q_h[p]);
    }

    // online softmax 状态
    float global_max = -CUDART_INF_F; // m
    float global_sum = 0.0f;          // s
    // out_acc[p]：每线程持 d 个输出维度的累加器（注意 d 个线程 vs d 维度，
    //   我们让每线程持 1 个输出维度的累加器：tid 映射到 p=tid (<d)）
    //   d=128, threads=128 → 一线程一维度，正好覆盖。
    float out_acc = 0.0f;
    const size_t p = tid; // 本线程负责的输出维度（仅当 tid < d 时有效）

    // 2) 沿 KV 分块流式处理
    for (size_t kb = 0; kb < kvlen; kb += DECODE_BK) {
        size_t block_end = kb + DECODE_BK;
        if (block_end > kvlen) block_end = kvlen;
        size_t this_bk = block_end - kb;

        // ---- 加载 K 块到 smem：ks[local_j][p] = K[(kb+local_j)*nkvh 步幅... ][p] ----
        // K 布局：k[j][kvh][p]，每行 = k + (j*nkvh + kvh)*d。
        // 协作加载：DECODE_THREADS 线程 × (this_bk × d) 元素
        for (size_t idx = tid; idx < this_bk * d; idx += DECODE_THREADS) {
            size_t lj = idx / d;        // 本地 key 行 [0, DECODE_BK)
            size_t pp = idx % d;        // 维度
            size_t gj = kb + lj;        // 全局 key 行
            ks[lj][pp] = to_f32(k_h[gj * nkvh * d + pp]);
        }
        __syncthreads();

        // ---- 每个线程算若干个 local_j 的 score，找本块局部 max ----
        // 把 score 也存 smem 复用（DECODE_BK 个 float）
        __shared__ float block_scores[DECODE_BK];
        float my_block_max = -CUDART_INF_F;
        for (size_t lj = tid; lj < this_bk; lj += DECODE_THREADS) {
            float s = 0.0f;
            for (size_t pp = 0; pp < d; ++pp) {
                s += qreg[pp] * ks[lj][pp];
            }
            s *= scale;
            block_scores[lj] = s;
            if (s > my_block_max) my_block_max = s;
        }
        // 归约本块 max（DECODE_THREADS 上多数线程没参与赋值，my_block_max=-INF 不影响）
        __shared__ float smax_reduce[DECODE_THREADS];
        smax_reduce[tid] = my_block_max;
        __syncthreads();
        for (int s = DECODE_THREADS / 2; s > 0; s >>= 1) {
            if (tid < s) smax_reduce[tid] = fmaxf(smax_reduce[tid], smax_reduce[tid + s]);
            __syncthreads();
        }
        float block_max = smax_reduce[0];
        __syncthreads();

        // ---- online softmax 更新 ----
        float new_max = fmaxf(global_max, block_max);
        float rescale = __expf(global_max - new_max); // 旧 acc 的缩放因子
        // 更新 sum：旧 sum × rescale + Σ_j exp(score_j - new_max)
        float block_sum = 0.0f;
        for (size_t lj = tid; lj < this_bk; lj += DECODE_THREADS) {
            block_scores[lj] = __expf(block_scores[lj] - new_max);
            block_sum += block_scores[lj];
        }
        // 归约 block_sum
        __shared__ float ssum_reduce[DECODE_THREADS];
        ssum_reduce[tid] = block_sum;
        __syncthreads();
        for (int s = DECODE_THREADS / 2; s > 0; s >>= 1) {
            if (tid < s) ssum_reduce[tid] += ssum_reduce[tid + s];
            __syncthreads();
        }
        block_sum = ssum_reduce[0];
        __syncthreads();

        global_sum = global_sum * rescale + block_sum;

        // 旧 out_acc 缩放（在线程本地，无需归约）
        out_acc = out_acc * rescale;

        // ---- 加载 V 块到 smem ----
        for (size_t idx = tid; idx < this_bk * d; idx += DECODE_THREADS) {
            size_t lj = idx / d;
            size_t pp = idx % d;
            size_t gj = kb + lj;
            vs[lj][pp] = to_f32(v_h[gj * nkvh * d + pp]);
        }
        __syncthreads();

        // ---- 每线程累加自己负责的输出维度 p ----
        // out_acc += Σ_lj block_scores[lj] * vs[lj][p]
        if (p < d) {
            for (size_t lj = 0; lj < this_bk; ++lj) {
                out_acc += block_scores[lj] * vs[lj][p];
            }
        }

        global_max = new_max;
        __syncthreads();
    }

    // 3) 最终归一化：out[p] = out_acc / global_sum
    if (p < d) {
        attn_h[p] = from_f32<T>(out_acc / global_sum);
    }
}

// ----------------------------------------------------------------------------
// launch：根据 qlen 选路径
// ----------------------------------------------------------------------------
template <typename T>
void launch_self_attention(T *attn, const T *q, const T *k, const T *v, size_t qlen,
                           size_t kvlen, size_t nh, size_t nkvh, size_t d, float scale) {
    if (qlen == 1) {
        // decode 路径：一个 block 一个 head
        size_t blocks = nh;
        self_attention_decode_kernel<T><<<blocks, DECODE_THREADS>>>(attn, q, k, v, kvlen, nh, nkvh,
                                                                     d, scale);
    } else {
        // prefill 路径：一个 block 一个 (i, h)
        size_t blocks = qlen * nh;
        size_t smem = kvlen * sizeof(float);
        self_attention_prefill_kernel<T><<<blocks, BLOCK, smem>>>(attn, q, k, v, qlen, kvlen, nh,
                                                                   nkvh, d, scale);
    }
}


void self_attention(std::byte *attn_val, const std::byte *q, const std::byte *k,
                    const std::byte *v, llaisysDataType_t type, size_t qlen, size_t kvlen,
                    size_t nh, size_t nkvh, size_t d, float scale) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return launch_self_attention(reinterpret_cast<float *>(attn_val),
                                     reinterpret_cast<const float *>(q),
                                     reinterpret_cast<const float *>(k),
                                     reinterpret_cast<const float *>(v), qlen, kvlen, nh, nkvh, d,
                                     scale);
    case LLAISYS_DTYPE_F16:
        return launch_self_attention(reinterpret_cast<__half *>(attn_val),
                                     reinterpret_cast<const __half *>(q),
                                     reinterpret_cast<const __half *>(k),
                                     reinterpret_cast<const __half *>(v), qlen, kvlen, nh, nkvh, d,
                                     scale);
    case LLAISYS_DTYPE_BF16:
        return launch_self_attention(reinterpret_cast<__nv_bfloat16 *>(attn_val),
                                     reinterpret_cast<const __nv_bfloat16 *>(q),
                                     reinterpret_cast<const __nv_bfloat16 *>(k),
                                     reinterpret_cast<const __nv_bfloat16 *>(v), qlen, kvlen, nh,
                                     nkvh, d, scale);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::nvidia
