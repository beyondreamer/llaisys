// ============================================================================
// src/ops/linear/nvidia/linear_nvidia.cu — 全连接算子的 CUDA 实现（优化版）
// ----------------------------------------------------------------------------
// 公式：out[i][j] = Σ_p in[i][p] * weight[j][p] + bias[j]
//       （weight 未转置，形状 [n, k]，按行取 —— 与 CPU 层同一约定）
//
// 【本文件实现：阶段 A — Shared Memory Tiling GEMM（默认，全平台通用）】
//
// 相对原版（一元素一线程、weight 全程走全局内存）的核心改进：
//   * 一个 block 负责 out 的 [BM×BN] 子块，把对应 x[BM,BK] 与 w[BN,BK] 分片
//     加载到 shared memory，BM 个线程复用同一 w 行 → 全局读次数从 O(M·N·K)
//     降到 O(M·N·K/BK)，访存带宽不再是瓶颈
//   * 每线程持有 TM×TN 个寄存器累加器，内层循环只读写寄存器 + smem，
//     指令级并行（ILP）由编译器自动展开外积累加
//   * bf16/f16 输入 cast 到 f32 累加，最后一次舍入回 T
//     —— 与 CPU 层「bf16 → float 累加 → bf16」精度约定完全一致，不破坏 token 对齐
//
// 平台：纯 CUDA intrinsics，NVIDIA 4090D（sm_89）与沐曦 C500（cu-bridge）通用，
//       零平台专用代码（守住 mxcc_nvcc_wrapper.sh 复用方案）。
//
// 预留扩展（未启用，留作后续）：
//   * LLAISYS_USE_WMMA  → bf16/f16 走 Tensor Core（16×16×16 fragment，f32 accumulator）
//   * LLAISYS_USE_CUBLAS → 4090D 走 cuBLAS（峰值保底）
//   两套后端都用编译宏开关保护，缺省时本文件即完整可用。
// ============================================================================
#include "linear_nvidia.hpp"

#include "../../nvidia_common.cuh"
#include "../../../utils.hpp"

#include <cuda_runtime.h>

namespace llaisys::ops::nvidia {

// ----------------------------------------------------------------------------
// m=1（decode）专用 GEMV 快路径 —— warp 协作版
// ----------------------------------------------------------------------------
// decode 场景下 196 个 linear 几乎全是 m=1 的矩阵-向量乘（GEMV）。
// GEMV 的瓶颈是 weight 的全局带宽（w[n,k] 全读一遍）+ K 维串行累加的算力。
//
// 【优化策略：warp 协作 + K 维分块归约】
//   * x 向量（k 个元素）加载到 shared memory，所有线程共享复用
//   * 每个 warp（32 线程）协作算【一个】输出 j：32 线程沿 K 维跨步分段累加，
//     再用 __shfl_down_sync 做 warp 内归约求和 → 每线程只算 K/32 次乘加
//   * 一个 block 有 WARPS_PER_BLOCK 个 warp，每 block 算 WARPS_PER_BLOCK 个 j
//   * weight 按行连续读（coalesced），每元素只读一次
//
// 相对「每线程独立算一个 j」的优势：K 维并行化，串行乘加从 K 降到 K/32，
//   算力利用率提升约 32 倍（在算力受限时显著，带宽受限时收益小但仍不亏）。
//
// 精度：每 warp 内 32 段 f32 累加 + shuffle 归约，累加顺序与串行不同，
//   但 bf16 输出精度下结果一致（与 PyTorch/CPU 在 token 级对齐，已实测）。
constexpr int GEMV_WARP_SIZE = 32;
constexpr int WARPS_PER_BLOCK = 8;          // 每 block 8 个 warp = 256 线程
constexpr int GEMV_THREADS = WARPS_PER_BLOCK * GEMV_WARP_SIZE;

// warp 内归约求和（32 lane → lane 0）
__device__ inline float warp_reduce_sum_gemv(float val) {
    val += __shfl_down_sync(0xFFFFFFFFu, val, 16);
    val += __shfl_down_sync(0xFFFFFFFFu, val, 8);
    val += __shfl_down_sync(0xFFFFFFFFu, val, 4);
    val += __shfl_down_sync(0xFFFFFFFFu, val, 2);
    val += __shfl_down_sync(0xFFFFFFFFu, val, 1);
    return val;
}

template <typename T>
__global__ void linear_gemv_kernel(T *out, const T *x, const T *w, const T *bias,
                                   size_t n, size_t k) {
    extern __shared__ char smem_raw[];
    T *x_vec = reinterpret_cast<T *>(smem_raw); // k 个 T

    const int tid = threadIdx.x;
    const int lane = tid % GEMV_WARP_SIZE;
    const int warp_id = tid / GEMV_WARP_SIZE;
    // 本 block 负责的输出列：每个 warp 一个 j
    const size_t j = (size_t)blockIdx.x * WARPS_PER_BLOCK + warp_id;

    // 协作加载 x 向量到 smem
    for (size_t p = tid; p < k; p += GEMV_THREADS) {
        x_vec[p] = x[p];
    }
    __syncthreads();

    if (j >= n) return;

    // 本 warp 协作算 out[j] = Σ_p x[p]·w[j][p]
    // 每 lane 跨步处理 K 维的一段（步长 32），各自累加
    const T *w_row = w + j * k;
    float acc = 0.0f;
    for (size_t p = lane; p < k; p += GEMV_WARP_SIZE) {
        acc += to_f32(x_vec[p]) * to_f32(w_row[p]);
    }
    // warp 内归约求和
    acc = warp_reduce_sum_gemv(acc);
    // lane 0 写回结果（加 bias）
    if (lane == 0) {
        if (bias != nullptr) {
            acc += to_f32(bias[j]);
        }
        out[j] = from_f32<T>(acc);
    }
}

// m 较小（如 prefill qlen=2，m=2）时也用类似 GEMV 的逐行处理，
// 但每行独立。为简单起见，m<=GEMV_M_THRESHOLD 时循环调用 GEMV（每行一次）。
// 实际 prefill 只 2 步，开销可忽略，重点优化 m=1。
constexpr size_t GEMV_M_THRESHOLD = 8;

// 前向声明：tiling GEMM 路径（定义在下方），GEMV 的 smem 超限回退时会用到
template <typename T>
void launch_linear_tiling(T *out, const T *x, const T *w, const T *bias, size_t m, size_t n,
                          size_t k);

template <typename T>
void launch_linear_gemv(T *out, const T *x, const T *w, const T *bias, size_t m, size_t n,
                        size_t k) {
    // 每 block 算 WARPS_PER_BLOCK 个输出列（每 warp 一个 j）
    size_t blocks = (n + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
    size_t smem = k * sizeof(T);
    // smem 上限保护：k 过大（f32 时 > 16K 元素）回退 tiling。
    // Qwen2-1.5B 最大 k=8960（bf16 18KB / f32 35KB），均在 48KB 内安全。
    if (smem > 40 * 1024) { // 留 8KB 余量
        launch_linear_tiling<T>(out, x, w, bias, m, n, k);
        return;
    }
    // 对 m 的每一行各启动一次 GEMV（m 很小，开销可忽略）
    for (size_t i = 0; i < m; ++i) {
        linear_gemv_kernel<T><<<blocks, GEMV_THREADS, smem>>>(
            out + i * n, x + i * k, w, bias, n, k);
    }
}

// ----------------------------------------------------------------------------
// 分块参数（m 较大时用 tiling GEMM）
// ----------------------------------------------------------------------------
// BM=64, BN=64, BK=8：一个 block 负责 out 的 [64×64] 子块，沿 K 每次推进 8。
//   每线程累加器 TM×TN = 8×8（每线程管 64 个输出）。
//   block 线程数 = (BM/TM) × (BN/TN) = 8 × 8 = 64。
//   smem 占用（bf16）= (64×8 + 64×8) × 2B = 2048B ≈ 2KB，远低于 sm_80 的 48KB 上限。
//
// 选 64×64 输出子块的原因：
//   Qwen2-1.5B 的 GEMM 形状（m=1~seq, n=1536/8960/151936, k=1536/8960），
//   子块 64×64 对 n 维利用率高，对 m 维（decode 时 m=1）会用上 m-padding 分支，
//   但 m=1 时只有 1 个 block 行，整体仍是带宽受限，tiling 收益来自 n×k 的复用。
constexpr int BM = 64;
constexpr int BN = 64;
constexpr int BK = 8;
constexpr int TM = 8; // 每线程 M 维输出数
constexpr int TN = 8; // 每线程 N 维输出数
constexpr int TILING_THREADS = (BM / TM) * (BN / TN); // = 64

// ----------------------------------------------------------------------------
// 阶段 A：分块 GEMM kernel
// ----------------------------------------------------------------------------
// 输入布局：
//   x[m, k]：行主序，x[i][p] = x[i*k + p]
//   w[n, k]：行主序（weight 未转置），w[j][p] = w[j*k + p]
//   bias[n]（可选，nullptr 表示无 bias）
// 输出：out[m, n]，行主序
template <typename T>
__global__ void linear_tiling_kernel(T *out, const T *x, const T *w, const T *bias,
                                     size_t m, size_t n, size_t k) {
    // shared memory 分片：x tile [BM][BK] 与 w tile [BN][BK]
    __shared__ T xs[BM][BK];
    __shared__ T ws[BN][BK];

    // 本 block 负责的输出子块起点（全局坐标）
    const size_t bm_base = (size_t)blockIdx.x * BM;
    const size_t bn_base = (size_t)blockIdx.y * BN;

    // 本线程在 8×8 线程网格里的坐标
    // threadIdx.x ∈ [0, 64)，分解为 (tx ∈ [0,8), ty ∈ [0,8))
    const int tx = threadIdx.x / (BN / TN); // M 维线程索引（每线程跳 TM 行）
    const int ty = threadIdx.x % (BN / TN); // N 维线程索引（每线程跳 TN 列）

    // 每线程的 TM×TN 个 f32 累加器
    float acc[TM][TN];
    for (int i = 0; i < TM; ++i)
        for (int j = 0; j < TN; ++j)
            acc[i][j] = 0.0f;

    // 沿 K 维每次 BK 推进
    for (size_t bk = 0; bk < k; bk += BK) {
        // ---- 协作加载 x tile [BM, BK] ----
        // 共 BM*BK=512 元素，64 线程协作 → 每线程 8 元素，跨步 THREADS
        // 越界（m 不被 BM 整除）的位置填 0，保证乘加不影响结果
        for (int i = threadIdx.x; i < BM * BK; i += TILING_THREADS) {
            int r = i / BK;     // tile 内行号 [0, BM)
            int c = i % BK;     // tile 内列号 [0, BK)
            size_t gi = bm_base + (size_t)r;
            size_t gp = bk + (size_t)c;
            xs[r][c] = (gi < m && gp < k) ? x[gi * k + gp] : T(0);
        }
        // ---- 协作加载 w tile [BN, BK] ----
        // w 行主序 [n, k]，w[j][p] = w[j*k + p]，子块连续行加载，coalesced
        for (int i = threadIdx.x; i < BN * BK; i += TILING_THREADS) {
            int r = i / BK;     // tile 内行号 [0, BN) → 对应 n 维
            int c = i % BK;     // tile 内列号 [0, BK) → 对应 k 维
            size_t gj = bn_base + (size_t)r;
            size_t gp = bk + (size_t)c;
            ws[r][c] = (gj < n && gp < k) ? w[gj * k + gp] : T(0);
        }
        __syncthreads();

        // ---- 在这个 BK 分片上做乘加 ----
        // 每线程负责输出子块内 [tx*TM..tx*TM+TM, ty*TN..ty*TN+TN] 的累加
        // 先把本线程关心的 TM 个 x 行、TN 个 w 行的 BK 列读进寄存器，
        // 再做 TM×TN 外积累加（编译器可展开，ILP 充分）
        for (int p = 0; p < BK; ++p) {
            float xv[TM];
            for (int i = 0; i < TM; ++i) {
                xv[i] = to_f32(xs[tx * TM + i][p]);
            }
            float wv[TN];
            for (int j = 0; j < TN; ++j) {
                wv[j] = to_f32(ws[ty * TN + j][p]);
            }
            for (int i = 0; i < TM; ++i)
                for (int j = 0; j < TN; ++j)
                    acc[i][j] += xv[i] * wv[j];
        }
        __syncthreads();
    }

    // ---- 写回输出（加 bias）----
    // 越界位置（m 或 n 不被块大小整除）跳过写
    for (int i = 0; i < TM; ++i) {
        for (int j = 0; j < TN; ++j) {
            size_t gi = bm_base + (size_t)(tx * TM + i);
            size_t gj = bn_base + (size_t)(ty * TN + j);
            if (gi < m && gj < n) {
                float val = acc[i][j];
                if (bias != nullptr) {
                    val += to_f32(bias[gj]);
                }
                out[gi * n + gj] = from_f32<T>(val);
            }
        }
    }
}

template <typename T>
void launch_linear_tiling(T *out, const T *x, const T *w, const T *bias, size_t m, size_t n,
                          size_t k) {
    dim3 grid((m + BM - 1) / BM, (n + BN - 1) / BN, 1);
    dim3 block(TILING_THREADS, 1, 1);
    linear_tiling_kernel<T><<<grid, block>>>(out, x, w, bias, m, n, k);
}


void linear(std::byte *out, const std::byte *in, const std::byte *weight, const std::byte *bias,
            llaisysDataType_t type, size_t m, size_t n, size_t k) {
    // 按 m 分流：m 小（decode/prefill）走 GEMV 快路径；m 大走 tiling GEMM。
    // decode 时几乎所有 linear 都是 m=1，GEMV 远快于 tiling（无 padding 浪费）。
    const bool use_gemv = (m <= GEMV_M_THRESHOLD);
    switch (type) {
    case LLAISYS_DTYPE_F32:
        if (use_gemv)
            return launch_linear_gemv(reinterpret_cast<float *>(out),
                                      reinterpret_cast<const float *>(in),
                                      reinterpret_cast<const float *>(weight),
                                      reinterpret_cast<const float *>(bias), m, n, k);
        return launch_linear_tiling(reinterpret_cast<float *>(out),
                                    reinterpret_cast<const float *>(in),
                                    reinterpret_cast<const float *>(weight),
                                    reinterpret_cast<const float *>(bias), m, n, k);
    case LLAISYS_DTYPE_F16:
        if (use_gemv)
            return launch_linear_gemv(reinterpret_cast<__half *>(out),
                                      reinterpret_cast<const __half *>(in),
                                      reinterpret_cast<const __half *>(weight),
                                      reinterpret_cast<const __half *>(bias), m, n, k);
        return launch_linear_tiling(reinterpret_cast<__half *>(out),
                                    reinterpret_cast<const __half *>(in),
                                    reinterpret_cast<const __half *>(weight),
                                    reinterpret_cast<const __half *>(bias), m, n, k);
    case LLAISYS_DTYPE_BF16:
        if (use_gemv)
            return launch_linear_gemv(reinterpret_cast<__nv_bfloat16 *>(out),
                                      reinterpret_cast<const __nv_bfloat16 *>(in),
                                      reinterpret_cast<const __nv_bfloat16 *>(weight),
                                      reinterpret_cast<const __nv_bfloat16 *>(bias), m, n, k);
        return launch_linear_tiling(reinterpret_cast<__nv_bfloat16 *>(out),
                                    reinterpret_cast<const __nv_bfloat16 *>(in),
                                    reinterpret_cast<const __nv_bfloat16 *>(weight),
                                    reinterpret_cast<const __nv_bfloat16 *>(bias), m, n, k);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::nvidia
