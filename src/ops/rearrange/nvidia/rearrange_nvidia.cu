// ============================================================================
// src/ops/rearrange/nvidia/rearrange_nvidia.cu — 视图重排的 CUDA 实现（A4）
// 连续张量换形状 = 同一块内存换个解释方式，直接 D2D 拷贝即可（等价于 view）。
// ============================================================================
#include "rearrange_nvidia.hpp"

#include <cuda_runtime.h>
#include <iostream>
#include <stdexcept>

namespace llaisys::ops::nvidia {
// 校验 CUDA 调用，失败抛异常（与 runtime api 层同一风格）。
#define CHECK_CUDA(call)                                                                     \
    do {                                                                                     \
        cudaError_t err_ = (call);                                                           \
        if (err_ != cudaSuccess) {                                                           \
            std::cerr << "[ERROR] CUDA: " << cudaGetErrorString(err_) << " at " << __FILE__  \
                      << ":" << __LINE__ << std::endl;                                       \
            throw std::runtime_error("CUDA error");                                          \
        }                                                                                    \
    } while (0)

// 设备到设备的一整块拷贝（out/in 都在显存里）。
void rearrange(std::byte *out, const std::byte *in, size_t bytes) {
    CHECK_CUDA(cudaMemcpy(out, in, bytes, cudaMemcpyDeviceToDevice));
}
} // namespace llaisys::ops::nvidia
