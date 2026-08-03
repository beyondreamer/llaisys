// ============================================================================
// src/device/nvidia/nvidia_runtime_api.cu — NVIDIA/国产 CUDA 兼容设备 Runtime API
// ----------------------------------------------------------------------------
// 职责：实现 LlaisysRuntimeAPI 函数指针表里的 12 个函数（Assignment #4）。
// 参照 src/device/cpu/cpu_runtime_api.cpp 的结构，底层换成 CUDA 调用。
//
// 兼容性说明：
//   * 全部使用标准 CUDA Runtime API（cudaMalloc/cudaMemcpy/cudaStream...），
//     在英伟达上用 nvcc 编译；国产兼容平台（如沐曦 MACA）可直接用其
//     CUDA 兼容工具链编译同一份代码。
//   * get_device_count 在无 CUDA 设备/驱动时返回 0 而不是抛异常——
//     Context 构造时会枚举所有设备类型，本机没有 GPU 也必须能正常初始化。
//   * memcpyAsync 的签名必须带 stream 参数（官方桩少写了一个参数，
//     会导致函数指针类型不匹配编译失败，这里已修正）。
// ============================================================================
#include "../runtime_api.hpp"

#include <cuda_runtime.h>

#include <cstdlib>
#include <cstring>
#include <iostream>
#include <stdexcept>

namespace llaisys::device::nvidia {

namespace runtime_api {

// 检查 CUDA 调用是否成功，失败则打印错误信息并抛异常（与框架其他层一致）。
#define CHECK_CUDA(call)                                                                     \
    do {                                                                                     \
        cudaError_t err_ = (call);                                                           \
        if (err_ != cudaSuccess) {                                                           \
            std::cerr << "[ERROR] CUDA: " << cudaGetErrorString(err_) << " at " << __FILE__  \
                      << ":" << __LINE__ << std::endl;                                       \
            throw std::runtime_error("CUDA error");                                          \
        }                                                                                    \
    } while (0)

// 把框架的 llaisysMemcpyKind_t 映射成 CUDA 的 cudaMemcpyKind。
static cudaMemcpyKind to_cuda_kind(llaisysMemcpyKind_t kind) {
    switch (kind) {
    case LLAISYS_MEMCPY_H2H:
        return cudaMemcpyHostToHost;
    case LLAISYS_MEMCPY_H2D:
        return cudaMemcpyHostToDevice;
    case LLAISYS_MEMCPY_D2H:
        return cudaMemcpyDeviceToHost;
    case LLAISYS_MEMCPY_D2D:
        return cudaMemcpyDeviceToDevice;
    default:
        throw std::invalid_argument("Invalid memcpy kind");
    }
}

int getDeviceCount() {
    int count = 0;
    // 没有 GPU/驱动时 cudaGetDeviceCount 会报错：这里优雅降级返回 0，
    // 让 Context 初始化跳过 NVIDIA（本机无卡也能用 CPU 跑）。
    cudaError_t err = cudaGetDeviceCount(&count);
    if (err != cudaSuccess) {
        return 0;
    }
    return count;
}

void setDevice(int device_id) {
    CHECK_CUDA(cudaSetDevice(device_id));
}

void deviceSynchronize() {
    CHECK_CUDA(cudaDeviceSynchronize());
}

llaisysStream_t createStream() {
    cudaStream_t stream = nullptr;
    CHECK_CUDA(cudaStreamCreate(&stream));
    return reinterpret_cast<llaisysStream_t>(stream);
}

void destroyStream(llaisysStream_t stream) {
    if (stream != nullptr) {
        CHECK_CUDA(cudaStreamDestroy(reinterpret_cast<cudaStream_t>(stream)));
    }
}

void streamSynchronize(llaisysStream_t stream) {
    if (stream != nullptr) {
        CHECK_CUDA(cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(stream)));
    }
}

void *mallocDevice(size_t size) {
    void *ptr = nullptr;
    CHECK_CUDA(cudaMalloc(&ptr, size));
    return ptr;
}

void freeDevice(void *ptr) {
    CHECK_CUDA(cudaFree(ptr));
}

// 页锁定内存（pinned memory）：H2D/D2H 拷贝更快（A4 性能要点）。
void *mallocHost(size_t size) {
    void *ptr = nullptr;
    CHECK_CUDA(cudaMallocHost(&ptr, size));
    return ptr;
}

void freeHost(void *ptr) {
    CHECK_CUDA(cudaFreeHost(ptr));
}

void memcpySync(void *dst, const void *src, size_t size, llaisysMemcpyKind_t kind) {
    CHECK_CUDA(cudaMemcpy(dst, src, size, to_cuda_kind(kind)));
}

// 异步拷贝：把 memcpy 提交到指定 stream，不阻塞 host。
// ★ 注意必须带 stream 参数（官方桩漏了，这里修正为 5 个参数）。
void memcpyAsync(void *dst, const void *src, size_t size, llaisysMemcpyKind_t kind,
                 llaisysStream_t stream) {
    CHECK_CUDA(cudaMemcpyAsync(dst, src, size, to_cuda_kind(kind),
                               reinterpret_cast<cudaStream_t>(stream)));
}

static const LlaisysRuntimeAPI RUNTIME_API = {
    &getDeviceCount,
    &setDevice,
    &deviceSynchronize,
    &createStream,
    &destroyStream,
    &streamSynchronize,
    &mallocDevice,
    &freeDevice,
    &mallocHost,
    &freeHost,
    &memcpySync,
    &memcpyAsync};

} // namespace runtime_api

const LlaisysRuntimeAPI *getRuntimeAPI() {
    return &runtime_api::RUNTIME_API;
}
} // namespace llaisys::device::nvidia
