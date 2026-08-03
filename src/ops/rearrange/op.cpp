// ============================================================================
// rearrange/op.cpp — 算子调度层（Dispatcher）
// ----------------------------------------------------------------------------
// 职责（三层结构的第一层，另两层在 cpu/ 子目录）：
//   1. 校验：设备一致 / dtype 一致 / 形状约束 / 连续性
//   2. 分发：CPU 直接走 cpu:: 实现；其他设备（A4 的 NVIDIA）走对应分支
// 设计意义：上层（Python 调用方）和下层（具体设备实现）都只依赖这一层，
// 新增设备时只需在这里补一个 case，不用动调用方。
//
// 算子语义：视图重排：[seq, nh·dh] ⟷ [seq, nh, dh]（A3 推理必需，README 未列出的隐藏算子）
// 公式：纯内存布局视角转换，连续张量下等价于 view
// 注意点：
//   * 两个张量都要求连续且元素数一致
//   * 连续时直接 memcpy（等价于 view 的语义：把同一块内存换个形状解释）
//   * 没有官方测试文件，但 A3 的 q/k/v 投影和 attention 输出都要用它
// ============================================================================
#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/rearrange_cpu.hpp"
#include "nvidia/rearrange_nvidia.hpp"

namespace llaisys::ops {
// Reshape [seq, nhead*dh] <-> [seq, nhead, dh] without data movement:
// both views of the same contiguous memory are layout-identical.
void rearrange(tensor_t out, tensor_t in) {
    CHECK_SAME_DEVICE(out, in);
    CHECK_SAME_DTYPE(out->dtype(), in->dtype());
    CHECK_ARGUMENT(out->numel() == in->numel(), "rearrange: element count must match");
    ASSERT(out->isContiguous() && in->isContiguous(), "rearrange: all tensors must be contiguous.");

    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::rearrange(out->data(), in->data(), out->numel() * out->elementSize());
    }

    llaisys::core::context().setDevice(out->deviceType(), out->deviceId());
    switch (out->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::rearrange(out->data(), in->data(), out->numel() * out->elementSize());
#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA:
        // A4：CUDA 实现（英伟达/国产兼容平台通用）
        return nvidia::rearrange(out->data(), in->data(),
                                  out->numel() * out->elementSize());
#endif
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}
} // namespace llaisys::ops
