// ============================================================================
// swiglu/op.cpp — 算子调度层（Dispatcher）
// ----------------------------------------------------------------------------
// 职责（三层结构的第一层，另两层在 cpu/ 子目录）：
//   1. 校验：设备一致 / dtype 一致 / 形状约束 / 连续性
//   2. 分发：CPU 直接走 cpu:: 实现；其他设备（A4 的 NVIDIA）走对应分支
// 设计意义：上层（Python 调用方）和下层（具体设备实现）都只依赖这一层，
// 新增设备时只需在这里补一个 case，不用动调用方。
//
// 算子语义：SwiGLU 激活（Qwen2 MLP 用）
// 公式：out = up * SiLU(gate) = up * gate / (1 + e^(−gate))
// 注意点：
//   * 逐元素运算，三个张量同形状同 dtype
//   * gate 先 cast 到 float 计算 sigmoid，再乘 up，最后 cast 回 T
// ============================================================================
#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/swiglu_cpu.hpp"
#include "nvidia/swiglu_nvidia.hpp"

namespace llaisys::ops {
void swiglu(tensor_t out, tensor_t gate, tensor_t up) {
    CHECK_SAME_DEVICE(out, gate, up);
    CHECK_SAME_DTYPE(out->dtype(), gate->dtype(), up->dtype());
    CHECK_SAME_SHAPE(out->shape(), gate->shape(), up->shape());
    ASSERT(out->isContiguous() && gate->isContiguous() && up->isContiguous(),
           "swiglu: all tensors must be contiguous.");

    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::swiglu(out->data(), gate->data(), up->data(),
                           out->dtype(), out->numel());
    }

    llaisys::core::context().setDevice(out->deviceType(), out->deviceId());
    switch (out->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::swiglu(out->data(), gate->data(), up->data(),
                           out->dtype(), out->numel());
#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA:
        // A4：CUDA 实现（英伟达/国产兼容平台通用）
        return nvidia::swiglu(out->data(), gate->data(), up->data(),
                               out->dtype(), out->numel());
#endif
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}
} // namespace llaisys::ops
