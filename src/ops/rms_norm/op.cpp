// ============================================================================
// rms_norm/op.cpp — 算子调度层（Dispatcher）
// ----------------------------------------------------------------------------
// 职责（三层结构的第一层，另两层在 cpu/ 子目录）：
//   1. 校验：设备一致 / dtype 一致 / 形状约束 / 连续性
//   2. 分发：CPU 直接走 cpu:: 实现；其他设备（A4 的 NVIDIA）走对应分支
// 设计意义：上层（Python 调用方）和下层（具体设备实现）都只依赖这一层，
// 新增设备时只需在这里补一个 case，不用动调用方。
//
// 算子语义：RMS 归一化（Qwen2 用的不是 LayerNorm 而是 RMSNorm）
// 公式：out[i][j] = x[i][j] / sqrt(mean_j(x[i][j]²) + eps) * w[j]
// 注意点：
//   * 沿最后一维归一化，weight 是 1D 长度 = 行宽
//   * 两遍循环：先求 sum_sq，再算 rms 并缩放；float 累加
//   * eps 来自模型配置（Qwen2 用 1e-6）
// ============================================================================
#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/rms_norm_cpu.hpp"
#include "nvidia/rms_norm_nvidia.hpp"

namespace llaisys::ops {
void rms_norm(tensor_t out, tensor_t in, tensor_t weight, float eps) {
    CHECK_SAME_DEVICE(out, in, weight);
    CHECK_SAME_DTYPE(out->dtype(), in->dtype(), weight->dtype());
    CHECK_ARGUMENT(in->ndim() == 2 && out->ndim() == 2 && weight->ndim() == 1,
                   "rms_norm: expected in(2D), out(2D), weight(1D)");
    CHECK_ARGUMENT(out->shape() == in->shape(), "rms_norm: out shape must match in");
    CHECK_ARGUMENT(weight->numel() == in->shape()[1], "rms_norm: weight length must match last dim");
    ASSERT(out->isContiguous() && in->isContiguous() && weight->isContiguous(),
           "rms_norm: all tensors must be contiguous.");

    const size_t rows = in->shape()[0];
    const size_t d = in->shape()[1];
    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::rms_norm(out->data(), in->data(), weight->data(),
                             out->dtype(), rows, d, eps);
    }

    llaisys::core::context().setDevice(out->deviceType(), out->deviceId());
    switch (out->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::rms_norm(out->data(), in->data(), weight->data(),
                             out->dtype(), rows, d, eps);
#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA:
        // A4：CUDA 实现（英伟达/国产兼容平台通用）
        return nvidia::rms_norm(out->data(), in->data(), weight->data(),
                                  out->dtype(), rows, d, eps);
#endif
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}
} // namespace llaisys::ops
