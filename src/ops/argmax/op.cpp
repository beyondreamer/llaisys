// ============================================================================
// argmax/op.cpp — 算子调度层（Dispatcher）
// ----------------------------------------------------------------------------
// 职责（三层结构的第一层，另两层在 cpu/ 子目录）：
//   1. 校验：设备一致 / dtype 一致 / 形状约束 / 连续性
//   2. 分发：CPU 直接走 cpu:: 实现；其他设备（A4 的 NVIDIA）走对应分支
// 设计意义：上层（Python 调用方）和下层（具体设备实现）都只依赖这一层，
// 新增设备时只需在这里补一个 case，不用动调用方。
//
// 算子语义：1D 张量求最大值及其下标
// 公式：max_val = max(vals[i]); max_idx = argmax_i vals[i]
// 注意点：
//   * max_idx 必须是 i64 类型（测试与 PyTorch 的 max() 对照）
//   * f16/bf16 比较时先 cast 到 float，避免低精度比较出错
//   * 输入必须是 1D 且非空
// ============================================================================
#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/argmax_cpu.hpp"
#include "nvidia/argmax_nvidia.hpp"

namespace llaisys::ops {
void argmax(tensor_t max_idx, tensor_t max_val, tensor_t vals) {
    CHECK_SAME_DEVICE(max_idx, max_val, vals);
    CHECK_SAME_DTYPE(max_val->dtype(), vals->dtype());
    CHECK_ARGUMENT(max_idx->dtype() == LLAISYS_DTYPE_I64, "argmax: max_idx must be i64");
    CHECK_ARGUMENT(vals->ndim() == 1 && vals->numel() > 0, "argmax: vals must be a non-empty 1D tensor");
    CHECK_ARGUMENT(max_idx->numel() == 1 && max_val->numel() == 1, "argmax: outputs must have exactly 1 element");
    ASSERT(max_idx->isContiguous() && max_val->isContiguous() && vals->isContiguous(),
           "argmax: all tensors must be contiguous.");

    if (vals->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::argmax(max_idx->data(), max_val->data(), vals->data(),
                           vals->dtype(), vals->numel());
    }

    llaisys::core::context().setDevice(vals->deviceType(), vals->deviceId());
    switch (vals->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::argmax(max_idx->data(), max_val->data(), vals->data(),
                           vals->dtype(), vals->numel());
#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA:
        // A4：CUDA 实现（英伟达/国产兼容平台通用）
        return nvidia::argmax(max_idx->data(), max_val->data(), vals->data(),
                              vals->dtype(), vals->numel());
#endif
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}
} // namespace llaisys::ops
