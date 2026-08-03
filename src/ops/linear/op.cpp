// ============================================================================
// linear/op.cpp — 算子调度层（Dispatcher）
// ----------------------------------------------------------------------------
// 职责（三层结构的第一层，另两层在 cpu/ 子目录）：
//   1. 校验：设备一致 / dtype 一致 / 形状约束 / 连续性
//   2. 分发：CPU 直接走 cpu:: 实现；其他设备（A4 的 NVIDIA）走对应分支
// 设计意义：上层（Python 调用方）和下层（具体设备实现）都只依赖这一层，
// 新增设备时只需在这里补一个 case，不用动调用方。
//
// 算子语义：全连接层：Y = X·Wᵀ + b
// 公式：out[i][j] = Σ_p in[i][p] * weight[j][p] + bias[j]
// 注意点：
//   * ★ weight 未转置：按 W 的【行】取（weight 形状是 [n, k]），不是按列
//   * f16/bf16 内积必须 float 累加，最后再 cast 回 T（精度关键）
//   * bias 可选（A3 的 o_proj/mlp 都没有 bias，C 层传空指针）
// ============================================================================
#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/linear_cpu.hpp"
#include "nvidia/linear_nvidia.hpp"

namespace llaisys::ops {
void linear(tensor_t out, tensor_t in, tensor_t weight, tensor_t bias) {
    CHECK_SAME_DEVICE(out, in, weight);
    if (bias != nullptr) {
        CHECK_SAME_DEVICE(out, bias);
    }
    CHECK_SAME_DTYPE(out->dtype(), in->dtype(), weight->dtype());
    if (bias != nullptr) {
        CHECK_SAME_DTYPE(out->dtype(), bias->dtype());
    }
    CHECK_ARGUMENT(in->ndim() == 2 && weight->ndim() == 2 && out->ndim() == 2,
                   "linear: expected 2D in/weight/out");
    // out = (m, n), in = (m, k), weight = (n, k)
    const size_t m = in->shape()[0];
    const size_t k = in->shape()[1];
    const size_t n = weight->shape()[0];
    CHECK_ARGUMENT(weight->shape()[1] == k, "linear: weight inner dim must match in");
    CHECK_ARGUMENT(out->shape()[0] == m && out->shape()[1] == n, "linear: out shape mismatch");
    if (bias != nullptr) {
        CHECK_ARGUMENT(bias->numel() == n, "linear: bias length must equal out width");
    }
    ASSERT(out->isContiguous() && in->isContiguous() && weight->isContiguous(),
           "linear: all tensors must be contiguous.");
    if (bias != nullptr) {
        ASSERT(bias->isContiguous(), "linear: bias must be contiguous.");
    }

    const std::byte *bias_data = bias != nullptr ? bias->data() : nullptr;
    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::linear(out->data(), in->data(), weight->data(), bias_data,
                           out->dtype(), m, n, k);
    }

    llaisys::core::context().setDevice(out->deviceType(), out->deviceId());
    switch (out->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::linear(out->data(), in->data(), weight->data(), bias_data,
                           out->dtype(), m, n, k);
#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA:
        // A4：CUDA 实现（英伟达/国产兼容平台通用）
        return nvidia::linear(out->data(), in->data(), weight->data(), bias_data,
                               out->dtype(), m, n, k);
#endif
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}
} // namespace llaisys::ops
