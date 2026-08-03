// ============================================================================
// rope/op.cpp — 算子调度层（Dispatcher）
// ----------------------------------------------------------------------------
// 职责（三层结构的第一层，另两层在 cpu/ 子目录）：
//   1. 校验：设备一致 / dtype 一致 / 形状约束 / 连续性
//   2. 分发：CPU 直接走 cpu:: 实现；其他设备（A4 的 NVIDIA）走对应分支
// 设计意义：上层（Python 调用方）和下层（具体设备实现）都只依赖这一层，
// 新增设备时只需在这里补一个 case，不用动调用方。
//
// 算子语义：旋转位置编码（Rotary Position Embedding）
// 公式：φ = pos / θ^(2j/d)；out[j]=a·cosφ−b·sinφ；out[j+d/2]=b·cosφ+a·sinφ（a=x[:d/2], b=x[d/2:]）
// 注意点：
//   * d（head_dim）必须为偶数
//   * 角度频率与 PyTorch 完全一致：pos_ids 转 float32 再除以 θ^(2j/d)
//   * pos_ids 是 i64 1D，长度 = seq；大测试用例验证非零起点（pos 512..1023）
// ============================================================================
#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/rope_cpu.hpp"
#include "nvidia/rope_nvidia.hpp"

namespace llaisys::ops {
void rope(tensor_t out, tensor_t in, tensor_t pos_ids, float theta) {
    CHECK_SAME_DEVICE(out, in, pos_ids);
    CHECK_SAME_DTYPE(out->dtype(), in->dtype());
    CHECK_ARGUMENT(pos_ids->dtype() == LLAISYS_DTYPE_I64, "rope: pos_ids must be i64");
    CHECK_ARGUMENT(in->ndim() == 3 && out->ndim() == 3 && pos_ids->ndim() == 1,
                   "rope: expected in/out(3D), pos_ids(1D)");
    CHECK_ARGUMENT(in->shape() == out->shape(), "rope: out shape must match in");
    CHECK_ARGUMENT(in->shape()[2] % 2 == 0, "rope: head dim must be even");
    CHECK_ARGUMENT(pos_ids->numel() == in->shape()[0], "rope: pos_ids length must equal seq");
    ASSERT(out->isContiguous() && in->isContiguous() && pos_ids->isContiguous(),
           "rope: all tensors must be contiguous.");

    const size_t seq = in->shape()[0];
    const size_t nh = in->shape()[1];
    const size_t d = in->shape()[2];
    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::rope(out->data(), in->data(), pos_ids->data(),
                         out->dtype(), seq, nh, d, theta);
    }

    llaisys::core::context().setDevice(out->deviceType(), out->deviceId());
    switch (out->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::rope(out->data(), in->data(), pos_ids->data(),
                         out->dtype(), seq, nh, d, theta);
#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA:
        // A4：CUDA 实现（英伟达/国产兼容平台通用）
        return nvidia::rope(out->data(), in->data(), pos_ids->data(),
                              out->dtype(), seq, nh, d, theta);
#endif
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}
} // namespace llaisys::ops
