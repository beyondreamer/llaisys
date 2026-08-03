// ============================================================================
// embedding/op.cpp — 算子调度层（Dispatcher）
// ----------------------------------------------------------------------------
// 职责（三层结构的第一层，另两层在 cpu/ 子目录）：
//   1. 校验：设备一致 / dtype 一致 / 形状约束 / 连续性
//   2. 分发：CPU 直接走 cpu:: 实现；其他设备（A4 的 NVIDIA）走对应分支
// 设计意义：上层（Python 调用方）和下层（具体设备实现）都只依赖这一层，
// 新增设备时只需在这里补一个 case，不用动调用方。
//
// 算子语义：查表：out[i,:] = weight[index[i],:]
// 公式：out[i, j] = weight[index[i], j]
// 注意点：
//   * index 必须是 i64 类型（1D）
//   * 整行 memcpy 保证与 PyTorch 严格一致（测试用 strict=True）
//   * weight 是 2D [vocab, d]，out 是 2D [nidx, d]
// ============================================================================
#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/embedding_cpu.hpp"
#include "nvidia/embedding_nvidia.hpp"

namespace llaisys::ops {
void embedding(tensor_t out, tensor_t index, tensor_t weight) {
    CHECK_SAME_DEVICE(out, index, weight);
    CHECK_SAME_DTYPE(out->dtype(), weight->dtype());
    CHECK_ARGUMENT(index->dtype() == LLAISYS_DTYPE_I64, "embedding: index must be i64");
    CHECK_ARGUMENT(index->ndim() == 1 && weight->ndim() == 2 && out->ndim() == 2,
                   "embedding: expected index(1D), weight(2D), out(2D)");
    CHECK_ARGUMENT(out->shape()[0] == index->numel() && out->shape()[1] == weight->shape()[1],
                   "embedding: shape mismatch");
    ASSERT(out->isContiguous() && index->isContiguous() && weight->isContiguous(),
           "embedding: all tensors must be contiguous.");

    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::embedding(out->data(), index->data(), weight->data(),
                              out->dtype(), index->numel(), weight->shape()[1]);
    }

    llaisys::core::context().setDevice(out->deviceType(), out->deviceId());
    switch (out->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::embedding(out->data(), index->data(), weight->data(),
                              out->dtype(), index->numel(), weight->shape()[1]);
#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA:
        // A4：CUDA 实现（英伟达/国产兼容平台通用）
        return nvidia::embedding(out->data(), index->data(), weight->data(),
                                  out->dtype(), index->numel(), weight->shape()[1]);
#endif
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}
} // namespace llaisys::ops
