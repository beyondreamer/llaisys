// ============================================================================
// self_attention/op.cpp — 算子调度层（Dispatcher）
// ----------------------------------------------------------------------------
// 职责（三层结构的第一层，另两层在 cpu/ 子目录）：
//   1. 校验：设备一致 / dtype 一致 / 形状约束 / 连续性
//   2. 分发：CPU 直接走 cpu:: 实现；其他设备（A4 的 NVIDIA）走对应分支
// 设计意义：上层（Python 调用方）和下层（具体设备实现）都只依赖这一层，
// 新增设备时只需在这里补一个 case，不用动调用方。
//
// 算子语义：带因果掩码的 Grouped Query Attention
// 公式：attn = softmax(Q·Kᵀ·scale + mask) · V
// 注意点：
//   * ★ GQA：query 头 h 使用 kv 头 h/groups（groups=nh/nkvh），对应 PyTorch repeat_interleave
//   * ★ causal mask 对角线 = kvlen−qlen：query i 只能 attend 到 key j ≤ i+(kvlen−qlen)；qlen==1 生成时能看到全部历史
//   * softmax 必须 float32 + max-subtract（数值稳定，且与 PyTorch 的 softmax(dtype=float32) 对齐）
//   * score 在 float 下累加再乘 scale
// ============================================================================
#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/self_attention_cpu.hpp"
#include "nvidia/self_attention_nvidia.hpp"

namespace llaisys::ops {
void self_attention(tensor_t attn_val, tensor_t q, tensor_t k, tensor_t v, float scale) {
    CHECK_SAME_DEVICE(attn_val, q, k, v);
    CHECK_SAME_DTYPE(attn_val->dtype(), q->dtype(), k->dtype(), v->dtype());
    CHECK_ARGUMENT(q->ndim() == 3 && k->ndim() == 3 && v->ndim() == 3 && attn_val->ndim() == 3,
                   "self_attention: expected 3D q/k/v/attn_val");
    const size_t qlen = q->shape()[0];
    const size_t nh = q->shape()[1];
    const size_t d = q->shape()[2];
    const size_t kvlen = k->shape()[0];
    const size_t nkvh = k->shape()[1];
    CHECK_ARGUMENT(q->shape()[2] == k->shape()[2] && k->shape()[2] == v->shape()[2],
                   "self_attention: head dims must match");
    CHECK_ARGUMENT(k->shape()[0] == v->shape()[0] && k->shape()[1] == v->shape()[1],
                   "self_attention: k/v shapes must match");
    CHECK_ARGUMENT(nh % nkvh == 0, "self_attention: nh must be divisible by nkvh");
    CHECK_ARGUMENT(attn_val->shape()[0] == qlen && attn_val->shape()[1] == nh &&
                       attn_val->shape()[2] == d,
                   "self_attention: attn_val shape mismatch");
    ASSERT(attn_val->isContiguous() && q->isContiguous() && k->isContiguous() && v->isContiguous(),
           "self_attention: all tensors must be contiguous.");

    if (attn_val->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::self_attention(attn_val->data(), q->data(), k->data(), v->data(),
                                   attn_val->dtype(), qlen, kvlen, nh, nkvh, d, scale);
    }

    llaisys::core::context().setDevice(attn_val->deviceType(), attn_val->deviceId());
    switch (attn_val->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::self_attention(attn_val->data(), q->data(), k->data(), v->data(),
                                   attn_val->dtype(), qlen, kvlen, nh, nkvh, d, scale);
#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA:
        // A4：CUDA 实现（英伟达/国产兼容平台通用）
        return nvidia::self_attention(attn_val->data(), q->data(), k->data(),
                                        v->data(), attn_val->dtype(), qlen,
                                        kvlen, nh, nkvh, d, scale);
#endif
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}
} // namespace llaisys::ops
