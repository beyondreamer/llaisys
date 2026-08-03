// ============================================================================
// src/models/qwen2/qwen2.cpp — Qwen2 模型层实现（Assignment #3 核心）
// ----------------------------------------------------------------------------
// 本文件完成两件事：
//   1. 构造函数：按 Qwen2Meta 在目标设备上分配全部权重 + 每层 KV cache
//   2. infer()：一次完整前向（prefill 传 n>1 个 token，decode 传 1 个 token），
//      返回下一个 token id；内部按标准 Qwen2 拓扑调用 A2 的算子
//
// 关键点：
//   * KV cache 按"行"写入/读取：写入位置由 kv_len 决定，读取用 slice 保持连续
//   * pos_ids = [kv_len, kv_len+n)，必须用全局位置（不是 0..n-1）
//   * 每个中间张量都临时分配（正确性优先；性能优化留给 A4/项目阶段）
// ============================================================================
#include "qwen2.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../ops/add/op.hpp"
#include "../../ops/argmax/op.hpp"
#include "../../ops/embedding/op.hpp"
#include "../../ops/linear/op.hpp"
#include "../../ops/rearrange/op.hpp"
#include "../../ops/rms_norm/op.hpp"
#include "../../ops/rope/op.hpp"
#include "../../ops/self_attention/op.hpp"
#include "../../ops/swiglu/op.hpp"
#include "../../utils.hpp"

#include <cmath>

namespace llaisys::models {
namespace {
// 设备内拷贝的方向：CPU 上用 H2H，其他设备（NVIDIA）用 D2D。
llaisysMemcpyKind_t device_kind(llaisysDeviceType_t dev) {
    return dev == LLAISYS_DEVICE_CPU ? LLAISYS_MEMCPY_H2H : LLAISYS_MEMCPY_D2D;
}

llaisysMemcpyKind_t to_host_kind(llaisysDeviceType_t dev) {
    return dev == LLAISYS_DEVICE_CPU ? LLAISYS_MEMCPY_H2H : LLAISYS_MEMCPY_D2H;
}

// 在指定设备上做一次 memcpy：先 setDevice 切到目标设备，
// 再通过当前 Runtime 的函数指针表调用 memcpy_sync。
void device_memcpy(std::byte *dst, const std::byte *src, size_t bytes,
                   llaisysDeviceType_t dev, int dev_id, llaisysMemcpyKind_t kind) {
    core::context().setDevice(dev, dev_id);
    core::context().runtime().api()->memcpy_sync(dst, src, bytes, kind);
}
} // namespace

Qwen2Model::Qwen2Model(const Qwen2Meta &meta_, llaisysDeviceType_t device_, int device_id_)
    : meta(meta_), device(device_), device_id(device_id_) {
    const size_t hs = meta.hs;
    const size_t nlayer = meta.nlayer;
    const size_t voc = meta.voc;
    const size_t di = meta.di;
    const size_t nh = meta.nh;
    const size_t nkvh = meta.nkvh;
    const size_t dh = meta.dh;
    const size_t maxseq = meta.maxseq;
    const llaisysDataType_t dtype = meta.dtype;

    // 顶层权重：词嵌入表、lm_head（独立，不与 embedding 绑定）、最终层 RMSNorm 权重
    in_embed = Tensor::create({voc, hs}, dtype, device, device_id);
    out_embed = Tensor::create({voc, hs}, dtype, device, device_id);
    out_norm_w = Tensor::create({hs}, dtype, device, device_id);

    layers.resize(nlayer);
    kv_caches.resize(nlayer);
    for (size_t l = 0; l < nlayer; ++l) {
        auto &w = layers[l];
        // attention 部分：q_proj 输出 nh*dh，k/v_proj 输出 nkvh*dh（GQA！）
        // 每个 linear 都配 bias（DeepSeek-R1-Distill-Qwen 的 q/k/v 有 bias）
        w.attn_norm_w = Tensor::create({hs}, dtype, device, device_id);
        w.attn_q_w = Tensor::create({nh * dh, hs}, dtype, device, device_id);
        w.attn_q_b = Tensor::create({nh * dh}, dtype, device, device_id);
        w.attn_k_w = Tensor::create({nkvh * dh, hs}, dtype, device, device_id);
        w.attn_k_b = Tensor::create({nkvh * dh}, dtype, device, device_id);
        w.attn_v_w = Tensor::create({nkvh * dh, hs}, dtype, device, device_id);
        w.attn_v_b = Tensor::create({nkvh * dh}, dtype, device, device_id);
        w.attn_o_w = Tensor::create({hs, hs}, dtype, device, device_id);
        w.mlp_norm_w = Tensor::create({hs}, dtype, device, device_id);
        w.mlp_gate_w = Tensor::create({di, hs}, dtype, device, device_id);
        w.mlp_up_w = Tensor::create({di, hs}, dtype, device, device_id);
        w.mlp_down_w = Tensor::create({hs, di}, dtype, device, device_id);

        // 每层一份 KV cache：形状 [maxseq, nkvh, dh]，只缓存 nkvh 个头的 K/V（GQA 省内存）
        kv_caches[l].k = Tensor::create({maxseq, nkvh, dh}, dtype, device, device_id);
        kv_caches[l].v = Tensor::create({maxseq, nkvh, dh}, dtype, device, device_id);
    }
}

int64_t Qwen2Model::infer(const std::vector<int64_t> &token_ids) {
    const size_t n = token_ids.size();
    CHECK_ARGUMENT(n > 0, "infer: empty input");
    const llaisysDataType_t dtype = meta.dtype;
    const size_t hs = meta.hs, nh = meta.nh, nkvh = meta.nkvh, dh = meta.dh, voc = meta.voc;
    const size_t di = meta.di;

    // --- 第 1 步：embedding ---
    // 把 token id 装进 i64 张量，查词嵌入表得到 hidden states [n, hs]
    auto index = Tensor::create({n}, LLAISYS_DTYPE_I64, device, device_id);
    index->load(token_ids.data());
    auto hidden = Tensor::create({n, hs}, dtype, device, device_id);
    ops::embedding(hidden, index, in_embed);

    // 位置编码：必须用【全局】位置 [kv_len, kv_len+n)，而不是 0..n-1。
    // 解码阶段 kv_len 不断增长，RoPE 的相位由全局位置决定，用错位置整条序列都会错。
    auto pos_ids = Tensor::create({n}, LLAISYS_DTYPE_I64, device, device_id);
    {
        std::vector<int64_t> pos(n);
        for (size_t i = 0; i < n; ++i) pos[i] = kv_len + (int64_t)i;
        pos_ids->load(pos.data());
    }

    for (size_t l = 0; l < meta.nlayer; ++l) {
        const auto &w = layers[l];
        auto &cache = kv_caches[l];

        // ① 输入 RMSNorm（Qwen2 无 bias 的 pre-norm）
        auto ln = Tensor::create({n, hs}, dtype, device, device_id);
        ops::rms_norm(ln, hidden, w.attn_norm_w, meta.epsilon);

        // ② q/k/v 投影：q 输出 nh*dh 维，k/v 输出 nkvh*dh 维（GQA 的维度差在这里体现）
        auto q = Tensor::create({n, nh * dh}, dtype, device, device_id);
        auto k = Tensor::create({n, nkvh * dh}, dtype, device, device_id);
        auto v = Tensor::create({n, nkvh * dh}, dtype, device, device_id);
        ops::linear(q, ln, w.attn_q_w, w.attn_q_b);
        ops::linear(k, ln, w.attn_k_w, w.attn_k_b);
        ops::linear(v, ln, w.attn_v_w, w.attn_v_b);

        // ③ rearrange：把扁平投影输出重排成多头形状 [n, nh, dh] / [n, nkvh, dh]
        // （连续张量，rearrange 内部就是一次 memcpy，等价于 view）
        auto qr = Tensor::create({n, nh, dh}, dtype, device, device_id);
        auto kr = Tensor::create({n, nkvh, dh}, dtype, device, device_id);
        auto v3 = Tensor::create({n, nkvh, dh}, dtype, device, device_id);
        ops::rearrange(qr, q);
        ops::rearrange(kr, k);
        ops::rearrange(v3, v);

        // ④ RoPE：对 q 和 k 施加旋转位置编码（v 不需要）
        auto q_rot = Tensor::create({n, nh, dh}, dtype, device, device_id);
        auto k_rot = Tensor::create({n, nkvh, dh}, dtype, device, device_id);
        ops::rope(q_rot, qr, pos_ids, meta.theta);
        ops::rope(k_rot, kr, pos_ids, meta.theta);

        // ⑤ 写 KV cache：把本步的 k_rot / v3 按行拷到 cache 的第 kv_len..kv_len+n 行。
        // 行偏移 = 行号 * strides[0] * elementSize()（strides 元素单位 × 字节数）。
        // 这就是"增量计算"：历史 token 的 K/V 存着，每次只写新 token 的那几行。
        {
            const size_t row_bytes = nkvh * dh * cache.k->elementSize();
            const size_t dst_stride = (size_t)cache.k->strides()[0] * cache.k->elementSize();
            const size_t k_src_stride = (size_t)k_rot->strides()[0] * k_rot->elementSize();
            const size_t v_src_stride = (size_t)v3->strides()[0] * v3->elementSize();
            for (size_t i = 0; i < n; ++i) {
                device_memcpy(cache.k->data() + (size_t)(kv_len + (int64_t)i) * dst_stride,
                              k_rot->data() + i * k_src_stride, row_bytes, device, device_id,
                              device_kind(device));
                device_memcpy(cache.v->data() + (size_t)(kv_len + (int64_t)i) * dst_stride,
                              v3->data() + i * v_src_stride, row_bytes, device, device_id,
                              device_kind(device));
            }
        }

        // ⑥ self-attention：对"全部已缓存"的 K/V（0..total）做注意力。
        // slice(0, 0, total) 沿第 0 维截取，仍是连续张量，可直接传给算子。
        // causal 掩码在算子内部处理（qlen==n 时对角线 = total - n）。
        const int64_t total = kv_len + (int64_t)n;
        auto ck = cache.k->slice(0, 0, (size_t)total);
        auto cv = cache.v->slice(0, 0, (size_t)total);
        auto attn = Tensor::create({n, nh, dh}, dtype, device, device_id);
        ops::self_attention(attn, q_rot, ck, cv, 1.0f / std::sqrt((float)dh));

        // ⑦ o_proj：attention 输出重排回 [n, hs] 再过线性层（无 bias），残差相加
        auto attn_2d = Tensor::create({n, hs}, dtype, device, device_id);
        ops::rearrange(attn_2d, attn);
        auto attn_out = Tensor::create({n, hs}, dtype, device, device_id);
        ops::linear(attn_out, attn_2d, w.attn_o_w, nullptr);
        ops::add(hidden, hidden, attn_out);

        // ⑧ MLP：post_attention_layernorm -> gate/up 两个线性 -> SwiGLU -> down 线性 -> 残差
        auto ln2 = Tensor::create({n, hs}, dtype, device, device_id);
        ops::rms_norm(ln2, hidden, w.mlp_norm_w, meta.epsilon);
        auto gate = Tensor::create({n, di}, dtype, device, device_id);
        auto up = Tensor::create({n, di}, dtype, device, device_id);
        ops::linear(gate, ln2, w.mlp_gate_w, nullptr);
        ops::linear(up, ln2, w.mlp_up_w, nullptr);
        auto act = Tensor::create({n, di}, dtype, device, device_id);
        ops::swiglu(act, gate, up);
        auto down = Tensor::create({n, hs}, dtype, device, device_id);
        ops::linear(down, act, w.mlp_down_w, nullptr);
        ops::add(hidden, hidden, down);
    }

    // ⑨ 收尾：最终 RMSNorm -> 取【最后一行】（生成时只关心最后一个 token 的 logits）
    //    -> lm_head 线性（vocab 维）-> argmax 得到下一个 token
    auto normed = Tensor::create({n, hs}, dtype, device, device_id);
    ops::rms_norm(normed, hidden, out_norm_w, meta.epsilon);
    auto last = normed->slice(0, n - 1, n); // [1, hs]
    auto logits = Tensor::create({1, voc}, dtype, device, device_id);
    ops::linear(logits, last, out_embed, nullptr);
    // argmax 算子要求 1D 输入，logits [1, voc] 连续，view 成 [voc] 即可
    auto logits_1d = logits->view({voc});
    auto max_idx = Tensor::create({1}, LLAISYS_DTYPE_I64, device, device_id);
    auto max_val = Tensor::create({1}, dtype, device, device_id);
    ops::argmax(max_idx, max_val, logits_1d);

    // 把 argmax 结果（i64）拷回 host（CPU 上就是 memcpy；设备上是 D2H）
    int64_t next = 0;
    device_memcpy(reinterpret_cast<std::byte *>(&next), max_idx->data(), sizeof(int64_t),
                  device, device_id, to_host_kind(device));

    // 已缓存 token 数前进 n 步，供下一轮 infer 使用
    kv_len += (int64_t)n;
    return next;
}

} // namespace llaisys::models
