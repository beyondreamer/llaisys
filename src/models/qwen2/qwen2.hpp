// ============================================================================
// src/models/qwen2/qwen2.hpp — Qwen2 模型层声明（Assignment #3 核心）
// ----------------------------------------------------------------------------
// 职责：用 A1 的 Tensor 和 A2 的算子，拼装出 Qwen2 完整推理引擎。
// 推理 = 算子拼装（与 PyTorch 的 modeling_qwen2.py 一一对应）：
//   embed -> 28 层[rms_norm -> q/k/v linear -> rearrange -> rope
//            -> 写 KV cache -> self_attention -> o_proj -> residual
//            -> mlp(gate/up/swiglu/down) -> residual] -> final_norm
//            -> lm_head -> argmax
//
// 数据结构：
//   Qwen2Meta          模型超参（从 config.json 读入）
//   Qwen2LayerWeights  每一层的 12 个权重张量
//   Qwen2KVCache       每一层的 K/V 缓存（避免重复计算历史 token）
//   Qwen2Model         模型本体：所有权重 + 每层 KV cache + 推理入口
//
// 注意：k_proj/v_proj 的输出维度是 nkvh*dh（GQA！），不是 hs；
//       lm_head 独立（tie_word_embeddings=false）。
// ============================================================================
#pragma once

#include "llaisys.h"

#include "../../tensor/tensor.hpp"

#include <vector>

namespace llaisys::models {

// 模型超参。字段顺序与 C 头文件 include/llaisys/models/qwen2.h 的
// LlaisysQwen2Meta 一一对应（ctypes 结构体也按此顺序声明）。
struct Qwen2Meta {
    llaisysDataType_t dtype;                 // 权重精度（本项目 bf16）
    size_t nlayer;                           // 隐藏层层数（28）
    size_t hs;                               // hidden_size（1536）
    size_t nh;                               // query 头数（12）
    size_t nkvh;                             // KV 头数（2，GQA）
    size_t dh;                               // head_dim（128 = hs/nh）
    size_t di;                               // intermediate_size（8960）
    size_t maxseq;                           // KV cache 最大序列长度
    size_t voc;                              // vocab_size（151936）
    float epsilon;                           // RMSNorm eps（1e-6）
    float theta;                             // RoPE theta（10000）
    int64_t end_token;                       // EOS token id（151643）
};

struct Qwen2LayerWeights {
    tensor_t attn_norm_w; // input_layernorm.weight
    tensor_t attn_q_w, attn_q_b;
    tensor_t attn_k_w, attn_k_b;
    tensor_t attn_v_w, attn_v_b;
    tensor_t attn_o_w; // no bias
    tensor_t mlp_norm_w; // post_attention_layernorm.weight
    tensor_t mlp_gate_w, mlp_up_w, mlp_down_w; // no bias
};

struct Qwen2KVCache {
    tensor_t k; // [maxseq, nkvh, dh]
    tensor_t v; // [maxseq, nkvh, dh]
};

class Qwen2Model {
public:
    Qwen2Meta meta;
    llaisysDeviceType_t device;
    int device_id;

    tensor_t in_embed;   // [voc, hs]
    tensor_t out_embed;  // [voc, hs] (lm_head, not tied)
    tensor_t out_norm_w; // model.norm.weight [hs]

    std::vector<Qwen2LayerWeights> layers;
    std::vector<Qwen2KVCache> kv_caches;
    int64_t kv_len = 0;

    Qwen2Model(const Qwen2Meta &meta, llaisysDeviceType_t device, int device_id = 0);

    // Run one forward pass over `token_ids` (prefill if >1, decode if ==1)
    // and return the next token id.
    int64_t infer(const std::vector<int64_t> &token_ids);

    void setKvLen(int64_t len) { kv_len = len; }
    int64_t getKvLen() const { return kv_len; }
};

} // namespace llaisys::models
