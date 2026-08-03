// ============================================================================
// src/llaisys/qwen2.cc — Qwen2 模型的 C API 薄封装层
// ----------------------------------------------------------------------------
// 职责：把 C++ 的 llaisys::models::Qwen2Model 暴露成 C 结构体 + C 函数，
//       供 Python ctypes 调用（C++ 代码的"国境线"就到这里为止）。
//
// 关键设计：
//   * LlaisysQwen2Model 是"句柄包装器"：持有 C++ 模型指针 + 权重句柄 +
//     句柄数组，销毁时统一释放，避免内存泄漏/重复释放
//   * LlaisysQwen2Weights 里的每个 llaisysTensor_t 是 new 出来的句柄
//     （内部包一个 shared_ptr），Python 端通过它把权重数据 load 进 C++ 张量
//   * 所有 helper（make_handle/fill_array）放在 extern "C" 块【外面】——
//     模板函数不允许 C linkage
// ============================================================================
#include "llaisys/models/qwen2.h"

#include "llaisys_tensor.hpp"

#include "../models/qwen2/qwen2.hpp"

#include <cstring>
#include <vector>

// 句柄包装器：C 端只看到一个不透明指针，内部实际是它。
//   model   -> C++ 模型本体（所有权）
//   weights -> 暴露给 Python 的权重句柄结构体（Python 用 llaisysQwen2ModelWeights 拿）
//   handles -> 所有 new 出来的 LlaisysTensor 句柄，析构时逐个 delete
//   arrays  -> 所有 new[] 出来的句柄数组（attn_norm_w 等），析构时逐个 delete[]
struct LlaisysQwen2Model {
    llaisys::models::Qwen2Model *model;
    LlaisysQwen2Weights *weights;
    std::vector<llaisysTensor_t> handles;       // owned tensor handles
    std::vector<llaisysTensor_t *> arrays;      // owned weight arrays (delete[])
};

namespace {
// 把一个 C++ shared_ptr<Tensor> 包装成 C 句柄（new 一个 LlaisysTensor 结构体）。
// 句柄只负责"转发"，Tensor 的生命周期仍由 shared_ptr 管理（与模型共存亡）。
llaisysTensor_t make_handle(const llaisys::tensor_t &tensor) {
    return new LlaisysTensor{tensor};
}

// 通用填充器：对每一层调用 getter(layers[i]) 取出该层的某个权重张量，
// 包装成句柄塞进数组，并记录到 handles 以便统一销毁。
// 12 个层权重数组（attn_norm_w .. mlp_down_w）都用它填充，避免重复代码。
template <typename Getter>
void fill_array(llaisysTensor_t *&array, size_t nlayer,
                const std::vector<llaisys::models::Qwen2LayerWeights> &layers,
                Getter getter, std::vector<llaisysTensor_t> &handles) {
    array = new llaisysTensor_t[nlayer];
    for (size_t i = 0; i < nlayer; ++i) {
        auto handle = make_handle(getter(layers[i]));
        array[i] = handle;
        handles.push_back(handle);
    }
    // (the caller records `array` in wrapper->arrays for cleanup)
}
} // namespace

__C {

struct LlaisysQwen2Model *llaisysQwen2ModelCreate(const LlaisysQwen2Meta *meta,
                                                  llaisysDeviceType_t device,
                                                  int *device_ids, int ndevice) {
    llaisys::models::Qwen2Meta m;
    m.dtype = meta->dtype;
    m.nlayer = meta->nlayer;
    m.hs = meta->hs;
    m.nh = meta->nh;
    m.nkvh = meta->nkvh;
    m.dh = meta->dh;
    m.di = meta->di;
    m.maxseq = meta->maxseq;
    m.voc = meta->voc;
    m.epsilon = meta->epsilon;
    m.theta = meta->theta;
    m.end_token = meta->end_token;

    // device_ids 预留了多卡支持：目前只用第一张卡（A4 可扩展为张量并行）
    const int device_id = (device_ids != nullptr && ndevice > 0) ? device_ids[0] : 0;

    auto *wrapper = new LlaisysQwen2Model;
    wrapper->model = new llaisys::models::Qwen2Model(m, device, device_id);
    wrapper->weights = new LlaisysQwen2Weights{};

    auto &model = *wrapper->model;
    const size_t nlayer = model.meta.nlayer;

    // 顶层 3 个权重（词嵌入/lm_head/最终 norm）是单句柄，直接赋值
    wrapper->weights->in_embed = make_handle(model.in_embed);
    wrapper->weights->out_embed = make_handle(model.out_embed);
    wrapper->weights->out_norm_w = make_handle(model.out_norm_w);
    wrapper->handles.push_back(wrapper->weights->in_embed);
    wrapper->handles.push_back(wrapper->weights->out_embed);
    wrapper->handles.push_back(wrapper->weights->out_norm_w);

    fill_array(wrapper->weights->attn_norm_w, nlayer, model.layers,
               [](const auto &w) { return w.attn_norm_w; }, wrapper->handles);
    wrapper->arrays.push_back(wrapper->weights->attn_norm_w);
    fill_array(wrapper->weights->attn_q_w, nlayer, model.layers,
               [](const auto &w) { return w.attn_q_w; }, wrapper->handles);
    wrapper->arrays.push_back(wrapper->weights->attn_q_w);
    fill_array(wrapper->weights->attn_q_b, nlayer, model.layers,
               [](const auto &w) { return w.attn_q_b; }, wrapper->handles);
    wrapper->arrays.push_back(wrapper->weights->attn_q_b);
    fill_array(wrapper->weights->attn_k_w, nlayer, model.layers,
               [](const auto &w) { return w.attn_k_w; }, wrapper->handles);
    wrapper->arrays.push_back(wrapper->weights->attn_k_w);
    fill_array(wrapper->weights->attn_k_b, nlayer, model.layers,
               [](const auto &w) { return w.attn_k_b; }, wrapper->handles);
    wrapper->arrays.push_back(wrapper->weights->attn_k_b);
    fill_array(wrapper->weights->attn_v_w, nlayer, model.layers,
               [](const auto &w) { return w.attn_v_w; }, wrapper->handles);
    wrapper->arrays.push_back(wrapper->weights->attn_v_w);
    fill_array(wrapper->weights->attn_v_b, nlayer, model.layers,
               [](const auto &w) { return w.attn_v_b; }, wrapper->handles);
    wrapper->arrays.push_back(wrapper->weights->attn_v_b);
    fill_array(wrapper->weights->attn_o_w, nlayer, model.layers,
               [](const auto &w) { return w.attn_o_w; }, wrapper->handles);
    wrapper->arrays.push_back(wrapper->weights->attn_o_w);
    fill_array(wrapper->weights->mlp_norm_w, nlayer, model.layers,
               [](const auto &w) { return w.mlp_norm_w; }, wrapper->handles);
    wrapper->arrays.push_back(wrapper->weights->mlp_norm_w);
    fill_array(wrapper->weights->mlp_gate_w, nlayer, model.layers,
               [](const auto &w) { return w.mlp_gate_w; }, wrapper->handles);
    wrapper->arrays.push_back(wrapper->weights->mlp_gate_w);
    fill_array(wrapper->weights->mlp_up_w, nlayer, model.layers,
               [](const auto &w) { return w.mlp_up_w; }, wrapper->handles);
    wrapper->arrays.push_back(wrapper->weights->mlp_up_w);
    fill_array(wrapper->weights->mlp_down_w, nlayer, model.layers,
               [](const auto &w) { return w.mlp_down_w; }, wrapper->handles);
    wrapper->arrays.push_back(wrapper->weights->mlp_down_w);

    return wrapper;
}

// 析构顺序很重要：先删句柄（new）和句柄数组（new[]），再删权重结构体、
// C++ 模型（其内部 shared_ptr 会随析构释放所有权重/KV cache 内存），最后删包装器。
void llaisysQwen2ModelDestroy(struct LlaisysQwen2Model *model) {
    if (model == nullptr) return;
    for (auto handle : model->handles) {
        delete handle;
    }
    for (auto array : model->arrays) {
        delete[] array;
    }
    delete model->weights;
    delete model->model;
    delete model;
}

struct LlaisysQwen2Weights *llaisysQwen2ModelWeights(struct LlaisysQwen2Model *model) {
    return model->weights;
}

// 推理入口：把 C 数组转成 vector 交给 C++ 模型，返回下一个 token id。
int64_t llaisysQwen2ModelInfer(struct LlaisysQwen2Model *model, int64_t *token_ids, size_t ntoken) {
    std::vector<int64_t> ids(token_ids, token_ids + ntoken);
    return model->model->infer(ids);
}

// KV cache 长度控制：generate 开始时 SetKvLen(0) 复位，实现"一次生成 = 一轮对话"。
void llaisysQwen2ModelSetKvLen(struct LlaisysQwen2Model *model, int64_t len) {
    model->model->setKvLen(len);
}

int64_t llaisysQwen2ModelGetKvLen(struct LlaisysQwen2Model *model) {
    return model->model->getKvLen();
}

} // extern "C"
