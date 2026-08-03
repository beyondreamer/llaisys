# ============================================================================
# models/qwen2.py — Qwen2 模型的 Python 包装（A3 重写）
# ----------------------------------------------------------------------------
# 职责：给用户一个"像 transformers 一样"的 Qwen2 类：
#   Qwen2(model_path)  -> 读 config、建 C++ 模型、把 safetensors 权重灌进去
#   model.generate()   -> 逐 token 推理（计算全部在 C++ 后端，Python 只做搬运）
#
# 数据流：
#   safetensors 权重(torch bf16) -> 原始字节 -> tensorLoad() 拷进 C++ 张量
#   token 序列 -> llaisysQwen2ModelInfer() -> 下一个 token id
#
# 重要约束：推理【不允许】在 Python 里用 PyTorch 计算 —— 全部走 C++ 算子，
#          这样才能检验 A1/A2/A3 的 C++ 实现是否正确。
# ============================================================================
import json
import re
from pathlib import Path
from typing import Sequence

from ctypes import POINTER, c_int64

import safetensors.torch
import torch

from ..libllaisys import LIB_LLAISYS, DataType, DeviceType
from ..libllaisys.qwen2 import LlaisysQwen2Meta


# 权重名映射表：safetensors 里的每层权重名 -> LlaisysQwen2Weights 的字段名。
# 例如 "model.layers.3.self_attn.q_proj.weight" -> 第 3 层的 attn_q_w 句柄。
_LAYER_FIELDS = {
    "input_layernorm.weight": "attn_norm_w",
    "self_attn.q_proj.weight": "attn_q_w",
    "self_attn.q_proj.bias": "attn_q_b",
    "self_attn.k_proj.weight": "attn_k_w",
    "self_attn.k_proj.bias": "attn_k_b",
    "self_attn.v_proj.weight": "attn_v_w",
    "self_attn.v_proj.bias": "attn_v_b",
    "self_attn.o_proj.weight": "attn_o_w",
    "post_attention_layernorm.weight": "mlp_norm_w",
    "mlp.gate_proj.weight": "mlp_gate_w",
    "mlp.up_proj.weight": "mlp_up_w",
    "mlp.down_proj.weight": "mlp_down_w",
}

_LAYER_RE = re.compile(r"^model\.layers\.(\d+)\.(.*)$")


def _to_raw_bytes(t: torch.Tensor):
    """把 torch 张量转成"原始字节"视图，供 tensorLoad 直接拷贝。

    bf16/fp16 是 2 字节类型，但 numpy 没有 bf16/fp16 原生 dtype，
    所以先 view 成 int16（位模式不变），再拿 numpy 缓冲区的地址。
    """
    t = t.contiguous()
    if t.dtype in (torch.bfloat16, torch.float16):
        t = t.view(torch.int16)
    return t.numpy()


class Qwen2:
    def __init__(self, model_path, device: DeviceType = DeviceType.CPU):
        model_path = Path(model_path)
        with open(model_path / "config.json", encoding="utf-8") as f:
            cfg = json.load(f)

        # 从 config.json 读超参；head_dim 未显式给出时按 hs/nh 计算（本项目 = 128）
        dtype_name = cfg.get("torch_dtype", "bfloat16")
        dtype = DataType.BF16 if dtype_name in ("bfloat16", "bf16") else DataType.F16
        hs = int(cfg["hidden_size"])
        nh = int(cfg["num_attention_heads"])
        nkvh = int(cfg["num_key_value_heads"])
        dh = int(cfg.get("head_dim", hs // nh))

        # 收集停止 token：config.json + generation_config.json 里的 eos_token_id
        # （可能是单个 int，也可能是 list），生成循环在这些 token 处停止
        stop_ids = {int(cfg.get("eos_token_id", 151643))}
        gc_path = model_path / "generation_config.json"
        if gc_path.exists():
            with open(gc_path, encoding="utf-8") as f:
                gc = json.load(f)
            eos = gc.get("eos_token_id")
            if isinstance(eos, list):
                stop_ids.update(int(x) for x in eos)
            elif eos is not None:
                stop_ids.add(int(eos))
        self._stop_ids = stop_ids

        # 组装 C 结构体 LlaisysQwen2Meta（字段顺序必须与 C 头文件一致），
        # 然后创建 C++ 模型并拿到权重句柄结构体
        meta = LlaisysQwen2Meta(
            dtype=int(dtype),
            nlayer=int(cfg["num_hidden_layers"]),
            hs=hs,
            nh=nh,
            nkvh=nkvh,
            dh=dh,
            di=int(cfg["intermediate_size"]),
            maxseq=2048,
            voc=int(cfg["vocab_size"]),
            epsilon=float(cfg["rms_norm_eps"]),
            theta=float(cfg.get("rope_theta", 10000.0)),
            end_token=sorted(stop_ids)[0],
        )

        self._device = device
        self._meta = meta
        # 建模型（device_ids 传 None -> C 端用 0 号设备），随后拿权重句柄并灌权重
        self._model = LIB_LLAISYS.llaisysQwen2ModelCreate(
            POINTER(LlaisysQwen2Meta)(meta), int(device), None, 0
        )
        self._weights = LIB_LLAISYS.llaisysQwen2ModelWeights(self._model).contents
        self._load_weights(model_path)

    def _load_weights(self, model_path: Path):
        """遍历所有 safetensors 分片，按权重名灌进对应的 C++ 张量。"""
        weights = self._weights
        for file in sorted(model_path.glob("*.safetensors")):
            tensors = safetensors.torch.load_file(str(file), device="cpu")
            for name, tensor in tensors.items():
                # 顶层权重：词嵌入 / lm_head / 最终 norm，都是单句柄
                if name == "model.embed_tokens.weight":
                    handle = weights.in_embed
                elif name == "lm_head.weight":
                    handle = weights.out_embed
                elif name == "model.norm.weight":
                    handle = weights.out_norm_w
                else:
                    # 层权重：解析 "model.layers.<i>.<rest>"，按映射表取该层对应句柄
                    m = _LAYER_RE.match(name)
                    if m is None:
                        continue
                    field = _LAYER_FIELDS.get(m.group(2))
                    if field is None:
                        continue
                    arr = getattr(weights, field)  # POINTER(llaisysTensor_t) 句柄数组
                    handle = arr[int(m.group(1))]
                # 把 bf16 的原始字节拷进 C++ 张量（shape 由 C++ 端创建时决定，字节数一致）
                LIB_LLAISYS.tensorLoad(handle, _to_raw_bytes(tensor).ctypes.data)
        print(f"[llaisys] weights loaded from {model_path}")

    def __del__(self):
        # Python 对象销毁时释放 C++ 模型（权重/KV cache 内存随 C++ 析构释放）
        if getattr(self, "_model", None) is not None:
            LIB_LLAISYS.llaisysQwen2ModelDestroy(self._model)
            self._model = None

    def generate(
        self,
        inputs: Sequence[int],
        max_new_tokens: int = None,
        top_k: int = 1,
        top_p: float = 0.8,
        temperature: float = 0.8,
    ):
        # 推理全部在 C++ 后端完成；Python 只负责驱动 token 循环。
        # top_k=1（--test 模式）等价于 argmax，正是 C++ 端返回的 token。
        # 首轮 prefill 一次喂 n 个输入 token，之后每轮只喂 1 个新 token（解码）。
        LIB_LLAISYS.llaisysQwen2ModelSetKvLen(self._model, 0)  # 复位 KV cache
        tokens = list(inputs)
        n = len(tokens)
        while max_new_tokens is None or len(tokens) - len(inputs) < max_new_tokens:
            arr = (c_int64 * n)(*tokens[-n:])
            next_id = int(LIB_LLAISYS.llaisysQwen2ModelInfer(self._model, arr, n))
            tokens.append(next_id)
            if next_id in self._stop_ids:  # 命中 EOS -> 结束生成（与 HF 行为一致）
                break
            n = 1  # 之后每次只生成 1 个 token
        return tokens  # 返回完整序列（输入 + 生成），与 HF 的 outputs[0].tolist() 对齐
