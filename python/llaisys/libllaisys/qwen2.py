# ============================================================================
# libllaisys/qwen2.py — Qwen2 模型的 ctypes 绑定层（A3 新增）
# ----------------------------------------------------------------------------
# 职责：把 C 头文件 include/llaisys/models/qwen2.h 里的结构体和函数，
#       逐字段/逐参数翻译成 ctypes 声明，让 Python 能调用共享库。
#
# 两个结构体必须与 C 端【完全一致】（字段顺序、类型、对齐），
# 否则 ctypes 读出来的是错位的数据 —— 这是 ctypes 绑定最常见的坑。
#
#   LlaisysQwen2Meta   模型超参（对应 C 的 LlaisysQwen2Meta）
#   LlaisysQwen2Weights 全部权重句柄（对应 C 的 LlaisysQwen2Weights，
#                       其中层权重是指向"句柄数组"的指针）
# ============================================================================
from ctypes import Structure, POINTER, c_int, c_size_t, c_float, c_int64, c_void_p
from .tensor import llaisysTensor_t
from .llaisys_types import llaisysDeviceType_t


# 与 C 结构体一一对应：dtype 是枚举(int)，维度是 size_t，epsilon/theta 是 float，
# end_token 是 int64。ctypes 里分别用 c_int / c_size_t / c_float / c_int64。
class LlaisysQwen2Meta(Structure):
    _fields_ = [
        ("dtype", c_int),
        ("nlayer", c_size_t),
        ("hs", c_size_t),
        ("nh", c_size_t),
        ("nkvh", c_size_t),
        ("dh", c_size_t),
        ("di", c_size_t),
        ("maxseq", c_size_t),
        ("voc", c_size_t),
        ("epsilon", c_float),
        ("theta", c_float),
        ("end_token", c_int64),
    ]


# 层权重（attn_norm_w 等 11 个）在 C 端是 llaisysTensor_t* 数组，
# 所以这里用 POINTER(llaisysTensor_t)；顶层 3 个是单个句柄（llaisysTensor_t）。
class LlaisysQwen2Weights(Structure):
    _fields_ = [
        ("in_embed", llaisysTensor_t),
        ("out_embed", llaisysTensor_t),
        ("out_norm_w", llaisysTensor_t),
        ("attn_norm_w", POINTER(llaisysTensor_t)),
        ("attn_q_w", POINTER(llaisysTensor_t)),
        ("attn_q_b", POINTER(llaisysTensor_t)),
        ("attn_k_w", POINTER(llaisysTensor_t)),
        ("attn_k_b", POINTER(llaisysTensor_t)),
        ("attn_v_w", POINTER(llaisysTensor_t)),
        ("attn_v_b", POINTER(llaisysTensor_t)),
        ("attn_o_w", POINTER(llaisysTensor_t)),
        ("mlp_norm_w", POINTER(llaisysTensor_t)),
        ("mlp_gate_w", POINTER(llaisysTensor_t)),
        ("mlp_up_w", POINTER(llaisysTensor_t)),
        ("mlp_down_w", POINTER(llaisysTensor_t)),
    ]


# 注册函数签名：argtypes 定义参数类型，restype 定义返回类型。
# 注意 ModelCreate 传结构体要用 POINTER(...)（传引用），Python 端持有 meta 对象
# 直到 Create 返回（C 端同步拷贝字段，不会保留指针）。
def load_qwen2(lib):
    lib.llaisysQwen2ModelCreate.argtypes = [
        POINTER(LlaisysQwen2Meta),
        llaisysDeviceType_t,
        POINTER(c_int),
        c_int,
    ]
    lib.llaisysQwen2ModelCreate.restype = c_void_p

    lib.llaisysQwen2ModelDestroy.argtypes = [c_void_p]
    lib.llaisysQwen2ModelDestroy.restype = None

    lib.llaisysQwen2ModelWeights.argtypes = [c_void_p]
    lib.llaisysQwen2ModelWeights.restype = POINTER(LlaisysQwen2Weights)

    lib.llaisysQwen2ModelInfer.argtypes = [c_void_p, POINTER(c_int64), c_size_t]
    lib.llaisysQwen2ModelInfer.restype = c_int64

    lib.llaisysQwen2ModelSetKvLen.argtypes = [c_void_p, c_int64]
    lib.llaisysQwen2ModelSetKvLen.restype = None

    lib.llaisysQwen2ModelGetKvLen.argtypes = [c_void_p]
    lib.llaisysQwen2ModelGetKvLen.restype = c_int64
