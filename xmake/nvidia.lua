-- ============================================================================
-- xmake/nvidia.lua — NVIDIA/国产兼容 GPU 构建规则（Assignment #4）
-- ----------------------------------------------------------------------------
-- 由 xmake.lua 的 `--nv-gpu` 开关 include（同时定义 ENABLE_NVIDIA_API 宏）。
--
-- 最终方案（实机验证确定）：
--   CUDA 代码（src/device/nvidia/*.cu、src/ops/*/nvidia/*.cu）直接编进
--   主共享库 llaisys target（见 xmake.lua），链接时 xmake 自动用 nvcc 做
--   “设备链接”（devlink），生成 __cudaRegisterLinkedBinary 符号。
--   不再走静态库（静态库 + g++ 链接会导致这些符号缺失，.so 加载失败）。
--
-- 使用（在装有 CUDA 工具链的机器上）：
--   xmake f --nv-gpu=y -cv && xmake && xmake install
-- 国产兼容平台（如沐曦 MACA）：把其 CUDA 兼容编译器加入 PATH / CUDA_PATH 即可。
-- ============================================================================

-- 注意：
--   * 默认 sm_80（__nv_bfloat16 需要 sm_80+）；沐曦 MACA 上可用 --cuda-arch=compute_80 等
--   * -Xcompiler -fPIC：共享库需要位置无关代码（否则链接报 relocation 错误）
local cuda_arch = get_config("cuda-arch") or "sm_80"
add_cuflags("-arch=" .. cuda_arch, "-Xcompiler -fPIC")
