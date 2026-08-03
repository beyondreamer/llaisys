#!/bin/bash
# ============================================================================
# scripts/mxcc_nvcc_wrapper.sh — nvcc -> MXMACA mxcc 参数翻译包装（沐曦平台用）
# ----------------------------------------------------------------------------
# 为什么需要：xmake 的 CUDA 工具链按 nvcc 的参数风格调用编译器，沐曦 mxcc 是
# clang 系，参数不同。本脚本伪装成 nvcc 放进 PATH，做参数翻译：
#   nvcc 写法          -> mxcc 写法
#   -arch=sm_80        -> -offload-arch=xcore1500（C500；可用 MX_ARCH 覆盖）
#   -rdc=true          -> -fno-gpu-rdc（强制非-rdc，让 .cu.o 自包含 gpubin handle）
#   -Xcompiler <x>     -> <x>（直接透传，clang 原生接受 -Wall/-Werror/-O3 等）
#   -dlink             -> 用 gcc 编空源产出占位 .o（mxcc -dlink-obj 不接 .o 输入）
#   -m64               -> 丢弃（mxcc 不认，默认即 64 位）
# 另：mxcc 默认 C++14，项目需要 C++17（std::byte），故附加 -std=c++17 + cu-bridge include。
# 用法：cp 本脚本到 /usr/local/bin/nvcc（xmake 会自动找到）。
# ============================================================================
set -euo pipefail

MXCC="${MXCC:-/opt/maca-3.2.1/mxgpu_llvm/bin/mxcc}"
MX_ARCH="${MX_ARCH:-xcore1500}"
CU_BRIDGE_INC="${CU_BRIDGE_INC:-/opt/maca-3.2.1/tools/cu-bridge/include}"

# 设备链接（devlink）特殊处理：xmake 会传 -dlink + 各 .cu.o + -o gpucode.cu.o。
# mxcc 的 -dlink-obj 语义是“从源文件编译出 dlink obj”，不接收已编译的 .o 输入，
# 直接调用无产出。沐曦 .cu.o 的设备代码已是最终形态，各 .cu.o 直接链接进 .so 即可，
# 故用 gcc 编空源产出无符号占位 .o（合法 ELF，满足 xmake 要求，不与各 .cu.o 冲突）。
for a in "$@"; do
    if [ "$a" = "-dlink" ]; then
        OUT=""
        while [ $# -gt 0 ]; do
            case "$1" in
                -o) OUT="$2"; shift 2;;
                *) shift;;
            esac
        done
        mkdir -p "$(dirname "$OUT")"
        printf '//' > /tmp/_mx_empty.cpp
        exec gcc -x c++ -c /tmp/_mx_empty.cpp -o "$OUT"
    fi
done

# 普通编译调用：翻译 nvcc 参数 -> mxcc 参数（-dlink 已被上面拦截，不会到这里）
ARGS=()
for a in "$@"; do
    case "$a" in
        -arch=*)          ARGS+=("-offload-arch=$MX_ARCH") ;;
        -rdc=true)        ARGS+=("-fno-gpu-rdc") ;;   # 强制非-rdc：mxcc devlink 不兼容 nvcc
        -rdc=false)       ARGS+=("-fno-gpu-rdc") ;;
        -Xcompiler)       : ;;                        # 下一个参数直接透传
        -m64|-m32)        : ;;                        # mxcc 不认，丢弃
        cross-execution-space-call,reorder,deprecated-declarations) : ;;  # CUDA 专用警告，丢弃
        *)                ARGS+=("$a") ;;
    esac
done
ARGS+=("-std=c++17")
ARGS+=("-I$CU_BRIDGE_INC")
exec "$MXCC" "${ARGS[@]}"
