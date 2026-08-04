# LLAISYS 26S 作业总结报告

> 日期：2026-08-04
> 仓库：`beyondreamer/llaisys`（fork，导师查看 main 分支即可）
> 模型：DeepSeek-R1-Distill-Qwen-1.5B（bf16，28 层，hs=1536，nh=12，nkvh=2，dh=128，di=8960，voc=151936）
> 双平台**全新纯净镜像节点从零重跑**，A0-A4 全部通过

---

## 一、完成情况与实测数据

| 作业 | NVIDIA RTX 4090D（CUDA 12.8, 24GB） | 沐曦曦云 C500（MACA 3.2.1.3, 64GB） |
|---|---|---|
| A0 runtime | ✅ | ✅ |
| A1 tensor | ✅ | ✅ |
| A2 8 算子（f32/f16/bf16） | ✅ 全部 | ✅ 全部 |
| A3 推理 2 步 | ✅ token 一致 | ✅ token 一致 |
| A3 推理 128 步 | ✅ 一致（**LLAISYS 5.16s** / PyTorch 2.14s） | ✅ 一致（**LLAISYS 32.35s** / PyTorch 2.80s） |

CI（GitHub Actions，ubuntu + windows）：A0-A3 自动验证全绿。

---

## 二、作业 0：环境搭建

**过程**：装 xmake（v3.0.9）+ 编译器 + Python + torch/transformers → `xmake && xmake install && pip install ./python/` → `python test/test_runtime.py --device cpu` 通过。

**实机节点镜像自带（优先用）**：
- 沐曦：`/opt/conda` python 3.10.10 + torch 2.6.0+metax3.2.1.3（cuda True）+ MACA 3.2.1.3
- NVIDIA：`/usr/bin/python3` 3.12.3 + torch 2.6.0a0（NGC）+ CUDA 12.8（`/usr/local/cuda`）

**坑**：xmake 在线脚本装失败 → 下 `gz.run` 手动装；国内网络慢 → aliyun pip mirror + modelscope 下模型（github/HF 不可达）。

---

## 三、作业 1：Tensor（`src/tensor/tensor.cpp`）

**设计**：Tensor = `meta{dtype,shape,strides}` + `storage`(shared_ptr 共享) + `offset`(字节)。实现 8 方法：
- `isContiguous`：从末维往前验证 `strides[i] == 前序维度乘积`
- `load`：连续 memcpy（H2H/H2D），非连续按 strides scatter
- `view`：要求连续 + numel 一致，非连续抛错（无拷贝 reshape 只对连续有效）
- `permute`：校验合法排列，shape/strides 按 order 重排，offset 不变
- `slice`：`offset += start·strides[dim]·elementSize`（offset 字节、strides 元素）
- 挑战项 `contiguous`/`reshape`/`to`（A3/A4 依赖）

**数据**：`test_tensor.py` 的 load/view(6,10)/permute(2,0,1)/slice(2,1,4) 全部与 PyTorch 的 shape/stride/数据一致，双平台均 Test passed。

**坑**：offset 是字节、strides 是元素（slice 忘乘 elementSize 是经典错）；view 必须检查连续（非连续强行 view 会乱序）。

---

## 四、作业 2：CPU 算子（`src/ops/*/cpu/`）

**设计**：三层结构 = `op.cpp`（校验 + 按设备分发）+ `cpu/*.cpp`（模板实现，`utils::cast<float>` 中转、float 累加）。8 算子支持 f32/f16/bf16：
- argmax/embedding/swiglu：线性扫描 / 整行 memcpy / 逐元素
- linear：`Y=XWᵀ+b`，**weight 未转置**（按 W 的行取）
- rms_norm：每行 `W·X/√(mean(X²)+eps)`
- rope：`φ=p/θ^(2j/d)`，a/b 对半旋转
- self_attention：**GQA**（query 头 h 用 kv 头 h/groups）+ **causal**（对角线 kvlen-qlen）+ float32 softmax(max-subtract)
- rearrange：`[seq,nh·dh]⟷[seq,nh,dh]`（A3 必需，README 未列）

**数据**：`test/ops/*.py`，8 算子 × f32/f16/bf16 × 小/大尺寸（512×4096）全部通过，双平台一致。

**坑**：linear weight 未转置（`out[i][j]=Σ x[i][p]·w[j][p]`）；GQA 映射是 h/groups 不是 h%nkvh（PyTorch repeat_interleave 连续复制）；causal 对角线 kvlen-qlen（生成场景 qlen=1 要能看全部历史）；softmax 必须 float32 + max-subtract（A3 逐 token 对齐的前提）。

---

## 五、作业 3：Qwen2 推理（`src/models/qwen2/`）

**设计**：C++ 模型层 `infer()`：embed → 28×(rms_norm→q/k/v 线性→rearrange→rope→**KV cache**→self_attention→o_proj→residual→mlp) → final_norm → lm_head → argmax。
- KV cache：每层 k/v `[maxseq,nkvh,dh]`，共享 kv_len 计数器，prefill + decode 通用（每步只算新 token 的 K/V，历史复用）
- C API（`include/llaisys/models/qwen2.h`）+ ctypes 绑定 + Python 包装（safetensors 权重加载 + generate）

**数据**：`test_infer.py --test`，128 步生成与 PyTorch **逐 token 完全一致**（argmax 采样）。NVIDIA 5.16s，沐曦 32.35s。

**坑**：bf16 精度对齐——所有算子 float 累加；用 tensor.debug() 逐层对比 hidden state 定位发散；KV cache 写入位置（kv_len 偏移）易错。

---

## 六、作业 4：CUDA 集成（NVIDIA + 沐曦）

**设计**：NVIDIA 与沐曦**共用同一套 CUDA 代码**（`src/device/nvidia/*.cu` runtime API + `src/ops/*/nvidia/*.cu` 9 个内核）。
- 构建：`xmake/nvidia.lua` + `--nv-gpu` 开关 + `--cuda-arch` option；CUDA 代码直接编进主共享库（nvcc 设备链接）
- 沐曦靠 `scripts/mxcc_nvcc_wrapper.sh` 把 nvcc 参数翻译成 mxcc，**零沐曦专用算子代码**

**数据**：两平台 runtime + 8 算子（--device nvidia）+ 推理 128 步全过（见第一节）。

**坑与修复**：

*沐曦（MACA cu-bridge 兼容，坑较多，全在 wrapper 与运行时 env）*：
1. mxcc 默认 C++14（`std::byte` 未定义）→ wrapper 加 `-std=c++17`
2. `-m64` unknown（mxcc 不认）→ wrapper 丢弃
3. mxcc `-dlink-obj` 不接 .o（语义是从源编译）→ wrapper 检测 `-dlink` 时用 gcc 编空源产出占位 .o
4. `-lcudart_static`/`-lcudadevrt` 找不到（cu-bridge 不提供）→ `/usr/local/lib64` 造空归档 stub
5. `-rdc=true` 模式 gpubin handle 需 devlink（mxcc 不兼容）→ 强制 `-fno-gpu-rdc`，.cu.o 自包含
6. `undefined __wcuda_version_internal__` → 缺 libruntime_cu（沐曦 cudart 等价物），运行时 `LD_PRELOAD=/opt/maca/lib/libruntime_cu.so`
7. triton `maca_home_dirs None` → 设 `MACA_PATH=/opt/maca-3.2.1`
8. torch 找不到 libmxomp → `LD_LIBRARY_PATH` 加 `/opt/maca-3.2.1/mxgpu_llvm/lib`
9. `python -m modelscope` 不可执行 → 用 `modelscope` CLI
10. test_infer device_map 要 accelerate → `pip install accelerate`

*NVIDIA（NGC 镜像）*：
- 新申请节点驱动库干净（570.124.06），`torch.cuda.is_available()` 直接 True，**基本无坑**
- 历史（旧节点曾遇，记录备查）：驱动库 580 抢先（ldconfig 改回 580，需重命名 580 库锁 570）+ hpcx libucs 抢先（LD_LIBRARY_PATH hpcx 优先）+ CUDA forward-compat 触发 Error 804（GeForce 不支持，禁 `00-cuda-compat.conf`）

*通用*：
- self_attention 官方测试 GPU bug（mask 建在 CPU）：两平台都需本地补丁 `temp_mask = torch.ones(..., device=query.device)`（不推送）
- pip PEP 668（Ubuntu 系统 python）→ 用 `PYTHONPATH=<仓库>/python` 代替 pip install

---

## 七、复现流程

### CPU（CI / 本地，验 A0-A3）
```bash
xmake && xmake install && pip install ./python/
python test/test_runtime.py --device cpu
python test/test_tensor.py
for f in add argmax embedding linear rms_norm rope self_attention swiglu; do python test/ops/$f.py; done
python test/test_infer.py --model <DeepSeek-R1-Distill-Qwen-1.5B 路径> --test
```

### NVIDIA（实机，验 A4；NGC 镜像 + GeForce）
```bash
export PATH=/usr/local/cuda/bin:$PATH XMAKE_ROOT=y
xmake f --nv-gpu=y && xmake && xmake install
export PYTHONPATH=<仓库>/python LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
python test/test_runtime.py --device nvidia
for f in add argmax embedding linear rms_norm rope self_attention swiglu; do python test/ops/$f.py --device nvidia; done
python test/test_infer.py --model <模型> --test --device nvidia
```

### 沐曦（实机，验 A4）
```bash
cp scripts/mxcc_nvcc_wrapper.sh /usr/local/bin/nvcc && chmod +x /usr/local/bin/nvcc   # wrapper 装成 nvcc
export XMAKE_ROOT=y PATH=/usr/local/bin:/root/.local/bin:$PATH
xmake f --nv-gpu=y && xmake && xmake install
export LD_PRELOAD=/opt/maca/lib/libruntime_cu.so MACA_PATH=/opt/maca-3.2.1
export LD_LIBRARY_PATH=/opt/maca-3.2.1/mxgpu_llvm/lib:/opt/maca/lib:/opt/mxdriver/lib:/opt/maca-3.2.1/tools/cu-bridge/lib:$LD_LIBRARY_PATH
python test/test_runtime.py --device nvidia   # 与 NVIDIA 完全相同的测试命令
```

---

## 八、平台支持状态

| 平台 | 状态 | 备注 |
|---|---|---|
| Linux CPU（CI） | ✅ | A0-A3，ubuntu + windows 自动验 |
| NVIDIA RTX 4090D | ✅ | A4，CUDA 12.8，NGC 镜像 |
| 沐曦曦云 C500 | ✅ | A4，MACA 3.2.1.3，靠 wrapper 复用 CUDA 代码 |
