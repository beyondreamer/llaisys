// ============================================================================
// tensor.cpp — Tensor 张量类实现（Assignment #1）
// ----------------------------------------------------------------------------
// 本文件是 A1 作业的核心：实现 Tensor 类的 8 个方法。
//
// 关键背景知识：
//   * Tensor = meta(shape/strides/dtype) + storage(共享内存块) + offset(字节偏移)
//   * strides 是【元素】单位，offset 是【字节】单位 —— 本文件最容易错的地方
//   * view / permute / slice 都是"视图"操作：只改 meta 和 offset，不搬数据，
//     多个 Tensor 可以共享同一个 Storage（shared_ptr 管理生命周期）
//   * 本文件的 helper 通过 Runtime API（函数指针表）做内存拷贝，因此同一套
//     代码在 CPU / CUDA 上都能工作（A4 之后无需改动）
//
// 实现清单：
//   isContiguous()  判断内存布局是否连续
//   permute()       按 order 重排维度（视图）
//   view()          不搬数据地 reshape（仅限连续张量）
//   slice()         沿某维切一段（视图，调整 offset）
//   load()          从 host 内存拷入（连续/非连续两种路径）
//   contiguous()    把非连续张量复制成连续布局（挑战项）
//   reshape()       连续->view，非连续->先 contiguous 再 view（挑战项）
//   to()            跨设备搬运（挑战项）
// ============================================================================
#include "tensor.hpp"

#include "../utils.hpp"

#include <cstring>
#include <numeric>
#include <sstream>

namespace llaisys {

Tensor::Tensor(TensorMeta meta, core::storage_t storage, size_t offset)
    : _meta(std::move(meta)), _storage(std::move(storage)), _offset(offset) {}

tensor_t Tensor::create(const std::vector<size_t> &shape,
                        llaisysDataType_t dtype,
                        llaisysDeviceType_t device_type,
                        int device) {
    size_t ndim_ = shape.size();
    std::vector<ptrdiff_t> strides(ndim_);
    size_t stride = 1;
    for (size_t i = 1; i <= ndim_; i++) {
        strides[ndim_ - i] = stride;
        stride *= shape[ndim_ - i];
    }
    TensorMeta meta{dtype, shape, strides};
    size_t total_elems = stride;
    size_t dtype_size = utils::dsize(dtype);

    if (device_type == LLAISYS_DEVICE_CPU && core::context().runtime().deviceType() != LLAISYS_DEVICE_CPU) {
        auto storage = core::context().runtime().allocateHostStorage(total_elems * dtype_size);
        return std::shared_ptr<Tensor>(new Tensor(meta, storage));
    } else {
        core::context().setDevice(device_type, device);
        auto storage = core::context().runtime().allocateDeviceStorage(total_elems * dtype_size);
        return std::shared_ptr<Tensor>(new Tensor(meta, storage));
    }
}

std::byte *Tensor::data() {
    return _storage->memory() + _offset;
}

const std::byte *Tensor::data() const {
    return _storage->memory() + _offset;
}

size_t Tensor::ndim() const {
    return _meta.shape.size();
}

const std::vector<size_t> &Tensor::shape() const {
    return _meta.shape;
}

const std::vector<ptrdiff_t> &Tensor::strides() const {
    return _meta.strides;
}

llaisysDataType_t Tensor::dtype() const {
    return _meta.dtype;
}

llaisysDeviceType_t Tensor::deviceType() const {
    return _storage->deviceType();
}

int Tensor::deviceId() const {
    return _storage->deviceId();
}

size_t Tensor::numel() const {
    return std::accumulate(_meta.shape.begin(), _meta.shape.end(), size_t(1), std::multiplies<size_t>());
}

size_t Tensor::elementSize() const {
    return utils::dsize(_meta.dtype);
}

std::string Tensor::info() const {
    std::stringstream ss;

    ss << "Tensor: "
       << "shape[ ";
    for (auto s : this->shape()) {
        ss << s << " ";
    }
    ss << "] strides[ ";
    for (auto s : this->strides()) {
        ss << s << " ";
    }
    ss << "] dtype=" << this->dtype();

    return ss.str();
}

template <typename T>
void print_data(const T *data, const std::vector<size_t> &shape, const std::vector<ptrdiff_t> &strides, size_t dim) {
    if (dim == shape.size() - 1) {
        for (size_t i = 0; i < shape[dim]; i++) {
            if constexpr (std::is_same_v<T, bf16_t> || std::is_same_v<T, fp16_t>) {
                std::cout << utils::cast<float>(data[i * strides[dim]]) << " ";
            } else {
                std::cout << data[i * strides[dim]] << " ";
            }
        }
        std::cout << std::endl;
    } else if (dim < shape.size() - 1) {
        for (size_t i = 0; i < shape[dim]; i++) {
            print_data(data + i * strides[dim], shape, strides, dim + 1);
        }
    }
}

void debug_print(const std::byte *data, const std::vector<size_t> &shape, const std::vector<ptrdiff_t> &strides, llaisysDataType_t dtype) {
    switch (dtype) {
    case LLAISYS_DTYPE_BYTE:
        return print_data(reinterpret_cast<const char *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_BOOL:
        return print_data(reinterpret_cast<const bool *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_I8:
        return print_data(reinterpret_cast<const int8_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_I16:
        return print_data(reinterpret_cast<const int16_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_I32:
        return print_data(reinterpret_cast<const int32_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_I64:
        return print_data(reinterpret_cast<const int64_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_U8:
        return print_data(reinterpret_cast<const uint8_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_U16:
        return print_data(reinterpret_cast<const uint16_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_U32:
        return print_data(reinterpret_cast<const uint32_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_U64:
        return print_data(reinterpret_cast<const uint64_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_F16:
        return print_data(reinterpret_cast<const fp16_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_F32:
        return print_data(reinterpret_cast<const float *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_F64:
        return print_data(reinterpret_cast<const double *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_BF16:
        return print_data(reinterpret_cast<const bf16_t *>(data), shape, strides, 0);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}

void Tensor::debug() const {
    core::context().setDevice(this->deviceType(), this->deviceId());
    core::context().runtime().api()->device_synchronize();
    std::cout << this->info() << std::endl;
    if (this->deviceType() == LLAISYS_DEVICE_CPU) {
        debug_print(this->data(), this->shape(), this->strides(), this->dtype());
    } else {
        auto tmp_tensor = create({this->_storage->size()}, this->dtype());
        core::context().runtime().api()->memcpy_sync(
            tmp_tensor->data(),
            this->data(),
            this->numel() * this->elementSize(),
            LLAISYS_MEMCPY_D2H);
        debug_print(tmp_tensor->data(), this->shape(), this->strides(), this->dtype());
    }
}

namespace {
// 通用 strided 拷贝：把逻辑 shape 的每一个元素，从 src（按 src_strides 布局）
// 复制到 dst（按 dst_strides 布局）。
// 实现方式：按坐标遍历所有元素，用 idx[] 计算两端的内存偏移，再通过 Runtime 的
// memcpy API 逐元素拷贝 —— 这样同一段代码既能在 CPU（H2H）也能在设备（D2D）上工作。
// 注意：偏移 = Σ idx[d] * strides[d]（元素单位），最后要乘以 elem_size 得到字节偏移；
// 本框架暂不支持负 strides（test_utils 里也明确 TODO 了）。
void copy_strided_bytes(llaisysMemcpyKind_t kind, std::byte *dst,
                        const std::vector<ptrdiff_t> &dst_strides,
                        const std::byte *src,
                        const std::vector<ptrdiff_t> &src_strides,
                        const std::vector<size_t> &shape, size_t elem_size) {
    const auto *api = core::context().runtime().api();
    const size_t n = shape.size();
    std::vector<size_t> idx(n, 0);
    size_t total = 1;
    for (size_t s : shape) total *= s;
    for (size_t t = 0; t < total; ++t) {
        size_t doff = 0, soff = 0;
        for (size_t d = 0; d < n; ++d) {
            doff += idx[d] * (size_t)dst_strides[d];
            soff += idx[d] * (size_t)src_strides[d];
        }
        api->memcpy_sync(dst + doff * elem_size, src + soff * elem_size, elem_size, kind);
        for (size_t d = n; d-- > 0;) {
            if (++idx[d] < shape[d]) break;
            idx[d] = 0;
        }
    }
}

// 推导"连续布局"的 strides：strides[i] = Π shape[i+1:]，从最后一维往前乘。
// 与 Tensor::create() 里的推导逻辑完全一致，用于 strided copy 时表示"源/目标端
// 是连续行主序"的那一侧。
std::vector<ptrdiff_t> contiguous_strides(const std::vector<size_t> &shape) {
    std::vector<ptrdiff_t> strides(shape.size());
    size_t stride = 1;
    for (size_t i = 1; i <= shape.size(); ++i) {
        strides[shape.size() - i] = (ptrdiff_t)stride;
        stride *= shape[shape.size() - i];
    }
    return strides;
}

// 根据"源设备 → 目标设备"选择 memcpy 的方向（H2H/H2D/D2H/D2D）。
// CPU 的 memcpy_sync 内部就是 std::memcpy，kind 只是占位；CUDA 版（A4）会真正
// 按 kind 映射到 cudaMemcpy 的拷贝类型。
llaisysMemcpyKind_t memcpy_kind(llaisysDeviceType_t src_dev, llaisysDeviceType_t dst_dev) {
    // 四种组合：CPU→CPU、CPU→设备、设备→CPU、设备→设备
    if (src_dev == LLAISYS_DEVICE_CPU && dst_dev == LLAISYS_DEVICE_CPU) return LLAISYS_MEMCPY_H2H;
    if (src_dev == LLAISYS_DEVICE_CPU) return LLAISYS_MEMCPY_H2D;
    if (dst_dev == LLAISYS_DEVICE_CPU) return LLAISYS_MEMCPY_D2H;
    return LLAISYS_MEMCPY_D2D;
}
} // namespace

bool Tensor::isContiguous() const {
    // 判断标准：从最后一维往第一维走，strides[i] 必须恰好等于"后面所有维度大小
    // 的乘积"。例如 shape (3,4,5) 的连续 strides 是 (20,5,1)：
    //   strides[2]=1 == 1                    ✓
    //   strides[1]=5 == shape[2]=5           ✓
    //   strides[0]=20 == 4*5=20              ✓
    // 任何一步对不上就说明布局被打散（比如 permute/slice 之后），返回 false。
    // ndim=0（标量）时循环不执行，恒为 true。
    ptrdiff_t expected = 1;
    for (size_t i = ndim(); i-- > 0;) {
        if (_meta.strides[i] != expected) return false;
        expected = (ptrdiff_t)_meta.shape[i] * expected;
    }
    return true;
}

tensor_t Tensor::permute(const std::vector<size_t> &order) const {
    const size_t n = ndim();
    CHECK_ARGUMENT(order.size() == n, "permute: order length must equal ndim");

    // Validate that `order` is a permutation of 0..n-1 (no duplicates, no OOB).
    std::vector<bool> seen(n, false);
    for (size_t i = 0; i < n; ++i) {
        CHECK_ARGUMENT(order[i] < n, "permute: order index out of range");
        CHECK_ARGUMENT(!seen[order[i]], "permute: duplicate index in order");
        seen[order[i]] = true;
    }

    // 按 order 重排 shape 和 strides：新 shape[i] = 原 shape[order[i]]，
    // 新 strides[i] = 原 strides[order[i]]（例如 permute(2,0,1) 后 strides 变 (1,20,5)）。
    TensorMeta meta{_meta.dtype, {}, {}};
    meta.shape.resize(n);
    meta.strides.resize(n);
    for (size_t i = 0; i < n; ++i) {
        meta.shape[i] = _meta.shape[order[i]];
        meta.strides[i] = _meta.strides[order[i]];
    }
    // 纯视图操作：共享原 storage，offset 不变，不搬任何数据 —— 所以转置是 O(1) 的。
    return std::shared_ptr<Tensor>(new Tensor(meta, _storage, _offset));
}

tensor_t Tensor::view(const std::vector<size_t> &shape) const {
    CHECK_ARGUMENT(isContiguous(), "view: tensor must be contiguous to view without copying");

    size_t new_numel = 1;
    for (size_t s : shape) new_numel *= s;
    CHECK_ARGUMENT(new_numel == numel(), "view: element count must not change");

    // 新 strides 按"连续布局"推导（与 create() 相同），例如 (6,10) -> (10,1)。
    // 因为原张量连续，新的连续 strides 一定和真实内存布局吻合，所以可以安全共享 storage。
    TensorMeta meta{_meta.dtype, shape, {}};
    meta.strides.resize(shape.size());
    size_t stride = 1;
    for (size_t i = 1; i <= shape.size(); ++i) {
        meta.strides[shape.size() - i] = (ptrdiff_t)stride;
        stride *= shape[shape.size() - i];
    }
    return std::shared_ptr<Tensor>(new Tensor(meta, _storage, _offset));
}

tensor_t Tensor::slice(size_t dim, size_t start, size_t end) const {
    CHECK_ARGUMENT(dim < ndim(), "slice: dim out of range");
    CHECK_ARGUMENT(start <= end && end <= _meta.shape[dim], "slice: invalid [start, end) range");

    TensorMeta meta = _meta;
    // 只有被切的这一维长度变化，strides 不变（切出来的子张量还是"隔 N 个取一个"）。
    meta.shape[dim] = end - start;
    // ★ 最经典的坑：offset 是【字节】单位，strides 是【元素】单位。
    // 起始位置前移 start 个元素 = start * strides[dim]（元素）* elementSize()（字节/元素）。
    // 漏乘 elementSize() 会让数据整体错位，测试立刻失败。
    size_t new_offset = _offset + start * (size_t)_meta.strides[dim] * elementSize();
    return std::shared_ptr<Tensor>(new Tensor(meta, _storage, new_offset));
}

void Tensor::load(const void *src_) {
    core::context().setDevice(deviceType(), deviceId());
    const std::byte *src = static_cast<const std::byte *>(src_);
    const llaisysMemcpyKind_t kind =
        deviceType() == LLAISYS_DEVICE_CPU ? LLAISYS_MEMCPY_H2H : LLAISYS_MEMCPY_H2D;

    // 情况一：连续张量 -> 一整块 memcpy 就完事（最快路径）。
    if (isContiguous()) {
        core::context().runtime().api()->memcpy_sync(
            data(), src, numel() * elementSize(), kind);
    } else {
        // 情况二：非连续张量（如 slice/permute 后的视图）。
        // 约定 src 是该张量"逻辑 shape"的连续行主序数据，
        // 需要按本张量的 strides 逐元素 scatter 到正确位置。
        // 例子：shape(3,4,5) strides(20,5,1) 的 slice(2,1,4)，
        // 源是 3*4*3=36 个连续元素，目标要写到第 1~3 列的位置。
        copy_strided_bytes(kind, data(), _meta.strides, src,
                           contiguous_strides(_meta.shape), _meta.shape, elementSize());
    }
}

tensor_t Tensor::contiguous() const {
    if (isContiguous()) {
        return std::shared_ptr<Tensor>(new Tensor(_meta, _storage, _offset));
    }
    // 分配一块新的连续 storage，再把数据按 strides 逐元素拷过去。
    // create() 内部会 setDevice 到目标设备；这里再显式 setDevice 一次是为了
    // 保证接下来 copy_strided_bytes 里 core::context().runtime() 指向正确的设备。
    auto out = create(_meta.shape, _meta.dtype, deviceType(), deviceId());
    core::context().setDevice(deviceType(), deviceId());
    // 同设备拷贝：CPU 用 H2H，设备用 D2D（kind 对 CPU 实现无实际影响）
    copy_strided_bytes(memcpy_kind(deviceType(), deviceType()),
                       out->data(), contiguous_strides(_meta.shape),
                       data(), _meta.strides, _meta.shape, elementSize());
    return out;
}

tensor_t Tensor::reshape(const std::vector<size_t> &shape) const {
    size_t new_numel = 1;
    for (size_t s : shape) new_numel *= s;
    CHECK_ARGUMENT(new_numel == numel(), "reshape: element count must not change");

    if (shape == _meta.shape) {
        return std::shared_ptr<Tensor>(new Tensor(_meta, _storage, _offset));
    }
    if (isContiguous()) {
        return view(shape);
    }
    return contiguous()->view(shape);
}

tensor_t Tensor::to(llaisysDeviceType_t device_type, int device) const {
    const int target_id = device >= 0 ? device : deviceId();
    // 同一个设备 -> 直接共享 storage 返回视图，无需搬运。
    if (deviceType() == device_type && deviceId() == target_id) {
        return std::shared_ptr<Tensor>(new Tensor(_meta, _storage, _offset));
    }

    // 跨设备搬运前必须保证源是连续的：非连续就先 contiguous() 落成一块连续内存，
    // 否则 memcpy 一整块会拷到错误位置。
    tensor_t src = isContiguous()
                       ? std::shared_ptr<Tensor>(new Tensor(_meta, _storage, _offset))
                       : contiguous();
    // 在目标设备上建同 shape 的连续张量，然后按设备方向选 memcpy kind 一次搬完。
    auto out = create(_meta.shape, _meta.dtype, device_type, target_id);
    core::context().setDevice(device_type, target_id);
    core::context().runtime().api()->memcpy_sync(
        out->data(), src->data(), numel() * elementSize(),
        memcpy_kind(deviceType(), device_type));
    return out;
}

} // namespace llaisys
