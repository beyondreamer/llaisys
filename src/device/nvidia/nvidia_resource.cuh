#pragma once

#include "../device_resource.hpp"

namespace llaisys::device::nvidia {
// 设备资源类（A4）：记录设备类型/编号。
// 后续可扩展持有 cuBLAS/cuDNN handle 等长生命周期的设备资源
// （如线性层用 cuBLAS 加速时，把 cublasHandle_t 放这里避免反复创建）。
class Resource : public llaisys::device::DeviceResource {
public:
    Resource(int device_id);
    ~Resource();
};
} // namespace llaisys::device::nvidia
