#pragma once

#include "nexus_common.hpp"
#include <string>
#include <vector>
#include <memory>
#include <functional>

namespace nexus {

enum class ComputeDeviceType : uint32_t {
    DEVICE_CPU = 0,
    DEVICE_CUDA = 1,
    DEVICE_APPLE_SILICON_METAL = 2,
    DEVICE_AMD_ROCM = 3,
    DEVICE_INTEL_ARC_SYCL = 4,
    DEVICE_OPENCL = 5
};

struct DeviceCapabilities {
    ComputeDeviceType type = ComputeDeviceType::DEVICE_CPU;
    std::string device_name = "Host CPU (SIMD)";
    std::string vendor = "Generic";
    std::string architecture = "x86_64";
    size_t total_vram_bytes = 0;
    size_t available_vram_bytes = 0;
    bool is_unified_memory = false;
    bool supports_fp16 = true;
    bool supports_int4 = true;
    bool supports_async_dma = true;
    std::string driver_version = "N/A";
    std::string api_version = "C++20 SIMD";
    std::string description;
};

// Abstract Compute Backend Interface
class INexusComputeBackend {
public:
    virtual ~INexusComputeBackend() = default;

    virtual ComputeDeviceType get_type() const = 0;
    virtual const DeviceCapabilities& get_capabilities() const = 0;

    virtual bool initialize() = 0;
    virtual void shutdown() = 0;
    virtual void synchronize() = 0;

    // Memory Allocation & DMA
    virtual void* allocate_device_memory(size_t bytes) = 0;
    virtual void free_device_memory(void* ptr) = 0;
    virtual bool copy_host_to_device_async(void* dst_device, const void* src_host, size_t bytes) = 0;
    virtual bool copy_device_to_host_async(void* dst_host, const void* src_device, size_t bytes) = 0;

    // Compute Kernels
    virtual void gemv_dispatch(
        const void* W,
        DataType dt,
        const float* x,
        float* y,
        size_t rows,
        size_t cols
    ) = 0;

    virtual void paged_attention_decode(
        const float* q,
        const std::vector<int32_t>& block_table,
        class PagedKVCacheManager& kv_mgr,
        size_t layer_idx,
        size_t position,
        size_t num_heads,
        size_t num_kv_heads,
        size_t head_dim,
        float* out
    ) = 0;

    virtual void rmsnorm(
        const float* x,
        const void* weight,
        DataType dt,
        float* out,
        size_t dim,
        float eps
    ) = 0;

    virtual void swiglu(
        const float* gate,
        const float* up,
        float* out,
        size_t dim
    ) = 0;
};

// Central Hardware Backend Registry & Runtime Discovery
class BackendRegistry {
public:
    static BackendRegistry& instance();

    // Query all hardware devices detected on the host system
    std::vector<DeviceCapabilities> get_available_devices();

    // Instantiate backend for specified device type
    std::shared_ptr<INexusComputeBackend> create_backend(ComputeDeviceType type);

    // Auto-detect optimal backend (CUDA -> Apple Silicon Metal -> AMD ROCm -> Intel Arc SYCL -> CPU)
    std::shared_ptr<INexusComputeBackend> create_default_backend();

    // String helpers
    static ComputeDeviceType parse_device_type(const std::string& name);
    static const char* device_type_to_string(ComputeDeviceType type);

private:
    BackendRegistry() = default;
};

} // namespace nexus
