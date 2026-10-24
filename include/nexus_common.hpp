#pragma once

#include <cstdint>
#include <cstddef>
#include <string>
#include <vector>
#include <memory>
#include <chrono>
#include <atomic>
#include <mutex>
#include <unordered_map>
#include <functional>
#include <cstring>
#include <iostream>

#if defined(_WIN32) || defined(_WIN64)
    #define NEXUS_PLATFORM_WINDOWS 1
    #ifndef WIN32_LEAN_AND_MEAN
        #define WIN32_LEAN_AND_MEAN
    #endif
    #include <windows.h>
#else
    #define NEXUS_PLATFORM_POSIX 1
    #include <sys/mman.h>
    #include <unistd.h>
#endif

// Cache line and SIMD alignment
constexpr size_t NEXUS_ALIGNMENT = 64; 

namespace nexus {

enum class DataType : uint32_t {
    FP32 = 0,
    FP16 = 1,
    BF16 = 2,
    FP8_E4M3 = 3,
    INT8 = 4,
    INT4_AWQ = 5,
    INT4_GPTQ = 6,
    GGUF_Q4_0 = 7,
    GGUF_Q4_K = 8
};

inline size_t get_element_size_bits(DataType dt) {
    switch (dt) {
        case DataType::FP32: return 32;
        case DataType::FP16: return 16;
        case DataType::BF16: return 16;
        case DataType::FP8_E4M3: return 8;
        case DataType::INT8: return 8;
        case DataType::INT4_AWQ: return 4;
        case DataType::INT4_GPTQ: return 4;
        case DataType::GGUF_Q4_0: return 4; // average bits
        case DataType::GGUF_Q4_K: return 4;
        default: return 32;
    }
}

enum class MemoryTier : uint32_t {
    TIER0_ACTIVE_WORKING_SET = 0, // Fast Accelerator / Pinned L1 cache
    TIER1_HOST_SYSTEM_RAM   = 1, // Paged Host RAM (for inactive weights/layers/experts)
    TIER2_NVME_FLASH        = 2  // Flash / Memory-mapped SSD backing
};

enum class ModelFormat : uint32_t {
    SAFETENSORS = 0,
    GGUF        = 1,
    PYTORCH_BIN = 2,
    ONNX        = 3,
    RAW_BINARY  = 4
};

struct TensorShape {
    std::vector<size_t> dims;

    size_t num_elements() const {
        if (dims.empty()) return 0;
        size_t count = 1;
        for (auto d : dims) count *= d;
        return count;
    }

    size_t total_bytes(DataType dt) const {
        size_t bits = num_elements() * get_element_size_bits(dt);
        return (bits + 7) / 8;
    }
};

struct TensorView {
    std::string name;
    DataType dtype;
    TensorShape shape;
    void* host_ptr = nullptr;
    void* active_ptr = nullptr;
    MemoryTier current_tier = MemoryTier::TIER1_HOST_SYSTEM_RAM;
    size_t byte_size = 0;
    bool is_active = false;
    std::atomic<uint32_t> access_count{0};
    uint64_t last_accessed_step = 0;
};

// Error codes addressing root causes from arXiv:2506.09713
enum class NexusStatus : int32_t {
    SUCCESS = 0,
    ERROR_OUT_OF_HOST_MEMORY = -1,
    ERROR_OUT_OF_TIER0_MEMORY = -2,
    ERROR_FORMAT_UNSUPPORTED = -3,
    ERROR_TENSOR_NOT_FOUND = -4,
    ERROR_SHAPE_MISMATCH = -5,
    ERROR_INVALID_KV_BLOCK = -6,
    ERROR_NUMERICAL_OVERFLOW = -7,
    ERROR_ASYNC_PREFETCH_FAILED = -8
};

inline const char* status_to_string(NexusStatus status) {
    switch (status) {
        case NexusStatus::SUCCESS: return "SUCCESS";
        case NexusStatus::ERROR_OUT_OF_HOST_MEMORY: return "ERROR_OUT_OF_HOST_MEMORY";
        case NexusStatus::ERROR_OUT_OF_TIER0_MEMORY: return "ERROR_OUT_OF_TIER0_MEMORY";
        case NexusStatus::ERROR_FORMAT_UNSUPPORTED: return "ERROR_FORMAT_UNSUPPORTED";
        case NexusStatus::ERROR_TENSOR_NOT_FOUND: return "ERROR_TENSOR_NOT_FOUND";
        case NexusStatus::ERROR_SHAPE_MISMATCH: return "ERROR_SHAPE_MISMATCH";
        case NexusStatus::ERROR_INVALID_KV_BLOCK: return "ERROR_INVALID_KV_BLOCK";
        case NexusStatus::ERROR_NUMERICAL_OVERFLOW: return "ERROR_NUMERICAL_OVERFLOW";
        case NexusStatus::ERROR_ASYNC_PREFETCH_FAILED: return "ERROR_ASYNC_PREFETCH_FAILED";
        default: return "UNKNOWN_ERROR";
    }
}

// Microsecond local latency timer
class ScopedTimer {
public:
    ScopedTimer(double* out_duration_us) : out_duration_(out_duration_us), start_(std::chrono::high_resolution_clock::now()) {}
    ~ScopedTimer() {
        if (out_duration_) {
            auto end = std::chrono::high_resolution_clock::now();
            *out_duration_ = std::chrono::duration<double, std::micro>(end - start_).count();
        }
    }
private:
    double* out_duration_;
    std::chrono::time_point<std::chrono::high_resolution_clock> start_;
};

} // namespace nexus
