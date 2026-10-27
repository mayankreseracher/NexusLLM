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
#include <cstdlib>
#include <new>
#include <iostream>
#include <fstream>
#include <sstream>
#include <thread>

// ============================================================================
// OS & Platform Identification
// ============================================================================
#if defined(_WIN32) || defined(_WIN64)
    #define NEXUS_PLATFORM_WINDOWS 1
    #ifndef WIN32_LEAN_AND_MEAN
        #define WIN32_LEAN_AND_MEAN
    #endif
    #include <windows.h>
#elif defined(__APPLE__) && defined(__MACH__)
    #define NEXUS_PLATFORM_MACOS 1
    #define NEXUS_PLATFORM_POSIX 1
    #include <sys/mman.h>
    #include <sys/types.h>
    #include <sys/sysctl.h>
    #include <unistd.h>
    #include <mach/mach.h>
#elif defined(__linux__)
    #define NEXUS_PLATFORM_LINUX 1
    #define NEXUS_PLATFORM_POSIX 1
    #include <sys/mman.h>
    #include <sys/types.h>
    #include <unistd.h>
    #include <sched.h>
#else
    #define NEXUS_PLATFORM_POSIX 1
    #include <sys/mman.h>
    #include <unistd.h>
#endif

// ============================================================================
// CPU Architecture & Vector Engine Identification
// ============================================================================
#if defined(__aarch64__) || defined(_M_ARM64)
    #define NEXUS_ARCH_AARCH64 1
    #if defined(__ARM_NEON) || defined(__APPLE__) || defined(__linux__)
        #define NEXUS_HAVE_ARM_NEON 1
        #include <arm_neon.h>
    #endif
#elif defined(__x86_64__) || defined(_M_X64)
    #define NEXUS_ARCH_X86_64 1
    #if (defined(__AVX2__) && defined(__FMA__)) || defined(_MSC_VER)
        #if !defined(NEXUS_NO_AVX2)
            #define NEXUS_HAVE_AVX2 1
            #include <immintrin.h>
        #endif
    #endif
#endif

// Cache line and SIMD alignment
constexpr size_t NEXUS_ALIGNMENT = 64; 

inline void* nexus_aligned_alloc(size_t alignment, size_t size) {
    size_t sz = (size + alignment - 1) & ~(alignment - 1);
#if defined(NEXUS_PLATFORM_WINDOWS)
    return VirtualAlloc(NULL, sz, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
#elif defined(NEXUS_PLATFORM_POSIX)
    void* ptr = nullptr;
    if (posix_memalign(&ptr, alignment, sz) == 0) {
        return ptr;
    }
    return nullptr;
#else
    return std::malloc(sz);
#endif
}

inline void nexus_aligned_free(void* ptr) {
    if (!ptr) return;
#if defined(NEXUS_PLATFORM_WINDOWS)
    VirtualFree(ptr, 0, MEM_RELEASE);
#else
    std::free(ptr);
#endif
} 

namespace nexus {

enum class OperatingSystem : uint32_t {
    OS_WINDOWS = 0,
    OS_MACOS_DARWIN = 1,
    OS_LINUX_UBUNTU_DEBIAN = 2,
    OS_LINUX_RHEL_CENTOS_ROCKY = 3,
    OS_LINUX_FEDORA = 4,
    OS_LINUX_ARCH = 5,
    OS_LINUX_ALPINE = 6,
    OS_LINUX_GENERIC = 7,
    OS_OTHER_POSIX = 8
};

enum class HardwareVectorISA : uint32_t {
    ISA_SCALAR_FALLBACK = 0,
    ISA_ARM_NEON = 1,
    ISA_AVX2_FMA = 2,
    ISA_AVX512 = 3
};

struct PlatformHardwareProfile {
    OperatingSystem os = OperatingSystem::OS_OTHER_POSIX;
    std::string os_name = "Unknown POSIX";
    std::string cpu_arch = "generic";
    HardwareVectorISA vector_isa = HardwareVectorISA::ISA_SCALAR_FALLBACK;
    std::string vector_isa_name = "Scalar";
    
    bool is_unified_memory = false;          // True on Apple Silicon (M1-M4) & APUs
    bool has_transparent_hugepages = false;  // Linux THP (2MB/1GB pages)
    bool has_posix_madvise = false;          // madvise support (WILLNEED, DONTNEED, HUGEPAGE)
    bool has_memory_pinning = false;         // mlock / VirtualLock
    uint32_t numa_nodes_count = 1;
    uint32_t hardware_concurrency = 1;
    double theoretical_bandwidth_gbps = 0.0;
    std::string performance_tier_description;

    static PlatformHardwareProfile detect() {
        PlatformHardwareProfile profile;
        profile.hardware_concurrency = std::thread::hardware_concurrency();
        if (profile.hardware_concurrency == 0) profile.hardware_concurrency = 1;

#if defined(NEXUS_PLATFORM_WINDOWS)
        profile.os = OperatingSystem::OS_WINDOWS;
        profile.os_name = "Microsoft Windows (NT Kernel)";
        profile.has_memory_pinning = true;
        profile.has_posix_madvise = false;
        profile.has_transparent_hugepages = false;
        profile.is_unified_memory = false;
        profile.theoretical_bandwidth_gbps = 64.0; // Typical Dual/Quad Channel DDR5
        profile.performance_tier_description = "Windows Tiered Host-VRAM Pipeline via VirtualAlloc & PCIe DMA";
#elif defined(NEXUS_PLATFORM_MACOS)
        profile.os = OperatingSystem::OS_MACOS_DARWIN;
        profile.os_name = "Apple macOS Darwin";
        profile.has_posix_madvise = true;
        profile.has_memory_pinning = true;

#if defined(NEXUS_ARCH_AARCH64)
        profile.is_unified_memory = true; // Apple Silicon Unified Memory Architecture
        profile.cpu_arch = "ARM64 (Apple Silicon M-Series)";
        profile.vector_isa = HardwareVectorISA::ISA_ARM_NEON;
        profile.vector_isa_name = "ARM NEON 128-bit + Apple AMX";
        profile.theoretical_bandwidth_gbps = 410.0; // Base: 100-150 GB/s, Max: 410 GB/s, Ultra: 820 GB/s
        profile.performance_tier_description = "Apple Silicon Zero-Copy UMA Tier (Shared DRAM bus, 0 PCIe latency)";
#else
        profile.is_unified_memory = false;
        profile.cpu_arch = "x86_64 (Intel Mac)";
        profile.vector_isa = HardwareVectorISA::ISA_AVX2_FMA;
        profile.vector_isa_name = "AVX2 + FMA";
        profile.theoretical_bandwidth_gbps = 50.0;
        profile.performance_tier_description = "macOS Intel POSIX mmap Tier";
#endif

#elif defined(NEXUS_PLATFORM_LINUX)
        profile.os = OperatingSystem::OS_LINUX_GENERIC;
        profile.os_name = "Linux Generic (GNU/Linux)";
        profile.has_posix_madvise = true;
        profile.has_memory_pinning = true;

        // Distro identification via /etc/os-release
        std::ifstream os_release("/etc/os-release");
        if (os_release.is_open()) {
            std::string line;
            while (std::getline(os_release, line)) {
                if (line.rfind("ID=", 0) == 0) {
                    std::string id = line.substr(3);
                    if (id.size() >= 2 && id.front() == '"' && id.back() == '"') {
                        id = id.substr(1, id.size() - 2);
                    }
                    if (id == "ubuntu" || id == "debian") {
                        profile.os = OperatingSystem::OS_LINUX_UBUNTU_DEBIAN;
                        profile.os_name = (id == "ubuntu") ? "Ubuntu Linux" : "Debian GNU/Linux";
                    } else if (id == "rhel" || id == "centos" || id == "rocky" || id == "almalinux") {
                        profile.os = OperatingSystem::OS_LINUX_RHEL_CENTOS_ROCKY;
                        profile.os_name = "RHEL/Rocky/CentOS Enterprise Linux";
                    } else if (id == "fedora") {
                        profile.os = OperatingSystem::OS_LINUX_FEDORA;
                        profile.os_name = "Fedora Linux";
                    } else if (id == "arch" || id == "manjaro") {
                        profile.os = OperatingSystem::OS_LINUX_ARCH;
                        profile.os_name = "Arch Linux";
                    } else if (id == "alpine") {
                        profile.os = OperatingSystem::OS_LINUX_ALPINE;
                        profile.os_name = "Alpine Linux (musl libc)";
                    }
                }
            }
        }

        // Check Transparent Huge Pages (THP) support
        std::ifstream thp_file("/sys/kernel/mm/transparent_hugepage/enabled");
        if (thp_file.is_open()) {
            profile.has_transparent_hugepages = true;
        }

        profile.is_unified_memory = false;
        profile.theoretical_bandwidth_gbps = 85.0; // Server/Desktop high-speed DDR5 or PCIe Gen5
        profile.performance_tier_description = "Linux Transparent Huge Pages (THP 2MB) + madvise(WILLNEED) Tier";
#else
        profile.os = OperatingSystem::OS_OTHER_POSIX;
        profile.os_name = "Generic POSIX";
        profile.has_posix_madvise = true;
        profile.is_unified_memory = false;
        profile.theoretical_bandwidth_gbps = 32.0;
        profile.performance_tier_description = "Standard POSIX mmap Fallback Tier";
#endif

#if defined(NEXUS_ARCH_AARCH64) && !defined(NEXUS_PLATFORM_MACOS)
        profile.cpu_arch = "ARM64 (aarch64)";
        profile.vector_isa = HardwareVectorISA::ISA_ARM_NEON;
        profile.vector_isa_name = "ARM NEON";
#elif defined(NEXUS_ARCH_X86_64) && !defined(NEXUS_PLATFORM_MACOS)
        profile.cpu_arch = "x86_64";
        profile.vector_isa = HardwareVectorISA::ISA_AVX2_FMA;
        profile.vector_isa_name = "AVX2 + FMA";
#elif !defined(NEXUS_ARCH_AARCH64) && !defined(NEXUS_ARCH_X86_64)
        profile.cpu_arch = "Unknown ISA";
        profile.vector_isa = HardwareVectorISA::ISA_SCALAR_FALLBACK;
        profile.vector_isa_name = "Scalar Fallback";
#endif
        return profile;
    }
};

enum class DataType : uint32_t {
    FP32 = 0,
    FP16 = 1,
    BF16 = 2,
    FP8_E4M3 = 3,
    INT8 = 4,
    INT4_AWQ = 5,
    INT4_GPTQ = 6,
    GGUF_Q4_0 = 7,
    GGUF_Q4_K = 8,
    GGUF_Q8_0 = 9,
    GGUF_Q5_0 = 10,
    GGUF_Q5_K = 11,
    GGUF_Q6_K = 12,
    GGUF_Q2_K = 13,
    GGUF_Q3_K = 14
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
        case DataType::GGUF_Q2_K: return 3;
        case DataType::GGUF_Q3_K: return 4;
        case DataType::GGUF_Q4_0: return 5;
        case DataType::GGUF_Q4_K: return 5;
        case DataType::GGUF_Q5_0: return 6;
        case DataType::GGUF_Q5_K: return 6;
        case DataType::GGUF_Q6_K: return 7;
        case DataType::GGUF_Q8_0: return 9;
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
constexpr int32_t NEXUS_ERR_MISSING_TENSOR = -4;

enum class NexusStatus : int32_t {
    SUCCESS = 0,
    ERROR_OUT_OF_HOST_MEMORY = -1,
    ERROR_OUT_OF_TIER0_MEMORY = -2,
    ERROR_FORMAT_UNSUPPORTED = -3,
    ERROR_TENSOR_NOT_FOUND = -4,
    ERROR_SHAPE_MISMATCH = -5,
    ERROR_INVALID_KV_BLOCK = -6,
    ERROR_NUMERICAL_OVERFLOW = -7,
    ERROR_ASYNC_PREFETCH_FAILED = -8,
    ERROR_INVALID_ARGUMENT = -9
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
        case NexusStatus::ERROR_INVALID_ARGUMENT: return "ERROR_INVALID_ARGUMENT";
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
