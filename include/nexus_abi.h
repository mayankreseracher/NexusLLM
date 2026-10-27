#pragma once

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#if defined(_WIN32) || defined(_WIN64)
    #ifdef NEXUS_BUILD_DLL
        #define NEXUS_API __declspec(dllexport)
    #else
        #define NEXUS_API __declspec(dllimport)
    #endif
#else
    #define NEXUS_API __attribute__((visibility("default")))
#endif

typedef void* NexusEngineHandle;
typedef void* NexusTierMgrHandle;
typedef void* NexusModelLoaderHandle;

typedef struct {
    size_t tier0_allocated_bytes;
    size_t tier0_capacity_bytes;
    size_t tier1_host_ram_bytes;
    size_t tier1_inactive_bytes;
    size_t bytes_transferred_tier1_to_tier0;
    size_t prefetch_hits;
    size_t prefetch_misses;
    double pcie_bandwidth_gbps;
    double prefix_cache_hit_rate;
} NexusLocalMemoryStats;

typedef struct {
    uint32_t os_type;            // 0=Windows, 1=macOS, 2=Ubuntu/Debian, 3=RHEL/CentOS, 4=Fedora, 5=Arch, 6=Alpine, 7=Generic Linux, 8=POSIX
    char os_name[64];
    char cpu_arch[32];
    uint32_t vector_isa;         // 0=Scalar, 1=NEON, 2=AVX2+FMA, 3=AVX512
    char vector_isa_name[32];
    uint32_t is_unified_memory;  // 1 if Apple Silicon UMA / APU, 0 if discrete PCIe
    uint32_t has_transparent_hugepages; // 1 if Linux THP enabled
    uint32_t hardware_concurrency;
    double theoretical_bandwidth_gbps;
    char performance_tier_desc[128];
} NexusPlatformInfo;

typedef struct {
    uint32_t device_type;       // 0=CPU, 1=CUDA, 2=Apple Silicon Metal, 3=AMD ROCm, 4=Intel Arc SYCL, 5=OpenCL
    char device_name[64];
    char vendor[64];
    char architecture[32];
    uint32_t is_unified_memory;
    uint32_t supports_fp16;
    uint32_t supports_int4;
    char driver_version[64];
    char description[128];
} NexusDeviceInfo;

// Lifecycle
NEXUS_API NexusEngineHandle nexus_create_engine(size_t tier0_bytes, size_t tier1_bytes, size_t max_kv_blocks);
NEXUS_API void nexus_destroy_engine(NexusEngineHandle engine);

// Platform & Hardware Capability Discovery
NEXUS_API void nexus_get_platform_info(NexusPlatformInfo* out_info);

// Hardware Accelerator Backend Management (Phase 5: CUDA, Apple Silicon Metal/CuMetal, AMD ROCm, Intel Arc SYCL)
NEXUS_API uint32_t nexus_get_available_device_count(void);
NEXUS_API int32_t nexus_get_device_info(uint32_t device_idx, NexusDeviceInfo* out_info);
NEXUS_API int32_t nexus_set_compute_device(NexusEngineHandle engine, const char* device_name);
NEXUS_API const char* nexus_get_active_device_name(NexusEngineHandle engine);

// Overlapped Double-Buffering Layer Prefetch (Phase 4)
NEXUS_API int32_t nexus_prefetch_layer_index(NexusEngineHandle engine, uint32_t layer_idx);
NEXUS_API int32_t nexus_deactivate_layer_index(NexusEngineHandle engine, uint32_t layer_idx);

// Model loading
NEXUS_API int32_t nexus_load_model_auto(NexusEngineHandle engine, const char* filepath);
NEXUS_API int32_t nexus_get_loaded_tensor_count(NexusEngineHandle engine);
NEXUS_API const char* nexus_get_model_architecture(NexusEngineHandle engine);

// Memory tiering & Inactive RAM paging
NEXUS_API int32_t nexus_prefetch_layer(NexusEngineHandle engine, const char* layer_name);
NEXUS_API int32_t nexus_deactivate_layer(NexusEngineHandle engine, const char* layer_name);
NEXUS_API void* nexus_get_active_buffer(NexusEngineHandle engine, const char* tensor_name);

// Inference execution & continuous batching
NEXUS_API int32_t nexus_submit_request(NexusEngineHandle engine, 
                                      const char* request_id, 
                                      const int32_t* prompt_tokens, 
                                      size_t prompt_len, 
                                      size_t max_new_tokens);

NEXUS_API int32_t nexus_step_continuous_batch(NexusEngineHandle engine, 
                                             char** out_finished_ids, 
                                             int32_t* out_tokens, 
                                             size_t max_out);

NEXUS_API int32_t nexus_abort_request(NexusEngineHandle engine, const char* request_id);

// Direct Model Forward Pass & Sampling (Embedding -> [RMSNorm -> QKV -> RoPE -> FlashAttention -> Wo -> RMSNorm -> SwiGLU]xL -> FinalRMSNorm -> LM Head -> Sample)
NEXUS_API int32_t nexus_forward_token(NexusEngineHandle engine, int32_t current_token, size_t position, float temperature);
NEXUS_API int32_t nexus_forward_sequence(NexusEngineHandle engine, const int32_t* prompt_tokens, size_t num_tokens, float temperature);

// Speculative Decoding Verification (Medusa / EAGLE-2 / MTP)
NEXUS_API int32_t nexus_step_speculative_verify(
    NexusEngineHandle engine,
    int32_t current_token,
    size_t position,
    const int32_t* draft_tokens,
    size_t num_drafts,
    int32_t* out_accepted_tokens,
    size_t* out_num_accepted,
    int32_t* out_correction_token
);

// Mathematical Verification: Computes max absolute difference between FlashAttention Tiled Online Softmax and Naive Attention
NEXUS_API float nexus_verify_flash_attention(void);

// Local hardware memory accounting (strictly offline / zero telemetry)
NEXUS_API void nexus_get_memory_stats(NexusEngineHandle engine, NexusLocalMemoryStats* out_stats);

// Direct SIMD Quantized Kernel Execution
NEXUS_API void nexus_fused_gemv_int4(const uint32_t* W_packed, 
                                    const float* x, 
                                    const float* scales, 
                                    const float* zeros, 
                                    float* y, 
                                    size_t rows, 
                                    size_t cols, 
                                    size_t group_size);

NEXUS_API void nexus_fused_gemv_fp8(const uint8_t* W_fp8, 
                                   const float* x, 
                                   float* y, 
                                   size_t rows, 
                                   size_t cols, 
                                   float scale);

#ifdef __cplusplus
}
#endif
