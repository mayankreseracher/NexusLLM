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

// Lifecycle
NEXUS_API NexusEngineHandle nexus_create_engine(size_t tier0_bytes, size_t tier1_bytes, size_t max_kv_blocks);
NEXUS_API void nexus_destroy_engine(NexusEngineHandle engine);

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
