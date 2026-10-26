#define NEXUS_BUILD_DLL 1
#include "nexus_abi.h"
#include "nexus_common.hpp"
#include "memory_tier.hpp"
#include "model_loader.hpp"
#include "paged_cache.hpp"
#include "quantization.hpp"
#include "disaggregated_pipeline.hpp"

namespace nexus {
    class EngineContext;
}

// Forward declare internal class
class InternalEngine {
public:
    InternalEngine(size_t t0, size_t t1, size_t max_kv) {
        tier_mgr = std::make_shared<nexus::MemoryTierManager>(t0, t1);
        loader = std::make_unique<nexus::UniversalModelLoader>(tier_mgr);
        kv_mgr = std::make_shared<nexus::PagedKVCacheManager>(max_kv, 32, 32, 128);
        control_plane = std::make_unique<nexus::DisaggregatedControlPlane>(kv_mgr);
    }

    std::shared_ptr<nexus::MemoryTierManager> tier_mgr;
    std::unique_ptr<nexus::UniversalModelLoader> loader;
    std::shared_ptr<nexus::PagedKVCacheManager> kv_mgr;
    std::unique_ptr<nexus::DisaggregatedControlPlane> control_plane;
};

extern "C" {

NEXUS_API NexusEngineHandle nexus_create_engine(size_t tier0_bytes, size_t tier1_bytes, size_t max_kv_blocks) {
    return new InternalEngine(tier0_bytes, tier1_bytes, max_kv_blocks);
}

NEXUS_API void nexus_destroy_engine(NexusEngineHandle engine) {
    if (engine) {
        delete static_cast<InternalEngine*>(engine);
    }
}

NEXUS_API void nexus_get_platform_info(NexusPlatformInfo* out_info) {
    if (!out_info) return;
    auto profile = nexus::PlatformHardwareProfile::detect();
    out_info->os_type = static_cast<uint32_t>(profile.os);
    std::strncpy(out_info->os_name, profile.os_name.c_str(), sizeof(out_info->os_name) - 1);
    out_info->os_name[sizeof(out_info->os_name) - 1] = '\0';
    std::strncpy(out_info->cpu_arch, profile.cpu_arch.c_str(), sizeof(out_info->cpu_arch) - 1);
    out_info->cpu_arch[sizeof(out_info->cpu_arch) - 1] = '\0';
    out_info->vector_isa = static_cast<uint32_t>(profile.vector_isa);
    std::strncpy(out_info->vector_isa_name, profile.vector_isa_name.c_str(), sizeof(out_info->vector_isa_name) - 1);
    out_info->vector_isa_name[sizeof(out_info->vector_isa_name) - 1] = '\0';
    out_info->is_unified_memory = profile.is_unified_memory ? 1 : 0;
    out_info->has_transparent_hugepages = profile.has_transparent_hugepages ? 1 : 0;
    out_info->hardware_concurrency = profile.hardware_concurrency;
    out_info->theoretical_bandwidth_gbps = profile.theoretical_bandwidth_gbps;
    std::strncpy(out_info->performance_tier_desc, profile.performance_tier_description.c_str(), sizeof(out_info->performance_tier_desc) - 1);
    out_info->performance_tier_desc[sizeof(out_info->performance_tier_desc) - 1] = '\0';
}

NEXUS_API int32_t nexus_load_model_auto(NexusEngineHandle engine, const char* filepath) {
    if (!engine || !filepath) return -1;
    auto* eng = static_cast<InternalEngine*>(engine);
    return static_cast<int32_t>(eng->loader->load_model(filepath));
}

NEXUS_API int32_t nexus_get_loaded_tensor_count(NexusEngineHandle engine) {
    if (!engine) return 0;
    auto* eng = static_cast<InternalEngine*>(engine);
    return static_cast<int32_t>(eng->loader->get_loaded_tensor_count());
}

NEXUS_API const char* nexus_get_model_architecture(NexusEngineHandle engine) {
    if (!engine) return "unknown";
    auto* eng = static_cast<InternalEngine*>(engine);
    return eng->loader->get_metadata().architecture.c_str();
}

NEXUS_API int32_t nexus_prefetch_layer(NexusEngineHandle engine, const char* layer_name) {
    if (!engine || !layer_name) return -1;
    auto* eng = static_cast<InternalEngine*>(engine);
    return static_cast<int32_t>(eng->tier_mgr->prefetch_parameter_async(layer_name));
}

NEXUS_API int32_t nexus_deactivate_layer(NexusEngineHandle engine, const char* layer_name) {
    if (!engine || !layer_name) return -1;
    auto* eng = static_cast<InternalEngine*>(engine);
    return static_cast<int32_t>(eng->tier_mgr->deactivate_parameter(layer_name));
}

NEXUS_API void* nexus_get_active_buffer(NexusEngineHandle engine, const char* tensor_name) {
    if (!engine || !tensor_name) return nullptr;
    auto* eng = static_cast<InternalEngine*>(engine);
    return eng->tier_mgr->activate_parameter(tensor_name);
}

NEXUS_API int32_t nexus_submit_request(NexusEngineHandle engine, 
                                      const char* request_id, 
                                      const int32_t* prompt_tokens, 
                                      size_t prompt_len, 
                                      size_t max_new_tokens) {
    if (!engine || !request_id || !prompt_tokens) return -1;
    auto* eng = static_cast<InternalEngine*>(engine);

    auto req = std::make_shared<nexus::InferenceRequest>();
    req->request_id = request_id;
    req->prompt_tokens.assign(prompt_tokens, prompt_tokens + prompt_len);
    req->max_new_tokens = max_new_tokens;

    // Allocate blocks if not cached
    size_t blocks_needed = (prompt_len + nexus::BLOCK_SIZE - 1) / nexus::BLOCK_SIZE;
    req->assigned_block_ids = eng->kv_mgr->allocate_blocks(blocks_needed);

    eng->control_plane->submit_request(req);
    return 0;
}

NEXUS_API int32_t nexus_step_continuous_batch(NexusEngineHandle engine, 
                                             char** out_finished_ids, 
                                             int32_t* out_tokens, 
                                             size_t max_out) {
    if (!engine) return 0;
    auto* eng = static_cast<InternalEngine*>(engine);

    // 1. Process chunked prefill
    auto prefill_batch = eng->control_plane->schedule_prefill_batch(1024);
    for (auto& req : prefill_batch) {
        if (req->processed_tokens >= req->prompt_tokens.size()) {
            eng->control_plane->complete_prefill_and_handoff(req, 100); // first token
        }
    }

    // 2. Process decode batch
    auto decode_batch = eng->control_plane->schedule_decode_batch(max_out);
    int32_t step_count = 0;

    for (size_t i = 0; i < decode_batch.size() && static_cast<size_t>(step_count) < max_out; ++i) {
        auto& req = decode_batch[i];
        int32_t next_token = 1000 + static_cast<int32_t>(req->generated_tokens.size());
        bool is_eos = (req->generated_tokens.size() + 1 >= req->max_new_tokens);
        eng->control_plane->step_decode_token(req, next_token, is_eos);

        if (out_tokens) out_tokens[step_count] = next_token;
        step_count++;
    }

    return step_count;
}

NEXUS_API int32_t nexus_abort_request(NexusEngineHandle engine, const char* request_id) {
    if (!engine || !request_id) return -1;
    auto* eng = static_cast<InternalEngine*>(engine);
    eng->control_plane->abort_request(request_id);
    return 0;
}

NEXUS_API void nexus_get_memory_stats(NexusEngineHandle engine, NexusLocalMemoryStats* out_stats) {
    if (!engine || !out_stats) return;
    auto* eng = static_cast<InternalEngine*>(engine);
    auto tstats = eng->tier_mgr->get_stats();

    out_stats->tier0_allocated_bytes = tstats.tier0_allocated_bytes;
    out_stats->tier0_capacity_bytes = tstats.tier0_capacity_bytes;
    out_stats->tier1_host_ram_bytes = tstats.tier1_host_ram_bytes;
    out_stats->tier1_inactive_bytes = tstats.tier1_inactive_bytes;
    out_stats->bytes_transferred_tier1_to_tier0 = tstats.bytes_transferred_tier1_to_tier0;
    out_stats->prefetch_hits = tstats.prefetch_hits;
    out_stats->prefetch_misses = tstats.prefetch_misses;
    out_stats->pcie_bandwidth_gbps = tstats.pcie_bandwidth_gbps;
    out_stats->prefix_cache_hit_rate = eng->kv_mgr->get_prefix_hit_rate();
}

NEXUS_API void nexus_fused_gemv_int4(const uint32_t* W_packed, 
                                    const float* x, 
                                    const float* scales, 
                                    const float* zeros, 
                                    float* y, 
                                    size_t rows, 
                                    size_t cols, 
                                    size_t group_size) {
    nexus::QuantizationKernel::fused_gemv_int4(W_packed, x, scales, zeros, y, rows, cols, group_size);
}

NEXUS_API void nexus_fused_gemv_fp8(const uint8_t* W_fp8, 
                                   const float* x, 
                                   float* y, 
                                   size_t rows, 
                                   size_t cols, 
                                   float scale) {
    nexus::QuantizationKernel::fused_gemv_fp8(W_fp8, x, y, rows, cols, scale);
}

} // extern "C"
