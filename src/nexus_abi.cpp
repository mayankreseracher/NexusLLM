#define NEXUS_BUILD_DLL 1
#include "nexus_abi.h"
#include "nexus_common.hpp"
#include "memory_tier.hpp"
#include "model_loader.hpp"
#include "paged_cache.hpp"
#include "quantization.hpp"
#include "disaggregated_pipeline.hpp"
#include "transformer_engine.hpp"
#include "forward_pass.hpp"
#include "nexus_backend.hpp"

namespace nexus {
    class EngineContext;
}

// Forward declare internal class
class InternalEngine {
public:
    InternalEngine(size_t t0, size_t t1, size_t max_kv) {
        tier_mgr = std::make_shared<nexus::MemoryTierManager>(t0, t1);
        loader = std::make_unique<nexus::UniversalModelLoader>(tier_mgr);
        kv_mgr = std::make_shared<nexus::PagedKVCacheManager>(max_kv, 4, 4, 64);
        control_plane = std::make_unique<nexus::DisaggregatedControlPlane>(kv_mgr);
        backend = nexus::BackendRegistry::instance().create_default_backend();
        transformer = std::make_unique<nexus::TransformerEngine>(tier_mgr, kv_mgr, loader->get_metadata());
        if (transformer && backend) {
            transformer->set_compute_backend(backend);
        }
        forward_executor = std::make_unique<nexus::ForwardPassExecutor>(tier_mgr, kv_mgr, loader->get_metadata());
    }

    std::shared_ptr<nexus::MemoryTierManager> tier_mgr;
    std::unique_ptr<nexus::UniversalModelLoader> loader;
    std::shared_ptr<nexus::PagedKVCacheManager> kv_mgr;
    std::unique_ptr<nexus::DisaggregatedControlPlane> control_plane;
    std::unique_ptr<nexus::TransformerEngine> transformer;
    std::unique_ptr<nexus::ForwardPassExecutor> forward_executor;
    std::shared_ptr<nexus::INexusComputeBackend> backend;
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

NEXUS_API uint32_t nexus_get_available_device_count(void) {
    return static_cast<uint32_t>(nexus::BackendRegistry::instance().get_available_devices().size());
}

NEXUS_API int32_t nexus_get_device_info(uint32_t device_idx, NexusDeviceInfo* out_info) {
    if (!out_info) return -1;
    auto devices = nexus::BackendRegistry::instance().get_available_devices();
    if (device_idx >= devices.size()) return -1;

    const auto& d = devices[device_idx];
    out_info->device_type = static_cast<uint32_t>(d.type);
    std::strncpy(out_info->device_name, d.device_name.c_str(), sizeof(out_info->device_name) - 1);
    out_info->device_name[sizeof(out_info->device_name) - 1] = '\0';
    std::strncpy(out_info->vendor, d.vendor.c_str(), sizeof(out_info->vendor) - 1);
    out_info->vendor[sizeof(out_info->vendor) - 1] = '\0';
    std::strncpy(out_info->architecture, d.architecture.c_str(), sizeof(out_info->architecture) - 1);
    out_info->architecture[sizeof(out_info->architecture) - 1] = '\0';
    out_info->is_unified_memory = d.is_unified_memory ? 1 : 0;
    out_info->supports_fp16 = d.supports_fp16 ? 1 : 0;
    out_info->supports_int4 = d.supports_int4 ? 1 : 0;
    std::strncpy(out_info->driver_version, d.driver_version.c_str(), sizeof(out_info->driver_version) - 1);
    out_info->driver_version[sizeof(out_info->driver_version) - 1] = '\0';
    std::strncpy(out_info->description, d.description.c_str(), sizeof(out_info->description) - 1);
    out_info->description[sizeof(out_info->description) - 1] = '\0';
    return 0;
}

NEXUS_API int32_t nexus_set_compute_device(NexusEngineHandle engine, const char* device_name) {
    if (!engine || !device_name) return -1;
    auto* eng = static_cast<InternalEngine*>(engine);
    auto dev_type = nexus::BackendRegistry::parse_device_type(device_name);
    eng->backend = nexus::BackendRegistry::instance().create_backend(dev_type);
    if (eng->transformer && eng->backend) {
        eng->transformer->set_compute_backend(eng->backend);
    }
    return 0;
}

NEXUS_API const char* nexus_get_active_device_name(NexusEngineHandle engine) {
    if (!engine) return "None";
    auto* eng = static_cast<InternalEngine*>(engine);
    if (!eng->backend) return "None";
    return eng->backend->get_capabilities().device_name.c_str();
}

NEXUS_API int32_t nexus_prefetch_layer_index(NexusEngineHandle engine, uint32_t layer_idx) {
    if (!engine) return -1;
    auto* eng = static_cast<InternalEngine*>(engine);
    return static_cast<int32_t>(eng->tier_mgr->prefetch_layer_async(layer_idx));
}

NEXUS_API int32_t nexus_deactivate_layer_index(NexusEngineHandle engine, uint32_t layer_idx) {
    if (!engine) return -1;
    auto* eng = static_cast<InternalEngine*>(engine);
    return static_cast<int32_t>(eng->tier_mgr->deactivate_layer(layer_idx));
}

NEXUS_API int32_t nexus_load_model_auto(NexusEngineHandle engine, const char* filepath) {
    if (!engine || !filepath) return -1;
    auto* eng = static_cast<InternalEngine*>(engine);
    auto status = eng->loader->load_model(filepath);
    if (status == nexus::NexusStatus::SUCCESS) {
        const auto& meta = eng->loader->get_metadata();
        size_t n_layers = meta.num_hidden_layers > 0 ? meta.num_hidden_layers : 4;
        size_t n_heads = meta.num_attention_heads > 0 ? meta.num_attention_heads : 4;
        size_t n_kv_heads = meta.num_key_value_heads > 0 ? meta.num_key_value_heads : n_heads;
        size_t h_dim = (n_heads > 0 && meta.hidden_size > 0) ? (meta.hidden_size / n_heads) : 64;
        eng->kv_mgr = std::make_shared<nexus::PagedKVCacheManager>(eng->kv_mgr->get_total_blocks_count(), n_layers, n_kv_heads, h_dim);
        eng->control_plane = std::make_unique<nexus::DisaggregatedControlPlane>(eng->kv_mgr);
        eng->transformer = std::make_unique<nexus::TransformerEngine>(
            eng->tier_mgr, eng->kv_mgr, meta
        );
        if (eng->backend) {
            eng->transformer->set_compute_backend(eng->backend);
        }
        eng->forward_executor = std::make_unique<nexus::ForwardPassExecutor>(
            eng->tier_mgr, eng->kv_mgr, meta
        );
    }
    return static_cast<int32_t>(status);
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

    // Submits to control plane which matches prefix and dynamically allocates needed blocks
    eng->control_plane->submit_request(req);
    return 0;
}

NEXUS_API int32_t nexus_step_continuous_batch(NexusEngineHandle engine, 
                                             char** out_finished_ids, 
                                             int32_t* out_tokens, 
                                             size_t max_out) {
    if (!engine) return 0;
    auto* eng = static_cast<InternalEngine*>(engine);

    // Continuous Batching Interleaved Schedule (Sarathi-Serve / Orca: chunked prefill + decode in same step)
    auto interleaved = eng->control_plane->schedule_interleaved_iteration(1024);
    if (eng->forward_executor) {
        return eng->forward_executor->execute_interleaved_batch(interleaved, *eng->control_plane, out_tokens, max_out);
    }

    int32_t step_count = 0;

    // 1. Process chunked prefills in this iteration
    for (auto& item : interleaved.prefill_chunks) {
        auto req = item.first;
        if (req->processed_tokens >= req->prompt_tokens.size()) {
            int32_t first_token = eng->transformer->forward_prefill(
                req->prompt_tokens,
                req->assigned_block_ids,
                req->temperature
            );
            eng->control_plane->complete_prefill_and_handoff(req, first_token);
            if (out_tokens && static_cast<size_t>(step_count) < max_out) {
                out_tokens[step_count++] = first_token;
            }
        }
    }

    // 2. Process active single-token decodes in this iteration
    for (size_t i = 0; i < interleaved.decode_requests.size() && static_cast<size_t>(step_count) < max_out; ++i) {
        auto& req = interleaved.decode_requests[i];
        if (req->is_finished) continue;

        int32_t last_token = req->generated_tokens.empty() ? 
            (req->prompt_tokens.empty() ? 1 : req->prompt_tokens.back()) :
            req->generated_tokens.back();

        size_t current_pos = req->prompt_tokens.size() + req->generated_tokens.size();

        // Ensure dynamic PagedAttention block allocation if sequence crosses block boundary
        size_t needed_block_idx = current_pos / nexus::BLOCK_SIZE;
        while (needed_block_idx >= req->assigned_block_ids.size()) {
            auto new_blocks = eng->kv_mgr->allocate_blocks(1);
            if (!new_blocks.empty()) {
                req->assigned_block_ids.push_back(new_blocks[0]);
            } else {
                break;
            }
        }

        // Full Transformer forward pass (FlashAttention CPU Tiled Online Softmax + SwiGLU FFN + Sampling)
        int32_t next_token = eng->transformer->forward_token(
            last_token,
            current_pos,
            req->assigned_block_ids,
            req->temperature
        );

        bool is_eos = (next_token == 2 || (req->max_new_tokens > 0 && req->generated_tokens.size() >= req->max_new_tokens));
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

NEXUS_API int32_t nexus_forward_token(NexusEngineHandle engine, int32_t current_token, size_t position, float temperature) {
    if (!engine) return 0;
    auto* eng = static_cast<InternalEngine*>(engine);
    size_t blocks_needed = (position + nexus::BLOCK_SIZE) / nexus::BLOCK_SIZE;
    auto blocks = eng->kv_mgr->allocate_blocks(blocks_needed > 0 ? blocks_needed : 1);
    if (eng->forward_executor) {
        return eng->forward_executor->forward_token(current_token, position, blocks, temperature);
    }
    return eng->transformer->forward_token(current_token, position, blocks, temperature);
}

NEXUS_API int32_t nexus_forward_sequence(NexusEngineHandle engine, const int32_t* prompt_tokens, size_t num_tokens, float temperature) {
    if (!engine || !prompt_tokens || num_tokens == 0) return 0;
    auto* eng = static_cast<InternalEngine*>(engine);
    std::vector<int32_t> tokens(prompt_tokens, prompt_tokens + num_tokens);
    size_t blocks_needed = (num_tokens + nexus::BLOCK_SIZE) / nexus::BLOCK_SIZE;
    auto blocks = eng->kv_mgr->allocate_blocks(blocks_needed > 0 ? blocks_needed : 1);
    if (eng->forward_executor) {
        return eng->forward_executor->forward_prefill(tokens, blocks, temperature);
    }
    return eng->transformer->forward_prefill(tokens, blocks, temperature);
}

NEXUS_API int32_t nexus_step_speculative_verify(
    NexusEngineHandle engine,
    int32_t current_token,
    size_t position,
    const int32_t* draft_tokens,
    size_t num_drafts,
    int32_t* out_accepted_tokens,
    size_t* out_num_accepted,
    int32_t* out_correction_token
) {
    if (!engine || !draft_tokens || num_drafts == 0) return -1;
    auto* eng = static_cast<InternalEngine*>(engine);

    std::vector<int32_t> drafts(draft_tokens, draft_tokens + num_drafts);
    size_t blocks_needed = (position + num_drafts + nexus::BLOCK_SIZE) / nexus::BLOCK_SIZE;
    auto blocks = eng->kv_mgr->allocate_blocks(blocks_needed > 0 ? blocks_needed : 1);

    auto result = eng->transformer->forward_speculative_verify(current_token, position, drafts, blocks, 0.0f);

    if (out_accepted_tokens) {
        for (size_t i = 0; i < result.accepted_tokens.size(); ++i) {
            out_accepted_tokens[i] = result.accepted_tokens[i];
        }
    }
    if (out_num_accepted) {
        *out_num_accepted = result.num_accepted;
    }
    if (out_correction_token) {
        *out_correction_token = result.bonus_or_correction_token;
    }

    return 0;
}

// Numerical Verification Unit-Test ABI only:
// Tests FlashAttention tiled online softmax against naive attention using deterministically varied
// synthetic vectors (sin/cos) across multi-block sequences to guarantee exact mathematical equivalence.
// Never invoked during model inference or forward decoding.
NEXUS_API float nexus_verify_flash_attention(void) {
    const size_t num_heads = 4;
    const size_t num_kv_heads = 4;
    const size_t head_dim = 64;
    const size_t seq_len = 35; // spans 3 blocks

    nexus::PagedKVCacheManager kv_mgr(16, 1, num_kv_heads, head_dim);
    size_t blocks_needed = (seq_len + nexus::BLOCK_SIZE) / nexus::BLOCK_SIZE;
    auto block_table = kv_mgr.allocate_blocks(blocks_needed);

    // Populate KV cache with deterministically varied vectors
    for (size_t pos = 0; pos <= seq_len; ++pos) {
        size_t blk_idx = pos / nexus::BLOCK_SIZE;
        size_t slot = pos % nexus::BLOCK_SIZE;
        auto* blk = kv_mgr.get_block(block_table[blk_idx]);
        size_t stride = num_kv_heads * nexus::BLOCK_SIZE * head_dim;
        for (size_t h = 0; h < num_kv_heads; ++h) {
            float* k_ptr = blk->key_data + (h * nexus::BLOCK_SIZE * head_dim) + (slot * head_dim);
            float* v_ptr = blk->value_data + (h * nexus::BLOCK_SIZE * head_dim) + (slot * head_dim);
            for (size_t d = 0; d < head_dim; ++d) {
                k_ptr[d] = std::sin(static_cast<float>(pos * 17 + h * 13 + d));
                v_ptr[d] = std::cos(static_cast<float>(pos * 19 + h * 7 + d));
            }
        }
    }

    std::vector<float> Q(num_heads * head_dim);
    for (size_t i = 0; i < Q.size(); ++i) {
        Q[i] = std::sin(static_cast<float>(i * 3 + 1));
    }

    std::vector<float> out_flash(num_heads * head_dim, 0.0f);
    std::vector<float> out_naive(num_heads * head_dim, 0.0f);

    nexus::TransformerOps::paged_attention_decode(
        Q.data(), block_table, kv_mgr, 0, seq_len, num_heads, num_kv_heads, head_dim, out_flash.data()
    );

    nexus::TransformerOps::paged_attention_naive(
        Q.data(), block_table, kv_mgr, 0, seq_len, num_heads, num_kv_heads, head_dim, out_naive.data()
    );

    float max_diff = 0.0f;
    for (size_t i = 0; i < out_flash.size(); ++i) {
        float diff = std::abs(out_flash[i] - out_naive[i]);
        if (diff > max_diff) max_diff = diff;
    }
    return max_diff;
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
