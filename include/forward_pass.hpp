#pragma once

#include "nexus_common.hpp"
#include "memory_tier.hpp"
#include "paged_cache.hpp"
#include "model_loader.hpp"
#include "disaggregated_pipeline.hpp"
#include "quantization.hpp"
#include "transformer_ops.hpp"
#include <vector>
#include <memory>
#include <string>

namespace nexus {

class ForwardPassExecutor {
public:
    ForwardPassExecutor(
        std::shared_ptr<MemoryTierManager> tier_mgr,
        std::shared_ptr<PagedKVCacheManager> kv_mgr,
        const ModelMetadata& metadata
    );

    // Core single-token forward evaluation loop (matches llama.cpp's llama_eval)
    // Orchestrates:
    // hidden = embed(tokens)
    // for layer in layers:
    //     hidden = rmsnorm(hidden)
    //     q, k, v = gemv(hidden, w_qkv)          # calls fused_gemv_*
    //     q, k = rope(q, k, pos)
    //     kv_mgr->write_block(block_id)           # write K and V
    //     attn = paged_attention_cpu(q)          # calls kv_mgr->read_block(block_id)
    //     hidden = hidden + gemv(attn, w_o)
    //     hidden = rmsnorm(hidden)
    //     ff = gemv(hidden, w_gate) * gemv(hidden, w_up)
    //     hidden = hidden + gemv(ff, w_down)
    // logits = gemv(hidden, w_lm_head)
    // token = sample(logits)
    int32_t forward_token(
        int32_t token,
        size_t pos,
        const std::vector<int32_t>& block_table,
        float temperature = 0.0f
    );

    // Tiled PagedAttention CPU kernel: reads K and V directly from kv_mgr->read_block(block_id)
    void paged_attention_cpu(
        const float* Q,
        const std::vector<int32_t>& block_table,
        size_t layer_idx,
        size_t seq_len,
        float* attn_out
    );

    // Continuous Batching Orchestration:
    // Takes InterleavedBatch from DisaggregatedControlPlane scheduler,
    // runs chunked prefill + decode forward passes, and advances requests
    int32_t execute_interleaved_batch(
        DisaggregatedControlPlane::InterleavedBatch& batch,
        DisaggregatedControlPlane& control_plane,
        int32_t* out_tokens,
        size_t max_out
    );

    int32_t forward_prefill(
        const std::vector<int32_t>& prompt_tokens,
        const std::vector<int32_t>& block_table,
        float temperature = 0.0f
    );

private:
    std::shared_ptr<MemoryTierManager> tier_mgr_;
    std::shared_ptr<PagedKVCacheManager> kv_mgr_;
    ModelMetadata metadata_;

    size_t hidden_size_ = 256;
    size_t intermediate_size_ = 512;
    size_t num_layers_ = 4;
    size_t num_heads_ = 4;
    size_t num_kv_heads_ = 4;
    size_t head_dim_ = 64;
    size_t vocab_size_ = 32000;

    // Intermediate activation buffers
    std::vector<float> act_hidden_;
    std::vector<float> act_norm_;
    std::vector<float> act_q_;
    std::vector<float> act_k_;
    std::vector<float> act_v_;
    std::vector<float> act_attn_out_;
    std::vector<float> act_attn_proj_;
    std::vector<float> act_gate_;
    std::vector<float> act_up_;
    std::vector<float> act_swiglu_;
    std::vector<float> act_down_;
    std::vector<float> act_logits_;

    // Default weight buffers
    bool weights_initialized_ = false;
    std::vector<float> default_embed_;
    std::vector<std::vector<float>> default_layer_data_;
    std::vector<float> default_out_norm_;
    std::vector<float> default_lm_head_;

    void initialize_default_weights();
};

} // namespace nexus
