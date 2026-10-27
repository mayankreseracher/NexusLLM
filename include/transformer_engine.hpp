#pragma once

#include "nexus_common.hpp"
#include "memory_tier.hpp"
#include "paged_cache.hpp"
#include "transformer_ops.hpp"
#include "model_loader.hpp"
#include <vector>
#include <string>
#include <memory>

namespace nexus {

struct TransformerLayerWeights {
    const float* attn_norm = nullptr;
    const float* wq = nullptr;
    const float* wk = nullptr;
    const float* wv = nullptr;
    const float* wo = nullptr;
    const float* ffn_norm = nullptr;
    const float* w_gate = nullptr;
    const float* w_up = nullptr;
    const float* w_down = nullptr;
};

// Speculative Decoding Verification Output (Medusa / EAGLE-2 / MTP)
struct SpeculativeVerifyResult {
    std::vector<int32_t> accepted_tokens; // Draft tokens confirmed by target model
    int32_t bonus_or_correction_token = 0; // Final bonus token (if all accepted) or target correction token
    size_t num_accepted = 0;               // Count of accepted draft tokens
    bool all_drafts_accepted = false;      // True if all proposed drafts matched target logits
};

class TransformerEngine {
public:
    TransformerEngine(std::shared_ptr<MemoryTierManager> tier_mgr,
                      std::shared_ptr<PagedKVCacheManager> kv_mgr,
                      const ModelMetadata& metadata);

    // Initialize real model weights into MemoryTierManager if not loaded from file
    void initialize_default_weights();

    // Full Transformer Forward Pass for a single token at position
    // Computes: Embedding -> [RMSNorm -> QKV -> RoPE -> FlashAttention -> Wo -> RMSNorm -> SwiGLU]xLayers -> FinalRMSNorm -> Logits -> Sample
    int32_t forward_token(
        int32_t current_token,
        size_t position,
        const std::vector<int32_t>& block_table,
        float temperature = 0.0f
    );

    // Prefill full sequence of prompt tokens and return next generated token
    int32_t forward_prefill(
        const std::vector<int32_t>& prompt_tokens,
        const std::vector<int32_t>& block_table,
        float temperature = 0.0f
    );

    // Forward evaluate arbitrary token sequence and return logits of final token
    std::vector<float> forward_sequence_logits(
        const std::vector<int32_t>& tokens,
        const std::vector<int32_t>& block_table
    );

    // Multi-Token Prediction (MTP) / Medusa Draft Proposal (arXiv:2401.10774 / arXiv:2402.12567)
    // Predicts top-1 candidate draft tokens for subsequent K positions
    std::vector<int32_t> predict_mtp_draft(
        int32_t current_token,
        size_t current_pos,
        size_t num_draft_tokens
    );

    // Speculative Decoding Verification Loop (EAGLE-2 / Medusa / MTP)
    // Verifies proposed draft tokens against target model forward pass.
    // Automatically accepts matching tokens, emits correction on rejection,
    // and rolls back KV cache state in O(1) time.
    SpeculativeVerifyResult forward_speculative_verify(
        int32_t current_token,
        size_t position,
        const std::vector<int32_t>& draft_tokens,
        const std::vector<int32_t>& block_table,
        float temperature = 0.0f
    );

    const ModelMetadata& get_metadata() const { return metadata_; }

    void set_compute_backend(std::shared_ptr<class INexusComputeBackend> backend) {
        backend_ = backend;
    }
    std::shared_ptr<class INexusComputeBackend> get_compute_backend() const {
        return backend_;
    }

private:
    std::shared_ptr<MemoryTierManager> tier_mgr_;
    std::shared_ptr<PagedKVCacheManager> kv_mgr_;
    std::shared_ptr<class INexusComputeBackend> backend_;
    ModelMetadata metadata_;

    size_t hidden_size_ = 256;
    size_t intermediate_size_ = 512;
    size_t num_layers_ = 4;
    size_t num_heads_ = 4;
    size_t num_kv_heads_ = 4;
    size_t head_dim_ = 64;
    size_t vocab_size_ = 32000;

    // Buffer allocations for forward pass activations
    std::vector<float> act_x_;
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

    // Default weight storage if not loaded from disk
    std::vector<float> default_embed_table_;
    std::vector<float> default_lm_head_;
    std::vector<std::vector<float>> default_layer_data_;
    bool weights_ready_ = false;
};

} // namespace nexus
