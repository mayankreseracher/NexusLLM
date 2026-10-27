#include "transformer_engine.hpp"
#include "nexus_backend.hpp"
#include <iostream>
#include <cmath>
#include <random>

namespace nexus {

TransformerEngine::TransformerEngine(
    std::shared_ptr<MemoryTierManager> tier_mgr,
    std::shared_ptr<PagedKVCacheManager> kv_mgr,
    const ModelMetadata& metadata)
    : tier_mgr_(tier_mgr), kv_mgr_(kv_mgr), metadata_(metadata) {

    // Configure dimensions
    bool has_external_weights = (tier_mgr_ && tier_mgr_->get_stats().tier1_inactive_bytes > 0);
    if (has_external_weights && metadata_.hidden_size > 0) {
        hidden_size_ = metadata_.hidden_size;
        intermediate_size_ = metadata_.intermediate_size > 0 ? metadata_.intermediate_size : (hidden_size_ * 2);
        num_layers_ = metadata_.num_hidden_layers > 0 ? metadata_.num_hidden_layers : 4;
        num_heads_ = metadata_.num_attention_heads > 0 ? metadata_.num_attention_heads : 4;
        num_kv_heads_ = metadata_.num_key_value_heads > 0 ? metadata_.num_key_value_heads : num_heads_;
        head_dim_ = hidden_size_ / num_heads_;
        vocab_size_ = metadata_.vocab_size > 0 ? metadata_.vocab_size : 32000;
    } else {
        // Fast, deterministic dimensions for standalone micro-benchmarks / tests
        hidden_size_ = (metadata_.hidden_size > 0 && metadata_.hidden_size <= 512) ? metadata_.hidden_size : 256;
        intermediate_size_ = (metadata_.intermediate_size > 0 && metadata_.intermediate_size <= 1024) ? metadata_.intermediate_size : 512;
        num_layers_ = (metadata_.num_hidden_layers > 0 && metadata_.num_hidden_layers <= 8) ? metadata_.num_hidden_layers : 4;
        num_heads_ = (metadata_.num_attention_heads > 0 && metadata_.num_attention_heads <= 8) ? metadata_.num_attention_heads : 4;
        num_kv_heads_ = (metadata_.num_key_value_heads > 0 && metadata_.num_key_value_heads <= 8) ? metadata_.num_key_value_heads : num_heads_;
        head_dim_ = hidden_size_ / num_heads_;
        vocab_size_ = (metadata_.vocab_size > 0 && metadata_.vocab_size <= 2000) ? metadata_.vocab_size : 1000;
    }

    // Allocate forward activation buffers
    act_x_.resize(hidden_size_, 0.0f);
    act_norm_.resize(hidden_size_, 0.0f);
    act_q_.resize(num_heads_ * head_dim_, 0.0f);
    act_k_.resize(num_kv_heads_ * head_dim_, 0.0f);
    act_v_.resize(num_kv_heads_ * head_dim_, 0.0f);
    act_attn_out_.resize(num_heads_ * head_dim_, 0.0f);
    act_attn_proj_.resize(hidden_size_, 0.0f);
    act_gate_.resize(intermediate_size_, 0.0f);
    act_up_.resize(intermediate_size_, 0.0f);
    act_swiglu_.resize(intermediate_size_, 0.0f);
    act_down_.resize(hidden_size_, 0.0f);
    act_logits_.resize(vocab_size_, 0.0f);

    initialize_default_weights();
}

void TransformerEngine::initialize_default_weights() {
    if (weights_ready_ || metadata_.is_loaded) {
        weights_ready_ = true;
        return;
    }

    // If real weights are already registered in tier_mgr_, do NOT overwrite with defaults
    if (tier_mgr_ && (tier_mgr_->get_tensor_view("output.weight") != nullptr ||
                      tier_mgr_->get_tensor_view("lm_head") != nullptr ||
                      tier_mgr_->get_tensor_view("token_embd.weight") != nullptr ||
                      tier_mgr_->get_tensor_view("token_embd") != nullptr ||
                      tier_mgr_->get_tensor_view("blk.0.attn_q.weight") != nullptr ||
                      tier_mgr_->get_tensor_view("layers.0.wq") != nullptr)) {
        weights_ready_ = true;
        return;
    }

    // Synthetic parameter initialization for standalone offline unit tests and
    // scheduler micro-benchmarks when no external GGUF/SafeTensors model is loaded.
    std::mt19937 rng(42);
    std::normal_distribution<float> dist(0.0f, 0.02f);

    default_embed_table_.resize(vocab_size_ * hidden_size_);
    for (auto& w : default_embed_table_) w = dist(rng);

    default_lm_head_.resize(vocab_size_ * hidden_size_);
    for (auto& w : default_lm_head_) w = dist(rng);

    default_out_norm_.resize(hidden_size_, 1.0f);

    default_layer_data_.resize(num_layers_);
    for (size_t l = 0; l < num_layers_; ++l) {
        // Size: norm(h) + wq(h*h) + wk(kv*h) + wv(kv*h) + wo(h*h) + ffn_norm(h) + gate(inter*h) + up(inter*h) + down(h*inter)
        size_t total_sz = hidden_size_ + (hidden_size_ * hidden_size_) +
                          2 * (num_kv_heads_ * head_dim_ * hidden_size_) +
                          (hidden_size_ * hidden_size_) + hidden_size_ +
                          2 * (intermediate_size_ * hidden_size_) +
                          (hidden_size_ * intermediate_size_);
        default_layer_data_[l].resize(total_sz);
        for (auto& w : default_layer_data_[l]) w = dist(rng);

        for (size_t i = 0; i < hidden_size_; ++i) {
            default_layer_data_[l][i] = 1.0f; // attn_norm
        }
        size_t ffn_norm_off = hidden_size_ + (hidden_size_ * hidden_size_) +
                              2 * (num_kv_heads_ * head_dim_ * hidden_size_) +
                              (hidden_size_ * hidden_size_);
        for (size_t i = 0; i < hidden_size_; ++i) {
            default_layer_data_[l][ffn_norm_off + i] = 1.0f; // ffn_norm
        }
    }

    // Register parameters into MemoryTierManager (Tier 1 Host System RAM)
    // so they are tracked, tiered, prefetched, and activated through Tier 0 working buffers
    if (tier_mgr_) {
        TensorShape emb_shape;
        emb_shape.dims = { vocab_size_, hidden_size_ };
        tier_mgr_->register_parameter("token_embd.weight", DataType::FP32, emb_shape, default_embed_table_.data());
        tier_mgr_->register_parameter("token_embd", DataType::FP32, emb_shape, default_embed_table_.data());

        TensorShape head_shape;
        head_shape.dims = { vocab_size_, hidden_size_ };
        tier_mgr_->register_parameter("output.weight", DataType::FP32, head_shape, default_lm_head_.data());
        tier_mgr_->register_parameter("lm_head", DataType::FP32, head_shape, default_lm_head_.data());

        TensorShape out_norm_shape;
        out_norm_shape.dims = { hidden_size_ };
        tier_mgr_->register_parameter("output_norm.weight", DataType::FP32, out_norm_shape, default_out_norm_.data());
        tier_mgr_->register_parameter("output_norm", DataType::FP32, out_norm_shape, default_out_norm_.data());

        for (size_t l = 0; l < num_layers_; ++l) {
            std::string pfx = "blk." + std::to_string(l) + ".";
            std::string canon_pfx = "layers." + std::to_string(l) + ".";
            const float* lw = default_layer_data_[l].data();
            size_t off = 0;

            TensorShape norm_shape; norm_shape.dims = { hidden_size_ };
            tier_mgr_->register_parameter(pfx + "attn_norm.weight", DataType::FP32, norm_shape, lw + off);
            tier_mgr_->register_parameter(canon_pfx + "attn_norm", DataType::FP32, norm_shape, lw + off);
            off += hidden_size_;

            TensorShape q_shape; q_shape.dims = { num_heads_ * head_dim_, hidden_size_ };
            tier_mgr_->register_parameter(pfx + "attn_q.weight", DataType::FP32, q_shape, lw + off);
            tier_mgr_->register_parameter(canon_pfx + "wq", DataType::FP32, q_shape, lw + off);
            off += (hidden_size_ * hidden_size_);

            TensorShape k_shape; k_shape.dims = { num_kv_heads_ * head_dim_, hidden_size_ };
            tier_mgr_->register_parameter(pfx + "attn_k.weight", DataType::FP32, k_shape, lw + off);
            tier_mgr_->register_parameter(canon_pfx + "wk", DataType::FP32, k_shape, lw + off);
            off += (num_kv_heads_ * head_dim_ * hidden_size_);

            TensorShape v_shape; v_shape.dims = { num_kv_heads_ * head_dim_, hidden_size_ };
            tier_mgr_->register_parameter(pfx + "attn_v.weight", DataType::FP32, v_shape, lw + off);
            tier_mgr_->register_parameter(canon_pfx + "wv", DataType::FP32, v_shape, lw + off);
            off += (num_kv_heads_ * head_dim_ * hidden_size_);

            TensorShape o_shape; o_shape.dims = { hidden_size_, hidden_size_ };
            tier_mgr_->register_parameter(pfx + "attn_output.weight", DataType::FP32, o_shape, lw + off);
            tier_mgr_->register_parameter(canon_pfx + "wo", DataType::FP32, o_shape, lw + off);
            off += (hidden_size_ * hidden_size_);

            TensorShape ffn_norm_shape; ffn_norm_shape.dims = { hidden_size_ };
            tier_mgr_->register_parameter(pfx + "ffn_norm.weight", DataType::FP32, ffn_norm_shape, lw + off);
            tier_mgr_->register_parameter(canon_pfx + "ffn_norm", DataType::FP32, ffn_norm_shape, lw + off);
            off += hidden_size_;

            TensorShape gate_shape; gate_shape.dims = { intermediate_size_, hidden_size_ };
            tier_mgr_->register_parameter(pfx + "ffn_gate.weight", DataType::FP32, gate_shape, lw + off);
            tier_mgr_->register_parameter(canon_pfx + "ffn_gate", DataType::FP32, gate_shape, lw + off);
            off += (intermediate_size_ * hidden_size_);

            TensorShape up_shape; up_shape.dims = { intermediate_size_, hidden_size_ };
            tier_mgr_->register_parameter(pfx + "ffn_up.weight", DataType::FP32, up_shape, lw + off);
            tier_mgr_->register_parameter(canon_pfx + "ffn_up", DataType::FP32, up_shape, lw + off);
            off += (intermediate_size_ * hidden_size_);

            TensorShape down_shape; down_shape.dims = { hidden_size_, intermediate_size_ };
            tier_mgr_->register_parameter(pfx + "ffn_down.weight", DataType::FP32, down_shape, lw + off);
            tier_mgr_->register_parameter(canon_pfx + "ffn_down", DataType::FP32, down_shape, lw + off);
        }
    }

    weights_ready_ = true;
}

int32_t TransformerEngine::forward_token(
    int32_t current_token,
    size_t position,
    const std::vector<int32_t>& block_table,
    float temperature) {

    if (block_table.empty()) return 0;

    auto get_param = [&](const std::string& key1, const std::string& key2 = "", const std::string& key3 = "") -> void* {
        if (!tier_mgr_) return nullptr;
        void* ptr = tier_mgr_->activate_parameter(key1);
        if (!ptr && !key2.empty()) ptr = tier_mgr_->activate_parameter(key2);
        if (!ptr && !key3.empty()) ptr = tier_mgr_->activate_parameter(key3);
        return ptr;
    };

    auto get_view = [&](const std::string& key1, const std::string& key2 = "", const std::string& key3 = "") -> TensorView* {
        if (!tier_mgr_) return nullptr;
        TensorView* tv = tier_mgr_->get_tensor_view(key1);
        if (!tv && !key2.empty()) tv = tier_mgr_->get_tensor_view(key2);
        if (!tv && !key3.empty()) tv = tier_mgr_->get_tensor_view(key3);
        return tv;
    };

    // 1. Embedding lookup
    size_t safe_tok = static_cast<size_t>(std::abs(current_token)) % vocab_size_;
    void* emb_raw = get_param("token_embd", "token_embd.weight", "model.embed_tokens.weight");
    TensorView* tv_emb = get_view("token_embd", "token_embd.weight", "model.embed_tokens.weight");
    DataType dt_emb = tv_emb ? tv_emb->dtype : DataType::FP32;

    if (metadata_.is_loaded && !emb_raw) {
        std::cerr << "[FATAL] Missing embedding weights (token_embd). Check tensor name mapping in model_loader.cpp." << std::endl;
        return NEXUS_ERR_MISSING_TENSOR;
    }

    if (emb_raw) {
        if (dt_emb == DataType::FP32) {
            const float* emb_row = static_cast<const float*>(emb_raw) + (safe_tok * hidden_size_);
            std::memcpy(act_x_.data(), emb_row, hidden_size_ * sizeof(float));
        } else if (dt_emb == DataType::FP16) {
            const uint16_t* emb_row = static_cast<const uint16_t*>(emb_raw) + (safe_tok * hidden_size_);
            for (size_t i = 0; i < hidden_size_; ++i) {
                act_x_[i] = QuantizationKernel::fp16_to_fp32(emb_row[i]);
            }
        } else if (dt_emb == DataType::GGUF_Q4_0) {
            size_t blocks_per_row = hidden_size_ / 32;
            const BlockQ4_0* blocks = static_cast<const BlockQ4_0*>(emb_raw) + (safe_tok * blocks_per_row);
            QuantizationKernel::dequantize_gguf_q4_0(blocks, act_x_.data(), blocks_per_row);
        } else if (dt_emb == DataType::GGUF_Q8_0) {
            size_t blocks_per_row = hidden_size_ / 32;
            const BlockQ8_0* blocks = static_cast<const BlockQ8_0*>(emb_raw) + (safe_tok * blocks_per_row);
            for (size_t b = 0; b < blocks_per_row; ++b) {
                float d = QuantizationKernel::fp16_to_fp32(blocks[b].d);
                for (size_t j = 0; j < 32; ++j) {
                    act_x_[b * 32 + j] = blocks[b].qs[j] * d;
                }
            }
        } else {
            const float* emb_row = static_cast<const float*>(emb_raw) + (safe_tok * hidden_size_);
            std::memcpy(act_x_.data(), emb_row, hidden_size_ * sizeof(float));
        }
    } else {
        const float* emb_row = default_embed_table_.data() + (safe_tok * hidden_size_);
        std::memcpy(act_x_.data(), emb_row, hidden_size_ * sizeof(float));
    }

    // 2. Multi-Layer Transformer Blocks
    for (size_t l = 0; l < num_layers_; ++l) {
        // Dynamic Tiering Overlapped Double-Buffering: trigger async prefetch for entire layer l+1
        if (l + 1 < num_layers_ && tier_mgr_) {
            tier_mgr_->prefetch_layer_async(l + 1);
        }

        // Pointer offsets into layer weights or dynamically activated Tier 0 parameters
        const float* l_weights = (!metadata_.is_loaded && default_layer_data_.size() > l) ? default_layer_data_[l].data() : nullptr;
        size_t offset = 0;

        std::string canon_pfx = "layers." + std::to_string(l) + ".";
        std::string blk_pfx = "blk." + std::to_string(l) + ".";
        void* attn_norm_raw = get_param(canon_pfx + "attn_norm", blk_pfx + "attn_norm.weight");
        void* wq_raw = get_param(canon_pfx + "wq", blk_pfx + "attn_q.weight");
        void* wk_raw = get_param(canon_pfx + "wk", blk_pfx + "attn_k.weight");
        void* wv_raw = get_param(canon_pfx + "wv", blk_pfx + "attn_v.weight");
        void* wo_raw = get_param(canon_pfx + "wo", blk_pfx + "attn_output.weight", blk_pfx + "attn_out.weight");
        void* ffn_norm_raw = get_param(canon_pfx + "ffn_norm", blk_pfx + "ffn_norm.weight");
        void* w_gate_raw = get_param(canon_pfx + "ffn_gate", blk_pfx + "ffn_gate.weight", blk_pfx + "ffn_gate_exps.weight");
        void* w_up_raw = get_param(canon_pfx + "ffn_up", blk_pfx + "ffn_up.weight", blk_pfx + "ffn_up_exps.weight");
        void* w_down_raw = get_param(canon_pfx + "ffn_down", blk_pfx + "ffn_down.weight", blk_pfx + "ffn_down_exps.weight");

        TensorView* tv_attn_norm = get_view(canon_pfx + "attn_norm", blk_pfx + "attn_norm.weight");
        DataType dt_attn_norm = tv_attn_norm ? tv_attn_norm->dtype : DataType::FP32;

        const void* attn_norm = attn_norm_raw ? attn_norm_raw : (l_weights ? (l_weights + offset) : nullptr); 
        if (l_weights) offset += hidden_size_;

        const void* wq = wq_raw ? wq_raw : (l_weights ? (l_weights + offset) : nullptr);
        if (l_weights) offset += (hidden_size_ * hidden_size_);

        const void* wk = wk_raw ? wk_raw : (l_weights ? (l_weights + offset) : nullptr);
        if (l_weights) offset += (num_kv_heads_ * head_dim_ * hidden_size_);

        const void* wv = wv_raw ? wv_raw : (l_weights ? (l_weights + offset) : nullptr);
        if (l_weights) offset += (num_kv_heads_ * head_dim_ * hidden_size_);

        const void* wo = wo_raw ? wo_raw : (l_weights ? (l_weights + offset) : nullptr);
        if (l_weights) offset += (hidden_size_ * hidden_size_);

        TensorView* tv_ffn_norm = get_view(canon_pfx + "ffn_norm", blk_pfx + "ffn_norm.weight");
        DataType dt_ffn_norm = tv_ffn_norm ? tv_ffn_norm->dtype : DataType::FP32;

        const void* ffn_norm = ffn_norm_raw ? ffn_norm_raw : (l_weights ? (l_weights + offset) : nullptr);
        if (l_weights) offset += hidden_size_;

        const void* w_gate = w_gate_raw ? w_gate_raw : (l_weights ? (l_weights + offset) : nullptr);
        if (l_weights) offset += (intermediate_size_ * hidden_size_);

        const void* w_up = w_up_raw ? w_up_raw : (l_weights ? (l_weights + offset) : nullptr);
        if (l_weights) offset += (intermediate_size_ * hidden_size_);

        const void* w_down = w_down_raw ? w_down_raw : (l_weights ? (l_weights + offset) : nullptr);

        // Strict tensor validation if real model is loaded from disk
        if (metadata_.is_loaded || !l_weights) {
            if (wq == nullptr || wk == nullptr || wv == nullptr || wo == nullptr) {
                std::cerr << "[FATAL] Layer " << l << ": missing attention weights. Check tensor name mapping in model_loader.cpp." << std::endl;
                std::cerr << "[TransformerEngine Error] Required tensor missing for layer " << l << ":";
                if (!wq) std::cerr << " wq";
                if (!wk) std::cerr << " wk";
                if (!wv) std::cerr << " wv";
                if (!wo) std::cerr << " wo";
                std::cerr << std::endl;
                return NEXUS_ERR_MISSING_TENSOR;
            }
            if (w_gate == nullptr || w_up == nullptr || w_down == nullptr) {
                std::cerr << "[FATAL] Layer " << l << ": missing FFN weights. Check tensor name mapping in model_loader.cpp." << std::endl;
                std::cerr << "[TransformerEngine Error] Required tensor missing for layer " << l << ":";
                if (!w_gate) std::cerr << " w_gate";
                if (!w_up) std::cerr << " w_up";
                if (!w_down) std::cerr << " w_down";
                std::cerr << std::endl;
                return NEXUS_ERR_MISSING_TENSOR;
            }
            if (attn_norm == nullptr || ffn_norm == nullptr) {
                std::cerr << "[FATAL] Layer " << l << ": missing norm weights. Check tensor name mapping in model_loader.cpp." << std::endl;
                std::cerr << "[TransformerEngine Error] Required tensor missing for layer " << l << ":";
                if (!attn_norm) std::cerr << " attn_norm";
                if (!ffn_norm) std::cerr << " ffn_norm";
                std::cerr << std::endl;
                return NEXUS_ERR_MISSING_TENSOR;
            }
        }

        // Pre-Attention RMSNorm
        if (backend_) {
            backend_->rmsnorm(act_x_.data(), attn_norm, dt_attn_norm, act_norm_.data(), hidden_size_, metadata_.rms_norm_eps);
        } else {
            TransformerOps::rmsnorm(act_x_.data(), attn_norm, dt_attn_norm, act_norm_.data(), hidden_size_, metadata_.rms_norm_eps);
        }

        TensorView* tv_wq = get_view(canon_pfx + "wq", blk_pfx + "attn_q.weight");
        DataType dt_wq = tv_wq ? tv_wq->dtype : DataType::FP32;

        TensorView* tv_wk = get_view(canon_pfx + "wk", blk_pfx + "attn_k.weight");
        DataType dt_wk = tv_wk ? tv_wk->dtype : DataType::FP32;

        TensorView* tv_wv = get_view(canon_pfx + "wv", blk_pfx + "attn_v.weight");
        DataType dt_wv = tv_wv ? tv_wv->dtype : DataType::FP32;

        TensorView* tv_wo = get_view(canon_pfx + "wo", blk_pfx + "attn_output.weight", blk_pfx + "attn_out.weight");
        DataType dt_wo = tv_wo ? tv_wo->dtype : DataType::FP32;

        TensorView* tv_gate = get_view(canon_pfx + "ffn_gate", blk_pfx + "ffn_gate.weight", blk_pfx + "ffn_gate_exps.weight");
        DataType dt_gate = tv_gate ? tv_gate->dtype : DataType::FP32;

        TensorView* tv_up = get_view(canon_pfx + "ffn_up", blk_pfx + "ffn_up.weight", blk_pfx + "ffn_up_exps.weight");
        DataType dt_up = tv_up ? tv_up->dtype : DataType::FP32;

        TensorView* tv_down = get_view(canon_pfx + "ffn_down", blk_pfx + "ffn_down.weight", blk_pfx + "ffn_down_exps.weight");
        DataType dt_down = tv_down ? tv_down->dtype : DataType::FP32;

        // Compute Q, K, V projections via matrix-vector multiplication (supporting FP32, FP16, Q4_0, Q8_0)
        if (backend_) {
            backend_->gemv_dispatch(wq, dt_wq, act_norm_.data(), act_q_.data(), num_heads_ * head_dim_, hidden_size_);
            backend_->gemv_dispatch(wk, dt_wk, act_norm_.data(), act_k_.data(), num_kv_heads_ * head_dim_, hidden_size_);
            backend_->gemv_dispatch(wv, dt_wv, act_norm_.data(), act_v_.data(), num_kv_heads_ * head_dim_, hidden_size_);
        } else {
            TransformerOps::matmul_vec_dispatch(wq, dt_wq, act_norm_.data(), act_q_.data(), num_heads_ * head_dim_, hidden_size_);
            TransformerOps::matmul_vec_dispatch(wk, dt_wk, act_norm_.data(), act_k_.data(), num_kv_heads_ * head_dim_, hidden_size_);
            TransformerOps::matmul_vec_dispatch(wv, dt_wv, act_norm_.data(), act_v_.data(), num_kv_heads_ * head_dim_, hidden_size_);
        }

        // Apply Rotary Position Embeddings (RoPE)
        TransformerOps::rope(act_q_.data(), act_k_.data(), head_dim_, num_heads_, num_kv_heads_, position, metadata_.rope_theta);

        // Store Key and Value into Paged KV Cache block table
        size_t blk_idx = position / BLOCK_SIZE;
        size_t slot = position % BLOCK_SIZE;
        if (blk_idx < block_table.size()) {
            int32_t blk_id = block_table[blk_idx];
            KVBlock* blk = kv_mgr_->get_block(blk_id);
            if (blk && blk->key_data && blk->value_data) {
                size_t stride = num_kv_heads_ * BLOCK_SIZE * head_dim_;
                for (size_t kv_h = 0; kv_h < num_kv_heads_; ++kv_h) {
                    float* k_dest = blk->key_data + (l * stride) + (kv_h * BLOCK_SIZE * head_dim_) + (slot * head_dim_);
                    float* v_dest = blk->value_data + (l * stride) + (kv_h * BLOCK_SIZE * head_dim_) + (slot * head_dim_);
                    const float* k_src = act_k_.data() + (kv_h * head_dim_);
                    const float* v_src = act_v_.data() + (kv_h * head_dim_);
                    std::memcpy(k_dest, k_src, head_dim_ * sizeof(float));
                    std::memcpy(v_dest, v_src, head_dim_ * sizeof(float));
                }
                if (slot + 1 > blk->tokens_occupied) blk->tokens_occupied = slot + 1;
            }
        }

        // PagedAttention Multi-Head / Grouped-Query Decode
        if (backend_) {
            backend_->paged_attention_decode(
                act_q_.data(),
                block_table,
                *kv_mgr_,
                l,
                position,
                num_heads_,
                num_kv_heads_,
                head_dim_,
                act_attn_out_.data()
            );
        } else {
            TransformerOps::paged_attention_decode(
                act_q_.data(),
                block_table,
                *kv_mgr_,
                l,
                position,
                num_heads_,
                num_kv_heads_,
                head_dim_,
                act_attn_out_.data()
            );
        }

        // Attention output projection and residual accumulation
        if (backend_) {
            backend_->gemv_dispatch(wo, dt_wo, act_attn_out_.data(), act_attn_proj_.data(), hidden_size_, hidden_size_);
        } else {
            TransformerOps::matmul_vec_dispatch(wo, dt_wo, act_attn_out_.data(), act_attn_proj_.data(), hidden_size_, hidden_size_);
        }
        for (size_t i = 0; i < hidden_size_; ++i) {
            act_x_[i] += act_attn_proj_[i];
        }

        // Post-Attention RMSNorm
        if (backend_) {
            backend_->rmsnorm(act_x_.data(), ffn_norm, dt_ffn_norm, act_norm_.data(), hidden_size_, metadata_.rms_norm_eps);
        } else {
            TransformerOps::rmsnorm(act_x_.data(), ffn_norm, dt_ffn_norm, act_norm_.data(), hidden_size_, metadata_.rms_norm_eps);
        }

        // SwiGLU Feed-Forward Network: down(SiLU(gate(x)) * up(x))
        if (backend_) {
            backend_->gemv_dispatch(w_gate, dt_gate, act_norm_.data(), act_gate_.data(), intermediate_size_, hidden_size_);
            backend_->gemv_dispatch(w_up, dt_up, act_norm_.data(), act_up_.data(), intermediate_size_, hidden_size_);
            backend_->swiglu(act_gate_.data(), act_up_.data(), act_swiglu_.data(), intermediate_size_);
            backend_->gemv_dispatch(w_down, dt_down, act_swiglu_.data(), act_down_.data(), hidden_size_, intermediate_size_);
        } else {
            TransformerOps::matmul_vec_dispatch(w_gate, dt_gate, act_norm_.data(), act_gate_.data(), intermediate_size_, hidden_size_);
            TransformerOps::matmul_vec_dispatch(w_up, dt_up, act_norm_.data(), act_up_.data(), intermediate_size_, hidden_size_);
            TransformerOps::swiglu(act_gate_.data(), act_up_.data(), act_swiglu_.data(), intermediate_size_);
            TransformerOps::matmul_vec_dispatch(w_down, dt_down, act_swiglu_.data(), act_down_.data(), hidden_size_, intermediate_size_);
        }

        // Residual accumulation
        for (size_t i = 0; i < hidden_size_; ++i) {
            act_x_[i] += act_down_[i];
        }

        // Dynamic layer deactivation & double-buffer slot rotation
        if (tier_mgr_) {
            tier_mgr_->deactivate_layer(l);
        }
    }

    // 3. Final RMSNorm
    void* out_norm_raw = get_param("output_norm", "output_norm.weight", "model.norm.weight");
    if (metadata_.is_loaded && !out_norm_raw) {
        std::cerr << "[FATAL] Missing output norm weights (output_norm). Check tensor name mapping in model_loader.cpp." << std::endl;
        return NEXUS_ERR_MISSING_TENSOR;
    }
    const void* out_norm = out_norm_raw ? out_norm_raw : default_out_norm_.data();
    TensorView* tv_out_norm = get_view("output_norm", "output_norm.weight", "model.norm.weight");
    DataType dt_out_norm = tv_out_norm ? tv_out_norm->dtype : DataType::FP32;
    if (backend_) {
        backend_->rmsnorm(act_x_.data(), out_norm, dt_out_norm, act_norm_.data(), hidden_size_, metadata_.rms_norm_eps);
    } else {
        TransformerOps::rmsnorm(act_x_.data(), out_norm, dt_out_norm, act_norm_.data(), hidden_size_, metadata_.rms_norm_eps);
    }

    // 4. LM Head / Logits Projection
    void* head_raw = get_param("lm_head", "output.weight", "lm_head.weight");
    if (metadata_.is_loaded && !head_raw) {
        std::cerr << "[FATAL] Missing LM head weights (lm_head/output.weight). Check tensor name mapping in model_loader.cpp." << std::endl;
        return NEXUS_ERR_MISSING_TENSOR;
    }
    const void* lm_head = head_raw ? head_raw : default_lm_head_.data();
    TensorView* tv_head = get_view("lm_head", "output.weight", "lm_head.weight");
    DataType dt_head = tv_head ? tv_head->dtype : DataType::FP32;
    if (backend_) {
        backend_->gemv_dispatch(lm_head, dt_head, act_norm_.data(), act_logits_.data(), vocab_size_, hidden_size_);
    } else {
        TransformerOps::matmul_vec_dispatch(lm_head, dt_head, act_norm_.data(), act_logits_.data(), vocab_size_, hidden_size_);
    }

    // 5. Sampling: select actual next token
    int32_t next_token = TransformerOps::sample_logits(act_logits_.data(), vocab_size_, temperature);
    return next_token;
}

int32_t TransformerEngine::forward_prefill(
    const std::vector<int32_t>& prompt_tokens,
    const std::vector<int32_t>& block_table,
    float temperature) {

    if (prompt_tokens.empty()) return 0;

    int32_t gen_tok = 0;
    for (size_t pos = 0; pos < prompt_tokens.size(); ++pos) {
        gen_tok = forward_token(prompt_tokens[pos], pos, block_table, temperature);
    }
    return gen_tok;
}

std::vector<float> TransformerEngine::forward_sequence_logits(
    const std::vector<int32_t>& tokens,
    const std::vector<int32_t>& block_table) {

    if (tokens.empty()) return act_logits_;
    for (size_t pos = 0; pos < tokens.size(); ++pos) {
        forward_token(tokens[pos], pos, block_table, 0.0f);
    }
    return act_logits_;
}

std::vector<int32_t> TransformerEngine::predict_mtp_draft(
    int32_t current_token,
    size_t current_pos,
    size_t num_draft_tokens) {

    std::vector<int32_t> drafts;
    if (num_draft_tokens == 0) return drafts;
    drafts.reserve(num_draft_tokens);

    auto get_param = [&](const std::string& key1, const std::string& key2 = "", const std::string& key3 = "", const std::string& key4 = "") -> void* {
        if (!tier_mgr_) return nullptr;
        void* ptr = tier_mgr_->activate_parameter(key1);
        if (!ptr && !key2.empty()) ptr = tier_mgr_->activate_parameter(key2);
        if (!ptr && !key3.empty()) ptr = tier_mgr_->activate_parameter(key3);
        if (!ptr && !key4.empty()) ptr = tier_mgr_->activate_parameter(key4);
        return ptr;
    };

    auto get_view = [&](const std::string& key1, const std::string& key2 = "", const std::string& key3 = "", const std::string& key4 = "") -> TensorView* {
        if (!tier_mgr_) return nullptr;
        TensorView* tv = tier_mgr_->get_tensor_view(key1);
        if (!tv && !key2.empty()) tv = tier_mgr_->get_tensor_view(key2);
        if (!tv && !key3.empty()) tv = tier_mgr_->get_tensor_view(key3);
        if (!tv && !key4.empty()) tv = tier_mgr_->get_tensor_view(key4);
        return tv;
    };

    // Check for Multi-Token Prediction (MTP) / Medusa auxiliary heads
    int32_t prev_tok = current_token;
    for (size_t k = 0; k < num_draft_tokens; ++k) {
        std::string k_str = std::to_string(k);
        std::string name1 = "mtp." + k_str + ".weight";
        std::string name2 = "mtp_head." + k_str + ".weight";
        std::string name3 = "mtp." + k_str;
        std::string name4 = "mtp_head." + k_str;

        void* mtp_raw = get_param(name1, name2, name3, name4);
        TensorView* tv_mtp = get_view(name1, name2, name3, name4);
        const void* w_mtp = mtp_raw ? mtp_raw : (tv_mtp ? (tv_mtp->active_ptr ? tv_mtp->active_ptr : tv_mtp->host_ptr) : nullptr);
        if (tv_mtp && w_mtp) {
            std::vector<float> mtp_logits(vocab_size_, 0.0f);
            if (backend_) {
                backend_->gemv_dispatch(w_mtp, tv_mtp->dtype, act_norm_.data(), mtp_logits.data(), vocab_size_, hidden_size_);
            } else {
                TransformerOps::matmul_vec_dispatch(w_mtp, tv_mtp->dtype, act_norm_.data(), mtp_logits.data(), vocab_size_, hidden_size_);
            }
            int32_t pred = TransformerOps::sample_logits(mtp_logits.data(), vocab_size_, 0.0f);
            drafts.push_back(pred);
            prev_tok = pred;
        } else {
            // No MTP/Medusa auxiliary head registered for position k; stop drafting without hallucinating fake tokens
            break;
        }
    }
    return drafts;
}

SpeculativeVerifyResult TransformerEngine::forward_speculative_verify(
    int32_t current_token,
    size_t position,
    const std::vector<int32_t>& draft_tokens,
    const std::vector<int32_t>& block_table,
    float temperature) {

    SpeculativeVerifyResult res;
    if (draft_tokens.empty()) {
        res.bonus_or_correction_token = forward_token(current_token, position, block_table, temperature);
        return res;
    }

    int32_t active_tok = current_token;
    size_t curr_pos = position;

    for (size_t i = 0; i < draft_tokens.size(); ++i) {
        int32_t target_tok = forward_token(active_tok, curr_pos, block_table, temperature);
        int32_t candidate = draft_tokens[i];

        if (candidate == target_tok) {
            // Draft candidate matches target model prediction -> ACCEPT
            res.accepted_tokens.push_back(candidate);
            res.num_accepted++;
            active_tok = candidate;
            curr_pos++;
        } else {
            // Mismatch detected -> REJECT and emit target model correction
            res.bonus_or_correction_token = target_tok;
            res.all_drafts_accepted = false;

            // Roll back KV cache past curr_pos in O(1)
            if (kv_mgr_) {
                kv_mgr_->rollback_sequence(block_table, curr_pos + 1);
            }
            return res;
        }
    }

    // All draft tokens were accepted by target model!
    res.all_drafts_accepted = true;
    // Compute bonus token from the last accepted draft
    res.bonus_or_correction_token = forward_token(active_tok, curr_pos, block_table, temperature);
    return res;
}

} // namespace nexus
