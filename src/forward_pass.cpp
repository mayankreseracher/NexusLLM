#include "forward_pass.hpp"
#include <cmath>
#include <random>
#include <iostream>
#include <cstring>
#include <algorithm>

namespace nexus {

ForwardPassExecutor::ForwardPassExecutor(
    std::shared_ptr<MemoryTierManager> tier_mgr,
    std::shared_ptr<PagedKVCacheManager> kv_mgr,
    const ModelMetadata& metadata
) : tier_mgr_(tier_mgr), kv_mgr_(kv_mgr), metadata_(metadata) {

    if (metadata_.is_loaded) {
        hidden_size_ = metadata_.hidden_size;
        intermediate_size_ = metadata_.intermediate_size > 0 ? metadata_.intermediate_size : (hidden_size_ * 2);
        num_layers_ = metadata_.num_hidden_layers > 0 ? metadata_.num_hidden_layers : 4;
        num_heads_ = metadata_.num_attention_heads > 0 ? metadata_.num_attention_heads : 4;
        num_kv_heads_ = metadata_.num_key_value_heads > 0 ? metadata_.num_key_value_heads : num_heads_;
        head_dim_ = hidden_size_ / num_heads_;
        vocab_size_ = metadata_.vocab_size > 0 ? metadata_.vocab_size : 32000;
    } else {
        // Fast, deterministic dimensions for standalone micro-benchmarks / tests without an external model file
        hidden_size_ = 256;
        intermediate_size_ = 512;
        num_layers_ = 4;
        num_heads_ = 4;
        num_kv_heads_ = 4;
        head_dim_ = hidden_size_ / num_heads_;
        vocab_size_ = 1000;
    }

    // Allocate forward activation buffers
    act_hidden_.resize(hidden_size_, 0.0f);
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

void ForwardPassExecutor::initialize_default_weights() {
    if (weights_initialized_ || metadata_.is_loaded) {
        weights_initialized_ = true;
        return;
    }

    // Check if parameters already registered by loader or previous instance
    if (tier_mgr_ && (tier_mgr_->get_tensor_view("output.weight") != nullptr ||
                      tier_mgr_->get_tensor_view("token_embd.weight") != nullptr ||
                      tier_mgr_->get_tensor_view("blk.0.attn_q.weight") != nullptr)) {
        weights_initialized_ = true;
        return;
    }

    std::mt19937 rng(42);
    std::normal_distribution<float> dist(0.0f, 0.02f);

    default_embed_.resize(vocab_size_ * hidden_size_);
    for (auto& w : default_embed_) w = dist(rng);

    default_lm_head_.resize(vocab_size_ * hidden_size_);
    for (auto& w : default_lm_head_) w = dist(rng);

    default_out_norm_.resize(hidden_size_, 1.0f);

    default_layer_data_.resize(num_layers_);
    for (size_t l = 0; l < num_layers_; ++l) {
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

    // Register parameters into MemoryTierManager (Tier 1 Host RAM)
    if (tier_mgr_) {
        TensorShape emb_shape;
        emb_shape.dims = { vocab_size_, hidden_size_ };
        tier_mgr_->register_parameter("token_embd.weight", DataType::FP32, emb_shape, default_embed_.data());

        TensorShape head_shape;
        head_shape.dims = { vocab_size_, hidden_size_ };
        tier_mgr_->register_parameter("output.weight", DataType::FP32, head_shape, default_lm_head_.data());

        TensorShape out_norm_shape;
        out_norm_shape.dims = { hidden_size_ };
        tier_mgr_->register_parameter("output_norm.weight", DataType::FP32, out_norm_shape, default_out_norm_.data());

        for (size_t l = 0; l < num_layers_; ++l) {
            std::string pfx = "blk." + std::to_string(l) + ".";
            const float* lw = default_layer_data_[l].data();
            size_t off = 0;

            TensorShape norm_shape; norm_shape.dims = { hidden_size_ };
            tier_mgr_->register_parameter(pfx + "attn_norm.weight", DataType::FP32, norm_shape, lw + off);
            off += hidden_size_;

            TensorShape q_shape; q_shape.dims = { num_heads_ * head_dim_, hidden_size_ };
            tier_mgr_->register_parameter(pfx + "attn_q.weight", DataType::FP32, q_shape, lw + off);
            off += (hidden_size_ * hidden_size_);

            TensorShape k_shape; k_shape.dims = { num_kv_heads_ * head_dim_, hidden_size_ };
            tier_mgr_->register_parameter(pfx + "attn_k.weight", DataType::FP32, k_shape, lw + off);
            off += (num_kv_heads_ * head_dim_ * hidden_size_);

            TensorShape v_shape; v_shape.dims = { num_kv_heads_ * head_dim_, hidden_size_ };
            tier_mgr_->register_parameter(pfx + "attn_v.weight", DataType::FP32, v_shape, lw + off);
            off += (num_kv_heads_ * head_dim_ * hidden_size_);

            TensorShape o_shape; o_shape.dims = { hidden_size_, hidden_size_ };
            tier_mgr_->register_parameter(pfx + "attn_output.weight", DataType::FP32, o_shape, lw + off);
            off += (hidden_size_ * hidden_size_);

            TensorShape ffn_norm_shape; ffn_norm_shape.dims = { hidden_size_ };
            tier_mgr_->register_parameter(pfx + "ffn_norm.weight", DataType::FP32, ffn_norm_shape, lw + off);
            off += hidden_size_;

            TensorShape gate_shape; gate_shape.dims = { intermediate_size_, hidden_size_ };
            tier_mgr_->register_parameter(pfx + "ffn_gate.weight", DataType::FP32, gate_shape, lw + off);
            off += (intermediate_size_ * hidden_size_);

            TensorShape up_shape; up_shape.dims = { intermediate_size_, hidden_size_ };
            tier_mgr_->register_parameter(pfx + "ffn_up.weight", DataType::FP32, up_shape, lw + off);
            off += (intermediate_size_ * hidden_size_);

            TensorShape down_shape; down_shape.dims = { hidden_size_, intermediate_size_ };
            tier_mgr_->register_parameter(pfx + "ffn_down.weight", DataType::FP32, down_shape, lw + off);
        }
    }

    weights_initialized_ = true;
}

void ForwardPassExecutor::paged_attention_cpu(
    const float* Q,
    const std::vector<int32_t>& block_table,
    size_t layer_idx,
    size_t seq_len,
    float* attn_out
) {
    if (num_heads_ == 0 || num_kv_heads_ == 0 || head_dim_ == 0 || block_table.empty() || !kv_mgr_) return;
    size_t kv_mul = num_heads_ / num_kv_heads_;
    float scale = 1.0f / std::sqrt(static_cast<float>(head_dim_));
    size_t total_tokens = seq_len + 1;

    for (size_t h = 0; h < num_heads_; ++h) {
        size_t kv_h = h / kv_mul;
        const float* q_h = Q + (h * head_dim_);
        float* out_h = attn_out + (h * head_dim_);

        // FlashAttention CPU Tiled Online Softmax running state (Dao et al. 2022)
        float m = -1e30f;
        float l = 0.0f;
        std::memset(out_h, 0, head_dim_ * sizeof(float));

        size_t tokens_processed = 0;
        for (size_t blk_idx = 0; blk_idx < block_table.size() && tokens_processed < total_tokens; ++blk_idx) {
            int32_t blk_id = block_table[blk_idx];
            // Explicitly read block via kv_mgr_->read_block(blk_id)
            KVBlock* blk = kv_mgr_->read_block(blk_id);
            if (!blk || !blk->key_data || !blk->value_data) {
                tokens_processed += BLOCK_SIZE;
                continue;
            }

            size_t valid_in_block = std::min(BLOCK_SIZE, total_tokens - tokens_processed);
            size_t stride = num_kv_heads_ * BLOCK_SIZE * head_dim_;
            const float* k_base = blk->key_data + (layer_idx * stride) + (kv_h * BLOCK_SIZE * head_dim_);
            const float* v_base = blk->value_data + (layer_idx * stride) + (kv_h * BLOCK_SIZE * head_dim_);

            for (size_t slot = 0; slot < valid_in_block; ++slot) {
                const float* k_vec = k_base + (slot * head_dim_);
                const float* v_vec = v_base + (slot * head_dim_);

                // 1. Compute dot product: Q_h . K_vec
                float dot = 0.0f;
#if defined(NEXUS_HAVE_AVX2)
                __m256 vdot = _mm256_setzero_ps();
                size_t d = 0;
                for (; d + 8 <= head_dim_; d += 8) {
                    __m256 q_v = _mm256_loadu_ps(q_h + d);
                    __m256 k_v = _mm256_loadu_ps(k_vec + d);
                    vdot = _mm256_fmadd_ps(q_v, k_v, vdot);
                }
                alignas(32) float tmp[8];
                _mm256_storeu_ps(tmp, vdot);
                dot = tmp[0] + tmp[1] + tmp[2] + tmp[3] + tmp[4] + tmp[5] + tmp[6] + tmp[7];
                for (; d < head_dim_; ++d) {
                    dot += q_h[d] * k_vec[d];
                }
#elif defined(NEXUS_HAVE_ARM_NEON)
                float32x4_t vdot = vdupq_n_f32(0.0f);
                size_t d = 0;
                for (; d + 4 <= head_dim_; d += 4) {
                    float32x4_t q_v = vld1q_f32(q_h + d);
                    float32x4_t k_v = vld1q_f32(k_vec + d);
                    vdot = vmlaq_f32(vdot, q_v, k_v);
                }
                dot = vaddvq_f32(vdot);
                for (; d < head_dim_; ++d) {
                    dot += q_h[d] * k_vec[d];
                }
#else
                for (size_t d = 0; d < head_dim_; ++d) {
                    dot += q_h[d] * k_vec[d];
                }
#endif
                float s = dot * scale;

                // 2. Online Softmax update
                float m_prev = m;
                float m_new = (s > m_prev) ? s : m_prev;
                float alpha = std::exp(m_prev - m_new);
                float beta = std::exp(s - m_new);

                l = (l * alpha) + beta;
                m = m_new;

                // 3. Accumulator update: O = (O * alpha) + (beta * V_vec)
#if defined(NEXUS_HAVE_AVX2)
                __m256 valpha = _mm256_set1_ps(alpha);
                __m256 vbeta = _mm256_set1_ps(beta);
                size_t vd = 0;
                for (; vd + 8 <= head_dim_; vd += 8) {
                    __m256 o_v = _mm256_loadu_ps(out_h + vd);
                    __m256 v_v = _mm256_loadu_ps(v_vec + vd);
                    o_v = _mm256_fmadd_ps(v_v, vbeta, _mm256_mul_ps(o_v, valpha));
                    _mm256_storeu_ps(out_h + vd, o_v);
                }
                for (; vd < head_dim_; ++vd) {
                    out_h[vd] = (out_h[vd] * alpha) + (beta * v_vec[vd]);
                }
#elif defined(NEXUS_HAVE_ARM_NEON)
                float32x4_t valpha = vdupq_n_f32(alpha);
                float32x4_t vbeta = vdupq_n_f32(beta);
                size_t vd = 0;
                for (; vd + 4 <= head_dim_; vd += 4) {
                    float32x4_t o_v = vld1q_f32(out_h + vd);
                    float32x4_t v_v = vld1q_f32(v_vec + vd);
                    o_v = vmlaq_f32(vmulq_f32(o_v, valpha), v_v, vbeta);
                    vst1q_f32(out_h + vd, o_v);
                }
                for (; vd < head_dim_; ++vd) {
                    out_h[vd] = (out_h[vd] * alpha) + (beta * v_vec[vd]);
                }
#else
                for (size_t d = 0; d < head_dim_; ++d) {
                    out_h[d] = (out_h[d] * alpha) + (beta * v_vec[d]);
                }
#endif
            }
            tokens_processed += valid_in_block;
        }

        // 4. Final normalization: O = O / l
        float inv_l = (l > 0.0f) ? (1.0f / l) : 0.0f;
#if defined(NEXUS_HAVE_AVX2)
        __m256 vinv_l = _mm256_set1_ps(inv_l);
        size_t d = 0;
        for (; d + 8 <= head_dim_; d += 8) {
            __m256 o_v = _mm256_loadu_ps(out_h + d);
            _mm256_storeu_ps(out_h + d, _mm256_mul_ps(o_v, vinv_l));
        }
        for (; d < head_dim_; ++d) {
            out_h[d] *= inv_l;
        }
#elif defined(NEXUS_HAVE_ARM_NEON)
        float32x4_t vinv_l = vdupq_n_f32(inv_l);
        size_t d = 0;
        for (; d + 4 <= head_dim_; d += 4) {
            float32x4_t o_v = vld1q_f32(out_h + d);
            vst1q_f32(out_h + d, vmulq_f32(o_v, vinv_l));
        }
        for (; d < head_dim_; ++d) {
            out_h[d] *= inv_l;
        }
#else
        for (size_t d = 0; d < head_dim_; ++d) {
            out_h[d] *= inv_l;
        }
#endif
    }
}

int32_t ForwardPassExecutor::forward_token(
    int32_t token,
    size_t pos,
    const std::vector<int32_t>& block_table,
    float temperature
) {
    if (block_table.empty()) return 0;

    // 1. Embedding lookup: hidden = embed(tokens)
    TensorView* tv_emb = tier_mgr_ ? tier_mgr_->get_tensor_view("token_embd.weight") : nullptr;
    size_t cur_vocab = (tv_emb && !tv_emb->shape.dims.empty()) ? tv_emb->shape.dims[0] : vocab_size_;
    size_t safe_tok = static_cast<size_t>(std::abs(token)) % (cur_vocab > 0 ? cur_vocab : 1);
    void* emb_raw = tier_mgr_ ? tier_mgr_->activate_parameter("token_embd.weight") : nullptr;
    DataType dt_emb = tv_emb ? tv_emb->dtype : DataType::FP32;

    if (emb_raw) {
        if (dt_emb == DataType::FP32) {
            const float* emb_row = static_cast<const float*>(emb_raw) + (safe_tok * hidden_size_);
            std::memcpy(act_hidden_.data(), emb_row, hidden_size_ * sizeof(float));
        } else if (dt_emb == DataType::FP16) {
            const uint16_t* emb_row = static_cast<const uint16_t*>(emb_raw) + (safe_tok * hidden_size_);
            for (size_t i = 0; i < hidden_size_; ++i) {
                act_hidden_[i] = QuantizationKernel::fp16_to_fp32(emb_row[i]);
            }
        } else if (dt_emb == DataType::GGUF_Q4_0) {
            size_t blocks_per_row = hidden_size_ / 32;
            const BlockQ4_0* blocks = static_cast<const BlockQ4_0*>(emb_raw) + (safe_tok * blocks_per_row);
            QuantizationKernel::dequantize_gguf_q4_0(blocks, act_hidden_.data(), blocks_per_row);
        } else if (dt_emb == DataType::GGUF_Q8_0) {
            size_t blocks_per_row = hidden_size_ / 32;
            const BlockQ8_0* blocks = static_cast<const BlockQ8_0*>(emb_raw) + (safe_tok * blocks_per_row);
            for (size_t b = 0; b < blocks_per_row; ++b) {
                float d = QuantizationKernel::fp16_to_fp32(blocks[b].d);
                for (size_t j = 0; j < 32; ++j) {
                    act_hidden_[b * 32 + j] = blocks[b].qs[j] * d;
                }
            }
        } else {
            const float* emb_row = static_cast<const float*>(emb_raw) + (safe_tok * hidden_size_);
            std::memcpy(act_hidden_.data(), emb_row, hidden_size_ * sizeof(float));
        }
    } else {
        const float* emb_row = default_embed_.data() + (safe_tok * hidden_size_);
        std::memcpy(act_hidden_.data(), emb_row, hidden_size_ * sizeof(float));
    }

    // 2. Transformer Layer Loop (llama_eval loop orchestration)
    for (size_t l = 0; l < num_layers_; ++l) {
        // Double-buffering prefetch for layer l+1
        if (l + 1 < num_layers_ && tier_mgr_) {
            tier_mgr_->prefetch_layer_async(l + 1);
        }

        std::string pfx = "blk." + std::to_string(l) + ".";
        const float* l_weights = (default_layer_data_.size() > l) ? default_layer_data_[l].data() : nullptr;
        size_t offset = 0;

        void* attn_norm_raw = tier_mgr_ ? tier_mgr_->activate_parameter(pfx + "attn_norm.weight") : nullptr;
        void* wq_raw = tier_mgr_ ? tier_mgr_->activate_parameter(pfx + "attn_q.weight") : nullptr;
        void* wk_raw = tier_mgr_ ? tier_mgr_->activate_parameter(pfx + "attn_k.weight") : nullptr;
        void* wv_raw = tier_mgr_ ? tier_mgr_->activate_parameter(pfx + "attn_v.weight") : nullptr;
        void* wo_raw = tier_mgr_ ? tier_mgr_->activate_parameter(pfx + "attn_output.weight") : nullptr;
        if (!wo_raw && tier_mgr_) wo_raw = tier_mgr_->activate_parameter(pfx + "attn_out.weight");
        void* ffn_norm_raw = tier_mgr_ ? tier_mgr_->activate_parameter(pfx + "ffn_norm.weight") : nullptr;
        void* w_gate_raw = tier_mgr_ ? tier_mgr_->activate_parameter(pfx + "ffn_gate.weight") : nullptr;
        if (!w_gate_raw && tier_mgr_) w_gate_raw = tier_mgr_->activate_parameter(pfx + "ffn_gate_exps.weight");
        void* w_up_raw = tier_mgr_ ? tier_mgr_->activate_parameter(pfx + "ffn_up.weight") : nullptr;
        if (!w_up_raw && tier_mgr_) w_up_raw = tier_mgr_->activate_parameter(pfx + "ffn_up_exps.weight");
        void* w_down_raw = tier_mgr_ ? tier_mgr_->activate_parameter(pfx + "ffn_down.weight") : nullptr;
        if (!w_down_raw && tier_mgr_) w_down_raw = tier_mgr_->activate_parameter(pfx + "ffn_down_exps.weight");

        TensorView* tv_attn_norm = tier_mgr_ ? tier_mgr_->get_tensor_view(pfx + "attn_norm.weight") : nullptr;
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

        TensorView* tv_ffn_norm = tier_mgr_ ? tier_mgr_->get_tensor_view(pfx + "ffn_norm.weight") : nullptr;
        DataType dt_ffn_norm = tv_ffn_norm ? tv_ffn_norm->dtype : DataType::FP32;
        const void* ffn_norm = ffn_norm_raw ? ffn_norm_raw : (l_weights ? (l_weights + offset) : nullptr);
        if (l_weights) offset += hidden_size_;

        const void* w_gate = w_gate_raw ? w_gate_raw : (l_weights ? (l_weights + offset) : nullptr);
        if (l_weights) offset += (intermediate_size_ * hidden_size_);

        const void* w_up = w_up_raw ? w_up_raw : (l_weights ? (l_weights + offset) : nullptr);
        if (l_weights) offset += (intermediate_size_ * hidden_size_);

        const void* w_down = w_down_raw ? w_down_raw : (l_weights ? (l_weights + offset) : nullptr);

        TensorView* tv_wq = tier_mgr_ ? tier_mgr_->get_tensor_view(pfx + "attn_q.weight") : nullptr;
        DataType dt_wq = tv_wq ? tv_wq->dtype : DataType::FP32;
        TensorView* tv_wk = tier_mgr_ ? tier_mgr_->get_tensor_view(pfx + "attn_k.weight") : nullptr;
        DataType dt_wk = tv_wk ? tv_wk->dtype : DataType::FP32;
        TensorView* tv_wv = tier_mgr_ ? tier_mgr_->get_tensor_view(pfx + "attn_v.weight") : nullptr;
        DataType dt_wv = tv_wv ? tv_wv->dtype : DataType::FP32;
        TensorView* tv_wo = tier_mgr_ ? tier_mgr_->get_tensor_view(pfx + "attn_output.weight") : nullptr;
        if (!tv_wo && tier_mgr_) tv_wo = tier_mgr_->get_tensor_view(pfx + "attn_out.weight");
        DataType dt_wo = tv_wo ? tv_wo->dtype : DataType::FP32;
        TensorView* tv_gate = tier_mgr_ ? tier_mgr_->get_tensor_view(pfx + "ffn_gate.weight") : nullptr;
        if (!tv_gate && tier_mgr_) tv_gate = tier_mgr_->get_tensor_view(pfx + "ffn_gate_exps.weight");
        DataType dt_gate = tv_gate ? tv_gate->dtype : DataType::FP32;
        TensorView* tv_up = tier_mgr_ ? tier_mgr_->get_tensor_view(pfx + "ffn_up.weight") : nullptr;
        if (!tv_up && tier_mgr_) tv_up = tier_mgr_->get_tensor_view(pfx + "ffn_up_exps.weight");
        DataType dt_up = tv_up ? tv_up->dtype : DataType::FP32;
        TensorView* tv_down = tier_mgr_ ? tier_mgr_->get_tensor_view(pfx + "ffn_down.weight") : nullptr;
        if (!tv_down && tier_mgr_) tv_down = tier_mgr_->get_tensor_view(pfx + "ffn_down_exps.weight");
        DataType dt_down = tv_down ? tv_down->dtype : DataType::FP32;

        // Strict tensor validation if real model is loaded
        if (metadata_.is_loaded) {
            if (!wq_raw || !wk_raw || !wv_raw || !wo_raw || !w_gate_raw || !w_up_raw || !w_down_raw) {
                std::cerr << "[Nexus Engine Error] Required tensor missing for layer " << l << ":";
                if (!wq_raw) std::cerr << " wq";
                if (!wk_raw) std::cerr << " wk";
                if (!wv_raw) std::cerr << " wv";
                if (!wo_raw) std::cerr << " wo";
                if (!w_gate_raw) std::cerr << " w_gate";
                if (!w_up_raw) std::cerr << " w_up";
                if (!w_down_raw) std::cerr << " w_down";
                std::cerr << std::endl;
                return 0;
            }
        }

        // Pre-Attention RMSNorm: hidden = rmsnorm(hidden)
        TransformerOps::rmsnorm(act_hidden_.data(), attn_norm, dt_attn_norm, act_norm_.data(), hidden_size_, metadata_.rms_norm_eps);

        // Q, K, V projections via fused GEMV kernels: q, k, v = gemv(hidden, w_qkv)
        TransformerOps::matmul_vec_dispatch(wq, dt_wq, act_norm_.data(), act_q_.data(), num_heads_ * head_dim_, hidden_size_);
        TransformerOps::matmul_vec_dispatch(wk, dt_wk, act_norm_.data(), act_k_.data(), num_kv_heads_ * head_dim_, hidden_size_);
        TransformerOps::matmul_vec_dispatch(wv, dt_wv, act_norm_.data(), act_v_.data(), num_kv_heads_ * head_dim_, hidden_size_);

        // RoPE positional embeddings
        TransformerOps::rope(act_q_.data(), act_k_.data(), head_dim_, num_heads_, num_kv_heads_, pos, metadata_.rope_theta);

        // Write K and Value into kv_mgr_->write_block(block_id)
        size_t blk_idx = pos / BLOCK_SIZE;
        size_t slot = pos % BLOCK_SIZE;
        if (blk_idx < block_table.size()) {
            int32_t blk_id = block_table[blk_idx];
            KVBlock* blk = kv_mgr_->write_block(blk_id);
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

        // Tiled PagedAttention: attn = paged_attention_cpu(q) (calls kv_mgr_->read_block)
        paged_attention_cpu(act_q_.data(), block_table, l, pos, act_attn_out_.data());

        // Attention output projection and residual: hidden = hidden + gemv(attn, w_o)
        TransformerOps::matmul_vec_dispatch(wo, dt_wo, act_attn_out_.data(), act_attn_proj_.data(), hidden_size_, hidden_size_);
        for (size_t i = 0; i < hidden_size_; ++i) {
            act_hidden_[i] += act_attn_proj_[i];
        }

        // Post-Attention RMSNorm: hidden = rmsnorm(hidden)
        TransformerOps::rmsnorm(act_hidden_.data(), ffn_norm, dt_ffn_norm, act_norm_.data(), hidden_size_, metadata_.rms_norm_eps);

        // SwiGLU FFN: ff = gemv(hidden, w_gate) * gemv(hidden, w_up)
        TransformerOps::matmul_vec_dispatch(w_gate, dt_gate, act_norm_.data(), act_gate_.data(), intermediate_size_, hidden_size_);
        TransformerOps::matmul_vec_dispatch(w_up, dt_up, act_norm_.data(), act_up_.data(), intermediate_size_, hidden_size_);
        TransformerOps::swiglu(act_gate_.data(), act_up_.data(), act_swiglu_.data(), intermediate_size_);

        // FFN down projection and residual: hidden = hidden + gemv(ff, w_down)
        TransformerOps::matmul_vec_dispatch(w_down, dt_down, act_swiglu_.data(), act_down_.data(), hidden_size_, intermediate_size_);
        for (size_t i = 0; i < hidden_size_; ++i) {
            act_hidden_[i] += act_down_[i];
        }

        // Dynamic layer deactivation
        if (tier_mgr_) {
            tier_mgr_->deactivate_layer(l);
        }
    }

    // 3. Final RMSNorm
    void* out_norm_raw = tier_mgr_ ? tier_mgr_->activate_parameter("output_norm.weight") : nullptr;
    TensorView* tv_out_norm = tier_mgr_ ? tier_mgr_->get_tensor_view("output_norm.weight") : nullptr;
    DataType dt_out_norm = tv_out_norm ? tv_out_norm->dtype : DataType::FP32;
    TransformerOps::rmsnorm(act_hidden_.data(), out_norm_raw, dt_out_norm, act_norm_.data(), hidden_size_, metadata_.rms_norm_eps);

    // 4. LM Head: logits = gemv(hidden, w_lm_head)
    void* head_raw = tier_mgr_ ? tier_mgr_->activate_parameter("output.weight") : nullptr;
    const void* lm_head = head_raw ? head_raw : default_lm_head_.data();
    TensorView* tv_head = tier_mgr_ ? tier_mgr_->get_tensor_view("output.weight") : nullptr;
    DataType dt_head = tv_head ? tv_head->dtype : DataType::FP32;
    size_t out_vocab = (tv_head && !tv_head->shape.dims.empty()) ? tv_head->shape.dims[0] : vocab_size_;
    if (act_logits_.size() < out_vocab) {
        act_logits_.resize(out_vocab, 0.0f);
    }
    TransformerOps::matmul_vec_dispatch(lm_head, dt_head, act_norm_.data(), act_logits_.data(), out_vocab, hidden_size_);

    // 5. Sampling: token = sample(logits)
    int32_t next_token = TransformerOps::sample_logits(act_logits_.data(), out_vocab, temperature);
    return next_token;
}

int32_t ForwardPassExecutor::forward_prefill(
    const std::vector<int32_t>& prompt_tokens,
    const std::vector<int32_t>& block_table,
    float temperature
) {
    if (prompt_tokens.empty() || block_table.empty()) return 0;
    int32_t out_tok = 0;
    for (size_t pos = 0; pos < prompt_tokens.size(); ++pos) {
        float temp = (pos + 1 == prompt_tokens.size()) ? temperature : 0.0f;
        out_tok = forward_token(prompt_tokens[pos], pos, block_table, temp);
    }
    return out_tok;
}

int32_t ForwardPassExecutor::execute_interleaved_batch(
    DisaggregatedControlPlane::InterleavedBatch& batch,
    DisaggregatedControlPlane& control_plane,
    int32_t* out_tokens,
    size_t max_out
) {
    int32_t step_count = 0;

    // 1. Process chunked prefills
    for (auto& item : batch.prefill_chunks) {
        auto req = item.first;
        if (!req || req->is_finished) continue;

        if (req->processed_tokens >= req->prompt_tokens.size()) {
            int32_t first_token = forward_prefill(
                req->prompt_tokens,
                req->assigned_block_ids,
                req->temperature
            );
            control_plane.complete_prefill_and_handoff(req, first_token);
            if (out_tokens && static_cast<size_t>(step_count) < max_out) {
                out_tokens[step_count++] = first_token;
            }
        }
    }

    // 2. Process active decodes
    for (size_t i = 0; i < batch.decode_requests.size() && static_cast<size_t>(step_count) < max_out; ++i) {
        auto& req = batch.decode_requests[i];
        if (!req || req->is_finished) continue;

        int32_t last_token = req->generated_tokens.empty() ?
            (req->prompt_tokens.empty() ? 1 : req->prompt_tokens.back()) :
            req->generated_tokens.back();

        size_t current_pos = req->prompt_tokens.size() + req->generated_tokens.size();

        // Ensure dynamic PagedAttention block allocation if sequence crosses block boundary
        size_t needed_block_idx = current_pos / BLOCK_SIZE;
        while (needed_block_idx >= req->assigned_block_ids.size()) {
            auto new_blocks = kv_mgr_->allocate_blocks(1);
            if (!new_blocks.empty()) {
                req->assigned_block_ids.push_back(new_blocks[0]);
            } else {
                break;
            }
        }

        // Full forward evaluation step
        int32_t next_token = forward_token(
            last_token,
            current_pos,
            req->assigned_block_ids,
            req->temperature
        );

        bool is_eos = (next_token == 2 || (req->max_new_tokens > 0 && req->generated_tokens.size() >= req->max_new_tokens));
        control_plane.step_decode_token(req, next_token, is_eos);

        if (out_tokens) out_tokens[step_count] = next_token;
        step_count++;
    }

    return step_count;
}

} // namespace nexus
