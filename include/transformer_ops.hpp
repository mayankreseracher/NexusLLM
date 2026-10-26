#pragma once

#include "nexus_common.hpp"
#include "paged_cache.hpp"
#include <vector>
#include <cmath>
#include <cstring>
#include <algorithm>

namespace nexus {

class TransformerOps {
public:
    // Root Mean Square Normalization: out = (x / sqrt(mean(x^2) + eps)) * weight
    static void rmsnorm(const float* x, const float* weight, float* out, size_t dim, float eps = 1e-6f) {
        float sum_sq = 0.0f;
        for (size_t i = 0; i < dim; ++i) {
            sum_sq += x[i] * x[i];
        }
        float inv_rms = 1.0f / std::sqrt((sum_sq / static_cast<float>(dim)) + eps);
        for (size_t i = 0; i < dim; ++i) {
            out[i] = x[i] * inv_rms * (weight ? weight[i] : 1.0f);
        }
    }

    // Rotary Position Embedding (RoPE) applied to Q and K
    static void rope(float* q, float* k, size_t head_dim, size_t num_q_heads, size_t num_kv_heads, size_t position, float theta = 10000.0f) {
        // RoPE rotates consecutive pairs (x[2*i], x[2*i+1])
        for (size_t i = 0; i < head_dim; i += 2) {
            float freq = 1.0f / std::pow(theta, static_cast<float>(i) / static_cast<float>(head_dim));
            float val = static_cast<float>(position) * freq;
            float cos_val = std::cos(val);
            float sin_val = std::sin(val);

            // Rotate Query heads
            for (size_t h = 0; h < num_q_heads; ++h) {
                float* q_head = q + (h * head_dim);
                float q0 = q_head[i];
                float q1 = q_head[i + 1];
                q_head[i]     = q0 * cos_val - q1 * sin_val;
                q_head[i + 1] = q0 * sin_val + q1 * cos_val;
            }

            // Rotate Key heads
            for (size_t h = 0; h < num_kv_heads; ++h) {
                float* k_head = k + (h * head_dim);
                float k0 = k_head[i];
                float k1 = k_head[i + 1];
                k_head[i]     = k0 * cos_val - k1 * sin_val;
                k_head[i + 1] = k0 * sin_val + k1 * cos_val;
            }
        }
    }

    // SiLU activation: x / (1 + exp(-x))
    static inline float silu(float x) {
        return x / (1.0f + std::exp(-x));
    }

    // SwiGLU: out = SiLU(gate) * up
    static void swiglu(const float* gate, const float* up, float* out, size_t dim) {
        for (size_t i = 0; i < dim; ++i) {
            out[i] = silu(gate[i]) * up[i];
        }
    }

    // Softmax in-place over array
    static void softmax(float* x, size_t n) {
        if (n == 0) return;
        float max_val = x[0];
        for (size_t i = 1; i < n; ++i) {
            if (x[i] > max_val) max_val = x[i];
        }

        float sum = 0.0f;
        for (size_t i = 0; i < n; ++i) {
            x[i] = std::exp(x[i] - max_val);
            sum += x[i];
        }

        float inv_sum = 1.0f / (sum > 0.0f ? sum : 1e-6f);
        for (size_t i = 0; i < n; ++i) {
            x[i] *= inv_sum;
        }
    }

    // Matrix-Vector Multiplication: y = W * x
    // W is [rows, cols] row-major, x is [cols], y is [rows]
    static void matmul_vec(const float* W, const float* x, float* y, size_t rows, size_t cols) {
        for (size_t r = 0; r < rows; ++r) {
            float sum = 0.0f;
            const float* row = W + (r * cols);
            for (size_t c = 0; c < cols; ++c) {
                sum += row[c] * x[c];
            }
            y[r] = sum;
        }
    }

    // PagedAttention Multi-Head/Grouped-Query Attention Decode
    static void paged_attention_decode(
        const float* Q,               // [num_heads, head_dim]
        const std::vector<int32_t>& block_table,
        PagedKVCacheManager& kv_mgr,
        size_t layer_idx,
        size_t seq_len,               // Total tokens prior to current step
        size_t num_heads,
        size_t num_kv_heads,
        size_t head_dim,
        float* attn_out               // Output: [num_heads * head_dim]
    ) {
        size_t kv_mul = num_heads / num_kv_heads;
        float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

        std::vector<float> scores(seq_len + 1);

        for (size_t h = 0; h < num_heads; ++h) {
            size_t kv_h = h / kv_mul;
            const float* q_h = Q + (h * head_dim);

            // Compute attention scores against all past cached tokens
            for (size_t pos = 0; pos <= seq_len; ++pos) {
                size_t blk_idx = pos / BLOCK_SIZE;
                size_t slot = pos % BLOCK_SIZE;

                if (blk_idx >= block_table.size()) {
                    scores[pos] = -1e9f;
                    continue;
                }

                int32_t blk_id = block_table[blk_idx];
                KVBlock* blk = kv_mgr.get_block(blk_id);
                if (!blk || !blk->key_data) {
                    scores[pos] = -1e9f;
                    continue;
                }

                // Offset into block: layer_idx * num_kv_heads * BLOCK_SIZE * head_dim
                size_t stride = num_kv_heads * BLOCK_SIZE * head_dim;
                const float* k_vec = blk->key_data + (layer_idx * stride) + (kv_h * BLOCK_SIZE * head_dim) + (slot * head_dim);

                float dot = 0.0f;
                for (size_t d = 0; d < head_dim; ++d) {
                    dot += q_h[d] * k_vec[d];
                }
                scores[pos] = dot * scale;
            }

            // Softmax over sequence scores
            softmax(scores.data(), seq_len + 1);

            // Weighted sum of Value vectors
            float* out_h = attn_out + (h * head_dim);
            std::memset(out_h, 0, head_dim * sizeof(float));

            for (size_t pos = 0; pos <= seq_len; ++pos) {
                float weight = scores[pos];
                if (weight < 1e-7f) continue;

                size_t blk_idx = pos / BLOCK_SIZE;
                size_t slot = pos % BLOCK_SIZE;
                int32_t blk_id = block_table[blk_idx];
                KVBlock* blk = kv_mgr.get_block(blk_id);
                if (!blk || !blk->value_data) continue;

                size_t stride = num_kv_heads * BLOCK_SIZE * head_dim;
                const float* v_vec = blk->value_data + (layer_idx * stride) + (kv_h * BLOCK_SIZE * head_dim) + (slot * head_dim);

                for (size_t d = 0; d < head_dim; ++d) {
                    out_h[d] += weight * v_vec[d];
                }
            }
        }
    }

    // Sampling: Greedy argmax if temp <= 0.01, otherwise top-1 / categorical
    static int32_t sample_logits(const float* logits, size_t vocab_size, float temperature = 0.0f) {
        if (vocab_size == 0) return 0;
        if (temperature <= 0.01f) {
            // Greedy Argmax
            int32_t best_id = 0;
            float max_val = logits[0];
            for (size_t i = 1; i < vocab_size; ++i) {
                if (logits[i] > max_val) {
                    max_val = logits[i];
                    best_id = static_cast<int32_t>(i);
                }
            }
            return best_id;
        }

        // Temperature-scaled softmax
        std::vector<float> probs(logits, logits + vocab_size);
        for (size_t i = 0; i < vocab_size; ++i) {
            probs[i] /= temperature;
        }
        softmax(probs.data(), vocab_size);

        // Deterministic top-1 above probability threshold
        int32_t best_id = 0;
        float max_p = probs[0];
        for (size_t i = 1; i < vocab_size; ++i) {
            if (probs[i] > max_p) {
                max_p = probs[i];
                best_id = static_cast<int32_t>(i);
            }
        }
        return best_id;
    }
};

} // namespace nexus
