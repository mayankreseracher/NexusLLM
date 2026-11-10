#pragma once

#include "nexus_common.hpp"
#include "paged_cache.hpp"
#include "quantization.hpp"
#include <vector>
#include <cmath>
#include <cstring>
#include <algorithm>

namespace nexus {

// Spin (arXiv:2604.26837) Configuration parameters
struct SpinConfig {
    size_t min_blocks_for_pruning = 8; // Dense execution for <= 128 tokens
    size_t recent_window_blocks = 2;   // Always evaluate the most recent 32 tokens
    float retention_ratio = 0.40f;     // Retain top 40% of historical blocks
};

class TransformerOps {
public:
    // Root Mean Square Normalization with flexible DataType support (FP32, FP16)
    static void rmsnorm(const float* x, const void* weight, DataType dt, float* out, size_t dim, float eps = 1e-6f) {
        float sum_sq = 0.0f;
        for (size_t i = 0; i < dim; ++i) {
            sum_sq += x[i] * x[i];
        }
        float inv_rms = 1.0f / std::sqrt((sum_sq / static_cast<float>(dim)) + eps);
        if (!weight) {
            for (size_t i = 0; i < dim; ++i) {
                out[i] = x[i] * inv_rms;
            }
        } else if (dt == DataType::FP16) {
            const uint16_t* w16 = static_cast<const uint16_t*>(weight);
            for (size_t i = 0; i < dim; ++i) {
                out[i] = x[i] * inv_rms * QuantizationKernel::fp16_to_fp32(w16[i]);
            }
        } else {
            const float* w32 = static_cast<const float*>(weight);
            for (size_t i = 0; i < dim; ++i) {
                out[i] = x[i] * inv_rms * w32[i];
            }
        }
    }

    // Root Mean Square Normalization: out = (x / sqrt(mean(x^2) + eps)) * weight
    static void rmsnorm(const float* x, const float* weight, float* out, size_t dim, float eps = 1e-6f) {
        rmsnorm(x, static_cast<const void*>(weight), DataType::FP32, out, dim, eps);
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
#if defined(_OPENMP)
        #pragma omp parallel for schedule(static)
#endif
        for (size_t r = 0; r < rows; ++r) {
            float sum = 0.0f;
            const float* row = W + (r * cols);
#if defined(NEXUS_HAVE_AVX2)
            __m256 v_acc = _mm256_setzero_ps();
            size_t c = 0;
            for (; c + 8 <= cols; c += 8) {
                __m256 vw = _mm256_loadu_ps(row + c);
                __m256 vx = _mm256_loadu_ps(x + c);
                v_acc = _mm256_fmadd_ps(vw, vx, v_acc);
            }
            alignas(32) float arr[8];
            _mm256_store_ps(arr, v_acc);
            sum = arr[0] + arr[1] + arr[2] + arr[3] + arr[4] + arr[5] + arr[6] + arr[7];
            for (; c < cols; ++c) {
                sum += row[c] * x[c];
            }
#elif defined(NEXUS_HAVE_ARM_NEON)
            float32x4_t v_acc = vdupq_n_f32(0.0f);
            size_t c = 0;
            for (; c + 4 <= cols; c += 4) {
                float32x4_t vw = vld1q_f32(row + c);
                float32x4_t vx = vld1q_f32(x + c);
                v_acc = vfmaq_f32(v_acc, vw, vx);
            }
            sum = vaddvq_f32(v_acc);
            for (; c < cols; ++c) {
                sum += row[c] * x[c];
            }
#else
            for (size_t c = 0; c < cols; ++c) {
                sum += row[c] * x[c];
            }
#endif
            y[r] = sum;
        }
    }

    // Matrix-Vector Multiplication supporting FP32, FP16, GGUF Q4_0, Q8_0, Q5_K, and Q6_K
    static void matmul_vec_dispatch(const void* W, DataType dtype, const float* x, float* y, size_t rows, size_t cols) {
        if (!W) return;
        if (dtype == DataType::GGUF_Q4_0) {
            QuantizationKernel::fused_gemv_q4_0(static_cast<const BlockQ4_0*>(W), x, y, rows, cols);
        } else if (dtype == DataType::GGUF_Q8_0) {
            QuantizationKernel::fused_gemv_q8_0(static_cast<const BlockQ8_0*>(W), x, y, rows, cols);
        } else if (dtype == DataType::GGUF_Q5_K) {
            QuantizationKernel::fused_gemv_q5_k(static_cast<const BlockQ5_K*>(W), x, y, rows, cols);
        } else if (dtype == DataType::GGUF_Q6_K) {
            QuantizationKernel::fused_gemv_q6_k(static_cast<const BlockQ6_K*>(W), x, y, rows, cols);
        } else if (dtype == DataType::FP16) {
            QuantizationKernel::fused_gemv_fp16(static_cast<const uint16_t*>(W), x, y, rows, cols);
        } else {
            // Default FP32
            matmul_vec(static_cast<const float*>(W), x, y, rows, cols);
        }
    }

    // FlashAttention CPU Tiled Online Softmax (§3 of Dao et al., 2022 / vLLM §4.2)
    // Computes attention directly over Paged KV Cache blocks with O(1) workspace,
    // maintaining running maximum m, running sum l, and running accumulator O.
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
        if (num_heads == 0 || num_kv_heads == 0 || head_dim == 0) return;
        size_t kv_mul = num_heads / num_kv_heads;
        float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
        size_t total_tokens = seq_len + 1;

#if defined(_OPENMP)
        #pragma omp parallel for schedule(static)
#endif
        for (size_t h = 0; h < num_heads; ++h) {
            size_t kv_h = h / kv_mul;
            const float* q_h = Q + (h * head_dim);
            float* out_h = attn_out + (h * head_dim);

            // FlashAttention Running State
            float m = -1e30f;       // Running max
            float l = 0.0f;         // Running sum of exponentials
            std::memset(out_h, 0, head_dim * sizeof(float)); // Running output accumulator O

            size_t tokens_processed = 0;

            for (size_t blk_idx = 0; blk_idx < block_table.size() && tokens_processed < total_tokens; ++blk_idx) {
                int32_t blk_id = block_table[blk_idx];
                KVBlock* blk = kv_mgr.get_block(blk_id);
                if (!blk || !blk->key_data || !blk->value_data) {
                    tokens_processed += BLOCK_SIZE;
                    continue;
                }

                size_t valid_in_block = std::min(BLOCK_SIZE, total_tokens - tokens_processed);
                size_t stride = num_kv_heads * BLOCK_SIZE * head_dim;
                const float* k_base = blk->key_data + (layer_idx * stride) + (kv_h * BLOCK_SIZE * head_dim);
                const float* v_base = blk->value_data + (layer_idx * stride) + (kv_h * BLOCK_SIZE * head_dim);

                for (size_t slot = 0; slot < valid_in_block; ++slot) {
                    const float* k_vec = k_base + (slot * head_dim);
                    const float* v_vec = v_base + (slot * head_dim);

                    // 1. Compute dot product: Q_h . K_vec
                    float dot = 0.0f;
#if defined(NEXUS_HAVE_AVX2)
                    __m256 vdot = _mm256_setzero_ps();
                    size_t d = 0;
                    for (; d + 8 <= head_dim; d += 8) {
                        __m256 q_v = _mm256_loadu_ps(q_h + d);
                        __m256 k_v = _mm256_loadu_ps(k_vec + d);
                        vdot = _mm256_fmadd_ps(q_v, k_v, vdot);
                    }
                    alignas(32) float tmp[8];
                    _mm256_storeu_ps(tmp, vdot);
                    dot = tmp[0] + tmp[1] + tmp[2] + tmp[3] + tmp[4] + tmp[5] + tmp[6] + tmp[7];
                    for (; d < head_dim; ++d) {
                        dot += q_h[d] * k_vec[d];
                    }
#elif defined(NEXUS_HAVE_ARM_NEON)
                    float32x4_t vdot = vdupq_n_f32(0.0f);
                    size_t d = 0;
                    for (; d + 4 <= head_dim; d += 4) {
                        float32x4_t q_v = vld1q_f32(q_h + d);
                        float32x4_t k_v = vld1q_f32(k_vec + d);
                        vdot = vmlaq_f32(vdot, q_v, k_v);
                    }
                    dot = vaddvq_f32(vdot);
                    for (; d < head_dim; ++d) {
                        dot += q_h[d] * k_vec[d];
                    }
#else
                    for (size_t d = 0; d < head_dim; ++d) {
                        dot += q_h[d] * k_vec[d];
                    }
#endif
                    float s = dot * scale;

                    // 2. Online Softmax update (FlashAttention §3)
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
                    for (; vd + 8 <= head_dim; vd += 8) {
                        __m256 o_v = _mm256_loadu_ps(out_h + vd);
                        __m256 v_v = _mm256_loadu_ps(v_vec + vd);
                        o_v = _mm256_fmadd_ps(v_v, vbeta, _mm256_mul_ps(o_v, valpha));
                        _mm256_storeu_ps(out_h + vd, o_v);
                    }
                    for (; vd < head_dim; ++vd) {
                        out_h[vd] = (out_h[vd] * alpha) + (beta * v_vec[vd]);
                    }
#elif defined(NEXUS_HAVE_ARM_NEON)
                    float32x4_t valpha = vdupq_n_f32(alpha);
                    float32x4_t vbeta = vdupq_n_f32(beta);
                    size_t vd = 0;
                    for (; vd + 4 <= head_dim; vd += 4) {
                        float32x4_t o_v = vld1q_f32(out_h + vd);
                        float32x4_t v_v = vld1q_f32(v_vec + vd);
                        o_v = vmlaq_f32(vmulq_f32(o_v, valpha), v_v, vbeta);
                        vst1q_f32(out_h + vd, o_v);
                    }
                    for (; vd < head_dim; ++vd) {
                        out_h[vd] = (out_h[vd] * alpha) + (beta * v_vec[vd]);
                    }
#else
                    for (size_t vd = 0; vd < head_dim; ++vd) {
                        out_h[vd] = (out_h[vd] * alpha) + (beta * v_vec[vd]);
                    }
#endif
                }
                tokens_processed += valid_in_block;
            }

            // 4. Final normalization: O = O / l
            float inv_l = (l > 1e-12f) ? (1.0f / l) : 0.0f;
#if defined(NEXUS_HAVE_AVX2)
            __m256 vinv = _mm256_set1_ps(inv_l);
            size_t fd = 0;
            for (; fd + 8 <= head_dim; fd += 8) {
                __m256 o_v = _mm256_loadu_ps(out_h + fd);
                _mm256_storeu_ps(out_h + fd, _mm256_mul_ps(o_v, vinv));
            }
            for (; fd < head_dim; ++fd) {
                out_h[fd] *= inv_l;
            }
#elif defined(NEXUS_HAVE_ARM_NEON)
            float32x4_t vinv = vdupq_n_f32(inv_l);
            size_t fd = 0;
            for (; fd + 4 <= head_dim; fd += 4) {
                float32x4_t o_v = vld1q_f32(out_h + fd);
                vst1q_f32(out_h + fd, vmulq_f32(o_v, vinv));
            }
            for (; fd < head_dim; ++fd) {
                out_h[fd] *= inv_l;
            }
#else
            for (size_t fd = 0; fd < head_dim; ++fd) {
                out_h[fd] *= inv_l;
            }
#endif
        }
    }

    // Spin (arXiv:2604.26837) Sparse Paged Attention
    // Dynamically scores physical KV cache blocks using coarse Key centroids,
    // pruning 60% of cold blocks before token-level FlashAttention online softmax.
    static void paged_attention_decode_spin(
        const float* Q,
        const std::vector<int32_t>& block_table,
        PagedKVCacheManager& kv_mgr,
        size_t layer_idx,
        size_t seq_len,
        size_t num_heads,
        size_t num_kv_heads,
        size_t head_dim,
        float* attn_out,
        const SpinConfig& cfg = SpinConfig()
    ) {
        if (num_heads == 0 || num_kv_heads == 0 || head_dim == 0 || block_table.empty()) return;
        size_t total_blocks = block_table.size();

        // Small sequence fast-path: dense paged attention
        if (total_blocks <= cfg.min_blocks_for_pruning) {
            paged_attention_decode(Q, block_table, kv_mgr, layer_idx, seq_len, num_heads, num_kv_heads, head_dim, attn_out);
            return;
        }

        size_t kv_mul = num_heads / num_kv_heads;
        float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
        size_t c_stride = num_kv_heads * head_dim;
        size_t num_recent = std::min(cfg.recent_window_blocks, total_blocks);
        size_t historical_blocks = total_blocks - num_recent;
        size_t top_k_blocks = std::max<size_t>(1, static_cast<size_t>(historical_blocks * cfg.retention_ratio));

#if defined(_OPENMP)
        #pragma omp parallel for schedule(static)
#endif
        for (size_t h = 0; h < num_heads; ++h) {
            size_t kv_h = h / kv_mul;
            const float* q_h = Q + (h * head_dim);
            float* out_h = attn_out + (h * head_dim);

            // Stage 1: Coarse Centroid Filter across historical blocks
            std::vector<std::pair<float, int32_t>> scores;
            scores.reserve(historical_blocks);

            for (size_t b = 0; b < historical_blocks; ++b) {
                int32_t blk_id = block_table[b];
                KVBlock* blk = kv_mgr.get_block(blk_id);
                if (!blk || !blk->k_centroid || !blk->centroid_valid) {
                    scores.push_back({0.0f, blk_id});
                    continue;
                }
                const float* c_vec = blk->k_centroid + (layer_idx * c_stride) + (kv_h * head_dim);

                float dot = 0.0f;
#if defined(NEXUS_HAVE_AVX2)
                __m256 vdot = _mm256_setzero_ps();
                size_t d = 0;
                for (; d + 8 <= head_dim; d += 8) {
                    vdot = _mm256_fmadd_ps(_mm256_loadu_ps(q_h + d), _mm256_loadu_ps(c_vec + d), vdot);
                }
                alignas(32) float arr[8];
                _mm256_store_ps(arr, vdot);
                dot = arr[0] + arr[1] + arr[2] + arr[3] + arr[4] + arr[5] + arr[6] + arr[7];
                for (; d < head_dim; ++d) dot += q_h[d] * c_vec[d];
#else
                for (size_t d = 0; d < head_dim; ++d) dot += q_h[d] * c_vec[d];
#endif
                scores.push_back({dot, blk_id});
            }

            // Select top-K highest scoring historical blocks
            if (top_k_blocks < scores.size()) {
                std::partial_sort(scores.begin(), scores.begin() + top_k_blocks, scores.end(),
                                  [](const auto& a, const auto& b) { return a.first > b.first; });
            }

            // Stage 2: Fine-Grained FlashAttention online softmax over selected + recent window blocks
            float m = -1e30f;
            float l = 0.0f;
            std::memset(out_h, 0, head_dim * sizeof(float));

            auto evaluate_block = [&](int32_t blk_id) {
                KVBlock* blk = kv_mgr.get_block(blk_id);
                if (!blk || !blk->key_data || !blk->value_data || blk->tokens_occupied == 0) return;

                size_t stride = num_kv_heads * BLOCK_SIZE * head_dim;
                const float* k_base = blk->key_data + (layer_idx * stride) + (kv_h * BLOCK_SIZE * head_dim);
                const float* v_base = blk->value_data + (layer_idx * stride) + (kv_h * BLOCK_SIZE * head_dim);

                for (size_t slot = 0; slot < blk->tokens_occupied; ++slot) {
                    const float* k_vec = k_base + (slot * head_dim);
                    const float* v_vec = v_base + (slot * head_dim);

                    float dot = 0.0f;
#if defined(NEXUS_HAVE_AVX2)
                    __m256 vdot = _mm256_setzero_ps();
                    size_t d = 0;
                    for (; d + 8 <= head_dim; d += 8) {
                        vdot = _mm256_fmadd_ps(_mm256_loadu_ps(q_h + d), _mm256_loadu_ps(k_vec + d), vdot);
                    }
                    alignas(32) float arr[8];
                    _mm256_store_ps(arr, vdot);
                    dot = arr[0] + arr[1] + arr[2] + arr[3] + arr[4] + arr[5] + arr[6] + arr[7];
                    for (; d < head_dim; ++d) dot += q_h[d] * k_vec[d];
#else
                    for (size_t d = 0; d < head_dim; ++d) dot += q_h[d] * k_vec[d];
#endif
                    float s = dot * scale;
                    float m_prev = m;
                    float m_new = (s > m_prev) ? s : m_prev;
                    float alpha = std::exp(m_prev - m_new);
                    float beta = std::exp(s - m_new);

                    l = (l * alpha) + beta;
                    m = m_new;

#if defined(NEXUS_HAVE_AVX2)
                    __m256 valpha = _mm256_set1_ps(alpha);
                    __m256 vbeta = _mm256_set1_ps(beta);
                    size_t vd = 0;
                    for (; vd + 8 <= head_dim; vd += 8) {
                        __m256 o_v = _mm256_loadu_ps(out_h + vd);
                        __m256 v_v = _mm256_loadu_ps(v_vec + vd);
                        o_v = _mm256_fmadd_ps(v_v, vbeta, _mm256_mul_ps(o_v, valpha));
                        _mm256_storeu_ps(out_h + vd, o_v);
                    }
                    for (; vd < head_dim; ++vd) out_h[vd] = (out_h[vd] * alpha) + (beta * v_vec[vd]);
#else
                    for (size_t vd = 0; vd < head_dim; ++vd) out_h[vd] = (out_h[vd] * alpha) + (beta * v_vec[vd]);
#endif
                }
            };

            // Selected historical blocks
            size_t act_count = std::min(top_k_blocks, scores.size());
            for (size_t i = 0; i < act_count; ++i) {
                evaluate_block(scores[i].second);
            }
            // Invariant recent window blocks
            for (size_t i = historical_blocks; i < total_blocks; ++i) {
                evaluate_block(block_table[i]);
            }

            // Final normalization
            float inv_l = (l > 1e-12f) ? (1.0f / l) : 0.0f;
#if defined(NEXUS_HAVE_AVX2)
            __m256 vinv = _mm256_set1_ps(inv_l);
            size_t fd = 0;
            for (; fd + 8 <= head_dim; fd += 8) {
                __m256 o_v = _mm256_loadu_ps(out_h + fd);
                _mm256_storeu_ps(out_h + fd, _mm256_mul_ps(o_v, vinv));
            }
            for (; fd < head_dim; ++fd) out_h[fd] *= inv_l;
#else
            for (size_t fd = 0; fd < head_dim; ++fd) out_h[fd] *= inv_l;
#endif
        }
    }

    // Reference PagedAttention Decode (for numerical validation)
    static void paged_attention_naive(
        const float* Q,
        const std::vector<int32_t>& block_table,
        PagedKVCacheManager& kv_mgr,
        size_t layer_idx,
        size_t seq_len,
        size_t num_heads,
        size_t num_kv_heads,
        size_t head_dim,
        float* attn_out
    ) {
        size_t kv_mul = num_heads / num_kv_heads;
        float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
        std::vector<float> scores(seq_len + 1);

        for (size_t h = 0; h < num_heads; ++h) {
            size_t kv_h = h / kv_mul;
            const float* q_h = Q + (h * head_dim);

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

                size_t stride = num_kv_heads * BLOCK_SIZE * head_dim;
                const float* k_vec = blk->key_data + (layer_idx * stride) + (kv_h * BLOCK_SIZE * head_dim) + (slot * head_dim);

                float dot = 0.0f;
                for (size_t d = 0; d < head_dim; ++d) {
                    dot += q_h[d] * k_vec[d];
                }
                scores[pos] = dot * scale;
            }

            softmax(scores.data(), seq_len + 1);

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

    // Sampling: Greedy argmax if temp <= 0.01, otherwise top-k/top-p/stochastic categorical sampling
    static int32_t sample_logits(const float* logits, size_t vocab_size, float temperature = 0.0f, float top_p = 0.95f, int32_t top_k = 40) {
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

        // 1. Temperature-scaled probabilities
        std::vector<float> probs(logits, logits + vocab_size);
        for (size_t i = 0; i < vocab_size; ++i) {
            probs[i] /= temperature;
        }
        softmax(probs.data(), vocab_size);

        // 2. Top-K filtering
        std::vector<std::pair<float, int32_t>> indexed_probs;
        indexed_probs.reserve(vocab_size);
        for (size_t i = 0; i < vocab_size; ++i) {
            if (probs[i] > 1e-6f) {
                indexed_probs.push_back({probs[i], static_cast<int32_t>(i)});
            }
        }

        if (indexed_probs.empty()) return 0;

        size_t effective_k = (top_k > 0 && static_cast<size_t>(top_k) < indexed_probs.size()) 
                           ? static_cast<size_t>(top_k) 
                           : indexed_probs.size();

        std::partial_sort(indexed_probs.begin(),
                          indexed_probs.begin() + effective_k,
                          indexed_probs.end(),
                          [](const std::pair<float, int32_t>& a, const std::pair<float, int32_t>& b) {
                              return a.first > b.first;
                          });

        indexed_probs.resize(effective_k);

        // 3. Top-P (Nucleus) truncation
        float cumsum = 0.0f;
        size_t cutoff = indexed_probs.size();
        for (size_t i = 0; i < indexed_probs.size(); ++i) {
            cumsum += indexed_probs[i].first;
            if (cumsum >= top_p && i > 0) {
                cutoff = i + 1;
                break;
            }
        }
        indexed_probs.resize(cutoff);

        // 4. Renormalize remaining probabilities
        float sum_p = 0.0f;
        for (const auto& p : indexed_probs) sum_p += p.first;
        if (sum_p <= 0.0f) return indexed_probs[0].second;

        // 5. Stochastic sampling with deterministic linear congruential fallback generator
        static thread_local uint64_t rng_state = 13374269;
        rng_state = rng_state * 6364136223846793005ULL + 1442695040888963407ULL;
        float r = static_cast<float>((rng_state >> 32) & 0xFFFFFF) / 16777216.0f * sum_p;

        float acc = 0.0f;
        for (const auto& p : indexed_probs) {
            acc += p.first;
            if (r <= acc) {
                return p.second;
            }
        }
        return indexed_probs.back().second;
    }
};

} // namespace nexus
