#include "quantization.hpp"
#include <cmath>
#include <cstring>

namespace nexus {

// Helper: Convert FP16 uint16_t to FP32 float
static inline float fp16_to_fp32(uint16_t h) {
    return QuantizationKernel::fp16_to_fp32(h);
}

// Helper: Convert FP8 E4M3 to FP32
static inline float fp8_e4m3_to_fp32(uint8_t val) {
    if (val == 0x7F || val == 0xFF) return 0.0f; // NaN
    int sign = (val >> 7) & 1;
    int exp = (val >> 3) & 0xF;
    int mant = val & 0x7;

    float result;
    if (exp == 0) {
        result = std::ldexp(static_cast<float>(mant), -9); // subnormal
    } else {
        result = std::ldexp(1.0f + static_cast<float>(mant) / 8.0f, exp - 7);
    }
    return sign ? -result : result;
}

void QuantizationKernel::dequantize_int4_awq(const uint32_t* packed, 
                                           const float* scales, 
                                           const float* zeros, 
                                           float* output, 
                                           size_t count, 
                                           size_t group_size) {
    for (size_t i = 0; i < count; ++i) {
        size_t group_idx = i / group_size;
        float scale = scales[group_idx];
        float zero = zeros ? zeros[group_idx] : 0.0f;

        size_t word_idx = i / 8;
        size_t nibble_idx = i % 8;
        uint32_t w = packed[word_idx];
        uint32_t q = (w >> (nibble_idx * 4)) & 0x0F;

        output[i] = (static_cast<float>(q) - zero) * scale;
    }
}

void QuantizationKernel::dequantize_gguf_q4_0(const BlockQ4_0* blocks, 
                                            float* output, 
                                            size_t num_blocks) {
    for (size_t b = 0; b < num_blocks; ++b) {
        float d = fp16_to_fp32(blocks[b].d);
        const uint8_t* qs = blocks[b].qs;
        float* out = output + (b * 32);

#if defined(NEXUS_HAVE_ARM_NEON)
        // ARM NEON vector path for Apple Silicon M1-M4 & aarch64 Linux
        float32x4_t vd = vdupq_n_f32(d);
        for (size_t i = 0; i < 16; i += 4) {
            float vals0[4];
            float vals1[4];
            for (size_t k = 0; k < 4; ++k) {
                uint8_t byte = qs[i + k];
                vals0[k] = static_cast<float>((byte & 0x0F) - 8);
                vals1[k] = static_cast<float>((byte >> 4) - 8);
            }
            float32x4_t v0 = vmulq_f32(vld1q_f32(vals0), vd);
            float32x4_t v1 = vmulq_f32(vld1q_f32(vals1), vd);
            vst1q_f32(out + i, v0);
            vst1q_f32(out + i + 16, v1);
        }
#else
        for (size_t i = 0; i < 16; ++i) {
            uint8_t val = qs[i];
            int8_t v0 = (val & 0x0F) - 8;
            int8_t v1 = (val >> 4) - 8;
            out[i] = v0 * d;
            out[i + 16] = v1 * d;
        }
#endif
    }
}

void QuantizationKernel::dequantize_fp8_e4m3(const uint8_t* input, 
                                           float* output, 
                                           size_t count, 
                                           float scale) {
    for (size_t i = 0; i < count; ++i) {
        output[i] = fp8_e4m3_to_fp32(input[i]) * scale;
    }
}

void QuantizationKernel::dequantize_int8(const int8_t* input, 
                                       float* output, 
                                       size_t count, 
                                       float scale) {
    for (size_t i = 0; i < count; ++i) {
        output[i] = static_cast<float>(input[i]) * scale;
    }
}

void QuantizationKernel::fused_gemv_int4(const uint32_t* W_packed, 
                                       const float* x, 
                                       const float* scales, 
                                       const float* zeros, 
                                       float* y, 
                                       size_t rows, 
                                       size_t cols, 
                                       size_t group_size) {
    size_t packed_cols = cols / 8;
    for (size_t r = 0; r < rows; ++r) {
        const uint32_t* row_packed = W_packed + (r * packed_cols);
        size_t g_base = r * (cols / group_size);

#if defined(NEXUS_HAVE_ARM_NEON)
        // High-throughput ARM NEON vector accumulator
        float32x4_t v_acc = vdupq_n_f32(0.0f);
        size_t c = 0;
        for (; c + 4 <= cols; c += 4) {
            size_t g_idx = g_base + (c / group_size);
            float scale = scales[g_idx];
            float zero = zeros ? zeros[g_idx] : 0.0f;

            float w_deq[4];
            for (size_t k = 0; k < 4; ++k) {
                size_t col_idx = c + k;
                uint32_t word = row_packed[col_idx / 8];
                uint32_t q = (word >> ((col_idx % 8) * 4)) & 0x0F;
                w_deq[k] = (static_cast<float>(q) - zero) * scale;
            }
            float32x4_t vw = vld1q_f32(w_deq);
            float32x4_t vx = vld1q_f32(x + c);
            v_acc = vfmaq_f32(v_acc, vw, vx);
        }
        float sum = vaddvq_f32(v_acc);
        for (; c < cols; ++c) {
            size_t g_idx = g_base + (c / group_size);
            float scale = scales[g_idx];
            float zero = zeros ? zeros[g_idx] : 0.0f;
            uint32_t word = row_packed[c / 8];
            uint32_t q = (word >> ((c % 8) * 4)) & 0x0F;
            float w = (static_cast<float>(q) - zero) * scale;
            sum += w * x[c];
        }
        y[r] = sum;

#elif defined(NEXUS_HAVE_AVX2)
        // High-throughput AVX2 + FMA vector accumulator
        __m256 v_acc = _mm256_setzero_ps();
        size_t c = 0;
        for (; c + 8 <= cols; c += 8) {
            size_t g_idx = g_base + (c / group_size);
            float scale = scales[g_idx];
            float zero = zeros ? zeros[g_idx] : 0.0f;

            alignas(32) float w_deq[8];
            uint32_t word = row_packed[c / 8];
            for (size_t k = 0; k < 8; ++k) {
                uint32_t q = (word >> (k * 4)) & 0x0F;
                w_deq[k] = (static_cast<float>(q) - zero) * scale;
            }
            __m256 vw = _mm256_load_ps(w_deq);
            __m256 vx = _mm256_loadu_ps(x + c);
            v_acc = _mm256_fmadd_ps(vw, vx, v_acc);
        }

        // Horizontal reduction of 8 floats in __m256
        alignas(32) float acc_arr[8];
        _mm256_store_ps(acc_arr, v_acc);
        float sum = acc_arr[0] + acc_arr[1] + acc_arr[2] + acc_arr[3] +
                    acc_arr[4] + acc_arr[5] + acc_arr[6] + acc_arr[7];

        for (; c < cols; ++c) {
            size_t g_idx = g_base + (c / group_size);
            float scale = scales[g_idx];
            float zero = zeros ? zeros[g_idx] : 0.0f;
            uint32_t word = row_packed[c / 8];
            uint32_t q = (word >> ((c % 8) * 4)) & 0x0F;
            float w = (static_cast<float>(q) - zero) * scale;
            sum += w * x[c];
        }
        y[r] = sum;

#else
        // Portable scalar fallback
        float sum = 0.0f;
        for (size_t c = 0; c < cols; ++c) {
            size_t g_idx = g_base + (c / group_size);
            float scale = scales[g_idx];
            float zero = zeros ? zeros[g_idx] : 0.0f;

            uint32_t word = row_packed[c / 8];
            uint32_t q = (word >> ((c % 8) * 4)) & 0x0F;
            float w = (static_cast<float>(q) - zero) * scale;
            sum += w * x[c];
        }
        y[r] = sum;
#endif
    }
}

void QuantizationKernel::fused_gemv_fp8(const uint8_t* W_fp8, 
                                      const float* x, 
                                      float* y, 
                                      size_t rows, 
                                      size_t cols, 
                                      float weight_scale) {
    for (size_t r = 0; r < rows; ++r) {
        const uint8_t* row_weights = W_fp8 + (r * cols);

#if defined(NEXUS_HAVE_ARM_NEON)
        float32x4_t v_acc = vdupq_n_f32(0.0f);
        size_t c = 0;
        for (; c + 4 <= cols; c += 4) {
            float w_deq[4];
            for (size_t k = 0; k < 4; ++k) {
                w_deq[k] = fp8_e4m3_to_fp32(row_weights[c + k]) * weight_scale;
            }
            float32x4_t vw = vld1q_f32(w_deq);
            float32x4_t vx = vld1q_f32(x + c);
            v_acc = vfmaq_f32(v_acc, vw, vx);
        }
        float sum = vaddvq_f32(v_acc);
        for (; c < cols; ++c) {
            float w = fp8_e4m3_to_fp32(row_weights[c]) * weight_scale;
            sum += w * x[c];
        }
        y[r] = sum;

#elif defined(NEXUS_HAVE_AVX2)
        __m256 v_acc = _mm256_setzero_ps();
        size_t c = 0;
        for (; c + 8 <= cols; c += 8) {
            alignas(32) float w_deq[8];
            for (size_t k = 0; k < 8; ++k) {
                w_deq[k] = fp8_e4m3_to_fp32(row_weights[c + k]) * weight_scale;
            }
            __m256 vw = _mm256_load_ps(w_deq);
            __m256 vx = _mm256_loadu_ps(x + c);
            v_acc = _mm256_fmadd_ps(vw, vx, v_acc);
        }
        alignas(32) float acc_arr[8];
        _mm256_store_ps(acc_arr, v_acc);
        float sum = acc_arr[0] + acc_arr[1] + acc_arr[2] + acc_arr[3] +
                    acc_arr[4] + acc_arr[5] + acc_arr[6] + acc_arr[7];

        for (; c < cols; ++c) {
            float w = fp8_e4m3_to_fp32(row_weights[c]) * weight_scale;
            sum += w * x[c];
        }
        y[r] = sum;

#else
        float sum = 0.0f;
        for (size_t c = 0; c < cols; ++c) {
            float w = fp8_e4m3_to_fp32(row_weights[c]) * weight_scale;
            sum += w * x[c];
        }
        y[r] = sum;
#endif
    }
}

void QuantizationKernel::fused_gemv_q4_0(const BlockQ4_0* W_q4,
                                        const float* x,
                                        float* y,
                                        size_t rows,
                                        size_t cols) {
    size_t num_blocks_per_row = cols / 32;

    for (size_t r = 0; r < rows; ++r) {
        const BlockQ4_0* row_blocks = W_q4 + (r * num_blocks_per_row);
        float total_sum = 0.0f;

        for (size_t b = 0; b < num_blocks_per_row; ++b) {
            float d = fp16_to_fp32(row_blocks[b].d);
            const uint8_t* qs = row_blocks[b].qs;
            const float* x_block = x + (b * 32);
            float block_sum = 0.0f;

#if defined(NEXUS_HAVE_AVX2)
            __m256 acc0 = _mm256_setzero_ps();
            __m256 acc1 = _mm256_setzero_ps();

            alignas(32) float w_low[16];
            alignas(32) float w_high[16];
            for (size_t i = 0; i < 16; ++i) {
                uint8_t byte = qs[i];
                w_low[i] = static_cast<float>((byte & 0x0F) - 8);
                w_high[i] = static_cast<float>((byte >> 4) - 8);
            }

            acc0 = _mm256_fmadd_ps(_mm256_load_ps(w_low), _mm256_loadu_ps(x_block), acc0);
            acc0 = _mm256_fmadd_ps(_mm256_load_ps(w_low + 8), _mm256_loadu_ps(x_block + 8), acc0);

            acc1 = _mm256_fmadd_ps(_mm256_load_ps(w_high), _mm256_loadu_ps(x_block + 16), acc1);
            acc1 = _mm256_fmadd_ps(_mm256_load_ps(w_high + 8), _mm256_loadu_ps(x_block + 24), acc1);

            __m256 acc = _mm256_add_ps(acc0, acc1);
            alignas(32) float arr[8];
            _mm256_store_ps(arr, acc);
            block_sum = arr[0] + arr[1] + arr[2] + arr[3] + arr[4] + arr[5] + arr[6] + arr[7];

#elif defined(NEXUS_HAVE_ARM_NEON)
            float32x4_t acc0 = vdupq_n_f32(0.0f);
            float32x4_t acc1 = vdupq_n_f32(0.0f);
            for (size_t i = 0; i < 16; i += 4) {
                float w0[4], w1[4];
                for (size_t k = 0; k < 4; ++k) {
                    uint8_t byte = qs[i + k];
                    w0[k] = static_cast<float>((byte & 0x0F) - 8);
                    w1[k] = static_cast<float>((byte >> 4) - 8);
                }
                acc0 = vfmaq_f32(acc0, vld1q_f32(w0), vld1q_f32(x_block + i));
                acc1 = vfmaq_f32(acc1, vld1q_f32(w1), vld1q_f32(x_block + 16 + i));
            }
            block_sum = vaddvq_f32(vaddq_f32(acc0, acc1));
#else
            for (size_t i = 0; i < 16; ++i) {
                uint8_t byte = qs[i];
                block_sum += static_cast<float>((byte & 0x0F) - 8) * x_block[i];
                block_sum += static_cast<float>((byte >> 4) - 8) * x_block[i + 16];
            }
#endif
            total_sum += d * block_sum;
        }

        y[r] = total_sum;
    }
}

void QuantizationKernel::fused_gemv_q8_0(const BlockQ8_0* W_q8,
                                        const float* x,
                                        float* y,
                                        size_t rows,
                                        size_t cols) {
    size_t num_blocks_per_row = cols / 32;

    for (size_t r = 0; r < rows; ++r) {
        const BlockQ8_0* row_blocks = W_q8 + (r * num_blocks_per_row);
        float total_sum = 0.0f;

        for (size_t b = 0; b < num_blocks_per_row; ++b) {
            float d = fp16_to_fp32(row_blocks[b].d);
            const int8_t* qs = row_blocks[b].qs;
            const float* x_block = x + (b * 32);
            float block_sum = 0.0f;

#if defined(NEXUS_HAVE_AVX2)
            __m256 acc = _mm256_setzero_ps();
            for (size_t i = 0; i < 32; i += 8) {
                alignas(32) float w_f[8];
                for (size_t k = 0; k < 8; ++k) {
                    w_f[k] = static_cast<float>(qs[i + k]);
                }
                acc = _mm256_fmadd_ps(_mm256_load_ps(w_f), _mm256_loadu_ps(x_block + i), acc);
            }
            alignas(32) float arr[8];
            _mm256_store_ps(arr, acc);
            block_sum = arr[0] + arr[1] + arr[2] + arr[3] + arr[4] + arr[5] + arr[6] + arr[7];
#elif defined(NEXUS_HAVE_ARM_NEON)
            float32x4_t acc = vdupq_n_f32(0.0f);
            for (size_t i = 0; i < 32; i += 4) {
                float w_f[4];
                for (size_t k = 0; k < 4; ++k) w_f[k] = static_cast<float>(qs[i + k]);
                acc = vfmaq_f32(acc, vld1q_f32(w_f), vld1q_f32(x_block + i));
            }
            block_sum = vaddvq_f32(acc);
#else
            for (size_t i = 0; i < 32; ++i) {
                block_sum += static_cast<float>(qs[i]) * x_block[i];
            }
#endif
            total_sum += d * block_sum;
        }

        y[r] = total_sum;
    }
}

void QuantizationKernel::fused_gemv_fp16(const uint16_t* W_fp16,
                                        const float* x,
                                        float* y,
                                        size_t rows,
                                        size_t cols) {
    for (size_t r = 0; r < rows; ++r) {
        const uint16_t* row = W_fp16 + (r * cols);
        float sum = 0.0f;
        size_t c = 0;

#if defined(NEXUS_HAVE_AVX2)
        __m256 v_acc = _mm256_setzero_ps();
        for (; c + 8 <= cols; c += 8) {
            alignas(32) float w_f[8];
            for (size_t k = 0; k < 8; ++k) {
                w_f[k] = fp16_to_fp32(row[c + k]);
            }
            v_acc = _mm256_fmadd_ps(_mm256_load_ps(w_f), _mm256_loadu_ps(x + c), v_acc);
        }
        alignas(32) float arr[8];
        _mm256_store_ps(arr, v_acc);
        sum = arr[0] + arr[1] + arr[2] + arr[3] + arr[4] + arr[5] + arr[6] + arr[7];
#endif

        for (; c < cols; ++c) {
            sum += fp16_to_fp32(row[c]) * x[c];
        }
        y[r] = sum;
    }
}

// -----------------------------------------------------------------------------
// GGUF Q5_K & Q6_K Dequantization and Fused GEMV Implementation
// Follows standard GGML k-quants specification (256 elements per super-block)
// -----------------------------------------------------------------------------

// Helper to unpack 16 6-bit scales and 16 6-bit mins from 12 bytes
static inline void unpack_q5_k_scales(const uint8_t* s, uint8_t* sc, uint8_t* m) {
    // 12 bytes hold 24 4-bit nibbles or packed 6-bit pairs
    // In GGUF Q5_K:
    // s[0..3] low 6 bits for scales 0..3, high 2 bits are low 2 bits of min 0..3
    // s[4..7] low 6 bits for scales 4..7, high 2 bits are high 4 bits of min 0..3
    // etc.
    for (int i = 0; i < 4; ++i) {
        sc[i]     = s[i] & 0x3F;
        m[i]      = s[i+4] & 0x3F;
        sc[i+4]   = (s[i+8] & 0x0F) | ((s[i] >> 6) << 4);
        m[i+4]    = (s[i+8] >> 4)   | ((s[i+4] >> 6) << 4);
    }
}

void QuantizationKernel::dequantize_gguf_q5_k(const BlockQ5_K* blocks,
                                             float* output,
                                             size_t num_blocks) {
    for (size_t b = 0; b < num_blocks; ++b) {
        const BlockQ5_K& blk = blocks[b];
        const float d = fp16_to_fp32(blk.d);
        const float dmin = fp16_to_fp32(blk.dmin);

        uint8_t sc[8];
        uint8_t m[8];
        unpack_q5_k_scales(blk.scales, sc, m);

        const uint8_t* qh = blk.qh;
        const uint8_t* qs = blk.qs;
        float* out = output + (b * QK_K);

        // 8 sub-blocks of 32 values = 256 values
        for (size_t sb = 0; sb < 8; ++sb) {
            const float scale = d * sc[sb];
            const float min_val = dmin * m[sb];
            const size_t qh_offset = sb * 4;
            const size_t qs_offset = sb * 16;
            float* sb_out = out + (sb * 32);

            for (size_t i = 0; i < 16; ++i) {
                uint8_t q_byte = qs[qs_offset + i];
                uint8_t h_byte = qh[qh_offset + (i / 4)];
                uint8_t h_shift_l = (i % 4) * 2;
                uint8_t h_shift_h = h_shift_l + 1;

                uint8_t h0 = (h_byte >> h_shift_l) & 1;
                uint8_t h1 = (h_byte >> h_shift_h) & 1;

                uint8_t v0 = (q_byte & 0x0F) | (h0 << 4);
                uint8_t v1 = (q_byte >> 4)   | (h1 << 4);

                sb_out[i]      = static_cast<float>(v0) * scale - min_val;
                sb_out[i + 16] = static_cast<float>(v1) * scale - min_val;
            }
        }
    }
}

void QuantizationKernel::dequantize_gguf_q6_k(const BlockQ6_K* blocks,
                                             float* output,
                                             size_t num_blocks) {
    for (size_t b = 0; b < num_blocks; ++b) {
        const BlockQ6_K& blk = blocks[b];
        const float d = fp16_to_fp32(blk.d);
        const uint8_t* ql = blk.ql;
        const uint8_t* qh = blk.qh;
        const int8_t* sc = blk.scales;
        float* out = output + (b * QK_K);

        // 16 sub-blocks of 16 values = 256 values
        for (size_t sb = 0; sb < 16; ++sb) {
            const float scale = d * static_cast<float>(sc[sb]);
            float* sb_out = out + (sb * 16);
            const size_t ql_offset = sb * 8;
            const size_t qh_offset = sb * 4;

            for (size_t i = 0; i < 8; ++i) {
                uint8_t l_byte = ql[ql_offset + i];
                uint8_t h_byte = qh[qh_offset + (i / 2)];
                uint8_t h_shift = (i % 2) * 4;

                uint8_t h0 = (h_byte >> h_shift) & 0x03;
                uint8_t h1 = (h_byte >> (h_shift + 2)) & 0x03;

                int8_t v0 = static_cast<int8_t>((l_byte & 0x0F) | (h0 << 4)) - 32;
                int8_t v1 = static_cast<int8_t>((l_byte >> 4)   | (h1 << 4)) - 32;

                sb_out[i]     = static_cast<float>(v0) * scale;
                sb_out[i + 8] = static_cast<float>(v1) * scale;
            }
        }
    }
}

void QuantizationKernel::fused_gemv_q5_k(const BlockQ5_K* W_q5_k,
                                        const float* x,
                                        float* y,
                                        size_t rows,
                                        size_t cols) {
    const size_t blocks_per_row = cols / QK_K;

#if defined(_OPENMP)
    #pragma omp parallel for schedule(static)
#endif
    for (size_t r = 0; r < rows; ++r) {
        const BlockQ5_K* row_blocks = W_q5_k + (r * blocks_per_row);
        float total_sum = 0.0f;

        for (size_t b = 0; b < blocks_per_row; ++b) {
            const BlockQ5_K& blk = row_blocks[b];
            const float d = fp16_to_fp32(blk.d);
            const float dmin = fp16_to_fp32(blk.dmin);

            uint8_t sc[8];
            uint8_t m[8];
            unpack_q5_k_scales(blk.scales, sc, m);

            const uint8_t* qh = blk.qh;
            const uint8_t* qs = blk.qs;
            const float* x_blk = x + (b * QK_K);

            for (size_t sb = 0; sb < 8; ++sb) {
                const float scale = d * sc[sb];
                const float min_val = dmin * m[sb];
                const size_t qh_offset = sb * 4;
                const size_t qs_offset = sb * 16;
                const float* x_sb = x_blk + (sb * 32);

#if defined(NEXUS_HAVE_AVX2)
                __m256 v_acc = _mm256_setzero_ps();
                __m256 v_scale = _mm256_set1_ps(scale);
                __m256 v_min = _mm256_set1_ps(min_val);

                alignas(32) float w_sb[32];
                for (size_t i = 0; i < 16; ++i) {
                    uint8_t q_byte = qs[qs_offset + i];
                    uint8_t h_byte = qh[qh_offset + (i / 4)];
                    uint8_t h_shift_l = (i % 4) * 2;
                    uint8_t h_shift_h = h_shift_l + 1;

                    uint8_t h0 = (h_byte >> h_shift_l) & 1;
                    uint8_t h1 = (h_byte >> h_shift_h) & 1;

                    w_sb[i]      = static_cast<float>((q_byte & 0x0F) | (h0 << 4));
                    w_sb[i + 16] = static_cast<float>((q_byte >> 4)   | (h1 << 4));
                }

                for (size_t vi = 0; vi < 32; vi += 8) {
                    __m256 vw = _mm256_fmsub_ps(_mm256_load_ps(w_sb + vi), v_scale, v_min);
                    __m256 vx = _mm256_loadu_ps(x_sb + vi);
                    v_acc = _mm256_fmadd_ps(vw, vx, v_acc);
                }

                alignas(32) float arr[8];
                _mm256_store_ps(arr, v_acc);
                total_sum += arr[0] + arr[1] + arr[2] + arr[3] + arr[4] + arr[5] + arr[6] + arr[7];
#elif defined(NEXUS_HAVE_ARM_NEON)
                float32x4_t v_acc = vdupq_n_f32(0.0f);
                float32x4_t v_scale = vdupq_n_f32(scale);
                float32x4_t v_min = vdupq_n_f32(min_val);

                float w_sb[32];
                for (size_t i = 0; i < 16; ++i) {
                    uint8_t q_byte = qs[qs_offset + i];
                    uint8_t h_byte = qh[qh_offset + (i / 4)];
                    uint8_t h_shift_l = (i % 4) * 2;
                    uint8_t h_shift_h = h_shift_l + 1;

                    uint8_t h0 = (h_byte >> h_shift_l) & 1;
                    uint8_t h1 = (h_byte >> h_shift_h) & 1;

                    w_sb[i]      = static_cast<float>((q_byte & 0x0F) | (h0 << 4));
                    w_sb[i + 16] = static_cast<float>((q_byte >> 4)   | (h1 << 4));
                }

                for (size_t vi = 0; vi < 32; vi += 4) {
                    float32x4_t vw = vmlsq_f32(v_min, vld1q_f32(w_sb + vi), v_scale); // wait, want w*scale - min
                    vw = vsubq_f32(vmulq_f32(vld1q_f32(w_sb + vi), v_scale), v_min);
                    float32x4_t vx = vld1q_f32(x_sb + vi);
                    v_acc = vfmaq_f32(v_acc, vw, vx);
                }
                total_sum += vaddvq_f32(v_acc);
#else
                for (size_t i = 0; i < 16; ++i) {
                    uint8_t q_byte = qs[qs_offset + i];
                    uint8_t h_byte = qh[qh_offset + (i / 4)];
                    uint8_t h_shift_l = (i % 4) * 2;
                    uint8_t h_shift_h = h_shift_l + 1;

                    uint8_t h0 = (h_byte >> h_shift_l) & 1;
                    uint8_t h1 = (h_byte >> h_shift_h) & 1;

                    float w0 = static_cast<float>((q_byte & 0x0F) | (h0 << 4)) * scale - min_val;
                    float w1 = static_cast<float>((q_byte >> 4)   | (h1 << 4)) * scale - min_val;

                    total_sum += w0 * x_sb[i];
                    total_sum += w1 * x_sb[i + 16];
                }
#endif
            }
        }
        y[r] = total_sum;
    }
}

void QuantizationKernel::fused_gemv_q6_k(const BlockQ6_K* W_q6_k,
                                        const float* x,
                                        float* y,
                                        size_t rows,
                                        size_t cols) {
    const size_t blocks_per_row = cols / QK_K;

#if defined(_OPENMP)
    #pragma omp parallel for schedule(static)
#endif
    for (size_t r = 0; r < rows; ++r) {
        const BlockQ6_K* row_blocks = W_q6_k + (r * blocks_per_row);
        float total_sum = 0.0f;

        for (size_t b = 0; b < blocks_per_row; ++b) {
            const BlockQ6_K& blk = row_blocks[b];
            const float d = fp16_to_fp32(blk.d);
            const uint8_t* ql = blk.ql;
            const uint8_t* qh = blk.qh;
            const int8_t* sc = blk.scales;
            const float* x_blk = x + (b * QK_K);

            for (size_t sb = 0; sb < 16; ++sb) {
                const float scale = d * static_cast<float>(sc[sb]);
                const size_t ql_offset = sb * 8;
                const size_t qh_offset = sb * 4;
                const float* x_sb = x_blk + (sb * 16);

#if defined(NEXUS_HAVE_AVX2)
                alignas(32) float w_sb[16];
                for (size_t i = 0; i < 8; ++i) {
                    uint8_t l_byte = ql[ql_offset + i];
                    uint8_t h_byte = qh[qh_offset + (i / 2)];
                    uint8_t h_shift = (i % 2) * 4;

                    uint8_t h0 = (h_byte >> h_shift) & 0x03;
                    uint8_t h1 = (h_byte >> (h_shift + 2)) & 0x03;

                    int8_t v0 = static_cast<int8_t>((l_byte & 0x0F) | (h0 << 4)) - 32;
                    int8_t v1 = static_cast<int8_t>((l_byte >> 4)   | (h1 << 4)) - 32;

                    w_sb[i]     = static_cast<float>(v0);
                    w_sb[i + 8] = static_cast<float>(v1);
                }

                __m256 v_scale = _mm256_set1_ps(scale);
                __m256 vw0 = _mm256_mul_ps(_mm256_load_ps(w_sb), v_scale);
                __m256 vw1 = _mm256_mul_ps(_mm256_load_ps(w_sb + 8), v_scale);

                __m256 v_acc = _mm256_fmadd_ps(vw0, _mm256_loadu_ps(x_sb), _mm256_setzero_ps());
                v_acc = _mm256_fmadd_ps(vw1, _mm256_loadu_ps(x_sb + 8), v_acc);

                alignas(32) float arr[8];
                _mm256_store_ps(arr, v_acc);
                total_sum += arr[0] + arr[1] + arr[2] + arr[3] + arr[4] + arr[5] + arr[6] + arr[7];
#elif defined(NEXUS_HAVE_ARM_NEON)
                float32x4_t v_scale = vdupq_n_f32(scale);
                float32x4_t v_acc = vdupq_n_f32(0.0f);

                for (size_t i = 0; i < 8; i += 2) {
                    uint8_t l0 = ql[ql_offset + i];
                    uint8_t l1 = ql[ql_offset + i + 1];
                    uint8_t h_byte = qh[qh_offset + (i / 2)];

                    uint8_t h00 = (h_byte) & 0x03;
                    uint8_t h01 = (h_byte >> 2) & 0x03;
                    uint8_t h10 = (h_byte >> 4) & 0x03;
                    uint8_t h11 = (h_byte >> 6) & 0x03;

                    float w[4];
                    w[0] = static_cast<float>(static_cast<int8_t>((l0 & 0x0F) | (h00 << 4)) - 32);
                    w[1] = static_cast<float>(static_cast<int8_t>((l0 >> 4)   | (h01 << 4)) - 32);
                    w[2] = static_cast<float>(static_cast<int8_t>((l1 & 0x0F) | (h10 << 4)) - 32);
                    w[3] = static_cast<float>(static_cast<int8_t>((l1 >> 4)   | (h11 << 4)) - 32);

                    float32x4_t vw = vmulq_f32(vld1q_f32(w), v_scale);
                    float32x4_t vx = vld1q_f32(x_sb + i * 2);
                    v_acc = vfmaq_f32(v_acc, vw, vx);
                }
                total_sum += vaddvq_f32(v_acc);
#else
                for (size_t i = 0; i < 8; ++i) {
                    uint8_t l_byte = ql[ql_offset + i];
                    uint8_t h_byte = qh[qh_offset + (i / 2)];
                    uint8_t h_shift = (i % 2) * 4;

                    uint8_t h0 = (h_byte >> h_shift) & 0x03;
                    uint8_t h1 = (h_byte >> (h_shift + 2)) & 0x03;

                    int8_t v0 = static_cast<int8_t>((l_byte & 0x0F) | (h0 << 4)) - 32;
                    int8_t v1 = static_cast<int8_t>((l_byte >> 4)   | (h1 << 4)) - 32;

                    total_sum += (static_cast<float>(v0) * scale) * x_sb[i];
                    total_sum += (static_cast<float>(v1) * scale) * x_sb[i + 8];
                }
#endif
            }
        }
        y[r] = total_sum;
    }
}

} // namespace nexus