#include "quantization.hpp"
#include <cmath>
#include <cstring>

namespace nexus {

// Helper: Convert FP16 uint16_t to FP32 float
static inline float fp16_to_fp32(uint16_t h) {
    uint32_t w = (h & 0x7fff) << 13;
    uint32_t sign = (h & 0x8000) << 16;
    uint32_t exp = (h & 0x7c00) >> 10;
    if (exp == 0x1f) {
        return 0.0f; // Inf/NaN simplified
    }
    float out;
    uint32_t val = sign | (w + (0x70 << 23));
    std::memcpy(&out, &val, 4);
    return out;
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

} // namespace nexus