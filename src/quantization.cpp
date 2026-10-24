#include "quantization.hpp"
#include <cmath>
#include <cstring>
#include <immintrin.h>

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

        for (size_t i = 0; i < 16; ++i) {
            uint8_t val = qs[i];
            int8_t v0 = (val & 0x0F) - 8;
            int8_t v1 = (val >> 4) - 8;
            out[i] = v0 * d;
            out[i + 16] = v1 * d;
        }
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
        float sum = 0.0f;
        const uint32_t* row_packed = W_packed + (r * packed_cols);
        size_t g_base = r * (cols / group_size);

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
    }
}

void QuantizationKernel::fused_gemv_fp8(const uint8_t* W_fp8, 
                                      const float* x, 
                                      float* y, 
                                      size_t rows, 
                                      size_t cols, 
                                      float weight_scale) {
    for (size_t r = 0; r < rows; ++r) {
        float sum = 0.0f;
        const uint8_t* row_weights = W_fp8 + (r * cols);
        for (size_t c = 0; c < cols; ++c) {
            float w = fp8_e4m3_to_fp32(row_weights[c]) * weight_scale;
            sum += w * x[c];
        }
        y[r] = sum;
    }
}

} // namespace nexus
