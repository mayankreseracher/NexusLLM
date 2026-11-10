#pragma once

#include "nexus_common.hpp"

namespace nexus {

// Quantization block for GGUF Q4_0: 32 values per block
struct BlockQ4_0 {
    uint16_t d;        // FP16 scale delta
    uint8_t qs[16];    // 32 4-bit nibbles
};

// Quantization block for GGUF Q8_0: 32 values per block
struct BlockQ8_0 {
    uint16_t d;        // FP16 scale delta
    int8_t qs[32];     // 32 signed 8-bit integers
};

// Quantization block for INT8: 32 values per block
struct BlockINT8 {
    float scale;
    int8_t qs[32];
};

// 256 values per block for k-quants
constexpr size_t QK_K = 256;

// Quantization block for GGUF Q5_K: 256 values per block
// 2-byte scale delta (FP16), 2-byte min delta (FP16), 12-byte sub-block scales/mins, 32-byte high bits (qh), 128-byte low nibbles (qs)
// Total size: 176 bytes
struct BlockQ5_K {
    uint16_t d;           // super-block scale (fp16)
    uint16_t dmin;        // super-block min (fp16)
    uint8_t scales[12];   // 16 6-bit scales and 16 6-bit mins packed into 12 bytes
    uint8_t qh[32];       // 256 1-bit high quant values
    uint8_t qs[QK_K / 2]; // 256 4-bit low quant values (128 bytes)
};

// Quantization block for GGUF Q6_K: 256 values per block
// 128-byte low 4 bits (ql), 64-byte high 2 bits (qh), 16-byte signed scales (scales), 2-byte scale delta (d)
// Total size: 210 bytes
struct BlockQ6_K {
    uint8_t ql[QK_K / 2];   // 256 4-bit low quant values
    uint8_t qh[QK_K / 4];   // 256 2-bit high quant values
    int8_t scales[QK_K / 16];// 16 8-bit sub-block scales
    uint16_t d;             // super-block scale (fp16)
};

// AWQ / GPTQ packed group
struct BlockAWQ_G128 {
    float scale;
    float zero;
    uint32_t packed_weights[16]; // 128 4-bit weights
};

class QuantizationKernel {
public:
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

    // Dequantize INT4 AWQ packed array into float32
    static void dequantize_int4_awq(const uint32_t* packed, 
                                   const float* scales, 
                                   const float* zeros, 
                                   float* output, 
                                   size_t count, 
                                   size_t group_size = 128);

    // Dequantize GGUF Q4_0 into float32
    static void dequantize_gguf_q4_0(const BlockQ4_0* blocks, 
                                    float* output, 
                                    size_t num_blocks);

    // Dequantize FP8 (E4M3) into float32
    static void dequantize_fp8_e4m3(const uint8_t* input, 
                                   float* output, 
                                   size_t count, 
                                   float scale = 1.0f);

    // Dequantize INT8 symmetric into float32
    static void dequantize_int8(const int8_t* input, 
                               float* output, 
                               size_t count, 
                               float scale);

    // High-performance Fused Quantized GEMV (Matrix-Vector multiplication)
    // Computes y = W * x directly from quantized weights W
    static void fused_gemv_int4(const uint32_t* W_packed, 
                               const float* x, 
                               const float* scales, 
                               const float* zeros, 
                               float* y, 
                               size_t rows, 
                               size_t cols, 
                               size_t group_size = 128);

    static void fused_gemv_fp8(const uint8_t* W_fp8, 
                              const float* x, 
                              float* y, 
                              size_t rows, 
                              size_t cols, 
                              float weight_scale = 1.0f);

    // GGUF Q4_0, Q8_0, and FP16 Fused GEMV kernels
    static void fused_gemv_q4_0(const BlockQ4_0* W_q4,
                                const float* x,
                                float* y,
                                size_t rows,
                                size_t cols);

    static void fused_gemv_q8_0(const BlockQ8_0* W_q8,
                                const float* x,
                                float* y,
                                size_t rows,
                                size_t cols);

    static void fused_gemv_fp16(const uint16_t* W_fp16,
                                const float* x,
                                float* y,
                                size_t rows,
                                size_t cols);

    // GGUF Q5_K and Q6_K Fused GEMV and Dequantization kernels
    static void dequantize_gguf_q5_k(const BlockQ5_K* blocks,
                                     float* output,
                                     size_t num_blocks);

    static void dequantize_gguf_q6_k(const BlockQ6_K* blocks,
                                     float* output,
                                     size_t num_blocks);

    static void fused_gemv_q5_k(const BlockQ5_K* W_q5_k,
                                const float* x,
                                float* y,
                                size_t rows,
                                size_t cols);

    static void fused_gemv_q6_k(const BlockQ6_K* W_q6_k,
                                const float* x,
                                float* y,
                                size_t rows,
                                size_t cols);
};

} // namespace nexus
