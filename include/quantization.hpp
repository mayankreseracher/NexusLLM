#pragma once

#include "nexus_common.hpp"

namespace nexus {

// Quantization block for GGUF Q4_0: 32 values per block
struct BlockQ4_0 {
    uint16_t d;        // FP16 scale delta
    uint8_t qs[16];    // 32 4-bit nibbles
};

// Quantization block for INT8: 32 values per block
struct BlockINT8 {
    float scale;
    int8_t qs[32];
};

// AWQ / GPTQ packed group
struct BlockAWQ_G128 {
    float scale;
    float zero;
    uint32_t packed_weights[16]; // 128 4-bit weights
};

class QuantizationKernel {
public:
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
};

} // namespace nexus
