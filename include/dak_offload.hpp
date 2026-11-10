#pragma once

#include "nexus_common.hpp"
#include <vector>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <memory>

#if defined(__CUDACC__) || defined(NEXUS_ENABLE_CUDA)
#include <cuda.h>
#include <cuda_runtime.h>
#else
// Portable type stubs for non-CUDA compilation environments (x86_64 CPU / ARM64 macOS CI)
typedef struct CUtensorMap_st {
    uint64_t opaque[16];
} CUtensorMap;

typedef enum cudaDataType_t {
    CUDA_R_16F = 2,
    CUDA_R_16BF = 14,
    CUDA_R_32F = 0
} cudaDataType_t;

typedef enum cudaError_enum {
    CUDA_SUCCESS = 0
} CUresult;

typedef int cudaError_t;
#define cudaSuccess 0
#define cudaErrorInvalidValue 1
#endif

namespace nexus {
namespace dak {

// ============================================================================
// DAK Hardware TMA Descriptor (arXiv:2604.26074 Section 3)
// 128-byte aligned tensor map descriptor for direct-access asynchronous DMA
// ============================================================================
struct alignas(64) TmaKVTiledDescriptor {
    CUtensorMap tensor_map;
    uint64_t host_dram_base_addr = 0;   // Registered pinned host virtual address (UVA)
    uint32_t block_id = 0;
    uint32_t tokens_per_block = 16;
    uint32_t num_kv_heads = 8;
    uint32_t head_dim = 128;
    uint32_t element_size_bytes = 2;    // FP16 / BF16
    bool is_valid = false;
};

// ============================================================================
// Multi-Head SMEM Allocation & Tiling Budget (Addresses Issue 8)
// Computes dynamic shared memory requirements scaled by num_kv_heads
// ============================================================================
struct SmemBudget {
    size_t smem_capacity_per_sm = 227 * 1024; // 228 KB max on Hopper (SM90a)
    size_t kernel_static_bytes = 16 * 1024;    // Internal warp scratch, logits, barriers
    size_t dynamic_smem_available = 0;
    size_t bytes_per_stage = 0;                // Size of multi-head K+V block stage
    uint32_t num_stages = 2;                   // Double-buffered (2) or Triple-buffered (3)

    static SmemBudget calculate(int device_id, size_t num_kv_heads, size_t head_dim, size_t block_size) {
        SmemBudget b;
#if defined(__CUDACC__) || defined(NEXUS_ENABLE_CUDA)
        cudaDeviceProp prop;
        if (cudaGetDeviceProperties(&prop, device_id) == cudaSuccess) {
            b.smem_capacity_per_sm = prop.sharedMemPerMultiprocessor;
        }
#else
        (void)device_id;
#endif
        b.dynamic_smem_available = (b.smem_capacity_per_sm > b.kernel_static_bytes)
            ? (b.smem_capacity_per_sm - b.kernel_static_bytes)
            : 64 * 1024;

        // Stage size accounts for all KV heads in Grouped-Query Attention (Issue 8)
        // 16 tokens * num_kv_heads * head_dim * sizeof(FP16) * 2 (Key + Value)
        b.bytes_per_stage = block_size * num_kv_heads * head_dim * sizeof(uint16_t) * 2;
        if (b.bytes_per_stage == 0) b.bytes_per_stage = 4096;

        b.num_stages = static_cast<uint32_t>(b.dynamic_smem_available / b.bytes_per_stage);
        if (b.num_stages > 3) b.num_stages = 3;
        if (b.num_stages < 2) b.num_stages = 2;
        return b;
    }
};

// ============================================================================
// Host-Locality-First Tile Scheduler (Addresses Issue 7)
// Temporally co-schedules blocks sharing host memory proximity to maximize
// TMA intra-cluster spatial multicast across SMs
// ============================================================================
struct LocalityScheduledBatch {
    std::vector<int32_t> ordered_block_ids;
    std::vector<uint8_t> is_tma_offload; // 1 = Direct-access TMA from Host DRAM, 0 = Resident HBM
};

inline LocalityScheduledBatch schedule_host_locality_tiles(
    const std::vector<int32_t>& selected_page_ids,
    const std::unordered_map<int32_t, uint64_t>& host_address_map,
    const std::unordered_set<int32_t>& hbm_resident_set
) {
    LocalityScheduledBatch batch;
    batch.ordered_block_ids = selected_page_ids;

    std::stable_sort(batch.ordered_block_ids.begin(), batch.ordered_block_ids.end(),
        [&](int32_t a, int32_t b) {
            bool a_in_hbm = hbm_resident_set.count(a) > 0;
            bool b_in_hbm = hbm_resident_set.count(b) > 0;
            if (a_in_hbm != b_in_hbm) return !a_in_hbm; // Schedule offloaded TMA blocks together
            auto ita = host_address_map.find(a);
            auto itb = host_address_map.find(b);
            uint64_t addra = (ita != host_address_map.end()) ? ita->second : static_cast<uint64_t>(a);
            uint64_t addrb = (itb != host_address_map.end()) ? itb->second : static_cast<uint64_t>(b);
            return addra < addrb;
        });

    batch.is_tma_offload.resize(batch.ordered_block_ids.size());
    for (size_t i = 0; i < batch.ordered_block_ids.size(); ++i) {
        batch.is_tma_offload[i] = (hbm_resident_set.count(batch.ordered_block_ids[i]) == 0) ? 1 : 0;
    }
    return batch;
}

// Function prototypes implemented in dak_offload.cu
cudaError_t create_tma_1d_kv_descriptor(
    TmaKVTiledDescriptor* out_desc,
    void* host_dram_pinned_ptr,
    size_t total_elements,
    size_t elements_per_block,
    cudaDataType_t data_type
);

void dak_paged_attention_tma_multicast_launch(
    const float* Q,
    const int32_t* block_table,
    size_t num_blocks,
    size_t num_heads,
    size_t num_kv_heads,
    size_t head_dim,
    float sm_scale,
    const TmaKVTiledDescriptor* tma_desc,
    const float* hbm_kv_pool,
    const uint8_t* is_tma_offloaded_mask,
    float* output
);

} // namespace dak
} // namespace nexus
