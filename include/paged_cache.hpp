#pragma once

#include "nexus_common.hpp"
#include <unordered_map>
#include <list>

namespace nexus {

constexpr size_t BLOCK_SIZE = 16; // 16 tokens per PagedAttention block

struct KVBlock {
    int32_t block_id = -1;
    uint32_t ref_count = 0;
    uint64_t last_accessed_timestamp = 0;
    
    // Pointers to key and value tensors
    float* key_data = nullptr;
    float* value_data = nullptr;
    size_t tokens_occupied = 0;
    std::vector<float> storage;

    KVBlock() = default;
    KVBlock(const KVBlock&) = default;
    KVBlock& operator=(const KVBlock&) = default;
    KVBlock(KVBlock&&) noexcept = default;
    KVBlock& operator=(KVBlock&&) noexcept = default;

    void ensure_allocated(size_t num_layers, size_t num_heads, size_t head_dim) {
        size_t half_stride = num_layers * num_heads * BLOCK_SIZE * head_dim;
        if (storage.size() < 2 * half_stride) {
            storage.resize(2 * half_stride, 0.0f);
        }
        key_data = storage.data();
        value_data = storage.data() + half_stride;
    }

    void reset() {
        ref_count = 0;
        tokens_occupied = 0;
        last_accessed_timestamp = 0;
    }
};

// Node in Radix Prefix Cache Tree
struct RadixTreeNode {
    std::vector<int32_t> token_subsequence;
    std::vector<int32_t> allocated_block_ids;
    std::unordered_map<int32_t, std::shared_ptr<RadixTreeNode>> children;
    std::weak_ptr<RadixTreeNode> parent;
    uint64_t last_accessed_time = 0;
    uint32_t match_count = 0;
};

class PagedKVCacheManager {
public:
    PagedKVCacheManager(size_t max_blocks, size_t num_layers, size_t num_heads, size_t head_dim);
    ~PagedKVCacheManager();

    // Allocate physical blocks for a request sequence
    std::vector<int32_t> allocate_blocks(size_t num_blocks_needed);
    
    // Increment reference count (shared prefix)
    void retain_block(int32_t block_id);

    // Release blocks safely (preventing memory leak bug root cause from arXiv:2506.09713)
    void release_blocks(const std::vector<int32_t>& block_ids);

    // Rollback sequence length in O(1) for speculative decoding rejection (Medusa/EAGLE-2)
    void rollback_sequence(const std::vector<int32_t>& block_table, size_t new_length);

    // Radix Tree Prefix Caching
    // Returns matched token count and existing block IDs
    std::pair<size_t, std::vector<int32_t>> match_prefix(const std::vector<int32_t>& prompt_tokens);

    // Insert newly generated or computed sequence into Radix Tree
    void insert_prefix(const std::vector<int32_t>& tokens, const std::vector<int32_t>& block_ids);

    // Get block pointer
    KVBlock* get_block(int32_t block_id);

    // Memory stats
    size_t get_free_blocks_count() const;
    size_t get_total_blocks_count() const { return max_blocks_; }
    double get_prefix_hit_rate() const;

    size_t get_num_layers() const { return num_layers_; }
    size_t get_num_heads() const { return num_heads_; }
    size_t get_head_dim() const { return head_dim_; }

private:
    size_t max_blocks_;
    size_t num_layers_;
    size_t num_heads_;
    size_t head_dim_;
    size_t block_byte_size_;

    std::vector<KVBlock> block_pool_;
    std::vector<int32_t> free_block_indices_;
    mutable std::mutex pool_mutex_;

    // Radix Tree root
    std::shared_ptr<RadixTreeNode> radix_root_;
    mutable std::mutex radix_mutex_;

    uint64_t total_requests_ = 0;
    uint64_t total_prefix_hits_ = 0;
};

} // namespace nexus
