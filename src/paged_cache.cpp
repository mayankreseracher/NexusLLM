#include "paged_cache.hpp"
#include <iostream>
#include <chrono>

namespace nexus {

PagedKVCacheManager::PagedKVCacheManager(size_t max_blocks, size_t num_layers, size_t num_heads, size_t head_dim)
    : max_blocks_(max_blocks), num_layers_(num_layers), num_heads_(num_heads), head_dim_(head_dim) {

    block_byte_size_ = num_layers_ * num_heads_ * BLOCK_SIZE * head_dim_ * sizeof(float);
    block_pool_.resize(max_blocks_);
    free_block_indices_.reserve(max_blocks_);

    for (size_t i = 0; i < max_blocks_; ++i) {
        block_pool_[i].block_id = static_cast<int32_t>(i);
        block_pool_[i].ref_count = 0;
        block_pool_[i].tokens_occupied = 0;
        block_pool_[i].key_data = nullptr;
        block_pool_[i].value_data = nullptr;
        free_block_indices_.push_back(static_cast<int32_t>(i));
    }

    radix_root_ = std::make_shared<RadixTreeNode>();
}

PagedKVCacheManager::~PagedKVCacheManager() {
    std::lock_guard<std::mutex> lock(radix_mutex_);
    for (auto& pair : radix_root_->children) {
        if (pair.second) {
            release_blocks(pair.second->allocated_block_ids);
        }
    }
    radix_root_->children.clear();
}

std::vector<int32_t> PagedKVCacheManager::allocate_blocks(size_t num_blocks_needed) {
    std::lock_guard<std::mutex> lock(pool_mutex_);
    if (free_block_indices_.size() < num_blocks_needed) {
        return {}; // Allocation failure - trigger eviction or wait
    }

    std::vector<int32_t> allocated;
    allocated.reserve(num_blocks_needed);

    for (size_t i = 0; i < num_blocks_needed; ++i) {
        int32_t blk_id = free_block_indices_.back();
        free_block_indices_.pop_back();

        KVBlock& blk = block_pool_[blk_id];
        blk.ref_count = 1;
        blk.tokens_occupied = 0;
        blk.ensure_allocated(num_layers_, num_heads_, head_dim_);
        blk.last_accessed_timestamp = std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::high_resolution_clock::now().time_since_epoch()).count();

        allocated.push_back(blk_id);
    }

    return allocated;
}

void PagedKVCacheManager::retain_block(int32_t block_id) {
    if (block_id >= 0 && static_cast<size_t>(block_id) < max_blocks_) {
        block_pool_[block_id].ref_count++;
    }
}

void PagedKVCacheManager::release_blocks(const std::vector<int32_t>& block_ids) {
    std::lock_guard<std::mutex> lock(pool_mutex_);
    for (int32_t blk_id : block_ids) {
        // Addressing arXiv:2506.09713 memory safety: strict boundary and double-free verification
        if (blk_id < 0 || static_cast<size_t>(blk_id) >= max_blocks_) {
            continue;
        }

        KVBlock& blk = block_pool_[blk_id];
        uint32_t current_rc = blk.ref_count;
        if (current_rc > 0) {
            uint32_t new_rc = --blk.ref_count;
            if (new_rc == 0) {
                blk.reset();
                free_block_indices_.push_back(blk_id);
            }
        }
    }
}

void PagedKVCacheManager::rollback_sequence(const std::vector<int32_t>& block_table, size_t new_length) {
    std::lock_guard<std::mutex> lock(pool_mutex_);
    size_t tokens_accounted = 0;
    for (size_t blk_idx = 0; blk_idx < block_table.size(); ++blk_idx) {
        int32_t blk_id = block_table[blk_idx];
        if (blk_id < 0 || static_cast<size_t>(blk_id) >= max_blocks_) continue;
        KVBlock& blk = block_pool_[blk_id];

        if (tokens_accounted >= new_length) {
            blk.tokens_occupied = 0;
        } else if (tokens_accounted + BLOCK_SIZE <= new_length) {
            blk.tokens_occupied = BLOCK_SIZE;
            tokens_accounted += BLOCK_SIZE;
        } else {
            blk.tokens_occupied = new_length - tokens_accounted;
            tokens_accounted = new_length;
        }
    }
}

std::pair<size_t, std::vector<int32_t>> PagedKVCacheManager::match_prefix(const std::vector<int32_t>& prompt_tokens) {
    std::lock_guard<std::mutex> lock(radix_mutex_);
    total_requests_++;

    if (prompt_tokens.empty()) return {0, {}};

    std::shared_ptr<RadixTreeNode> curr = radix_root_;
    size_t matched_tokens = 0;
    std::vector<int32_t> matched_blocks;

    size_t token_idx = 0;
    while (token_idx < prompt_tokens.size()) {
        int32_t next_tok = prompt_tokens[token_idx];
        auto it = curr->children.find(next_tok);
        if (it == curr->children.end()) break;

        std::shared_ptr<RadixTreeNode> child = it->second;
        // Verify sequence match
        bool mismatch = false;
        for (size_t k = 0; k < child->token_subsequence.size(); ++k) {
            if (token_idx + k >= prompt_tokens.size() || 
                prompt_tokens[token_idx + k] != child->token_subsequence[k]) {
                mismatch = true;
                break;
            }
        }

        if (mismatch) break;

        // Match found! Retain blocks
        token_idx += child->token_subsequence.size();
        matched_tokens += child->token_subsequence.size();
        for (int32_t b : child->allocated_block_ids) {
            retain_block(b);
            matched_blocks.push_back(b);
        }
        child->match_count++;
        curr = child;
    }

    if (matched_tokens > 0) {
        total_prefix_hits_++;
    }

    return {matched_tokens, matched_blocks};
}

void PagedKVCacheManager::insert_prefix(const std::vector<int32_t>& tokens, const std::vector<int32_t>& block_ids) {
    if (tokens.empty() || block_ids.empty()) return;

    std::lock_guard<std::mutex> lock(radix_mutex_);
    auto node = std::make_shared<RadixTreeNode>();
    node->token_subsequence = tokens;
    node->allocated_block_ids = block_ids;
    node->last_accessed_time = std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::high_resolution_clock::now().time_since_epoch()).count();

    // Retain cached blocks so request completion doesn't prematurely release them while held in the prefix tree
    for (int32_t b : block_ids) {
        retain_block(b);
    }

    auto it = radix_root_->children.find(tokens[0]);
    if (it != radix_root_->children.end()) {
        release_blocks(it->second->allocated_block_ids);
    }

    // Insert under root child indexed by first token
    radix_root_->children[tokens[0]] = node;
}

KVBlock* PagedKVCacheManager::get_block(int32_t block_id) {
    if (block_id >= 0 && static_cast<size_t>(block_id) < max_blocks_) {
        return &block_pool_[block_id];
    }
    return nullptr;
}

KVBlock* PagedKVCacheManager::read_block(int32_t block_id) {
    return get_block(block_id);
}

KVBlock* PagedKVCacheManager::write_block(int32_t block_id) {
    return get_block(block_id);
}

size_t PagedKVCacheManager::get_free_blocks_count() const {
    std::lock_guard<std::mutex> lock(pool_mutex_);
    return free_block_indices_.size();
}

double PagedKVCacheManager::get_prefix_hit_rate() const {
    if (total_requests_ == 0) return 0.0;
    return static_cast<double>(total_prefix_hits_) / static_cast<double>(total_requests_);
}

} // namespace nexus
