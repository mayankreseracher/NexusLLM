#pragma once

#include "nexus_common.hpp"
#include "paged_cache.hpp"
#include <vector>
#include <string>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <deque>

namespace nexus {

// Memory Partition Scope
enum class PartitionScope : uint32_t {
    PREFILL_DEDICATED = 0,
    DECODE_DEDICATED = 1,
    SHARED_TENANT = 2,
    ISOLATED_TENANT = 3
};

// Head-Wise Page abstraction: Decouples physical memory allocated for individual attention heads
struct HeadWisePage {
    int32_t physical_block_id = -1;
    size_t head_index = 0;
    size_t layer_index = 0;
    size_t tokens_occupied = 0;
    float* key_head_ptr = nullptr;
    float* val_head_ptr = nullptr;
    float* centroid_head_ptr = nullptr;
};

// Multi-Tier Bucketed LRU Cache Ring
// Tracks allocated blocks across distinct access temperature tiers:
// - Hot Bucket: frequently referenced blocks, protected from eviction
// - Warm Bucket: active generation blocks
// - Cold/Evictable Bucket: candidate blocks for reuse or tier migration
class BucketedLRUCache {
public:
    BucketedLRUCache() = default;

    void record_access(int32_t block_id, uint32_t ref_count) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (ref_count > 1) {
            // Shared prefix: promote directly to Hot tier
            promote_to(hot_bucket_, block_id);
        } else {
            // Generation decode token: promote to Warm tier
            promote_to(warm_bucket_, block_id);
        }
    }

    void mark_evictable(int32_t block_id) {
        std::lock_guard<std::mutex> lock(mutex_);
        remove_from_all(block_id);
        cold_bucket_.push_back(block_id);
    }

    int32_t evict_next() {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!cold_bucket_.empty()) {
            int32_t b = cold_bucket_.front();
            cold_bucket_.pop_front();
            return b;
        }
        if (!warm_bucket_.empty()) {
            int32_t b = warm_bucket_.front();
            warm_bucket_.pop_front();
            return b;
        }
        return -1; // No unpinned blocks evictable
    }

    size_t get_hot_count() const { std::lock_guard<std::mutex> lock(mutex_); return hot_bucket_.size(); }
    size_t get_warm_count() const { std::lock_guard<std::mutex> lock(mutex_); return warm_bucket_.size(); }
    size_t get_cold_count() const { std::lock_guard<std::mutex> lock(mutex_); return cold_bucket_.size(); }

private:
    void remove_from_all(int32_t b) {
        auto rm = [b](std::deque<int32_t>& d) {
            for (auto it = d.begin(); it != d.end(); ++it) {
                if (*it == b) { d.erase(it); break; }
            }
        };
        rm(hot_bucket_);
        rm(warm_bucket_);
        rm(cold_bucket_);
    }

    void promote_to(std::deque<int32_t>& target, int32_t b) {
        remove_from_all(b);
        target.push_back(b);
        // Cap hot and warm buckets to maintain temporal bounds
        if (target.size() > 2048) {
            int32_t demoted = target.front();
            target.pop_front();
            cold_bucket_.push_back(demoted);
        }
    }

    mutable std::mutex mutex_;
    std::deque<int32_t> hot_bucket_;
    std::deque<int32_t> warm_bucket_;
    std::deque<int32_t> cold_bucket_;
};

// Partition Abstraction: Manages isolated or shared sub-pools of KV blocks
class MemoryPartition {
public:
    MemoryPartition(const std::string& partition_id, PartitionScope scope, size_t block_budget, std::shared_ptr<PagedKVCacheManager> kv_mgr)
        : partition_id_(partition_id), scope_(scope), block_budget_(block_budget), kv_mgr_(kv_mgr) {}

    const std::string& get_partition_id() const { return partition_id_; }
    PartitionScope get_scope() const { return scope_; }
    size_t get_budget() const { return block_budget_; }
    size_t get_allocated_count() const { return allocated_blocks_.size(); }

    std::vector<int32_t> allocate_blocks(size_t count) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (allocated_blocks_.size() + count > block_budget_) {
            // Trigger bucketed LRU eviction if budget exceeded
            size_t needed = (allocated_blocks_.size() + count) - block_budget_;
            for (size_t i = 0; i < needed; ++i) {
                int32_t b = lru_.evict_next();
                if (b >= 0 && kv_mgr_) {
                    kv_mgr_->release_blocks({b});
                    for (auto it = allocated_blocks_.begin(); it != allocated_blocks_.end(); ++it) {
                        if (*it == b) { allocated_blocks_.erase(it); break; }
                    }
                }
            }
        }

        if (!kv_mgr_) return {};
        auto blocks = kv_mgr_->allocate_blocks(count);
        for (int32_t b : blocks) {
            allocated_blocks_.push_back(b);
            lru_.record_access(b, 1);
        }
        return blocks;
    }

    void release_blocks(const std::vector<int32_t>& block_ids) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (kv_mgr_) {
            kv_mgr_->release_blocks(block_ids);
        }
        for (int32_t b : block_ids) {
            for (auto it = allocated_blocks_.begin(); it != allocated_blocks_.end(); ++it) {
                if (*it == b) { allocated_blocks_.erase(it); break; }
            }
            lru_.mark_evictable(b);
        }
    }

    // Resolves head-wise page descriptors for granular head allocation
    HeadWisePage get_head_page(int32_t block_id, size_t layer_idx, size_t head_idx) {
        HeadWisePage hp;
        hp.physical_block_id = block_id;
        hp.layer_index = layer_idx;
        hp.head_index = head_idx;
        if (!kv_mgr_) return hp;

        KVBlock* blk = kv_mgr_->get_block(block_id);
        if (!blk || !blk->key_data || !blk->value_data) return hp;

        size_t n_heads = kv_mgr_->get_num_heads();
        size_t h_dim = kv_mgr_->get_head_dim();
        size_t stride = n_heads * BLOCK_SIZE * h_dim;

        hp.tokens_occupied = blk->tokens_occupied;
        hp.key_head_ptr = blk->key_data + (layer_idx * stride) + (head_idx * BLOCK_SIZE * h_dim);
        hp.val_head_ptr = blk->value_data + (layer_idx * stride) + (head_idx * BLOCK_SIZE * h_dim);
        if (blk->k_centroid) {
            hp.centroid_head_ptr = blk->k_centroid + (layer_idx * n_heads * h_dim) + (head_idx * h_dim);
        }
        return hp;
    }

    BucketedLRUCache& get_lru() { return lru_; }

private:
    std::string partition_id_;
    PartitionScope scope_;
    size_t block_budget_;
    std::shared_ptr<PagedKVCacheManager> kv_mgr_;
    std::vector<int32_t> allocated_blocks_;
    BucketedLRUCache lru_;
    mutable std::mutex mutex_;
};

// Global Partition Manager
class PartitionManager {
public:
    static PartitionManager& instance() {
        static PartitionManager mgr;
        return mgr;
    }

    std::shared_ptr<MemoryPartition> create_partition(const std::string& id, PartitionScope scope, size_t budget, std::shared_ptr<PagedKVCacheManager> kv_mgr) {
        std::lock_guard<std::mutex> lock(mutex_);
        auto part = std::make_shared<MemoryPartition>(id, scope, budget, kv_mgr);
        partitions_[id] = part;
        return part;
    }

    std::shared_ptr<MemoryPartition> get_partition(const std::string& id) {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = partitions_.find(id);
        if (it != partitions_.end()) return it->second;
        return nullptr;
    }

    void remove_partition(const std::string& id) {
        std::lock_guard<std::mutex> lock(mutex_);
        partitions_.erase(id);
    }

    void clear() {
        std::lock_guard<std::mutex> lock(mutex_);
        partitions_.clear();
    }

private:
    PartitionManager() = default;
    mutable std::mutex mutex_;
    std::unordered_map<std::string, std::shared_ptr<MemoryPartition>> partitions_;
};

} // namespace nexus
