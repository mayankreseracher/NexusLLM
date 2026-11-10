#pragma once

#include "nexus_common.hpp"
#include "paged_cache.hpp"
#include <vector>
#include <string>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <unordered_set>
#include <array>
#include <deque>
#include <algorithm>
#include <cmath>

namespace nexus {

// ============================================================================
// Issue 1: Spin Sparsity Partition & Sparse Indexer
// Maps coarse-grained sparse block selection -> physical page IDs for a query
// ============================================================================
struct SparsityPartition {
    uint32_t query_token_id = 0;
    std::vector<int32_t> selected_page_ids;  // Top-K physical pages selected for this query
    float score_threshold = 0.0f;           // Adaptive threshold based on query attention distribution
    uint32_t coarse_block_id = 0;           // Coarse centroid index identifying the cluster
};

// Abstract interface for sparse KV indexing algorithms (Spin, H2O, StreamingLLM, etc.)
class ISparseIndexer {
public:
    virtual ~ISparseIndexer() = default;

    virtual SparsityPartition index_query(
        uint32_t query_token_id,
        const float* query_vec,
        size_t head_dim,
        const std::vector<int32_t>& block_table,
        PagedKVCacheManager& kv_mgr,
        size_t layer_idx,
        size_t kv_head_idx,
        float retention_ratio,
        size_t recent_window_blocks
    ) = 0;
};

// Spin (arXiv:2604.26837) Centroid-Based Sparse Indexer implementation
class SpinCentroidIndexer : public ISparseIndexer {
public:
    SpinCentroidIndexer() = default;

    SparsityPartition index_query(
        uint32_t query_token_id,
        const float* query_vec,
        size_t head_dim,
        const std::vector<int32_t>& block_table,
        PagedKVCacheManager& kv_mgr,
        size_t layer_idx,
        size_t kv_head_idx,
        float retention_ratio,
        size_t recent_window_blocks
    ) override {
        SparsityPartition sp;
        sp.query_token_id = query_token_id;
        if (block_table.empty() || head_dim == 0) return sp;

        size_t total_blocks = block_table.size();
        size_t num_recent = std::min(recent_window_blocks, total_blocks);
        size_t historical_blocks = total_blocks - num_recent;
        size_t top_k = std::max<size_t>(1, static_cast<size_t>(historical_blocks * retention_ratio));

        size_t c_stride = kv_mgr.get_num_heads() * head_dim;
        std::vector<std::pair<float, int32_t>> scores;
        scores.reserve(historical_blocks);

        for (size_t b = 0; b < historical_blocks; ++b) {
            int32_t blk_id = block_table[b];
            KVBlock* blk = kv_mgr.get_block(blk_id);
            if (!blk || !blk->k_centroid || !blk->centroid_valid) {
                scores.push_back({0.0f, blk_id});
                continue;
            }
            const float* c_vec = blk->k_centroid + (layer_idx * c_stride) + (kv_head_idx * head_dim);

            float dot = 0.0f;
            size_t d = 0;
#if defined(NEXUS_HAVE_AVX2)
            __m256 vdot = _mm256_setzero_ps();
            for (; d + 8 <= head_dim; d += 8) {
                vdot = _mm256_fmadd_ps(_mm256_loadu_ps(query_vec + d), _mm256_loadu_ps(c_vec + d), vdot);
            }
            alignas(32) float arr[8];
            _mm256_store_ps(arr, vdot);
            dot = arr[0] + arr[1] + arr[2] + arr[3] + arr[4] + arr[5] + arr[6] + arr[7];
#endif
            for (; d < head_dim; ++d) {
                dot += query_vec[d] * c_vec[d];
            }
            // Normalize by 1/sqrt(head_dim) for consistent cross-model threshold behavior
            float norm_factor = 1.0f / std::sqrt(static_cast<float>(head_dim));
            scores.push_back({dot * norm_factor, blk_id});
        }

        if (top_k < scores.size()) {
            std::partial_sort(scores.begin(), scores.begin() + top_k, scores.end(),
                              [](const auto& a, const auto& b) { return a.first > b.first; });
            sp.score_threshold = scores[top_k - 1].first;
        } else if (!scores.empty()) {
            sp.score_threshold = scores.back().first;
        }

        sp.selected_page_ids.reserve(top_k + num_recent);
        size_t take_hist = std::min(top_k, scores.size());
        for (size_t i = 0; i < take_hist; ++i) {
            sp.selected_page_ids.push_back(scores[i].second);
        }
        for (size_t i = historical_blocks; i < total_blocks; ++i) {
            sp.selected_page_ids.push_back(block_table[i]);
        }

        if (!sp.selected_page_ids.empty()) {
            sp.coarse_block_id = static_cast<uint32_t>(sp.selected_page_ids[0]);
        }
        return sp;
    }
};

// ============================================================================
// Issue 2: Spin 64-Bucket Recency Ring LRU
// True temporal decay with O(1) access updates, pinning support, and O(64) bounded victim scan
// ============================================================================
class BucketedLRUCache {
public:
    static constexpr uint32_t NUM_BUCKETS = 64;

    BucketedLRUCache() = default;

    void record_access(int32_t block_id, uint32_t ref_count = 1) {
        std::lock_guard<std::mutex> lock(mutex_);
        current_tick_++;
        // If block has multiple references (shared system prompt prefix), boost into warm/hot sector
        uint32_t target_bucket = (ref_count > 1)
            ? ((current_tick_ + NUM_BUCKETS / 2) % NUM_BUCKETS)
            : (current_tick_ % NUM_BUCKETS);

        // Remove from previous bucket if already tracked (O(1) amortized via bucket index map)
        auto it = block_bucket_.find(block_id);
        if (it != block_bucket_.end()) {
            uint32_t old_bucket = it->second;
            auto& old_list = buckets_[old_bucket];
            for (auto dit = old_list.begin(); dit != old_list.end(); ++dit) {
                if (*dit == block_id) {
                    old_list.erase(dit);
                    break;
                }
            }
        }

        buckets_[target_bucket].push_back(block_id);
        block_bucket_[block_id] = target_bucket;
    }

    void mark_evictable(int32_t block_id) {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = block_bucket_.find(block_id);
        if (it != block_bucket_.end()) {
            uint32_t old_bucket = it->second;
            auto& old_list = buckets_[old_bucket];
            for (auto dit = old_list.begin(); dit != old_list.end(); ++dit) {
                if (*dit == block_id) {
                    old_list.erase(dit);
                    break;
                }
            }
        }
        // Place in oldest bucket for priority eviction (unpinning if was pinned)
        pinned_.erase(block_id);
        uint32_t oldest_bucket = (current_tick_ + 1) % NUM_BUCKETS;
        buckets_[oldest_bucket].push_front(block_id);
        block_bucket_[block_id] = oldest_bucket;
    }

    // Fully delete block from bucket and map to prevent double-free
    void remove(int32_t block_id) {
        std::lock_guard<std::mutex> lock(mutex_);
        pinned_.erase(block_id);
        auto it = block_bucket_.find(block_id);
        if (it != block_bucket_.end()) {
            uint32_t old_bucket = it->second;
            auto& old_list = buckets_[old_bucket];
            for (auto dit = old_list.begin(); dit != old_list.end(); ++dit) {
                if (*dit == block_id) {
                    old_list.erase(dit);
                    break;
                }
            }
            block_bucket_.erase(it);
        }
    }

    // Pinning support for mandatory pages
    void pin(int32_t block_id) {
        std::lock_guard<std::mutex> lock(mutex_);
        pinned_.insert(block_id);
    }

    void unpin(int32_t block_id) {
        std::lock_guard<std::mutex> lock(mutex_);
        pinned_.erase(block_id);
    }

    bool is_pinned(int32_t block_id) const {
        std::lock_guard<std::mutex> lock(mutex_);
        return pinned_.find(block_id) != pinned_.end();
    }

    // Victim selection: scans NUM_BUCKETS (64 slots) from oldest to newest tick, skipping pinned blocks
    int32_t evict_next() {
        std::lock_guard<std::mutex> lock(mutex_);
        for (uint32_t i = 0; i < NUM_BUCKETS; ++i) {
            uint32_t idx = (current_tick_ + 1 + i) % NUM_BUCKETS;
            while (!buckets_[idx].empty()) {
                int32_t victim = buckets_[idx].front();
                buckets_[idx].pop_front();
                if (pinned_.count(victim)) {
                    // Skip pinned blocks - re-queue to the end of the bucket so it is preserved
                    buckets_[idx].push_back(victim);
                    break;
                }
                block_bucket_.erase(victim);
                return victim;
            }
        }
        return -1; // No unpinned blocks evictable
    }

    size_t size() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return block_bucket_.size();
    }

    bool empty() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return block_bucket_.empty();
    }

private:
    uint32_t current_tick_ = 0;
    std::array<std::deque<int32_t>, NUM_BUCKETS> buckets_;
    std::unordered_map<int32_t, uint32_t> block_bucket_;
    std::unordered_set<int32_t> pinned_;
    mutable std::mutex mutex_;
};

// ============================================================================
// Issue 3: P2 Two-Level Paged Metadata Architecture (Design Specification)
// Scales to 10M+ KV blocks without flat std::vector<int32_t> blowup.
// Documented for multi-node / enterprise context scaling.
// ============================================================================
struct MetadataSegment {
    int32_t entries[256];       // 256 physical block indices per segment
    uint8_t flags[256];         // Residency flags, dirty state, reference bits
};

struct MetadataPageTable {
    MetadataSegment* segment_dir[256] = {nullptr};  // Level 0 directory (256 segments)
    // Level 1: segments allocated on demand (256 * 256 = 65,536 page entries per table)
    ~MetadataPageTable() {
        for (size_t i = 0; i < 256; ++i) {
            delete segment_dir[i];
            segment_dir[i] = nullptr;
        }
    }
};

// ============================================================================
// Issue 4: Mandatory vs Buffering Budget Split
// Protects recent sliding window & prefix blocks from being evicted under memory pressure
// ============================================================================
struct RequestBudget {
    uint32_t mandatory_pages = 0;   // Never evicted (e.g. recent context window, system prompt)
    uint32_t buffering_pages = 0;   // Dynamically resizable by PD scheduler under HBM pressure

    uint32_t total() const { return mandatory_pages + buffering_pages; }
};

// Memory Partition Scope for Tenancy & PD Isolation
enum class PartitionScope : uint32_t {
    PREFILL_DEDICATED = 0,
    DECODE_DEDICATED = 1,
    SHARED_TENANT = 2,
    ISOLATED_TENANT = 3
};

// Head-Wise Page View: Decouples physical page offsets down to individual attention heads
struct HeadWisePage {
    int32_t physical_block_id = -1;
    size_t head_index = 0;
    size_t layer_index = 0;
    size_t tokens_occupied = 0;
    float* key_head_ptr = nullptr;
    float* val_head_ptr = nullptr;
    float* centroid_head_ptr = nullptr;
};

// Memory Partition: Manages a budgeted slice of physical KV blocks with LRU decay
class MemoryPartition {
public:
    MemoryPartition(const std::string& partition_id, PartitionScope scope, size_t block_budget, std::shared_ptr<PagedKVCacheManager> kv_mgr)
        : partition_id_(partition_id), scope_(scope), kv_mgr_(kv_mgr) {
        budget_.buffering_pages = static_cast<uint32_t>(block_budget);
        budget_.mandatory_pages = 0;
    }

    MemoryPartition(const std::string& partition_id, PartitionScope scope, RequestBudget budget, std::shared_ptr<PagedKVCacheManager> kv_mgr)
        : partition_id_(partition_id), scope_(scope), budget_(budget), kv_mgr_(kv_mgr) {}

    const std::string& get_partition_id() const { return partition_id_; }
    PartitionScope get_scope() const { return scope_; }
    size_t get_budget() const { return budget_.total(); }
    RequestBudget get_request_budget() const { std::lock_guard<std::mutex> lock(mutex_); return budget_; }
    size_t get_allocated_count() const { std::lock_guard<std::mutex> lock(mutex_); return allocated_blocks_.size(); }

    // Dynamic buffering budget resize (called by PD scheduler under global memory pressure)
    void resize_buffering(uint32_t new_buffering_count) {
        std::vector<int32_t> to_evict;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (new_buffering_count < budget_.buffering_pages) {
                uint32_t diff = budget_.buffering_pages - new_buffering_count;
                budget_.buffering_pages = new_buffering_count;
                for (uint32_t i = 0; i < diff; ++i) {
                    int32_t victim = lru_.evict_next();
                    if (victim >= 0) {
                        allocated_blocks_.erase(victim);
                        auto it = std::find(allocation_order_.begin(), allocation_order_.end(), victim);
                        if (it != allocation_order_.end()) allocation_order_.erase(it);
                        to_evict.push_back(victim);
                    }
                }
            } else {
                budget_.buffering_pages = new_buffering_count;
            }
        }
        if (!to_evict.empty() && kv_mgr_) {
            kv_mgr_->release_blocks(to_evict);
        }
    }

    void set_mandatory_budget(uint32_t mandatory_count) {
        std::lock_guard<std::mutex> lock(mutex_);
        budget_.mandatory_pages = mandatory_count;
        update_pins_locked();
    }

    // Allocate physical blocks with eviction performed outside the partition mutex to avoid serialization
    std::vector<int32_t> allocate_blocks(size_t count) {
        std::vector<int32_t> to_evict;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            uint32_t total_budget = budget_.total();
            if (allocated_blocks_.size() + count > total_budget) {
                size_t needed = (allocated_blocks_.size() + count) - total_budget;
                for (size_t i = 0; i < needed; ++i) {
                    int32_t victim = lru_.evict_next();
                    if (victim >= 0) {
                        allocated_blocks_.erase(victim);
                        auto it = std::find(allocation_order_.begin(), allocation_order_.end(), victim);
                        if (it != allocation_order_.end()) allocation_order_.erase(it);
                        to_evict.push_back(victim);
                    }
                }
            }
        }

        // Evict outside lock
        if (!to_evict.empty() && kv_mgr_) {
            kv_mgr_->release_blocks(to_evict);
        }

        if (!kv_mgr_) return {};
        auto blocks = kv_mgr_->allocate_blocks(count);

        {
            std::lock_guard<std::mutex> lock(mutex_);
            for (int32_t b : blocks) {
                allocated_blocks_.insert(b);
                allocation_order_.push_back(b);
                lru_.record_access(b, 1);
            }
            // Incremental pinning: newly allocated blocks are the most recent;
            // pin them if mandatory_pages > 0 and unpin oldest mandatory if window shifted
            if (budget_.mandatory_pages > 0) {
                for (int32_t b : blocks) {
                    lru_.pin(b);
                }
                // If allocation order now exceeds mandatory_pages, unpin anything pushed out of mandatory zone
                if (allocation_order_.size() > budget_.mandatory_pages) {
                    size_t unpin_count = allocation_order_.size() - budget_.mandatory_pages;
                    for (size_t i = 0; i < unpin_count; ++i) {
                        lru_.unpin(allocation_order_[i]);
                    }
                }
            }
        }
        return blocks;
    }

    // O(1) release using std::unordered_set membership, full LRU removal, and O(N+M) allocation_order_ pruning
    void release_blocks(const std::vector<int32_t>& block_ids) {
        std::vector<int32_t> to_release;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            std::unordered_set<int32_t> release_set(block_ids.begin(), block_ids.end());
            for (int32_t b : block_ids) {
                if (allocated_blocks_.erase(b)) {
                    lru_.remove(b); // Fully remove from LRU ring
                    lru_.unpin(b);  // Unpin immediately
                    to_release.push_back(b);
                }
            }
            // Hash-set lookup: O(N + M) instead of O(N * M)
            for (auto it = allocation_order_.begin(); it != allocation_order_.end(); ) {
                if (release_set.count(*it)) {
                    it = allocation_order_.erase(it);
                } else {
                    ++it;
                }
            }
            // Re-pin the most recent mandatory_pages after removal
            size_t to_pin = std::min<size_t>(budget_.mandatory_pages, allocation_order_.size());
            for (size_t i = 0; i < to_pin; ++i) {
                size_t idx = allocation_order_.size() - 1 - i;
                lru_.pin(allocation_order_[idx]);
            }
        }

        // Release in underlying manager outside partition lock
        if (!to_release.empty() && kv_mgr_) {
            kv_mgr_->release_blocks(to_release);
        }
    }

    // Resolves head-wise page descriptors with verified KVBlock layout offsets
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

    // Thread-safe access to LRU statistics
    size_t get_lru_size() const { return lru_.size(); }

    void record_block_access(int32_t block_id, uint32_t ref_count = 1) {
        lru_.record_access(block_id, ref_count);
    }

    bool is_block_pinned(int32_t block_id) const {
        return lru_.is_pinned(block_id);
    }

private:
    void update_pins_locked() {
        // Unpin all first
        for (int32_t b : allocated_blocks_) {
            lru_.unpin(b);
        }
        // Pin the most recent mandatory_pages from allocation order
        size_t to_pin = std::min<size_t>(budget_.mandatory_pages, allocation_order_.size());
        for (size_t i = 0; i < to_pin; ++i) {
            size_t idx = allocation_order_.size() - 1 - i;
            lru_.pin(allocation_order_[idx]);
        }
    }

    std::string partition_id_;
    PartitionScope scope_;
    RequestBudget budget_;
    std::shared_ptr<PagedKVCacheManager> kv_mgr_;
    std::unordered_set<int32_t> allocated_blocks_;
    std::vector<int32_t> allocation_order_;
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

    std::shared_ptr<MemoryPartition> create_partition(
        const std::string& id, 
        PartitionScope scope, 
        size_t budget, 
        std::shared_ptr<PagedKVCacheManager> kv_mgr
    ) {
        std::lock_guard<std::mutex> lock(mutex_);
        auto part = std::make_shared<MemoryPartition>(id, scope, budget, kv_mgr);
        partitions_[id] = part;
        return part;
    }

    std::shared_ptr<MemoryPartition> create_partition_with_budget(
        const std::string& id, 
        PartitionScope scope, 
        RequestBudget budget, 
        std::shared_ptr<PagedKVCacheManager> kv_mgr
    ) {
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
