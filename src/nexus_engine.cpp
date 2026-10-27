#include "nexus_common.hpp"
#include "memory_tier.hpp"
#include "model_loader.hpp"
#include "paged_cache.hpp"
#include "quantization.hpp"
#include "disaggregated_pipeline.hpp"
#include <iostream>
#include <random>

namespace nexus {

class EngineContext {
public:
    EngineContext(size_t tier0_bytes, size_t tier1_bytes, size_t max_kv_blocks) {
        tier_mgr_ = std::make_shared<MemoryTierManager>(tier0_bytes, tier1_bytes);
        model_loader_ = std::make_unique<UniversalModelLoader>(tier_mgr_);
        kv_mgr_ = std::make_shared<PagedKVCacheManager>(max_kv_blocks, 32, 32, 128);
        control_plane_ = std::make_unique<DisaggregatedControlPlane>(kv_mgr_);
    }

    std::shared_ptr<MemoryTierManager> tier_mgr_;
    std::unique_ptr<UniversalModelLoader> model_loader_;
    std::shared_ptr<PagedKVCacheManager> kv_mgr_;
    std::unique_ptr<DisaggregatedControlPlane> control_plane_;

    std::atomic<uint64_t> total_tokens_generated{0};
    double total_inference_time_us = 0.0;
};

// DisaggregatedControlPlane Implementation
DisaggregatedControlPlane::DisaggregatedControlPlane(std::shared_ptr<PagedKVCacheManager> kv_mgr)
    : kv_mgr_(kv_mgr) {}

DisaggregatedControlPlane::~DisaggregatedControlPlane() {}

void DisaggregatedControlPlane::submit_request(std::shared_ptr<InferenceRequest> req) {
    std::lock_guard<std::mutex> lock(queue_mutex_);
    req->start_time_us = std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::high_resolution_clock::now().time_since_epoch()).count();
    
    // Check Radix prefix cache for instant prefix match
    auto [matched_tokens, matched_blocks] = kv_mgr_->match_prefix(req->prompt_tokens);
    req->processed_tokens = matched_tokens;
    req->assigned_block_ids = matched_blocks;

    // Allocate any additional blocks needed for the remaining uncached prompt tokens
    size_t total_blocks_needed = (req->prompt_tokens.size() + BLOCK_SIZE - 1) / BLOCK_SIZE;
    if (req->assigned_block_ids.size() < total_blocks_needed) {
        size_t extra_needed = total_blocks_needed - req->assigned_block_ids.size();
        auto extra_blocks = kv_mgr_->allocate_blocks(extra_needed);
        req->assigned_block_ids.insert(req->assigned_block_ids.end(), extra_blocks.begin(), extra_blocks.end());
    }

    if (req->processed_tokens >= req->prompt_tokens.size()) {
        // Entire prompt was cached! Skip prefill, immediately enter decode
        req->phase = TaskPhase::DECODE;
        active_decode_pool_.push_back(req);
    } else {
        req->phase = TaskPhase::PREFILL;
        pending_prefill_queue_.push(req);
    }

    all_requests_[req->request_id] = req;
}

std::vector<std::shared_ptr<InferenceRequest>> DisaggregatedControlPlane::schedule_prefill_batch(size_t max_tokens) {
    std::lock_guard<std::mutex> lock(queue_mutex_);
    std::vector<std::shared_ptr<InferenceRequest>> batch;
    size_t accumulated_tokens = 0;

    while (!pending_prefill_queue_.empty() && accumulated_tokens < max_tokens) {
        auto req = pending_prefill_queue_.front();
        size_t remaining = req->prompt_tokens.size() - req->processed_tokens;
        size_t chunk = std::min(remaining, max_tokens - accumulated_tokens);

        accumulated_tokens += chunk;
        batch.push_back(req);
        pending_prefill_queue_.pop();

        if (req->processed_tokens + chunk < req->prompt_tokens.size()) {
            req->processed_tokens += chunk;
            pending_prefill_queue_.push(req); // Re-queue remainder (chunked prefill)
        } else {
            req->processed_tokens += chunk;
        }
    }
    return batch;
}

DisaggregatedControlPlane::InterleavedBatch DisaggregatedControlPlane::schedule_interleaved_iteration(size_t token_budget) {
    std::lock_guard<std::mutex> lock(queue_mutex_);
    InterleavedBatch batch;
    size_t budget_used = 0;

    // 1. Prioritize active decodes (1 token compute per decode request) to eliminate decode jitter
    for (auto it = active_decode_pool_.begin(); it != active_decode_pool_.end() && budget_used < token_budget; ++it) {
        if (!(*it)->is_finished) {
            batch.decode_requests.push_back(*it);
            budget_used += 1;
        }
    }

    // 2. Fill remaining token budget with chunked prefill (Sarathi-Serve / Orca arXiv:2308.16369)
    size_t prefill_budget = (budget_used < token_budget) ? (token_budget - budget_used) : 0;
    size_t prefill_accum = 0;

    while (!pending_prefill_queue_.empty() && prefill_accum < prefill_budget) {
        auto req = pending_prefill_queue_.front();
        size_t remaining = req->prompt_tokens.size() - req->processed_tokens;
        size_t chunk = std::min(remaining, prefill_budget - prefill_accum);

        prefill_accum += chunk;
        batch.prefill_chunks.push_back({req, chunk});
        pending_prefill_queue_.pop();

        req->processed_tokens += chunk;
        if (req->processed_tokens < req->prompt_tokens.size()) {
            pending_prefill_queue_.push(req); // Re-queue remaining chunk
        }
    }

    return batch;
}

std::vector<std::shared_ptr<InferenceRequest>> DisaggregatedControlPlane::schedule_decode_batch(size_t max_batch_size) {
    std::lock_guard<std::mutex> lock(queue_mutex_);
    std::vector<std::shared_ptr<InferenceRequest>> batch;
    for (auto it = active_decode_pool_.begin(); it != active_decode_pool_.end() && batch.size() < max_batch_size; ++it) {
        if (!(*it)->is_finished) {
            batch.push_back(*it);
        }
    }
    return batch;
}

void DisaggregatedControlPlane::complete_prefill_and_handoff(std::shared_ptr<InferenceRequest> req, uint32_t first_token) {
    std::lock_guard<std::mutex> lock(queue_mutex_);
    req->first_token_time_us = std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::high_resolution_clock::now().time_since_epoch()).count();
    req->generated_tokens.push_back(first_token);
    req->phase = TaskPhase::DECODE;

    // Cache the prompt in Radix tree
    kv_mgr_->insert_prefix(req->prompt_tokens, req->assigned_block_ids);

    active_decode_pool_.push_back(req);
}

void DisaggregatedControlPlane::step_decode_token(std::shared_ptr<InferenceRequest> req, int32_t token, bool is_eos) {
    req->generated_tokens.push_back(token);
    if (is_eos || req->generated_tokens.size() >= req->max_new_tokens) {
        req->is_finished = true;
        req->finish_time_us = std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::high_resolution_clock::now().time_since_epoch()).count();
        // Safe release of blocks (solving memory leak root cause from arXiv:2506.09713)
        kv_mgr_->release_blocks(req->assigned_block_ids);
    }
}

void DisaggregatedControlPlane::abort_request(const std::string& request_id) {
    std::lock_guard<std::mutex> lock(queue_mutex_);
    auto it = all_requests_.find(request_id);
    if (it != all_requests_.end()) {
        it->second->is_finished = true;
        kv_mgr_->release_blocks(it->second->assigned_block_ids);
        all_requests_.erase(it);
    }
}

size_t DisaggregatedControlPlane::get_active_prefill_count() const {
    std::lock_guard<std::mutex> lock(queue_mutex_);
    return pending_prefill_queue_.size();
}

size_t DisaggregatedControlPlane::get_active_decode_count() const {
    std::lock_guard<std::mutex> lock(queue_mutex_);
    return active_decode_pool_.size();
}

} // namespace nexus
