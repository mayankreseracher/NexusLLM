#pragma once

#include "nexus_common.hpp"
#include "paged_cache.hpp"
#include <queue>
#include <future>

namespace nexus {

enum class TaskPhase {
    PREFILL = 0,
    DECODE = 1,
    SPECULATIVE_VERIFY = 2
};

struct InferenceRequest {
    std::string request_id;
    std::vector<int32_t> prompt_tokens;
    std::vector<int32_t> generated_tokens;
    size_t max_new_tokens = 128;
    float temperature = 0.7f;
    float top_p = 0.9f;
    TaskPhase phase = TaskPhase::PREFILL;
    std::vector<int32_t> assigned_block_ids;
    size_t processed_tokens = 0;
    bool is_finished = false;
    uint64_t start_time_us = 0;
    uint64_t first_token_time_us = 0;
    uint64_t finish_time_us = 0;
};

// Zero-copy KV transfer descriptor for Prefill-to-Decode disaggregation (arXiv:2609.23130)
struct KVTransferPacket {
    std::string request_id;
    std::vector<int32_t> block_ids;
    size_t prompt_length;
    uint32_t first_generated_token;
    void* kv_data_handle = nullptr;
    size_t kv_byte_size = 0;
};

class DisaggregatedControlPlane {
public:
    DisaggregatedControlPlane(std::shared_ptr<PagedKVCacheManager> kv_mgr);
    ~DisaggregatedControlPlane();

    // Submit new client request
    void submit_request(std::shared_ptr<InferenceRequest> req);

    // Pull next batch for Prefill worker (chunked prefill)
    std::vector<std::shared_ptr<InferenceRequest>> schedule_prefill_batch(size_t max_tokens_in_batch);

    // Pull next batch for Decode worker (continuous iteration-level batching)
    std::vector<std::shared_ptr<InferenceRequest>> schedule_decode_batch(size_t max_batch_size);

    // Interleaved Iteration Schedule for Orca / Sarathi-Serve (arXiv:2308.16369)
    // Combines active single-token decodes with chunked prefill within a single iteration budget
    struct InterleavedBatch {
        std::vector<std::shared_ptr<InferenceRequest>> decode_requests;
        std::vector<std::pair<std::shared_ptr<InferenceRequest>, size_t>> prefill_chunks;
    };
    InterleavedBatch schedule_interleaved_iteration(size_t token_budget);

    // Complete prefill phase and dispatch KV transfer packet to decode worker
    void complete_prefill_and_handoff(std::shared_ptr<InferenceRequest> req, uint32_t first_token);

    // Handle completed decode token
    void step_decode_token(std::shared_ptr<InferenceRequest> req, int32_t token, bool is_eos);

    // Abort request safely and immediately reclaim blocks (fixing bug root cause from arXiv:2506.09713)
    void abort_request(const std::string& request_id);

    size_t get_active_prefill_count() const;
    size_t get_active_decode_count() const;

private:
    std::shared_ptr<PagedKVCacheManager> kv_mgr_;

    mutable std::mutex queue_mutex_;
    std::queue<std::shared_ptr<InferenceRequest>> pending_prefill_queue_;
    std::vector<std::shared_ptr<InferenceRequest>> active_decode_pool_;
    std::unordered_map<std::string, std::shared_ptr<InferenceRequest>> all_requests_;

    std::queue<KVTransferPacket> kv_transfer_queue_;
    std::mutex transfer_mutex_;
};

} // namespace nexus
