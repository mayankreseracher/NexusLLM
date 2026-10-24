#pragma once

#include "nexus_common.hpp"
#include "memory_tier.hpp"
#include <vector>
#include <algorithm>
#include <cmath>

namespace nexus {

struct MoERoutingDecision {
    std::vector<uint32_t> selected_expert_ids;
    std::vector<float> routing_weights;
};

class MoEExpertRouter {
public:
    MoEExpertRouter(uint32_t num_experts, uint32_t top_k, std::shared_ptr<MemoryTierManager> tier_mgr);

    // Compute top-k routing decisions for a batch of tokens given gating logits
    std::vector<MoERoutingDecision> route_tokens(const float* gate_logits, size_t batch_size);

    // Stream the required experts into Tier 0 active buffers, keeping inactive ones in Tier 1 RAM
    NexusStatus ensure_active_experts(uint32_t layer_idx, const std::vector<uint32_t>& active_expert_ids);

    // Predictive prefetching for next layer's probable experts based on temporal locality
    void prefetch_next_layer_experts(uint32_t next_layer_idx, const std::vector<uint32_t>& historical_experts);

    uint32_t get_num_experts() const { return num_experts_; }
    uint32_t get_top_k() const { return top_k_; }

private:
    uint32_t num_experts_;
    uint32_t top_k_;
    std::shared_ptr<MemoryTierManager> tier_mgr_;
    std::unordered_set<uint32_t> currently_active_experts_;
};

} // namespace nexus
