#include "moe_router.hpp"
#include <numeric>
#include <iostream>

namespace nexus {

MoEExpertRouter::MoEExpertRouter(uint32_t num_experts, uint32_t top_k, std::shared_ptr<MemoryTierManager> tier_mgr)
    : num_experts_(num_experts), top_k_(top_k), tier_mgr_(tier_mgr) {}

std::vector<MoERoutingDecision> MoEExpertRouter::route_tokens(const float* gate_logits, size_t batch_size) {
    std::vector<MoERoutingDecision> decisions(batch_size);

    for (size_t b = 0; b < batch_size; ++b) {
        const float* logits = gate_logits + (b * num_experts_);
        
        // Find top-k expert indices
        std::vector<std::pair<float, uint32_t>> expert_scores;
        expert_scores.reserve(num_experts_);
        for (uint32_t e = 0; e < num_experts_; ++e) {
            expert_scores.push_back({logits[e], e});
        }

        std::partial_sort(expert_scores.begin(), 
                          expert_scores.begin() + top_k_, 
                          expert_scores.end(), 
                          std::greater<std::pair<float, uint32_t>>());

        // Compute softmax over top-k
        float max_score = expert_scores[0].first;
        float sum_exp = 0.0f;
        std::vector<float> exps(top_k_);
        for (size_t k = 0; k < top_k_; ++k) {
            exps[k] = std::exp(expert_scores[k].first - max_score);
            sum_exp += exps[k];
        }

        decisions[b].selected_expert_ids.resize(top_k_);
        decisions[b].routing_weights.resize(top_k_);

        for (size_t k = 0; k < top_k_; ++k) {
            decisions[b].selected_expert_ids[k] = expert_scores[k].second;
            decisions[b].routing_weights[k] = exps[k] / sum_exp;
        }
    }

    return decisions;
}

NexusStatus MoEExpertRouter::ensure_active_experts(uint32_t layer_idx, const std::vector<uint32_t>& active_expert_ids) {
    // Bring active experts into Tier 0, keeping rest in Tier 1 RAM
    for (uint32_t expert_id : active_expert_ids) {
        std::string expert_prefix = "layers." + std::to_string(layer_idx) + ".experts." + std::to_string(expert_id);
        tier_mgr_->activate_parameter(expert_prefix + ".gate_proj.weight");
        tier_mgr_->activate_parameter(expert_prefix + ".up_proj.weight");
        tier_mgr_->activate_parameter(expert_prefix + ".down_proj.weight");
    }
    return NexusStatus::SUCCESS;
}

void MoEExpertRouter::prefetch_next_layer_experts(uint32_t next_layer_idx, const std::vector<uint32_t>& historical_experts) {
    // Predictive asynchronous prefetch (arXiv:2609.16161)
    for (uint32_t expert_id : historical_experts) {
        std::string expert_prefix = "layers." + std::to_string(next_layer_idx) + ".experts." + std::to_string(expert_id);
        tier_mgr_->prefetch_parameter_async(expert_prefix + ".gate_proj.weight");
    }
}

} // namespace nexus
