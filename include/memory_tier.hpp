#pragma once

#include "nexus_common.hpp"
#include <thread>
#include <condition_variable>
#include <queue>
#include <unordered_set>

namespace nexus {

struct TierMemoryStats {
    size_t tier0_capacity_bytes = 0;
    size_t tier0_allocated_bytes = 0;
    size_t tier1_host_ram_bytes = 0;
    size_t tier1_inactive_bytes = 0;
    size_t bytes_transferred_tier1_to_tier0 = 0;
    size_t prefetch_hits = 0;
    size_t prefetch_misses = 0;
    double pcie_bandwidth_gbps = 0.0;
};

// Double-buffered layer slot for asynchronous ping-pong execution
struct WorkingBufferSlot {
    void* buffer_ptr = nullptr;
    size_t capacity_bytes = 0;
    size_t used_bytes = 0;
    std::string current_layer_id;
    bool is_ready = false;
    std::atomic<bool> in_use{false};
};

class MemoryTierManager {
public:
    MemoryTierManager(size_t tier0_capacity_bytes, size_t tier1_capacity_bytes);
    ~MemoryTierManager();

    // Register a model parameter tensor into Tier 1 (Host RAM)
    NexusStatus register_parameter(const std::string& name, 
                                   DataType dtype, 
                                   const TensorShape& shape, 
                                   const void* source_data = nullptr);

    // Dynamic activation: Bring tensor into Tier 0 (Active working set)
    void* activate_parameter(const std::string& name);

    // Predictive asynchronous prefetch: Start loading parameter into Tier 0 before needed
    NexusStatus prefetch_parameter_async(const std::string& name);

    // Evict parameter back to inactive state in Tier 1 RAM
    NexusStatus deactivate_parameter(const std::string& name);

    // Pin-pong buffer rotation for sequential transformer layer execution
    WorkingBufferSlot* acquire_active_slot(size_t required_bytes);
    void release_active_slot(WorkingBufferSlot* slot);

    // Local buffer residency statistics (strictly offline / zero-telemetry)
    TierMemoryStats get_stats() const;

    // Platform hardware and raw performance profile
    const PlatformHardwareProfile& get_platform_profile() const { return platform_profile_; }

    // Direct access to a tensor view
    TensorView* get_tensor_view(const std::string& name);
    const std::unordered_map<std::string, std::shared_ptr<TensorView>>& get_all_tensors() const {
        return registry_;
    }

private:
    void prefetch_worker_loop();

    PlatformHardwareProfile platform_profile_;
    size_t tier0_capacity_;
    size_t tier1_capacity_;
    
    void* tier0_raw_pool_ = nullptr;
    void* tier1_raw_pool_ = nullptr;

    size_t tier0_offset_ = 0;
    size_t tier1_offset_ = 0;

    mutable std::mutex registry_mutex_;
    std::unordered_map<std::string, std::shared_ptr<TensorView>> registry_;

    // Ping-pong double buffers for zero-stall streaming (Layer L compute, Layer L+1 prefetch)
    static constexpr size_t NUM_WORKING_SLOTS = 2;
    WorkingBufferSlot working_slots_[NUM_WORKING_SLOTS];
    size_t current_slot_idx_ = 0;

    // Asynchronous background prefetch queue
    std::thread prefetch_thread_;
    std::atomic<bool> stop_prefetch_thread_{false};
    std::queue<std::string> prefetch_queue_;
    std::mutex prefetch_mutex_;
    std::condition_variable prefetch_cv_;

    // Statistics
    mutable std::mutex stats_mutex_;
    TierMemoryStats stats_;
};

} // namespace nexus
