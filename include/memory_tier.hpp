#pragma once

#include "nexus_common.hpp"
#include "dak_offload.hpp"
#include <thread>
#include <condition_variable>
#include <queue>
#include <unordered_set>

namespace nexus {

// ============================================================================
// DAK PCIe Congestion Controller (arXiv:2604.26074 Section 4)
// Tracks in-flight TMA requests and applies backpressure to prevent PCIe saturation
// ============================================================================
class DakCongestionController {
public:
    explicit DakCongestionController(size_t max_inflight_bytes = 64 * 1024 * 1024,
                                     size_t max_outstanding_descriptors = 32)
        : max_inflight_bytes_(max_inflight_bytes),
          max_descriptors_(max_outstanding_descriptors), // Fixed constructor self-init bug (Issue 4)
          current_inflight_bytes_(0),
          current_descriptors_(0) {}

    bool try_acquire_transfer(size_t transfer_bytes) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (current_inflight_bytes_ + transfer_bytes > max_inflight_bytes_ ||
            current_descriptors_ + 1 > max_descriptors_) {
            return false;
        }
        current_inflight_bytes_ += transfer_bytes;
        current_descriptors_++;
        return true;
    }

    void acquire_transfer_blocking(size_t transfer_bytes) {
        std::unique_lock<std::mutex> lock(mutex_);
        cv_.wait(lock, [&]() {
            return (current_inflight_bytes_ + transfer_bytes <= max_inflight_bytes_) &&
                   (current_descriptors_ + 1 <= max_descriptors_);
        });
        current_inflight_bytes_ += transfer_bytes;
        current_descriptors_++;
    }

    void release_transfer(size_t transfer_bytes) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (current_inflight_bytes_ >= transfer_bytes) {
            current_inflight_bytes_ -= transfer_bytes;
        } else {
            current_inflight_bytes_ = 0;
        }
        if (current_descriptors_ > 0) current_descriptors_--;
        cv_.notify_all();
    }

    float get_saturation_ratio() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return static_cast<float>(current_inflight_bytes_) / static_cast<float>(max_inflight_bytes_);
    }

    size_t get_inflight_bytes() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return current_inflight_bytes_;
    }

private:
    size_t max_inflight_bytes_;
    size_t max_descriptors_;
    size_t current_inflight_bytes_;
    size_t current_descriptors_;
    mutable std::mutex mutex_;
    std::condition_variable cv_;
};

// ============================================================================
// DAK Analytical Greedy Offload Planner (Addresses Issue 6)
// Distinguishes compute-bound GEMM (~75% offload) vs memory-bound Attention (~25% offload)
// ============================================================================
class DakOffloadPlanner {
public:
    struct PlanResult {
        float offload_ratio = 0.0f;
        size_t tma_direct_blocks = 0;
        size_t hbm_resident_blocks = 0;
        bool bypass_hbm = false;
    };

    // Analytical model for Compute-Bound GEMM operations (Prefill / FFN / Projections)
    static PlanResult compute_ratio_gemm(
        size_t total_blocks,
        size_t M, size_t N, size_t K,
        float current_hbm_pressure,
        double pcie_bandwidth_gbps,
        double tflops_tensor_core
    ) {
        PlanResult res;
        if (total_blocks == 0) return res;

        // Arithmetic Intensity = 2 * M * N * K / (2 * (M*K + K*N + M*N))
        double flops = 2.0 * static_cast<double>(M) * N * K;
        double memory_bytes = 2.0 * (static_cast<double>(M) * K + static_cast<double>(K) * N + static_cast<double>(M) * N);
        double ai = (memory_bytes > 0.0) ? (flops / memory_bytes) : 100.0;

        // Compute-bound operations (AI >= 30 FLOP/Byte) comfortably allow high offload ratios (paper reports ~75%)
        float optimal_ratio = (ai >= 30.0) ? 0.75f : std::min(0.75f, static_cast<float>(ai / 40.0f));

        if (current_hbm_pressure > 0.85f) {
            optimal_ratio = std::min(1.0f, optimal_ratio + 0.15f);
        } else if (current_hbm_pressure < 0.30f) {
            optimal_ratio = std::max(0.20f, optimal_ratio * 0.7f);
        }

        (void)pcie_bandwidth_gbps;
        (void)tflops_tensor_core;

        res.offload_ratio = optimal_ratio;
        res.tma_direct_blocks = static_cast<size_t>(total_blocks * optimal_ratio);
        res.hbm_resident_blocks = total_blocks - res.tma_direct_blocks;
        res.bypass_hbm = (res.tma_direct_blocks > 0);
        return res;
    }

    // Analytical model for Memory-Bound Attention operations (Decode Phase)
    static PlanResult compute_ratio_attention(
        size_t total_blocks,
        float current_hbm_pressure,
        float pcie_congestion
    ) {
        PlanResult res;
        if (total_blocks == 0) return res;

        if (pcie_congestion > 0.80f) {
            res.offload_ratio = 0.0f;
            res.hbm_resident_blocks = total_blocks;
            return res;
        }

        // Attention has low arithmetic intensity; paper reports max ~25% offload on PCIe
        float base_ratio = 0.25f;
        if (current_hbm_pressure < 0.60f) {
            base_ratio = 0.15f;
        } else if (current_hbm_pressure > 0.85f) {
            base_ratio = 0.35f;
        }

        res.offload_ratio = base_ratio;
        res.tma_direct_blocks = static_cast<size_t>(total_blocks * base_ratio);
        res.hbm_resident_blocks = total_blocks - res.tma_direct_blocks;
        res.bypass_hbm = (res.tma_direct_blocks > 0);
        return res;
    }
};

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
    int32_t active_layer_idx = -1;
    std::string current_layer_id;
    std::vector<std::string> slot_tensors;
    std::atomic<bool> is_ready{false};
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

    // Register a memory-mapped parameter directly backed by OS page cache / NVMe file mapping
    NexusStatus register_parameter_mapped(const std::string& name, 
                                          DataType dtype, 
                                          const TensorShape& shape, 
                                          const void* mapped_ptr);

    // Dynamic activation: Bring tensor into Tier 0 (Active working set)
    void* activate_parameter(const std::string& name);

    // Predictive asynchronous prefetch: Start loading parameter into Tier 0 before needed
    NexusStatus prefetch_parameter_async(const std::string& name);

    // Prefetch all parameters of a layer into alternating double buffer slot
    NexusStatus prefetch_layer_async(size_t layer_idx);

    // Evict parameter back to inactive state in Tier 1 RAM
    NexusStatus deactivate_parameter(const std::string& name);

    // Deactivate all parameters of a layer and rotate double buffer slot
    NexusStatus deactivate_layer(size_t layer_idx);

    // Helper: list canonical tensor names for a layer index
    std::vector<std::string> get_layer_tensor_names(size_t layer_idx) const;

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

    // DAK (Direct-Access KV Offloading via TMA) configuration & congestion control
    bool is_tma_offload_enabled() const { return enable_tma_offload_; }
    void set_tma_offload_enabled(bool enabled) { enable_tma_offload_ = enabled; }
    DakCongestionController& get_congestion_controller() { return dak_congestion_; }
    const DakCongestionController& get_congestion_controller() const { return dak_congestion_; }

    // Clear registry and reset allocations (called when loading a new model to eliminate stale synthetic weights)
    void clear_registry();

private:
    void prefetch_worker_loop();

    PlatformHardwareProfile platform_profile_;
    size_t tier0_capacity_;
    size_t tier1_capacity_;
    
    void* tier0_raw_pool_ = nullptr;
    void* tier1_raw_pool_ = nullptr;
    bool tier0_is_mmap_ = false;
    bool tier1_is_mmap_ = false;

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

    // DAK Asynchronous TMA offload control plane
    bool enable_tma_offload_ = false;
    DakCongestionController dak_congestion_;
};

} // namespace nexus
