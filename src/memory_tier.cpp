#include "memory_tier.hpp"
#include <iostream>
#include <algorithm>

namespace nexus {

MemoryTierManager::MemoryTierManager(size_t tier0_capacity_bytes, size_t tier1_capacity_bytes)
    : platform_profile_(PlatformHardwareProfile::detect()),
      tier0_capacity_(tier0_capacity_bytes), 
      tier1_capacity_(tier1_capacity_bytes) {

#if defined(NEXUS_PLATFORM_WINDOWS)
    // Allocate Tier 0 (Active Working Set) in pinned memory
    tier0_raw_pool_ = VirtualAlloc(NULL, tier0_capacity_, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
    // Allocate Tier 1 (Host System RAM) for full model weights / inactive parameter residency
    tier1_raw_pool_ = VirtualAlloc(NULL, tier1_capacity_, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
#elif defined(NEXUS_PLATFORM_MACOS) || defined(NEXUS_PLATFORM_LINUX)
#if defined(NEXUS_PLATFORM_MACOS)
    tier0_raw_pool_ = mmap(NULL, tier0_capacity_, PROT_READ | PROT_WRITE, MAP_ANON | MAP_PRIVATE, -1, 0);
    tier1_raw_pool_ = mmap(NULL, tier1_capacity_, PROT_READ | PROT_WRITE, MAP_ANON | MAP_PRIVATE, -1, 0);
#else
    tier0_raw_pool_ = mmap(NULL, tier0_capacity_, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    tier1_raw_pool_ = mmap(NULL, tier1_capacity_, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
#endif

    if (tier0_raw_pool_ != MAP_FAILED && tier0_raw_pool_ != nullptr) {
        tier0_is_mmap_ = true;
    } else {
        tier0_raw_pool_ = nexus_aligned_alloc(NEXUS_ALIGNMENT, tier0_capacity_);
        tier0_is_mmap_ = false;
    }

    if (tier1_raw_pool_ != MAP_FAILED && tier1_raw_pool_ != nullptr) {
        tier1_is_mmap_ = true;
    } else {
        tier1_raw_pool_ = nexus_aligned_alloc(NEXUS_ALIGNMENT, tier1_capacity_);
        tier1_is_mmap_ = false;
    }

#if defined(MADV_HUGEPAGE)
    if (tier0_is_mmap_) madvise(tier0_raw_pool_, tier0_capacity_, MADV_HUGEPAGE);
    if (tier1_is_mmap_) madvise(tier1_raw_pool_, tier1_capacity_, MADV_HUGEPAGE);
#endif

#else
    tier0_raw_pool_ = nexus_aligned_alloc(NEXUS_ALIGNMENT, tier0_capacity_);
    tier1_raw_pool_ = nexus_aligned_alloc(NEXUS_ALIGNMENT, tier1_capacity_);
    tier0_is_mmap_ = false;
    tier1_is_mmap_ = false;
#endif

    stats_.tier0_capacity_bytes = tier0_capacity_;
    stats_.tier1_host_ram_bytes = tier1_capacity_;

    // Initialize ping-pong working slots
    size_t half_tier0 = tier0_capacity_ / NUM_WORKING_SLOTS;
    for (size_t i = 0; i < NUM_WORKING_SLOTS; ++i) {
        working_slots_[i].buffer_ptr = static_cast<char*>(tier0_raw_pool_) + (i * half_tier0);
        working_slots_[i].capacity_bytes = half_tier0;
        working_slots_[i].used_bytes = 0;
        working_slots_[i].in_use.store(false);
    }

    // Launch background prefetch worker thread
    prefetch_thread_ = std::thread(&MemoryTierManager::prefetch_worker_loop, this);
}

MemoryTierManager::~MemoryTierManager() {
    stop_prefetch_thread_.store(true);
    prefetch_cv_.notify_all();
    if (prefetch_thread_.joinable()) {
        prefetch_thread_.join();
    }

#if defined(NEXUS_PLATFORM_WINDOWS)
    if (tier0_raw_pool_) VirtualFree(tier0_raw_pool_, 0, MEM_RELEASE);
    if (tier1_raw_pool_) VirtualFree(tier1_raw_pool_, 0, MEM_RELEASE);
#elif defined(NEXUS_PLATFORM_POSIX)
    if (tier0_raw_pool_ && tier0_raw_pool_ != MAP_FAILED) {
        if (tier0_is_mmap_) munmap(tier0_raw_pool_, tier0_capacity_);
        else nexus_aligned_free(tier0_raw_pool_);
    }
    if (tier1_raw_pool_ && tier1_raw_pool_ != MAP_FAILED) {
        if (tier1_is_mmap_) munmap(tier1_raw_pool_, tier1_capacity_);
        else nexus_aligned_free(tier1_raw_pool_);
    }
#else
    if (tier0_raw_pool_) nexus_aligned_free(tier0_raw_pool_);
    if (tier1_raw_pool_) nexus_aligned_free(tier1_raw_pool_);
#endif
}

NexusStatus MemoryTierManager::register_parameter(const std::string& name, 
                                                DataType dtype, 
                                                const TensorShape& shape, 
                                                const void* source_data) {
    std::lock_guard<std::mutex> lock(registry_mutex_);
    size_t req_bytes = shape.total_bytes(dtype);

    if (tier1_offset_ + req_bytes > tier1_capacity_) {
        return NexusStatus::ERROR_OUT_OF_HOST_MEMORY;
    }

    auto tv = std::make_shared<TensorView>();
    tv->name = name;
    tv->dtype = dtype;
    tv->shape = shape;
    tv->byte_size = req_bytes;
    tv->current_tier = MemoryTier::TIER1_HOST_SYSTEM_RAM;
    tv->is_active = false;
    tv->host_ptr = static_cast<char*>(tier1_raw_pool_) + tier1_offset_;
    tier1_offset_ += (req_bytes + NEXUS_ALIGNMENT - 1) & ~(NEXUS_ALIGNMENT - 1);

    if (source_data) {
        std::memcpy(tv->host_ptr, source_data, req_bytes);
    } else {
        std::memset(tv->host_ptr, 0, req_bytes);
    }

    registry_[name] = tv;
    stats_.tier1_inactive_bytes += req_bytes;
    return NexusStatus::SUCCESS;
}

NexusStatus MemoryTierManager::register_parameter_mapped(const std::string& name, 
                                                        DataType dtype, 
                                                        const TensorShape& shape, 
                                                        const void* mapped_ptr) {
    if (!mapped_ptr) return NexusStatus::ERROR_INVALID_ARGUMENT;

    std::lock_guard<std::mutex> lock(registry_mutex_);
    size_t req_bytes = shape.total_bytes(dtype);

    auto tv = std::make_shared<TensorView>();
    tv->name = name;
    tv->dtype = dtype;
    tv->shape = shape;
    tv->byte_size = req_bytes;
    tv->current_tier = MemoryTier::TIER1_HOST_SYSTEM_RAM;
    tv->is_active = false;
    tv->host_ptr = const_cast<void*>(mapped_ptr);
    tier1_offset_ += req_bytes;

    registry_[name] = tv;
    stats_.tier1_inactive_bytes += req_bytes;
    return NexusStatus::SUCCESS;
}

void* MemoryTierManager::activate_parameter(const std::string& name) {
    std::lock_guard<std::mutex> lock(registry_mutex_);
    auto it = registry_.find(name);
    if (it == registry_.end()) return nullptr;

    auto tv = it->second;
    tv->access_count++;

    if (tv->is_active && tv->active_ptr) {
        std::lock_guard<std::mutex> sl(stats_mutex_);
        stats_.prefetch_hits++;
        return tv->active_ptr;
    }

    // High-performance UMA Zero-Copy Path for macOS Apple Silicon
    if (platform_profile_.is_unified_memory) {
        tv->active_ptr = tv->host_ptr;
        tv->is_active = true;
        tv->current_tier = MemoryTier::TIER0_ACTIVE_WORKING_SET;

        std::lock_guard<std::mutex> sl(stats_mutex_);
        stats_.prefetch_hits++;
        stats_.pcie_bandwidth_gbps = platform_profile_.theoretical_bandwidth_gbps;
        return tv->active_ptr;
    }

    // Discrete PCIe Copy Path (Windows / Linux discrete GPUs)
    WorkingBufferSlot* slot = acquire_active_slot(tv->byte_size);
    if (!slot) return nullptr;

    size_t aligned_req = (tv->byte_size + NEXUS_ALIGNMENT - 1) & ~(NEXUS_ALIGNMENT - 1);
    void* dest_ptr = static_cast<char*>(slot->buffer_ptr) + (slot->used_bytes - aligned_req);

    auto t0 = std::chrono::high_resolution_clock::now();
    std::memcpy(dest_ptr, tv->host_ptr, tv->byte_size);
    auto t1 = std::chrono::high_resolution_clock::now();
    double copy_time_s = std::chrono::duration<double>(t1 - t0).count();

    tv->active_ptr = dest_ptr;
    tv->is_active = true;
    tv->current_tier = MemoryTier::TIER0_ACTIVE_WORKING_SET;
    slot->current_layer_id = name;
    slot->slot_tensors.push_back(name);

    {
        std::lock_guard<std::mutex> sl(stats_mutex_);
        stats_.prefetch_misses++;
        stats_.bytes_transferred_tier1_to_tier0 += tv->byte_size;
        if (copy_time_s > 0.0) {
            double cur_gbps = (tv->byte_size / (1024.0 * 1024.0 * 1024.0)) / copy_time_s;
            stats_.pcie_bandwidth_gbps = (stats_.pcie_bandwidth_gbps * 0.9) + (cur_gbps * 0.1);
        }
    }

    return tv->active_ptr;
}

NexusStatus MemoryTierManager::prefetch_parameter_async(const std::string& name) {
#if defined(NEXUS_PLATFORM_POSIX) && defined(MADV_WILLNEED)
    {
        std::lock_guard<std::mutex> lock(registry_mutex_);
        auto it = registry_.find(name);
        if (it != registry_.end() && it->second->host_ptr) {
            long page_size = sysconf(_SC_PAGESIZE);
            if (page_size <= 0) page_size = 4096;
            uintptr_t addr = reinterpret_cast<uintptr_t>(it->second->host_ptr);
            uintptr_t aligned_addr = addr & ~(page_size - 1);
            size_t aligned_len = ((addr + it->second->byte_size + page_size - 1) & ~(page_size - 1)) - aligned_addr;
            madvise(reinterpret_cast<void*>(aligned_addr), aligned_len, MADV_WILLNEED);
        }
    }
#endif
    {
        std::lock_guard<std::mutex> lock(prefetch_mutex_);
        prefetch_queue_.push(name);
    }
    prefetch_cv_.notify_one();
    return NexusStatus::SUCCESS;
}

NexusStatus MemoryTierManager::deactivate_parameter(const std::string& name) {
    std::lock_guard<std::mutex> lock(registry_mutex_);
    auto it = registry_.find(name);
    if (it == registry_.end()) return NexusStatus::ERROR_TENSOR_NOT_FOUND;

    auto tv = it->second;
    if (tv->is_active) {
        tv->is_active = false;
        tv->active_ptr = nullptr;
        tv->current_tier = MemoryTier::TIER1_HOST_SYSTEM_RAM;

#if defined(NEXUS_PLATFORM_POSIX) && defined(MADV_COLD)
        if (tv->host_ptr) {
            long page_size = sysconf(_SC_PAGESIZE);
            if (page_size <= 0) page_size = 4096;
            uintptr_t addr = reinterpret_cast<uintptr_t>(tv->host_ptr);
            uintptr_t aligned_addr = addr & ~(page_size - 1);
            size_t aligned_len = ((addr + tv->byte_size + page_size - 1) & ~(page_size - 1)) - aligned_addr;
            madvise(reinterpret_cast<void*>(aligned_addr), aligned_len, MADV_COLD);
        }
#endif

        std::lock_guard<std::mutex> sl(stats_mutex_);
        if (stats_.tier0_allocated_bytes >= tv->byte_size) {
            stats_.tier0_allocated_bytes -= tv->byte_size;
        }
        stats_.tier1_inactive_bytes += tv->byte_size;
    }

    return NexusStatus::SUCCESS;
}

WorkingBufferSlot* MemoryTierManager::acquire_active_slot(size_t required_bytes) {
    size_t aligned_req = (required_bytes + NEXUS_ALIGNMENT - 1) & ~(NEXUS_ALIGNMENT - 1);

    // Try packing into the current active slot first
    size_t cur = current_slot_idx_ % NUM_WORKING_SLOTS;
    if (working_slots_[cur].capacity_bytes >= working_slots_[cur].used_bytes + aligned_req) {
        working_slots_[cur].in_use.store(true);
        working_slots_[cur].used_bytes += aligned_req;
        return &working_slots_[cur];
    }

    // Ping-pong double-buffering rotation: advance to next alternating slot
    current_slot_idx_++;
    size_t next = current_slot_idx_ % NUM_WORKING_SLOTS;

    // Reset previous contents in this slot
    if (!working_slots_[next].slot_tensors.empty()) {
        for (const auto& t_name : working_slots_[next].slot_tensors) {
            auto it = registry_.find(t_name);
            if (it != registry_.end()) {
                it->second->is_active = false;
                it->second->active_ptr = nullptr;
                it->second->current_tier = MemoryTier::TIER1_HOST_SYSTEM_RAM;
            }
        }
        working_slots_[next].slot_tensors.clear();
    }
    working_slots_[next].used_bytes = aligned_req;
    working_slots_[next].in_use.store(true);
    return &working_slots_[next];
}

void MemoryTierManager::release_active_slot(WorkingBufferSlot* slot) {
    if (slot) {
        slot->in_use.store(false);
        slot->used_bytes = 0;
        slot->current_layer_id.clear();
        slot->slot_tensors.clear();
        slot->is_ready.store(false);
    }
}

std::vector<std::string> MemoryTierManager::get_layer_tensor_names(size_t layer_idx) const {
    std::string pfx = "blk." + std::to_string(layer_idx) + ".";
    return {
        pfx + "attn_norm.weight",
        pfx + "attn_q.weight",
        pfx + "attn_k.weight",
        pfx + "attn_v.weight",
        pfx + "attn_output.weight",
        pfx + "ffn_norm.weight",
        pfx + "ffn_gate.weight",
        pfx + "ffn_up.weight",
        pfx + "ffn_down.weight"
    };
}

NexusStatus MemoryTierManager::prefetch_layer_async(size_t layer_idx) {
    auto names = get_layer_tensor_names(layer_idx);
    for (const auto& name : names) {
        prefetch_parameter_async(name);
    }
    return NexusStatus::SUCCESS;
}

NexusStatus MemoryTierManager::deactivate_layer(size_t layer_idx) {
    auto names = get_layer_tensor_names(layer_idx);
    for (const auto& name : names) {
        deactivate_parameter(name);
    }
    return NexusStatus::SUCCESS;
}

void MemoryTierManager::prefetch_worker_loop() {
    while (!stop_prefetch_thread_.load()) {
        std::string target_tensor;
        {
            std::unique_lock<std::mutex> lock(prefetch_mutex_);
            prefetch_cv_.wait(lock, [this]() {
                return !prefetch_queue_.empty() || stop_prefetch_thread_.load();
            });

            if (stop_prefetch_thread_.load()) break;

            target_tensor = prefetch_queue_.front();
            prefetch_queue_.pop();
        }

        // Perform background prefetch into Tier 0
        activate_parameter(target_tensor);
    }
}

TierMemoryStats MemoryTierManager::get_stats() const {
    std::lock_guard<std::mutex> sl(stats_mutex_);
    TierMemoryStats s = stats_;
    size_t active_bytes = 0;
    for (size_t i = 0; i < NUM_WORKING_SLOTS; ++i) {
        if (working_slots_[i].in_use.load()) {
            active_bytes += working_slots_[i].used_bytes;
        }
    }
    s.tier0_allocated_bytes = active_bytes;
    if (tier1_offset_ >= active_bytes) {
        s.tier1_inactive_bytes = tier1_offset_ - active_bytes;
    } else {
        s.tier1_inactive_bytes = tier1_offset_;
    }
    return s;
}

TensorView* MemoryTierManager::get_tensor_view(const std::string& name) {
    std::lock_guard<std::mutex> lock(registry_mutex_);
    auto it = registry_.find(name);
    return (it != registry_.end()) ? it->second.get() : nullptr;
}

void MemoryTierManager::clear_registry() {
    std::lock_guard<std::mutex> lock(registry_mutex_);
    registry_.clear();
    tier1_offset_ = 0;
    tier0_offset_ = 0;
    std::lock_guard<std::mutex> sl(stats_mutex_);
    stats_.tier1_inactive_bytes = 0;
    stats_.tier0_allocated_bytes = 0;
}

} // namespace nexus
