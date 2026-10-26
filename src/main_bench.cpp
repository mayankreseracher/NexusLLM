#include "nexus_abi.h"
#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>

int main(int argc, char** argv) {
    std::cout << "================================================================" << std::endl;
    std::cout << "   NEXUS-LLM HIGH-THROUGHPUT TIERED INFERENCE ENGINE BENCHMARK  " << std::endl;
    std::cout << "   Active Accelerator + Inactive Host RAM Dynamic Offloader     " << std::endl;
    std::cout << "================================================================" << std::endl;

    // 512 MB Tier 0 (working buffer), 4 GB Tier 1 (Host System RAM), 4096 KV blocks
    size_t tier0_bytes = 512ULL * 1024 * 1024;
    size_t tier1_bytes = 4ULL * 1024 * 1024 * 1024;
    size_t max_kv_blocks = 4096;

    NexusEngineHandle engine = nexus_create_engine(tier0_bytes, tier1_bytes, max_kv_blocks);
    if (!engine) {
        std::cerr << "Failed to initialize Nexus Engine!" << std::endl;
        return 1;
    }

    NexusPlatformInfo pinfo;
    nexus_get_platform_info(&pinfo);

    std::cout << "[Nexus Engine] Engine initialized successfully." << std::endl;
    std::cout << "[Platform HW]  Operating System       : " << pinfo.os_name << std::endl;
    std::cout << "[Platform HW]  CPU Architecture       : " << pinfo.cpu_arch << std::endl;
    std::cout << "[Platform HW]  Vector Engine          : " << pinfo.vector_isa_name << std::endl;
    std::cout << "[Platform HW]  Unified Memory (UMA)   : " << (pinfo.is_unified_memory ? "YES (Zero-Copy Enabled)" : "NO (PCIe Offload)") << std::endl;
    std::cout << "[Platform HW]  Transparent HugePages  : " << (pinfo.has_transparent_hugepages ? "YES (2MB Pages Enabled)" : "NO") << std::endl;
    std::cout << "[Platform HW]  Hardware Concurrency   : " << pinfo.hardware_concurrency << " threads" << std::endl;
    std::cout << "[Platform HW]  Theoretical Peak BW    : " << pinfo.theoretical_bandwidth_gbps << " GB/s" << std::endl;
    std::cout << "[Platform HW]  Performance Profile    : " << pinfo.performance_tier_desc << std::endl;
    std::cout << "----------------------------------------------------------------" << std::endl;
    std::cout << "[Nexus Engine] Tier 0 (Active Working Set): " << (tier0_bytes / (1024 * 1024)) << " MB" << std::endl;
    std::cout << "[Nexus Engine] Tier 1 (Host RAM Residency): " << (tier1_bytes / (1024 * 1024)) << " MB" << std::endl;

    // Submit a burst of concurrent requests to test continuous batching
    const size_t NUM_REQUESTS = 16;
    const size_t PROMPT_LEN = 128;
    const size_t GEN_LEN = 64;

    std::vector<int32_t> sample_prompt(PROMPT_LEN);
    for (size_t i = 0; i < PROMPT_LEN; ++i) {
        sample_prompt[i] = static_cast<int32_t>(100 + (i % 500));
    }

    std::cout << "\n[Benchmark] Submitting " << NUM_REQUESTS << " concurrent requests..." << std::endl;
    for (size_t r = 0; r < NUM_REQUESTS; ++r) {
        std::string req_id = "req-" + std::to_string(r);
        nexus_submit_request(engine, req_id.c_str(), sample_prompt.data(), PROMPT_LEN, GEN_LEN);
    }

    // Run continuous batching decode loop
    std::cout << "[Benchmark] Executing continuous iteration-level decode loop..." << std::endl;
    auto t_start = std::chrono::high_resolution_clock::now();

    size_t total_tokens = 0;
    std::vector<int32_t> out_tokens(NUM_REQUESTS);

    for (size_t step = 0; step < GEN_LEN + 10; ++step) {
        int32_t stepped = nexus_step_continuous_batch(engine, nullptr, out_tokens.data(), NUM_REQUESTS);
        total_tokens += stepped;
        if (stepped == 0 && step > 5) break; // Finished all
    }

    auto t_end = std::chrono::high_resolution_clock::now();
    double elapsed_sec = std::chrono::duration<double>(t_end - t_start).count();
    double tps = total_tokens / elapsed_sec;

    NexusLocalMemoryStats stats;
    nexus_get_memory_stats(engine, &stats);

    std::cout << "\n======================== RESULTS ===============================" << std::endl;
    std::cout << " Total Output Tokens Generated : " << total_tokens << std::endl;
    std::cout << " Total Elapsed Time            : " << std::fixed << std::setprecision(4) << elapsed_sec << " s" << std::endl;
    std::cout << " Throughput                    : " << std::fixed << std::setprecision(2) << tps << " tokens/sec (TPS)" << std::endl;
    std::cout << " Host RAM Inactive Weights     : " << (stats.tier1_inactive_bytes / (1024 * 1024)) << " MB" << std::endl;
    std::cout << " Active Tier 0 Working Set     : " << (stats.tier0_allocated_bytes / (1024 * 1024)) << " MB" << std::endl;
    std::cout << " Host-to-Active PCIe Bandwidth : " << std::fixed << std::setprecision(2) << stats.pcie_bandwidth_gbps << " GB/s" << std::endl;
    std::cout << " Radix Prefix Cache Hit Rate   : " << std::fixed << std::setprecision(1) << (stats.prefix_cache_hit_rate * 100.0) << " %" << std::endl;
    std::cout << "================================================================" << std::endl;

    nexus_destroy_engine(engine);
    return 0;
}
