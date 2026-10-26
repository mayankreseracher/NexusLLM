#include "nexus_abi.h"
#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <string>

int main(int argc, char** argv) {
    std::cout << "================================================================" << std::endl;
    std::cout << "   NEXUS-LLM HIGH-THROUGHPUT TIERED INFERENCE ENGINE BENCHMARK  " << std::endl;
    std::cout << "   Active Accelerator + Inactive Host RAM Dynamic Offloader     " << std::endl;
    std::cout << "================================================================" << std::endl;

    std::string model_path = "";
    std::string device_name = "";
    size_t num_requests = 16;
    size_t prompt_len = 128;
    size_t gen_len = 64;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--model" && i + 1 < argc) {
            model_path = argv[++i];
        } else if (arg == "--device" && i + 1 < argc) {
            device_name = argv[++i];
        } else if (arg == "--requests" && i + 1 < argc) {
            num_requests = std::stoull(argv[++i]);
        } else if (arg == "--prompt" && i + 1 < argc) {
            prompt_len = std::stoull(argv[++i]);
        } else if (arg == "--gen" && i + 1 < argc) {
            gen_len = std::stoull(argv[++i]);
        } else if (arg == "--help" || arg == "-h") {
            std::cout << "Usage: " << argv[0] << " [--model <path.gguf>] [--device <cpu|cuda|metal|rocm|sycl>] [--requests <N>] [--prompt <len>] [--gen <len>]" << std::endl;
            return 0;
        }
    }

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

    // Phase 5 Hardware Backend Discovery
    uint32_t dev_count = nexus_get_available_device_count();
    std::cout << "[Hardware Devices] Detected " << dev_count << " Compute Acceleration Interfaces:" << std::endl;
    for (uint32_t d = 0; d < dev_count; ++d) {
        NexusDeviceInfo dinfo;
        if (nexus_get_device_info(d, &dinfo) == 0) {
            std::cout << "  - [" << dinfo.device_name << "] (" << dinfo.vendor << ", " << dinfo.architecture << "): " << dinfo.description << std::endl;
        }
    }

    if (!device_name.empty()) {
        nexus_set_compute_device(engine, device_name.c_str());
    }
    std::cout << "[Compute Engine] Active Compute Backend : " << nexus_get_active_device_name(engine) << std::endl;
    std::cout << "----------------------------------------------------------------" << std::endl;

    if (!model_path.empty()) {
        std::cout << "[Model Loading] Loading model from: " << model_path << " ..." << std::endl;
        int32_t status = nexus_load_model_auto(engine, model_path.c_str());
        if (status != 0) {
            std::cerr << "[Model Loading Error] Failed to load model file (Code: " << status << ")" << std::endl;
            nexus_destroy_engine(engine);
            return 1;
        }
        int32_t t_count = nexus_get_loaded_tensor_count(engine);
        const char* arch = nexus_get_model_architecture(engine);
        std::cout << "[Model Loading] Successfully loaded " << t_count << " parameters (" << arch << " arch) into Tier 1 Host RAM." << std::endl;
    } else {
        std::cout << "[Model Loading] Running on registered tiered model weights (Tier 1 Host RAM -> Tier 0 Active DMA)." << std::endl;
    }

    NexusLocalMemoryStats initial_stats;
    nexus_get_memory_stats(engine, &initial_stats);
    std::cout << "[Nexus Engine] Tier 0 (Active Working Set): " << (tier0_bytes / (1024 * 1024)) << " MB" << std::endl;
    std::cout << "[Nexus Engine] Tier 1 (Host RAM Residency): " << (tier1_bytes / (1024 * 1024)) << " MB" << std::endl;
    std::cout << "[Model Weights] Parameter Residency in Host RAM: " << (initial_stats.tier1_inactive_bytes / (1024 * 1024)) << " MB" << std::endl;

    // Radix Prefix Cache Demonstration:
    // Warm up the Radix tree with a common system prompt prefix (e.g. system instructions)
    const size_t PREFIX_LEN = prompt_len / 2;
    std::vector<int32_t> system_prefix(PREFIX_LEN);
    for (size_t i = 0; i < PREFIX_LEN; ++i) {
        system_prefix[i] = static_cast<int32_t>(100 + (i % 250));
    }

    std::cout << "\n[Prefix Cache] Priming Radix tree with " << PREFIX_LEN << "-token shared system prompt..." << std::endl;
    nexus_submit_request(engine, "req-prime-prefix", system_prefix.data(), PREFIX_LEN, 1);
    int32_t prime_out = 0;
    nexus_step_continuous_batch(engine, nullptr, &prime_out, 1);

    // Submit batch of requests that share the common system prompt prefix
    std::cout << "[Benchmark] Submitting " << num_requests << " concurrent requests sharing prefix..." << std::endl;
    for (size_t r = 0; r < num_requests; ++r) {
        std::vector<int32_t> full_prompt = system_prefix;
        for (size_t i = PREFIX_LEN; i < prompt_len; ++i) {
            full_prompt.push_back(static_cast<int32_t>(300 + (r * 13 + i) % 500));
        }
        std::string req_id = "req-" + std::to_string(r);
        nexus_submit_request(engine, req_id.c_str(), full_prompt.data(), full_prompt.size(), gen_len);
    }

    // Run continuous batching decode loop
    std::cout << "[Benchmark] Executing continuous iteration-level decode loop..." << std::endl;
    auto t_start = std::chrono::high_resolution_clock::now();

    size_t total_tokens = 0;
    std::vector<int32_t> out_tokens(num_requests);

    for (size_t step = 0; step < gen_len + 10; ++step) {
        int32_t stepped = nexus_step_continuous_batch(engine, nullptr, out_tokens.data(), num_requests);
        total_tokens += stepped;
        if (stepped == 0 && step > 5) break; // Finished all
    }

    auto t_end = std::chrono::high_resolution_clock::now();
    double elapsed_sec = std::chrono::duration<double>(t_end - t_start).count();
    double tps = total_tokens / (elapsed_sec > 0.0 ? elapsed_sec : 1e-6);

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
