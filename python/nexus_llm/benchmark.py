"""
Live Hardware Benchmark Suite:
Measures real NexusLLM continuous batching throughput, iteration latency,
prefix caching efficiency, and tiered memory allocation on the local host.
"""

import time
import statistics
from typing import Dict, Any, List
from .bindings import NexusNativeEngine
from .memory_manager import DynamicMemoryTierManager
from .prefix_cache import RadixPrefixCache
from .continuous_batcher import ContinuousBatchEngine

def run_benchmark(concurrency: int = 16, prompt_len: int = 128, gen_len: int = 64) -> Dict[str, Any]:
    print("=====================================================================")
    print(f"   NEXUS-LLM LIVE BENCHMARK: Concurrency={concurrency}, Prompt={prompt_len}, Gen={gen_len}")
    print("=====================================================================")

    # Initialize Nexus Engine
    native_eng = NexusNativeEngine(tier0_mb=512, tier1_mb=8192)
    caps = native_eng.get_platform_capabilities()

    print(f" Detected OS            : {caps.os_name}")
    print(f" CPU Architecture       : {caps.cpu_arch}")
    print(f" Vector SIMD Engine     : {caps.vector_isa}")
    print(f" Unified Memory (UMA)   : {'YES (Zero-Copy)' if caps.is_unified_memory else 'NO (PCIe Tiered)'}")
    print(f" Transparent HugePages  : {'YES' if caps.has_transparent_hugepages else 'NO'}")
    print(f" Hardware Concurrency   : {caps.hardware_concurrency} threads")
    print(f" Theoretical Peak BW    : {caps.theoretical_bandwidth_gbps} GB/s")
    print(f" Platform Profile       : {caps.performance_tier_desc}")
    print("---------------------------------------------------------------------")

    mem_mgr = DynamicMemoryTierManager(tier0_capacity_mb=512, tier1_capacity_mb=8192)
    p_cache = RadixPrefixCache()
    batcher = ContinuousBatchEngine(
        native_engine=native_eng,
        prefix_cache=p_cache,
        memory_mgr=mem_mgr,
        speculative_draft_len=3
    )

    # Pre-populate prefix cache with common prompt prefix (half of prompt_len)
    prefix_tokens_count = min(64, prompt_len // 2)
    shared_prefix = [42 + (i % 100) for i in range(prefix_tokens_count)]
    p_cache.insert_prefix(shared_prefix, list(range(max(1, prefix_tokens_count // 16))))

    # Submit batch of concurrent requests
    req_ids = []
    for r in range(concurrency):
        user_suffix = [100 + (r * 7 + i) % 500 for i in range(prompt_len - prefix_tokens_count)]
        full_tokens = shared_prefix + user_suffix
        rid = batcher.add_request(full_tokens, max_new_tokens=gen_len)
        req_ids.append(rid)

    # Execute iteration-level decode loop
    t_start = time.perf_counter()
    step_count = 0
    while len(batcher.finished_requests) < concurrency:
        batcher.step()
        step_count += 1
        if step_count > (gen_len + 50):
            break

    t_end = time.perf_counter()
    total_time_s = max(t_end - t_start, 1e-6)
    total_tokens = sum(len(r.generated_tokens) for r in batcher.finished_requests.values())
    tps = total_tokens / total_time_s

    ttfts = [
        (r.first_token_time - r.start_time) * 1000.0
        for r in batcher.finished_requests.values()
        if r.first_token_time is not None
    ]
    tpots = [
        ((r.finish_time - r.first_token_time) / max(1, len(r.generated_tokens) - 1)) * 1000.0
        for r in batcher.finished_requests.values()
        if r.finish_time is not None and r.first_token_time is not None
    ]

    mean_ttft = statistics.mean(ttfts) if ttfts else 0.0
    mean_tpot = statistics.mean(tpots) if tpots else 0.0

    mem_stats = mem_mgr.get_stats()

    results = {
        "platform": {
            "os_name": caps.os_name,
            "cpu_arch": caps.cpu_arch,
            "vector_isa": caps.vector_isa,
            "is_unified_memory": caps.is_unified_memory,
            "has_transparent_hugepages": caps.has_transparent_hugepages,
            "theoretical_bandwidth_gbps": caps.theoretical_bandwidth_gbps
        },
        "workload": {
            "concurrent_requests": concurrency,
            "prompt_length_tokens": prompt_len,
            "generated_tokens_per_req": gen_len,
            "total_output_tokens": total_tokens,
            "total_execution_time_s": round(total_time_s, 4),
            "step_count": step_count
        },
        "metrics": {
            "throughput_tps": round(tps, 2),
            "mean_ttft_ms": round(mean_ttft, 2),
            "mean_tpot_ms": round(mean_tpot, 2),
            "prefix_cache_hit_rate_pct": round(p_cache.get_hit_rate() * 100.0, 1),
            "active_tier0_working_set_mb": mem_stats["tier0_allocated_mb"],
            "host_ram_inactive_weights_mb": mem_stats["tier1_inactive_mb"],
            "speculative_speedup_factor": batcher.get_throughput_stats()["speculative_speedup_factor"]
        }
    }

    print("\n------------------------- BENCHMARK SUMMARY -------------------------")
    print(f" Measured Engine Throughput     : {results['metrics']['throughput_tps']} tokens/sec (TPS)")
    print(f" Mean Time to First Token (TTFT): {results['metrics']['mean_ttft_ms']} ms")
    print(f" Mean Time Per Output Tok (TPOT): {results['metrics']['mean_tpot_ms']} ms")
    print(f" Active Tier 0 Working Set      : {results['metrics']['active_tier0_working_set_mb']} MB")
    print(f" Inactive Host RAM Residency    : {results['metrics']['host_ram_inactive_weights_mb']} MB")
    print(f" Radix Prefix Cache Hit Rate    : {results['metrics']['prefix_cache_hit_rate_pct']} %")
    print(f" Speculative Speedup Ratio      : {results['metrics']['speculative_speedup_factor']}x")
    print("---------------------------------------------------------------------")

    native_eng.close()
    mem_mgr.close()
    return results

if __name__ == "__main__":
    run_benchmark(concurrency=16, prompt_len=128, gen_len=64)
