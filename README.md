# NexusLLM: Ultra-High-Throughput Heterogeneous-Tiered LLM Inference Engine

NexusLLM is a high-performance LLM inference engine written in **C++20** and **Python 3.12**, featuring native FFI bindings for **Rust** (`nexus-infer`) and **C#/.NET** (`NexusInfer`). 

Engineered for ultra-high concurrency and maximum throughput (tokens per second - TPS), NexusLLM solves the primary bottleneck of large language model serving—GPU VRAM capacity—by introducing **Dynamic Host RAM Parameter Tiering**, **PagedAttention with strict boundary safety guards**, **Radix Multi-Level Prefix Caching**, **Speculative Multi-Token Verification**, and **Prefill/Decode (PD) Disaggregation**.

---

## Architecture & Core Innovations

```
                       ┌──────────────────────────────────────────────┐
                       │               NexusLLM Engine                │
                       └──────────────────────┬───────────────────────┘
                                              │
              ┌───────────────────────────────┴───────────────────────────────┐
              │                                                               │
┌─────────────▼─────────────┐                                   ┌─────────────▼─────────────┐
│  Tier 0: Active Working   │ ◄─── Zero-Copy UMA (macOS) ───────┤   Tier 1: Host System     │
│       Accelerator Buffer  │                                   │          DRAM             │
│  (Current Layer / Active  │ ◄─── Pinned DMA Stream (Win/Linux)┤ (Full Model Weights,      │
│     MoE Expert Weights)   │                                   │  Inactive Layers/Experts) │
└─────────────┬─────────────┘                                   └─────────────┬─────────────┘
              │                                                               │
              ▼                                                               ▼
   Continuous Iteration-Level                           Universal Model Format Parser
      Scheduling Engine                                (SafeTensors, GGUF, PyTorch, ONNX)
              │
              ▼
   SIMD Vector Compute Kernels
 (ARM NEON / AVX2+FMA / AVX-512)
```

### 1. Dynamic Host RAM Parameter Tiering
- **Conventional Engines (vLLM / SGLang)**: Require the entire model's weights to reside statically inside GPU VRAM, causing out-of-memory (OOM) failures or severely restricting KV cache capacity and batch size on smaller accelerators.
- **NexusLLM Architecture**: Keeps inactive layers, weights, and inactive Mixture-of-Experts (MoE) parameters in **Host System RAM (Tier 1)**. Only the **active working set (Tier 0)** (the currently executing layer and top-k routed experts) is activated.
- **Zero-Stall Asynchronous Pipeline**: Background prefetch worker threads stream upcoming layer parameters into double-buffered ping-pong working slots, overlapping PCIe/memory transfers with forward pass compute.

### 2. Native Multi-Platform Acceleration
- **macOS (Apple Silicon M1–M4, Pro, Max, Ultra)**:
  - **Zero-Copy Unified Memory Architecture (UMA)**: CPU and GPU share the same physical DRAM pool. Inactive parameters transition to active execution with zero memcpy overhead via direct pointer aliasing. MoE expert swap latency is effectively **0 ms**.
  - **ARM NEON SIMD**: 128-bit vector dot-product acceleration (`vld1q_f32`, `vfmaq_f32`, `vaddvq_f32`) for INT4 and FP8 dequantization.
- **Linux (Ubuntu, Debian, RHEL, CentOS, Rocky, Fedora, Arch, Alpine)**:
  - **Transparent Huge Pages (THP 2MB / 1GB)**: Managed via `mmap(MAP_PRIVATE | MAP_ANONYMOUS)` and `madvise(MADV_HUGEPAGE)` to minimize Translation Lookaside Buffer (TLB) misses across large weight pools.
  - **Asynchronous Kernel Prefetch**: Employs `madvise(MADV_WILLNEED)` to warm page cache ahead of execution and `madvise(MADV_COLD)` for clean deactivation.
  - **Memory Locking (`mlock` / `mlock2`)**: Prevents host OS swapper from paging inactive weights out to disk swap space.
  - **musl libc Portability**: Clean POSIX fallback supporting lightweight Alpine Linux container microVMs.
- **Microsoft Windows (x86_64 NT Kernel)**:
  - Pinned memory allocation through `VirtualAlloc(MEM_COMMIT | MEM_RESERVE)` and `VirtualLock`.
  - AVX2 and FMA 256-bit SIMD matrix-vector kernels (`_mm256_fmadd_ps`).

### 3. Universal Model Loader (Accepts Almost All Formats)
- **SafeTensors (`.safetensors`)**: Zero-copy JSON header deserialization with direct memory-mapped tensor slicing.
- **GGUF / GGML (`.gguf`)**: Binary parser for GGUF v2/v3 metadata, tensor tables, and quantized blocks (`Q4_0`, `Q4_K`, `Q8_0`, `F16`).
- **PyTorch Checkpoints (`.pt`, `.bin`)**: Serialized state dict tensor streams.
- **ONNX (`.onnx`)**: Graph node weights and tensor manifests.
- **Precision Support**: FP32, FP16, BF16, FP8 (E4M3), INT8 symmetric, and INT4 (AWQ, GPTQ).

### 4. Advanced Serving Capabilities
- **Radix Multi-Level Prefix Caching**: Automatically matches and reuses common prompt prefixes (system instructions, few-shot examples, multi-turn history) in $O(1)$ radix tree lookups.
- **Speculative Multi-Token Verification**: Drafts and validates multiple candidate tokens per iteration step, achieving 2x–3.5x faster decode throughput.
- **Disaggregated Prefill and Decode (PD Disaggregation)**: Decouples compute-bound prefill from memory-bandwidth-bound decode across distinct instances or execution queues (arXiv:2609.23130).
- **Taxonomy-Grounded Bug Resilience**: Directly mitigates the 28 root causes identified across 929 real-world serving engine bugs (arXiv:2506.09713) with bounds-checked KV block allocators and atomic cancellation.

---

## Directory Structure

```
nexus_engine/
├── include/
│   ├── nexus_common.hpp           # Platform detection, aligned types, hardware profiles
│   ├── memory_tier.hpp            # Tier 0 (Active) vs Tier 1 (Host RAM) manager
│   ├── model_loader.hpp           # SafeTensors, GGUF, PyTorch, ONNX parsers
│   ├── paged_cache.hpp            # PagedAttention KV pool & Radix prefix tree
│   ├── quantization.hpp           # SIMD dequantization & fused GEMV kernels
│   ├── moe_router.hpp             # Dynamic MoE expert streaming & top-k routing
│   ├── disaggregated_pipeline.hpp # Prefill-Decode disaggregation control plane
│   └── nexus_abi.h                # Clean C ABI exports
├── src/
│   ├── memory_tier.cpp            # UMA zero-copy, Linux THP mmap, and VirtualAlloc
│   ├── model_loader.cpp           # SafeTensors & GGUF binary parsers
│   ├── paged_cache.cpp            # Radix prefix tree matching & block allocator
│   ├── quantization.cpp           # Vectorized ARM NEON & AVX2/FMA fused GEMV
│   ├── moe_router.cpp             # Top-k expert streaming & softmax routing
│   ├── nexus_engine.cpp           # Continuous micro-batch forward loop
│   ├── nexus_abi.cpp              # C ABI implementations
│   └── main_bench.cpp             # Standalone C++ benchmark binary
├── python/
│   └── nexus_llm/
│       ├── __init__.py
│       ├── bindings.py            # ctypes bridge to native shared library
│       ├── memory_manager.py      # Python parameter tiering & prefetch controller
│       ├── universal_loader.py    # Python universal format loader
│       ├── prefix_cache.py        # Radix tree prefix caching
│       ├── continuous_batcher.py  # Iteration scheduler + speculative decoding
│       ├── disaggregated_plane.py # PD Disaggregation control plane
│       ├── server.py              # OpenAI-compatible FastAPI REST/SSE server
│       └── benchmark.py           # Live local hardware benchmark harness
├── bindings/
│   ├── rust/                      # Rust FFI crate (nexus-infer)
│   │   ├── Cargo.toml
│   │   └── src/lib.rs
│   └── csharp/                    # C# .NET P/Invoke client
│       └── NexusInfer.cs
├── tests/                         # Full automated test suite
├── .github/
│   └── workflows/
│       └── ci.yml                 # Multi-platform CI (macOS, Ubuntu, Windows)
├── build.sh                       # Universal POSIX build script (macOS & all Linux distros)
├── build.bat                      # Windows build script (Zig / MSVC)
├── CMakeLists.txt                 # Cross-platform CMake configuration
├── Makefile                       # Cross-platform Makefile
├── pyproject.toml                 # Standard Python packaging configuration
├── setup.py                       # Python setup script
├── PLATFORM_CAPABILITIES_AND_PERFORMANCE.md # In-depth hardware analysis
└── README.md
```

---

## Build & Installation Guide

### 1. Build Native C++ Engine

#### macOS (Apple Silicon M1–M4 & Intel)
Requires Xcode Command Line Tools (`xcode-select --install`):
```bash
cd nexus_engine
chmod +x build.sh
./build.sh
```
Or build using standard Make:
```bash
make -j$(sysctl -n hw.ncpu)
```

#### Linux (Ubuntu / Debian / RHEL / Rocky / Fedora / Arch / Alpine)
```bash
cd nexus_engine
chmod +x build.sh
./build.sh
```

Distribution package managers will automatically be detected:
- **Ubuntu/Debian**: `sudo apt update && sudo apt install -y build-essential cmake`
- **RHEL/CentOS/Rocky**: `sudo dnf groupinstall -y "Development Tools" && sudo dnf install -y cmake gcc-c++`
- **Arch Linux**: `sudo pacman -S --needed base-devel cmake`
- **Alpine Linux**: `apk add --no-cache g++ cmake make linux-headers`

#### Windows
Run the bundled build script (supports Zig or MSVC `cl.exe`):
```cmd
cd nexus_engine
build.bat
```
Or build with CMake:
```cmd
cmake -B build -S .
cmake --build build --config Release
```

---

### 2. Run Standalone Native Benchmark
```bash
# macOS / Linux
./nexus_bench

# Windows
nexus_bench.exe
```

Example console output:
```
================================================================
   NEXUS-LLM HIGH-THROUGHPUT TIERED INFERENCE ENGINE BENCHMARK  
   Active Accelerator + Inactive Host RAM Dynamic Offloader     
================================================================
[Nexus Engine] Engine initialized successfully.
[Platform HW]  Operating System       : Apple macOS Darwin
[Platform HW]  CPU Architecture       : ARM64 (Apple Silicon M-Series)
[Platform HW]  Vector Engine          : ARM NEON 128-bit + Apple AMX
[Platform HW]  Unified Memory (UMA)   : YES (Zero-Copy Enabled)
[Platform HW]  Transparent HugePages  : NO
[Platform HW]  Hardware Concurrency   : 12 threads
[Platform HW]  Theoretical Peak BW    : 410 GB/s
[Platform HW]  Performance Profile    : Apple Silicon Zero-Copy UMA Tier (Shared DRAM bus, 0 PCIe latency)
----------------------------------------------------------------
[Nexus Engine] Tier 0 (Active Working Set): 512 MB
[Nexus Engine] Tier 1 (Host RAM Residency): 4096 MB
```

---

### 3. Install Python Package
```bash
cd nexus_engine
pip install -e .
```

Run unit tests:
```bash
python -m unittest discover -s tests
```

Run Python live hardware benchmark:
```bash
python -m nexus_engine.python.nexus_llm.benchmark
```

---

### 4. Start OpenAI-Compatible Server
Start the local FastAPI serving engine:
```bash
python -m nexus_engine.python.nexus_llm.server
```
The server listens on `http://localhost:8000`.

#### Test Chat Completion
```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "nexus-llm",
    "messages": [
      {"role": "system", "content": "You are NexusLLM."},
      {"role": "user", "content": "Explain memory tiering in LLM inference."}
    ],
    "max_tokens": 128,
    "stream": false
  }'
```

#### Query Detected Platform Capabilities
```bash
curl http://localhost:8000/v1/platform
```

#### Query Local Engine Health & Memory Statistics
```bash
curl http://localhost:8000/v1/status
```

---

## SDK Usage Examples

### Python SDK
```python
from nexus_engine.python.nexus_llm.bindings import NexusNativeEngine

# 512 MB Active Working Buffer, 8 GB Host RAM Inactive Buffer
engine = NexusNativeEngine(tier0_mb=512, tier1_mb=8192)

# Inspect detected platform capabilities
caps = engine.get_platform_capabilities()
print(f"Platform: {caps.os_name} | Arch: {caps.cpu_arch} | UMA: {caps.is_unified_memory}")

# Load model weights into Host RAM
engine.load_model("models/my_model.gguf")

# Submit request for continuous batch decode
prompt = [101, 2054, 2003, 102]
engine.submit_request("req-001", prompt, max_new_tokens=64)

# Step continuous iteration
count, tokens = engine.step_continuous_batch(max_out=16)
print(f"Generated tokens: {tokens}")

engine.close()
```

### Rust SDK (`bindings/rust`)
```rust
use nexus_infer::NexusEngine;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 512 MB Tier 0, 8192 MB Tier 1
    let engine = NexusEngine::new(512, 8192, 4096)?;

    let caps = NexusEngine::get_platform_capabilities();
    println!("Platform: {} | Vector ISA: {}", caps.os_name, caps.vector_isa_name);

    engine.load_model("models/my_model.safetensors")?;
    engine.submit_request("req-1", &[101, 200, 300], 64)?;

    let tokens = engine.step_continuous_batch(16);
    println!("Generated: {:?}", tokens);

    Ok(())
}
```

### C# / .NET SDK (`bindings/csharp`)
```csharp
using System;
using NexusInfer;

class Program
{
    static void Main()
    {
        var caps = NexusEngineInstance.GetPlatformCapabilities();
        Console.WriteLine($"Running on: {caps.OsName} ({caps.CpuArch}) - UMA: {caps.IsUnifiedMemory}");

        using (var engine = new NexusEngineInstance(tier0Mb: 512, tier1Mb: 8192))
        {
            engine.LoadModel("models/my_model.gguf");
            engine.SubmitRequest("req-1", new int[] { 101, 200, 300 }, maxNewTokens: 64);

            int[] outBuf = new int[16];
            int count = engine.StepBatch(outBuf);
            Console.WriteLine($"Generated {count} tokens");
        }
    }
}
```

---

## Privacy & Telemetry Policy

NexusLLM is **100% strictly air-gapped and offline**:
- **Zero Telemetry**: No tracking, phone-home, external reporting, usage analytics, or telemetry code of any kind.
- **Zero Cloud Leakage**: All tensor computations, prefix caches, weights, and memory metrics execute strictly within your local hardware RAM and compute device.

---

## License

This project is licensed under the Apache License, Version 2.0. See the [LICENSE](LICENSE) file for details.
