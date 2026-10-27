# NexusLLM: Heterogeneous Memory-Tiered LLM Inference Engine (Research Prototype)

NexusLLM is an experimental LLM inference engine written in **C++20** and **Python 3.12**, featuring native FFI bindings for **Rust** (`nexus-infer`) and **C#/.NET** (`NexusInfer`).

The core research objective of NexusLLM is investigating **Dynamic Host RAM Parameter Tiering** for consumer and edge hardware: keeping inactive transformer layers and Mixture-of-Experts (MoE) parameters in **Host System RAM (Tier 1)** while dynamically streaming the active working set into fast accelerator or pinned memory buffers **(Tier 0)**, alongside **PagedAttention** KV cache management and **Radix Prefix Caching**.

---

## Implementation Status & Roadmap

| Subsystem / Capability | Current Implementation Status | Notes |
| :--- | :--- | :--- |
| **Dynamic Memory Tiering** | **Operational** | Pinned Host RAM (`VirtualAlloc` on Win, `mmap`/THP on Linux, UMA zero-copy on macOS Apple Silicon). |
| **Double-Buffered Overlap (Phase 4)**| **Operational** | Ping-pong layer slots with asynchronous background worker prefetching layer $l+1$ while layer $l$ computes. |
| **Paged KV Cache Management** | **Operational** | Block allocator with on-demand allocation, reference counting, and Radix tree prefix caching. |
| **Transformer Forward Pass** | **Operational** | Complete mathematical execution: Token Embedding -> RMSNorm -> QKV GEMV -> RoPE -> PagedAttention Dot-Product & Softmax -> Wo -> RMSNorm -> SwiGLU FFN -> LM Head Logits -> Sampling. |
| **Continuous Batching Scheduler**| **Operational** | Chunked prefill scheduling and iteration-level decode loop in C ABI and Python runtime. |
| **Quantization & SIMD Kernels** | **Operational** | Fused quantized GEMV for GGUF Q4_0, Q8_0, FP16 via AVX2+FMA (x86_64) and ARM NEON (ARM64). |
| **Model Formats (Phase 3)** | **Operational** | Zero-copy binary memory mapping of `.gguf` v2/v3 files directly into Host RAM with full metadata extraction. |
| **Hardware Backends (Phase 5)** | **Operational & Extensible** | Multi-device abstraction: NVIDIA CUDA, Apple Silicon Metal (with [CuMetal](https://github.com/Lulzx/cuda-metal) translation layer), AMD ROCm/HIP, Intel Arc oneAPI/SYCL, CPU SIMD fallback. |
| **Distributed PD Disaggregation** | **Single-Node Prototype** | Queue-level decoupling of prefill and decode phases; multi-node RDMA network transport on roadmap. |

---

## Architecture & Core Design

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
- **Concept**: Rather than requiring the entirety of a model's weights to reside statically inside GPU VRAM, NexusLLM stages inactive parameters in **Host System RAM (Tier 1)**. Only the **active working set (Tier 0)** (the currently executing layer or top-k routed experts) is activated.
- **Asynchronous Pipeline**: A background worker thread prefetches upcoming layer parameters into double-buffered working slots to overlap memory transfers with forward pass execution.

### 2. Native Multi-Platform Acceleration
- **macOS (Apple Silicon M1–M4)**:
  - **Zero-Copy Unified Memory Architecture (UMA)**: CPU and GPU share the same physical DRAM pool. Inactive parameters transition to active execution with zero memcpy overhead via direct pointer aliasing.
  - **ARM NEON SIMD**: 128-bit vector dot-product acceleration (`vld1q_f32`, `vfmaq_f32`, `vaddvq_f32`) for INT4 and FP8 dequantization.
- **Linux (Ubuntu, Debian, RHEL, CentOS, Rocky, Fedora, Arch, Alpine)**:
  - **Transparent Huge Pages (THP 2MB / 1GB)**: Managed via `mmap(MAP_PRIVATE | MAP_ANONYMOUS)` and `madvise(MADV_HUGEPAGE)` to minimize Translation Lookaside Buffer (TLB) misses across large weight pools.
  - **Asynchronous Kernel Prefetch**: Employs `madvise(MADV_WILLNEED)` to warm page cache ahead of execution and `madvise(MADV_COLD)` for clean deactivation.
  - **Memory Locking (`mlock` / `mlock2`)**: Prevents host OS swapper from paging inactive weights out to disk swap space.
  - **musl libc Portability**: Clean POSIX fallback supporting lightweight Alpine Linux container microVMs.
- **Microsoft Windows (x86_64 NT Kernel)**:
  - Pinned memory allocation through `VirtualAlloc(MEM_COMMIT | MEM_RESERVE)` and `VirtualLock`.
  - AVX2 and FMA 256-bit SIMD matrix-vector kernels (`_mm256_fmadd_ps`).

### 3. Universal Model Loader
- **SafeTensors (`.safetensors`)**: Header deserialization with direct memory-mapped tensor slicing.
- **GGUF / GGML (`.gguf`)**: Binary parser for GGUF v2/v3 metadata, tensor tables, and quantized blocks (`Q4_0`, `Q4_K`, `Q8_0`, `F16`).
- **PyTorch Checkpoints (`.pt`, `.bin`)**: State dict tensor streams.
- **ONNX (`.onnx`)**: Graph node weights and tensor manifests.

### 4. Serving Capabilities
- **Radix Multi-Level Prefix Caching**: Automatically matches and reuses common prompt prefixes (system instructions, few-shot examples, multi-turn history) in $O(1)$ radix tree lookups.
- **Disaggregated Prefill and Decode (PD Pipeline)**: Decouples prefill scheduling from memory-bandwidth-bound decode across distinct execution queues.
- **Bounds-Checked KV Block Allocator**: On-demand allocation with reference counting and atomic cancellation to avoid memory leaks.

### 5. Transformer Forward Execution Pipeline & Speculative Decoding
- **Full Model Forward Loop Orchestration (`llama_eval` architecture)**:
  - **Token Embedding Lookup**: Retrieves dense representations with on-the-fly FP32/FP16/Q4_0/Q8_0 dequantization.
  - **RMSNorm (Root Mean Square Normalization)**: Numerically stable pre- and post-attention normalization (`rms_norm_eps`).
  - **Q, K, V Matrix-Vector GEMV Projections**: Multi-Head Attention (MHA) and Grouped-Query Attention (GQA) head projections via SIMD quantized kernels (`fused_gemv_q4_0`, `fused_gemv_q8_0`, `fused_gemv_fp16`).
  - **Rotary Position Embeddings (RoPE)**: High-precision frequency rotation applied to Query and Key heads across arbitrary context windows.
  - **FlashAttention CPU Tiled Online Softmax ([Dao et al., 2022](https://arxiv.org/abs/2205.14135) / [vLLM](https://arxiv.org/abs/2309.06180) §4.2)**: Cache-tiled attention over physical `KVBlock` pointers with running max $m$, running sum $l$, and accumulator $O$. Operates with $O(1)$ workspace (zero dynamic heap allocations in inner decode loop) and AVX2/NEON fused multiply-accumulate vectorization.
  - **Attention Output Projection & Residual Connection**: Projects multi-head output through $W_o$ and accumulates into residual stream.
  - **SwiGLU Feed-Forward Network**: SiLU-gated up-projection multiplied element-wise and projected down (`w_down(SiLU(w_gate(x)) * w_up(x))`).
  - **Final RMSNorm & LM Head Logits Projection**: Dense matrix-vector projection into vocabulary space.
  - **Sampling**: Temperature-scaled categorical, top-k/top-p, and greedy argmax sampling.
- **Speculative Decoding Verification ([Medusa](https://arxiv.org/abs/2401.10774) / [EAGLE-2](https://arxiv.org/abs/2402.12567) / Multi-Token Prediction)**:
  - Proposes $K$ candidate draft tokens via MTP/Medusa auxiliary heads.
  - Verifies candidate draft chains against target model forward pass; accepts matching tokens sequentially and emits target model correction on first mismatch.
  - Executes $O(1)$ physical KV cache sequence rollback (`rollback_sequence`) on rejection, reclaiming discarded tokens without memory leaks.
- **Continuous Batching Interleaving ([Sarathi-Serve](https://arxiv.org/abs/2308.16369) / Orca)**:
  - Interleaves active single-token decodes with chunked prefill within a single unified iteration token budget, eliminating Time-Between-Tokens (TBT) jitter.
- **Dynamic Asynchronous Layer Prefetching**: Layer $l+1$ weights are proactively transferred into Tier 0 working buffers while layer $l$ is executing via alternating double-buffered ping-pong working slots.

### 6. Hardware Accelerator Backends (Phase 5)
NexusLLM features a modular hardware device abstraction (`INexusComputeBackend` and `BackendRegistry`) providing runtime hardware discovery and unified kernel dispatch across all accelerator ecosystems with zero failing backends:
- **NVIDIA CUDA**: Probes CUDA driver dynamically (`nvcuda.dll` / `libcuda.so`), sets up CUDA streams, asynchronous device allocation, Tensor Core GEMV dispatch, or activates the CUDA software emulation bridge on non-NVIDIA hosts.
- **Apple Silicon Metal**: Native Metal 3 on Apple Silicon M-series (M1-M4) Unified Memory Architecture (zero-copy DRAM access). Integrates the [CuMetal (Lulzx/cuda-metal)](https://github.com/Lulzx/cuda-metal) cross-platform translation bridge, allowing CUDA kernel code and PTX to transpile directly into Metal Shading Language (MSL) compute pipelines, with UMA zero-copy emulation across Windows and Linux.
- **AMD ROCm / HIP**: Probes AMD HIP runtime (`amdhip64.dll` / `libamdhip64.so`), AMD OpenCL compute runtime (supporting discrete Radeon and integrated RDNA APUs like `gfx1103`), or activates the HIP-CPU emulation bridge.
- **Intel Arc oneAPI / SYCL**: Probes Intel Level Zero (`ze_loader.dll` / `libze_loader.so`), Intel Graphics Compute runtime via OpenCL ICD (for Intel Arc A-Series, Battlemage, and Iris Xe), or activates the Intel oneAPI DPC++ / SYCL USM emulation bridge.
- **Universal OpenCL**: Universal cross-vendor GPU accelerator backend dynamically binding to OpenCL 3.0 (`OpenCL.dll` / `libOpenCL.so.1` / `OpenCL.framework`), providing immediate hardware GPU acceleration on any system without proprietary SDK installations.
- **CPU SIMD Reference Engine**: Highly optimized AVX2+FMA (x86_64) and ARM NEON (AArch64) fused quantized GEMV and PagedAttention kernels.

---

## Directory Structure

```
nexus_engine/
├── include/
│   ├── nexus_common.hpp           # Platform detection, aligned types, hardware profiles
│   ├── nexus_backend.hpp          # Hardware compute backends (CUDA, Metal, ROCm, SYCL, CPU)
│   ├── memory_tier.hpp            # Tier 0 (Active) vs Tier 1 (Host RAM) manager
│   ├── model_loader.hpp           # SafeTensors, GGUF v2/v3 zero-copy mmap parsers
│   ├── paged_cache.hpp            # PagedAttention KV pool & Radix prefix tree
│   ├── quantization.hpp           # SIMD dequantization & fused GEMV kernels
│   ├── moe_router.hpp             # Dynamic MoE expert streaming & top-k routing
│   ├── transformer_ops.hpp        # RMSNorm, RoPE, PagedAttention MHA/GQA, SwiGLU, Softmax, GEMV
│   ├── transformer_engine.hpp     # Full transformer forward pass pipeline with async prefetch
│   ├── disaggregated_pipeline.hpp # Prefill-Decode disaggregation control plane
│   └── nexus_abi.h                # Clean C ABI exports
├── src/
│   ├── memory_tier.cpp            # UMA zero-copy, Linux THP mmap, and VirtualAlloc
│   ├── nexus_backend.cpp          # Backend implementations & dynamic device registry
│   ├── model_loader.cpp           # SafeTensors & GGUF zero-copy memory-mapped loader
│   ├── paged_cache.cpp            # Radix prefix tree matching & block allocator
│   ├── quantization.cpp           # Vectorized ARM NEON & AVX2/FMA fused GEMV
│   ├── moe_router.cpp             # Top-k expert streaming & softmax routing
│   ├── transformer_engine.cpp     # End-to-end forward pass execution & token sampling
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
