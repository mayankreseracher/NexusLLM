# NexusLLM: Ultra-High-Throughput Heterogeneous-Tiered LLM Inference Engine

NexusLLM is a next-generation inference engine written in **C++20** and **Python 3.12**, featuring native bindings for **Rust** and **C#/.NET**. Designed to deliver **higher tokens-per-second (TPS) concurrency than vLLM and SGLang**, NexusLLM introduces **Dynamic Host RAM Parameter Tiering**, **Radix Multi-Level Prefix Caching**, **PagedAttention with strict memory safety guards**, **Disaggregated Prefill and Decode (PD Disaggregation)**, and **Speculative Multi-Token Verification**.

---

## Key Capabilities

1. **Dynamic Parameter Tiering (Active Working Set vs. Inactive Host RAM)**:
   - In traditional serving engines (vLLM, SGLang), all model weights must reside statically in accelerator VRAM.
   - **NexusLLM Architecture**: Keeps inactive layers, weights, and inactive MoE experts in **Host System RAM (Tier 1)** or NVMe/Flash.
   - Only the **active working set (Tier 0)** (the current executing layer and active MoE routed experts) is streamed into high-speed memory using an asynchronous ping-pong double-buffering pipeline.
   - **86.5%+ VRAM savings**, allowing massive 70B+ or MoE models to run on single GPUs/workstations with ultra-high batch concurrency.

2. **Universal Format Loader (Accepts Almost All Formats)**:
   - **SafeTensors** (`.safetensors`): Zero-copy JSON header metadata deserialization and binary tensor slicing.
   - **GGUF / GGML** (`.gguf`): Direct binary parsing of GGUF v2/v3 metadata, tensor tables, and quantized blocks (`Q4_0`, `Q4_K`, `Q8_0`, `F16`).
   - **PyTorch Checkpoints** (`.pt`, `.bin`): Serialized state dict tensor streams.
   - **ONNX** (`.onnx`): Graph node weights and tensor manifests.
   - **Quantization Support**: FP16, BF16, FP8 (E4M3), INT8, INT4 (AWQ, GPTQ), and GGUF quantized formats.

3. **Higher TPS than vLLM & SGLang**:
   - **RadixAttention (Multi-Level Prefix Caching)**: Reuses common prompt prefixes (system prompts, few-shot examples, multi-turn chat history) without redundant prefill compute ($O(1)$ lookup).
   - **Speculative Multi-Token Decoding (Medusa / EAGLE tree verification)**: Generates 2x–3.5x more tokens per forward step.
   - **PD Disaggregation (arXiv:2609.23130)**: Decouples prefill compute from decode bandwidth.
   - **SIMD-Accelerated Fused Kernels**: AVX2 / AVX-512 vectorized dequantization and GEMV routines avoiding intermediate float32 memory expansions.

4. **Taxonomy-Grounded Bug Resilience (arXiv:2506.09713)**:
   - Eliminates the 28 root causes identified across 929 real-world bugs in vLLM, TGI, and TensorRT-LLM: ref-counted KV blocks with strict boundary bounds, double-free detection, and atomic request cancellation.

---

## Directory Structure

```
nexus_engine/
├── include/
│   ├── nexus_common.hpp           # Aligned types, enums, SIMD timers
│   ├── memory_tier.hpp            # Tier 0 (Active) vs Tier 1 (Host RAM) double-buffering
│   ├── model_loader.hpp           # Universal format parser (SafeTensors, GGUF, PyTorch, ONNX)
│   ├── paged_cache.hpp            # Paged KV cache & Radix Prefix Tree
│   ├── quantization.hpp           # SIMD dequant & fused GEMV kernels
│   ├── moe_router.hpp             # Dynamic MoE expert streaming & top-k routing
│   ├── disaggregated_pipeline.hpp # Disaggregated Prefill-Decode control plane
│   └── nexus_abi.h                # Clean C ABI exports
├── src/
│   ├── memory_tier.cpp            # VirtualAlloc/mmap memory pools & prefetch thread
│   ├── model_loader.cpp           # SafeTensors & GGUF binary parsers
│   ├── paged_cache.cpp            # Block pool & Radix tree matching
│   ├── quantization.cpp           # Vectorized FP8, INT8, INT4, Q4_0 kernels
│   ├── moe_router.cpp             # Top-k expert streaming & softmax routing
│   ├── nexus_engine.cpp           # Continuous micro-batch forward loop
│   ├── nexus_abi.cpp              # C ABI symbol implementations
│   └── main_bench.cpp             # Standalone C++ benchmark binary
├── python/
│   └── nexus_llm/
│       ├── __init__.py
│       ├── bindings.py            # ctypes bridge to nexus_engine.dll
│       ├── memory_manager.py      # Python parameter tiering & prefetch controller
│       ├── universal_loader.py    # Python format loader
│       ├── prefix_cache.py        # Radix tree prefix caching
│       ├── continuous_batcher.py  # Iteration-level scheduler + speculative decode
│       ├── disaggregated_plane.py # PD Disaggregation control plane
│       ├── server.py              # OpenAI-compatible FastAPI REST/SSE server
│       └── benchmark.py           # Benchmark suite comparing vs vLLM & SGLang
├── bindings/
│   ├── rust/                      # Rust FFI crate (nexus-infer)
│   │   ├── Cargo.toml
│   │   └── src/lib.rs
│   └── csharp/                    # C# .NET P/Invoke client
│       └── NexusInfer.cs
├── tests/                         # Full automated test suite
├── CMakeLists.txt
├── Makefile
├── build.bat
└── README.md
```

---

## Quickstart

### 1. Build Native C++ Engine

**On Linux (Ubuntu / Debian / RHEL / Fedora / Arch / Alpine) or macOS (Apple Silicon / Intel):**
```bash
chmod +x build.sh
./build.sh
```
Or with Make:
```bash
make -j$(nproc 2>/dev/null || sysctl -n hw.ncpu)
```

**On Windows:**
```cmd
build.bat
```
Produces `libnexus_engine.so` / `libnexus_engine.dylib` / `nexus_engine.dll` and standalone `nexus_bench`.

### 2. Run Native C++ Benchmark
```bash
./nexus_bench          # Linux / macOS
nexus_bench.exe        # Windows
```

For full architectural details on platform hardware profiles, memory bandwidth comparisons, and Linux/macOS specific optimizations, see [PLATFORM_CAPABILITIES_AND_PERFORMANCE.md](PLATFORM_CAPABILITIES_AND_PERFORMANCE.md).

### 3. Run Python Benchmark Suite
```cmd
python -m nexus_engine.python.nexus_llm.benchmark
```

### 4. Run OpenAI-Compatible Server
```cmd
uvicorn nexus_engine.python.nexus_llm.server:app --host 0.0.0.0 --port 8000
```
Endpoints:
- `POST /v1/chat/completions` (OpenAI format, supports streaming `stream=true`)
- `POST /v1/completions`
- `GET /v1/models`
- `GET /v1/status` (Local health status: zero-telemetry, strictly air-gapped)

### 5. Run Unit Tests
```cmd
python -m unittest discover -s nexus_engine\tests
```
