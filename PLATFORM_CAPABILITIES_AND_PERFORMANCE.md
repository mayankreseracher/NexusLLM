# NexusLLM: Multi-Platform Capabilities & Raw Performance Architecture

## Executive Overview

NexusLLM is architected to decouple active model parameters (working set in accelerator/L1 buffer) from inactive model parameters (residing in host system RAM or memory-mapped backing). Because memory subsystem topology, interconnect bandwidth, and CPU vector engines diverge drastically across operating systems and hardware architectures, NexusLLM incorporates **architecture-aware execution kernels and memory management strategies** tailored to the raw performance characteristics of each platform:

1. **macOS (Darwin ARM64 / Apple Silicon M1–M4, Pro, Max, Ultra)**:
   - **Unified Memory Architecture (UMA) Zero-Copy Streaming**: Eliminates PCIe interconnect transfer stalls entirely. Host RAM *is* device memory; inactive experts and weights transition to the active set via zero-copy atomic pointer alias with zero memcpy overhead.
   - **ARM NEON 128-bit Vector Acceleration**: Specialized SIMD routines for INT4 and FP8 dequantization and Fused GEMV (`vld1q_f32`, `vfmaq_f32`, `vaddvq_f32`).
2. **Linux Distributions (Ubuntu, Debian, RHEL, CentOS, Rocky, Fedora, Arch, Alpine)**:
   - **Transparent Huge Pages (THP 2MB / 1GB)**: Allocated via `mmap(MAP_PRIVATE | MAP_ANONYMOUS)` with `madvise(MADV_HUGEPAGE)` to eliminate TLB miss penalties across tens of gigabytes of inactive weights.
   - **Predictive Kernel Prefetching**: Uses `madvise(MADV_WILLNEED)` during asynchronous background prefetch to force kernel page table population before layer execution.
   - **Swapper Lockdown (`mlock` / `mlock2`)**: Prevents host OS memory managers from paging out inactive weights to swap space under memory pressure.
   - **musl libc Portability**: Clean POSIX fallback supporting lightweight Alpine Linux microVMs and containerized inference pods.
3. **Microsoft Windows (x86_64 NT Kernel)**:
   - **Double-Buffered Ping-Pong DMA Streaming**: Pinned virtual memory pools via `VirtualAlloc(MEM_COMMIT | MEM_RESERVE)` and asynchronous PCIe streaming overlapping compute with layer-level transfers.
   - **AVX2 + FMA Vector Acceleration**: 256-bit SIMD matrix-vector kernels (`_mm256_fmadd_ps`).

---

## 1. Platform Performance & Architectural Matrix

| Platform / Operating System | Processor Architecture | Memory Topology | Theoretical Peak Bandwidth | MoE Expert Swap Latency (28.7 MB expert) | Vector SIMD Engine | Key Kernel / OS Primitives |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **macOS (Apple Silicon M4 Max)** | ARM64 (aarch64) | Unified Memory (UMA) | **410 GB/s** | **~0.00 ms** (Zero-Copy Pointer Swap) | ARM NEON + AMX | POSIX `mmap`, `sysctlbyname`, Mach VM |
| **macOS (Apple Silicon M2/M3 Ultra)**| ARM64 (aarch64) | Unified Memory (UMA) | **800–820 GB/s**| **~0.00 ms** (Zero-Copy Pointer Swap) | ARM NEON + AMX | POSIX `mmap`, `sysctlbyname`, Mach VM |
| **macOS (Apple Silicon M1/M2/M3/M4)**| ARM64 (aarch64) | Unified Memory (UMA) | **100–150 GB/s**| **~0.00 ms** (Zero-Copy Pointer Swap) | ARM NEON | POSIX `mmap`, `sysctlbyname`, Mach VM |
| **Linux (Ubuntu / Debian Server)** | x86_64 / EPYC / Xeon | Discrete PCIe Gen 5 | **64 GB/s** (x16 Link) | **~0.45 ms** | AVX2 / AVX-512 | `mmap`, `madvise(MADV_HUGEPAGE)`, `madvise(MADV_WILLNEED)` |
| **Linux (RHEL / Rocky / CentOS)** | x86_64 / ARM Neoverse | Enterprise NUMA | **64–85 GB/s** (DDR5) | **~0.34–0.45 ms** | AVX-512 / ARM NEON | `mlock2(MLOCK_ONFAULT)`, NUMA node affinity, THP |
| **Linux (Arch Linux / Rolling)** | x86_64 | Desktop / Workstation | **64 GB/s** (PCIe Gen 5) | **~0.45 ms** | AVX-512 / AVX10 | Bleeding-edge GCC/Clang, low-latency kernel scheduler |
| **Linux (Alpine Linux / musl)** | x86_64 / aarch64 | Container / MicroVM | Varies by Host VM | **~0.45–1.20 ms** | NEON / AVX2 / Scalar | musl-safe POSIX `mmap`, lightweight 10MB container base |
| **Windows 11 / Server** | x86_64 | Discrete PCIe Gen 4 | **14–32 GB/s** | **~0.90–2.05 ms** | AVX2 + FMA | `VirtualAlloc`, `VirtualLock`, I/O Completion Ports |

---

## 2. Apple Silicon (macOS) Raw Performance Capabilities

### 2.1 The Zero-Copy UMA Advantage
On conventional discrete GPU architectures (x86 + PCIe + NVIDIA GPU), offloading inactive model parameters to host RAM requires transferring weights across the PCIe bus before every layer or expert execution. Even over PCIe Gen 4 x16 (32 GB/s) or PCIe Gen 4 x8 (14 GB/s), streaming tens of gigabytes introduces significant transfer stalls.

On **macOS Apple Silicon (M1 through M4)**, the CPU, GPU, and Neural Engine share a single physically unified pool of high-bandwidth LPDDR5/LPDDR5X DRAM:
- When NexusLLM detects `is_unified_memory = true`, the engine engages the **Zero-Copy Direct Alias Path**.
- The `activate_parameter()` call does not allocate working scratch buffers or perform `memcpy`. Instead, the active pointer is bound directly to the host pointer:
  ```cpp
  if (platform_profile_.is_unified_memory) {
      tv->active_ptr = tv->host_ptr;
      tv->is_active = true;
      tv->current_tier = MemoryTier::TIER0_ACTIVE_WORKING_SET;
      return tv->active_ptr;
  }
  ```
- **Result**: MoE expert swapping latency is effectively **0 microseconds**, allowing models up to 128 GB or 192 GB (e.g. on M2/M3/M4 Max/Ultra) to execute with 0 PCIe bottleneck.

### 2.2 ARM NEON Vector Kernels
For CPU-based prefill and decode fallbacks, NexusLLM replaces x86 AVX intrinsics with native ARM NEON SIMD routines:
- Quantized GGUF Q4_0 dequantization unpacks 4-bit nibbles directly into 32-bit float vector registers using `vmulq_f32` and `vld1q_f32`.
- Fused INT4 GEMV computes dot-products using fused multiply-accumulate `vfmaq_f32(v_acc, vw, vx)` and collapses the accumulator with single-instruction horizontal reduction `vaddvq_f32`.

---

## 3. Linux Distribution Specific Capabilities

### 3.1 Transparent Huge Pages (THP) & madvise
During LLM inference with tiered memory, standard 4 KB OS virtual memory pages result in millions of page table entries. Stride-heavy weight activations (e.g. gathering MoE expert matrices) cause severe Translation Lookaside Buffer (TLB) thrashing.

On all supported Linux distros, NexusLLM automatically leverages:
1. **2 MB / 1 GB Transparent Huge Pages**:
   ```cpp
   #if defined(MADV_HUGEPAGE)
   madvise(tier0_raw_pool_, tier0_capacity_, MADV_HUGEPAGE);
   madvise(tier1_raw_pool_, tier1_capacity_, MADV_HUGEPAGE);
   #endif
   ```
   This reduces TLB entries by a factor of 512, ensuring deterministic memory latency.
2. **Predictive Asynchronous Kernel Prefetching (`MADV_WILLNEED`)**:
   When `prefetch_parameter_async()` is scheduled, NexusLLM notifies the Linux kernel page cache:
   ```cpp
   #if defined(MADV_WILLNEED)
   madvise(tv->host_ptr, tv->byte_size, MADV_WILLNEED);
   #endif
   ```
   The kernel faults the physical pages into RAM in advance, overlapping disk/page I/O with current layer GPU compute.
3. **Eviction Guidance (`MADV_COLD`)**:
   Deactivated parameters in Tier 1 RAM are marked with `MADV_COLD`, signaling to the kernel that the pages can be reclaimed if system RAM is exhausted without dropping the virtual memory mapping.

### 3.2 Distribution-Specific Profiles

- **Ubuntu & Debian**:
  Default cloud server profile. Optimally tunes continuous batching and integrates with standard `systemd` CPU and memory cgroups.
- **RHEL, CentOS, Rocky Linux & AlmaLinux**:
  Enterprise enterprise clusters with NUMA awareness. NexusLLM supports memory locking via `mlock` to ensure mission-critical LLM parameters are never evicted to swap partitions.
- **Arch Linux & Manjaro**:
  Leverages bleeding-edge toolchains (GCC 14+, Clang 18+) with full `-march=native` support, auto-activating AVX-512, AMX, and AVX10.
- **Alpine Linux (musl libc)**:
  Fully adheres to POSIX standards without reliance on glibc-specific headers. Allows deploying NexusLLM inside ultralight Docker containers or Firecracker microVMs with sub-second cold starts.

---

## 4. Hardware Vector Engine Comparison

```
                         Vector Width    Fused Multiply-Add    Throughput / Clock
ARM NEON (Apple / ARM64) 128-bit (4xFP32) vfmaq_f32             High Energy Efficiency
Intel / AMD AVX2 + FMA   256-bit (8xFP32) _mm256_fmadd_ps       High Compute Density
Intel / AMD AVX-512      512-bit (16xFP32)_mm512_fmadd_ps       Maximum FLOPS
Scalar Fallback          32-bit (1xFP32)  w * x + sum           Universal Portability
```

---

## 5. Universal Build & Installation Guide

### 5.1 macOS (Apple Silicon & Intel)
Ensure Xcode Command Line Tools are installed:
```bash
xcode-select --install
```
Build using the universal build script:
```bash
cd nexus_engine
chmod +x build.sh
./build.sh
```
Or build using standard Make:
```bash
make -j$(sysctl -n hw.ncpu)
```

### 5.2 Linux (Ubuntu / Debian)
```bash
sudo apt update
sudo apt install -y build-essential cmake
cd nexus_engine
./build.sh
```

### 5.3 Linux (RHEL / Rocky / CentOS / AlmaLinux)
```bash
sudo dnf groupinstall -y "Development Tools"
sudo dnf install -y cmake gcc-c++
cd nexus_engine
./build.sh
```

### 5.4 Linux (Arch Linux / Manjaro)
```bash
sudo pacman -S --needed base-devel cmake
cd nexus_engine
./build.sh
```

### 5.5 Linux (Alpine Linux)
```bash
apk add --no-cache g++ cmake make linux-headers bash
cd nexus_engine
./build.sh
```

### 5.6 Microsoft Windows
Using the provided `build.bat` with Zig or Visual Studio C++:
```cmd
cd nexus_engine
build.bat
```

---

## 6. Python Integration & Platform Discovery

The NexusLLM Python SDK automatically inspects the underlying operating system and hardware vector engines:

```python
from nexus_engine.python.nexus_llm.bindings import NexusNativeEngine

# Initialize tiered engine
engine = NexusNativeEngine(tier0_mb=512, tier1_mb=4096)

# Query detected platform hardware capabilities
caps = engine.get_platform_capabilities()
print(f"OS Name           : {caps.os_name}")
print(f"CPU Architecture  : {caps.cpu_arch}")
print(f"Vector Engine     : {caps.vector_isa}")
print(f"Unified Memory    : {caps.is_unified_memory}")
print(f"Transparent HP    : {caps.has_transparent_hugepages}")
print(f"Theoretical BW    : {caps.theoretical_bandwidth_gbps} GB/s")
print(f"Performance Tier  : {caps.performance_tier_desc}")
```

Example output on Apple Silicon:
```
OS Name           : Apple macOS Darwin
CPU Architecture  : ARM64 (Apple Silicon M-Series)
Vector Engine     : ARM NEON 128-bit + Apple AMX
Unified Memory    : True
Transparent HP    : False
Theoretical BW    : 410.0 GB/s
Performance Tier  : Apple Silicon Zero-Copy UMA Tier (Shared DRAM bus, 0 PCIe latency)
```

Example output on Linux Server:
```
OS Name           : Ubuntu Linux
CPU Architecture  : x86_64
Vector Engine     : AVX2 + FMA
Unified Memory    : False
Transparent HP    : True
Theoretical BW    : 85.0 GB/s
Performance Tier  : Linux Transparent Huge Pages (THP 2MB) + madvise(WILLNEED) Tier
```
