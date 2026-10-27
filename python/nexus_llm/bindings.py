import ctypes
import ctypes.util
import os
import sys
import platform
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, List, Tuple, Dict, Any

class NexusMemoryStatsStruct(ctypes.Structure):
    _fields_ = [
        ("tier0_allocated_bytes", ctypes.c_size_t),
        ("tier0_capacity_bytes", ctypes.c_size_t),
        ("tier1_host_ram_bytes", ctypes.c_size_t),
        ("tier1_inactive_bytes", ctypes.c_size_t),
        ("bytes_transferred_tier1_to_tier0", ctypes.c_size_t),
        ("prefetch_hits", ctypes.c_size_t),
        ("prefetch_misses", ctypes.c_size_t),
        ("pcie_bandwidth_gbps", ctypes.c_double),
        ("prefix_cache_hit_rate", ctypes.c_double),
    ]

class NexusPlatformInfoStruct(ctypes.Structure):
    _fields_ = [
        ("os_type", ctypes.c_uint32),
        ("os_name", ctypes.c_char * 64),
        ("cpu_arch", ctypes.c_char * 32),
        ("vector_isa", ctypes.c_uint32),
        ("vector_isa_name", ctypes.c_char * 32),
        ("is_unified_memory", ctypes.c_uint32),
        ("has_transparent_hugepages", ctypes.c_uint32),
        ("hardware_concurrency", ctypes.c_uint32),
        ("theoretical_bandwidth_gbps", ctypes.c_double),
        ("performance_tier_desc", ctypes.c_char * 128),
    ]

@dataclass
class NexusPlatformCapabilities:
    os_name: str
    cpu_arch: str
    vector_isa: str
    is_unified_memory: bool
    has_transparent_hugepages: bool
    hardware_concurrency: int
    theoretical_bandwidth_gbps: float
    performance_tier_desc: str

@dataclass
class NexusLocalStats:
    tier0_allocated_mb: float
    tier0_capacity_mb: float
    tier1_host_ram_mb: float
    tier1_inactive_mb: float
    bytes_transferred_mb: float
    prefetch_hits: int
    prefetch_misses: int
    pcie_bandwidth_gbps: float
    prefix_cache_hit_rate: float

class NexusNativeEngine:
    def __init__(self, tier0_mb: int = 512, tier1_mb: int = 4096, max_kv_blocks: int = 4096, dll_path: Optional[str] = None):
        self.tier0_bytes = tier0_mb * 1024 * 1024
        self.tier1_bytes = tier1_mb * 1024 * 1024
        self.max_kv_blocks = max_kv_blocks
        self._lib = None
        self._engine_handle = None

        searched_paths: List[str] = []
        if dll_path is None:
            # 1. Check explicit environment variable override
            env_path = os.environ.get("NEXUS_LIB_PATH")
            if env_path and os.path.exists(env_path):
                dll_path = env_path
            else:
                pkg_dir = Path(__file__).resolve().parent
                repo_dir = pkg_dir.parent.parent
                cwd = Path.cwd()

                names = ["nexus_engine.dll", "libnexus_engine.so", "libnexus_engine.dylib"]
                search_dirs = [
                    repo_dir,
                    repo_dir / "build",
                    repo_dir / "build" / "Release",
                    repo_dir / "build" / "Debug",
                    pkg_dir,
                    pkg_dir.parent,
                    cwd,
                    cwd / "build",
                    cwd / "build" / "Release",
                    Path("/usr/local/lib"),
                    Path("/usr/lib"),
                ]

                # Walk upward directories to locate root build folder if installed inside a virtualenv
                curr = pkg_dir
                for _ in range(5):
                    curr = curr.parent
                    search_dirs.append(curr)
                    search_dirs.append(curr / "build")

                for d in search_dirs:
                    for name in names:
                        candidate = d / name
                        candidate_str = str(candidate.resolve())
                        if candidate_str not in searched_paths:
                            searched_paths.append(candidate_str)
                        if candidate.exists():
                            dll_path = candidate_str
                            break
                    if dll_path:
                        break

                if dll_path is None:
                    found = ctypes.util.find_library("nexus_engine")
                    if found:
                        dll_path = found

        if not dll_path or not os.path.exists(dll_path):
            raise RuntimeError(
                "[NexusLLM Fatal] Native C++ engine library (libnexus_engine.so / nexus_engine.dll / libnexus_engine.dylib) "
                "could not be found on host system.\n"
                "Searched paths:\n" +
                "\n".join(f"  - {p}" for p in searched_paths[:12]) +
                "\n\nTo build the native engine, execute:\n"
                "  Linux/macOS: ./build.sh  or  make\n"
                "  Windows:     build.bat\n"
                "  CMake:       cmake -B build && cmake --build build\n"
                "Or set the NEXUS_LIB_PATH environment variable to the compiled library file."
            )

        try:
            self._lib = ctypes.CDLL(dll_path)
            self._setup_abi()
            self._engine_handle = self._lib.nexus_create_engine(
                ctypes.c_size_t(self.tier0_bytes),
                ctypes.c_size_t(self.tier1_bytes),
                ctypes.c_size_t(self.max_kv_blocks)
            )
            if not self._engine_handle:
                raise RuntimeError("nexus_create_engine returned null pointer.")
        except Exception as e:
            raise RuntimeError(
                f"[NexusLLM Fatal] Failed to bind and initialize native C++ engine at '{dll_path}': {e}.\n"
                f"Please ensure the library was compiled for this architecture and all symbols match."
            ) from e

    def _setup_abi(self):
        self._lib.nexus_create_engine.argtypes = [ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t]
        self._lib.nexus_create_engine.restype = ctypes.c_void_p

        self._lib.nexus_destroy_engine.argtypes = [ctypes.c_void_p]
        self._lib.nexus_destroy_engine.restype = None

        self._lib.nexus_load_model_auto.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
        self._lib.nexus_load_model_auto.restype = ctypes.c_int32

        self._lib.nexus_get_loaded_tensor_count.argtypes = [ctypes.c_void_p]
        self._lib.nexus_get_loaded_tensor_count.restype = ctypes.c_int32

        self._lib.nexus_get_model_architecture.argtypes = [ctypes.c_void_p]
        self._lib.nexus_get_model_architecture.restype = ctypes.c_char_p

        self._lib.nexus_prefetch_layer.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
        self._lib.nexus_prefetch_layer.restype = ctypes.c_int32

        self._lib.nexus_deactivate_layer.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
        self._lib.nexus_deactivate_layer.restype = ctypes.c_int32

        self._lib.nexus_get_active_buffer.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
        self._lib.nexus_get_active_buffer.restype = ctypes.c_void_p

        self._lib.nexus_submit_request.argtypes = [
            ctypes.c_void_p, ctypes.c_char_p, ctypes.POINTER(ctypes.c_int32), ctypes.c_size_t, ctypes.c_size_t
        ]
        self._lib.nexus_submit_request.restype = ctypes.c_int32

        self._lib.nexus_step_continuous_batch.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(ctypes.c_int32), ctypes.c_size_t
        ]
        self._lib.nexus_step_continuous_batch.restype = ctypes.c_int32

        self._lib.nexus_abort_request.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
        self._lib.nexus_abort_request.restype = ctypes.c_int32

        self._lib.nexus_get_memory_stats.argtypes = [ctypes.c_void_p, ctypes.POINTER(NexusMemoryStatsStruct)]
        self._lib.nexus_get_memory_stats.restype = None

        self._lib.nexus_get_platform_info.argtypes = [ctypes.POINTER(NexusPlatformInfoStruct)]
        self._lib.nexus_get_platform_info.restype = None

    def get_platform_capabilities(self) -> NexusPlatformCapabilities:
        """Returns detected raw hardware performance capabilities across macOS, Linux, and Windows."""
        if self._lib:
            try:
                raw_info = NexusPlatformInfoStruct()
                self._lib.nexus_get_platform_info(ctypes.byref(raw_info))
                return NexusPlatformCapabilities(
                    os_name=raw_info.os_name.decode("utf-8", errors="ignore"),
                    cpu_arch=raw_info.cpu_arch.decode("utf-8", errors="ignore"),
                    vector_isa=raw_info.vector_isa_name.decode("utf-8", errors="ignore"),
                    is_unified_memory=bool(raw_info.is_unified_memory),
                    has_transparent_hugepages=bool(raw_info.has_transparent_hugepages),
                    hardware_concurrency=int(raw_info.hardware_concurrency),
                    theoretical_bandwidth_gbps=float(raw_info.theoretical_bandwidth_gbps),
                    performance_tier_desc=raw_info.performance_tier_desc.decode("utf-8", errors="ignore"),
                )
            except Exception:
                pass

        # Fallback profile inspection
        sys_os = platform.system()
        arch = platform.machine()
        concurrency = os.cpu_count() or 4
        is_apple_silicon = (sys_os == "Darwin" and arch in ("arm64", "aarch64"))
        
        if is_apple_silicon:
            return NexusPlatformCapabilities(
                os_name="Apple macOS Darwin",
                cpu_arch="ARM64 (Apple Silicon M-Series)",
                vector_isa="ARM NEON 128-bit + Apple AMX",
                is_unified_memory=True,
                has_transparent_hugepages=False,
                hardware_concurrency=concurrency,
                theoretical_bandwidth_gbps=410.0,
                performance_tier_desc="Apple Silicon Zero-Copy UMA Tier (Shared DRAM bus, 0 PCIe latency)"
            )
        elif sys_os == "Linux":
            return NexusPlatformCapabilities(
                os_name="Linux Generic (GNU/Linux)",
                cpu_arch=arch,
                vector_isa="AVX2 + FMA" if "x86" in arch else "ARM NEON",
                is_unified_memory=False,
                has_transparent_hugepages=True,
                hardware_concurrency=concurrency,
                theoretical_bandwidth_gbps=85.0,
                performance_tier_desc="Linux Transparent Huge Pages (THP 2MB) + madvise(WILLNEED) Tier"
            )
        else:
            return NexusPlatformCapabilities(
                os_name=f"Microsoft Windows ({arch})",
                cpu_arch=arch,
                vector_isa="AVX2 + FMA",
                is_unified_memory=False,
                has_transparent_hugepages=False,
                hardware_concurrency=concurrency,
                theoretical_bandwidth_gbps=64.0,
                performance_tier_desc="Windows Tiered Host-VRAM Pipeline via VirtualAlloc & PCIe DMA"
            )

    def load_model(self, model_path: str) -> bool:
        if not self._lib or not self._engine_handle:
            raise RuntimeError("[NexusNativeEngine] Native engine is not initialized.")
        res = self._lib.nexus_load_model_auto(self._engine_handle, model_path.encode('utf-8'))
        return res == 0

    def prefetch_layer(self, layer_name: str) -> bool:
        if not self._lib or not self._engine_handle:
            raise RuntimeError("[NexusNativeEngine] Native engine is not initialized.")
        return self._lib.nexus_prefetch_layer(self._engine_handle, layer_name.encode('utf-8')) == 0

    def deactivate_layer(self, layer_name: str) -> bool:
        if not self._lib or not self._engine_handle:
            raise RuntimeError("[NexusNativeEngine] Native engine is not initialized.")
        return self._lib.nexus_deactivate_layer(self._engine_handle, layer_name.encode('utf-8')) == 0

    def submit_request(self, request_id: str, prompt_tokens: List[int], max_new_tokens: int = 128) -> bool:
        if not self._lib or not self._engine_handle:
            raise RuntimeError("[NexusNativeEngine] Native engine is not initialized.")
        arr_type = ctypes.c_int32 * len(prompt_tokens)
        arr = arr_type(*prompt_tokens)
        res = self._lib.nexus_submit_request(
            self._engine_handle,
            request_id.encode('utf-8'),
            arr,
            len(prompt_tokens),
            max_new_tokens
        )
        return res == 0

    def step_continuous_batch(self, max_out: int = 32) -> Tuple[int, List[int]]:
        if not self._lib or not self._engine_handle:
            raise RuntimeError("[NexusNativeEngine] Native engine is not initialized.")
        out_arr = (ctypes.c_int32 * max_out)()
        count = self._lib.nexus_step_continuous_batch(
            self._engine_handle,
            None,
            out_arr,
            max_out
        )
        return count, list(out_arr[:count])

    def get_local_stats(self) -> NexusLocalStats:
        if not self._lib or not self._engine_handle:
            raise RuntimeError("[NexusNativeEngine] Native engine is not initialized.")
        raw = NexusMemoryStatsStruct()
        self._lib.nexus_get_memory_stats(self._engine_handle, ctypes.byref(raw))
        return NexusLocalStats(
            tier0_allocated_mb=raw.tier0_allocated_bytes / (1024 * 1024),
            tier0_capacity_mb=raw.tier0_capacity_bytes / (1024 * 1024),
            tier1_host_ram_mb=raw.tier1_host_ram_bytes / (1024 * 1024),
            tier1_inactive_mb=raw.tier1_inactive_bytes / (1024 * 1024),
            bytes_transferred_mb=raw.bytes_transferred_tier1_to_tier0 / (1024 * 1024),
            prefetch_hits=raw.prefetch_hits,
            prefetch_misses=raw.prefetch_misses,
            pcie_bandwidth_gbps=raw.pcie_bandwidth_gbps,
            prefix_cache_hit_rate=raw.prefix_cache_hit_rate
        )

    def close(self):
        if self._lib and self._engine_handle:
            self._lib.nexus_destroy_engine(self._engine_handle)
            self._engine_handle = None
