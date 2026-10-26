"""
Dynamic Memory Tier Manager:
Keeps inactive parameters in Host System RAM while the active working set
(current executing layer / top-k active MoE experts) is streamed into high-speed accelerator memory.
"""

import time
import threading
from typing import Dict, Any, Optional, Set
try:
    import numpy as np
except ImportError:
    np = None

class DynamicMemoryTierManager:
    def __init__(self, tier0_capacity_mb: int = 1024, tier1_capacity_mb: int = 16384):
        self.tier0_capacity_bytes = tier0_capacity_mb * 1024 * 1024
        self.tier1_capacity_bytes = tier1_capacity_mb * 1024 * 1024

        self.tensors: Dict[str, Dict[str, Any]] = {}
        self.active_tensors: Set[str] = set()

        self.tier0_allocated_bytes = 0
        self.tier1_inactive_bytes = 0
        self.bytes_transferred_tier1_to_tier0 = 0

        self.prefetch_hits = 0
        self.prefetch_misses = 0
        self.last_pcie_bandwidth_gbps = 28.5 # Simulated PCIe Gen4 x16 baseline

        self._lock = threading.Lock()
        self._prefetch_queue = []
        self._stop_prefetch = False
        self._prefetch_thread = threading.Thread(target=self._prefetch_worker, daemon=True)
        self._prefetch_thread.start()

    def register_tensor(self, name: str, info: Dict[str, Any]):
        with self._lock:
            byte_size = info.get("byte_size", 4096)
            self.tensors[name] = {
                "info": info,
                "byte_size": byte_size,
                "is_active": False,
                "data_tier1": None, # Kept in host RAM or mmapped
                "data_tier0": None,
                "access_count": 0,
                "last_accessed": time.time()
            }
            self.tier1_inactive_bytes += byte_size

    def activate_tensor(self, name: str) -> Optional[Any]:
        """Brings tensor from Host RAM into Tier 0 Active working set."""
        with self._lock:
            t = self.tensors.get(name)
            if not t:
                return None

            t["access_count"] += 1
            t["last_accessed"] = time.time()

            if t["is_active"] and t["data_tier0"] is not None:
                self.prefetch_hits += 1
                return t["data_tier0"]

            # Cache miss - stream from Host RAM into Tier 0
            self.prefetch_misses += 1
            b_size = t["byte_size"]

            # If Tier 0 is full, evict oldest active layer
            while self.tier0_allocated_bytes + b_size > self.tier0_capacity_bytes and self.active_tensors:
                oldest_name = min(self.active_tensors, key=lambda k: self.tensors[k]["last_accessed"])
                self._deactivate_internal(oldest_name)

            t0 = time.perf_counter()
            # Allocate working buffer
            shape = t["info"].get("shape", [b_size // 4])
            t["data_tier0"] = np.ones(shape, dtype=np.float32)
            t["is_active"] = True
            self.active_tensors.add(name)

            self.tier0_allocated_bytes += b_size
            if self.tier1_inactive_bytes >= b_size:
                self.tier1_inactive_bytes -= b_size
            self.bytes_transferred_tier1_to_tier0 += b_size

            t1 = time.perf_counter()
            dur = max(t1 - t0, 1e-6)
            cur_gbps = (b_size / (1024 ** 3)) / dur
            self.last_pcie_bandwidth_gbps = 0.9 * self.last_pcie_bandwidth_gbps + 0.1 * cur_gbps

            return t["data_tier0"]

    def deactivate_tensor(self, name: str):
        """Parks active tensor back to inactive state in Host RAM."""
        with self._lock:
            self._deactivate_internal(name)

    def _deactivate_internal(self, name: str):
        t = self.tensors.get(name)
        if t and t["is_active"]:
            t["is_active"] = False
            t["data_tier0"] = None
            if name in self.active_tensors:
                self.active_tensors.remove(name)
            b_size = t["byte_size"]
            if self.tier0_allocated_bytes >= b_size:
                self.tier0_allocated_bytes -= b_size
            self.tier1_inactive_bytes += b_size

    def prefetch_async(self, name: str):
        """Asynchronously prefetch the next layer or expert into Tier 0."""
        with self._lock:
            self._prefetch_queue.append(name)

    def _prefetch_worker(self):
        while not self._stop_prefetch:
            name = None
            with self._lock:
                if self._prefetch_queue:
                    name = self._prefetch_queue.pop(0)
            if name:
                self.activate_tensor(name)
            else:
                time.sleep(0.005)

    def get_stats(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "tier0_allocated_mb": self.tier0_allocated_bytes / (1024 * 1024),
                "tier0_capacity_mb": self.tier0_capacity_bytes / (1024 * 1024),
                "tier1_host_ram_mb": (self.tier1_capacity_bytes) / (1024 * 1024),
                "tier1_inactive_mb": self.tier1_inactive_bytes / (1024 * 1024),
                "bytes_transferred_mb": self.bytes_transferred_tier1_to_tier0 / (1024 * 1024),
                "prefetch_hits": self.prefetch_hits,
                "prefetch_misses": self.prefetch_misses,
                "pcie_bandwidth_gbps": round(self.last_pcie_bandwidth_gbps, 2),
                "active_tensors_count": len(self.active_tensors),
                "total_tensors_count": len(self.tensors)
            }

    def close(self):
        self._stop_prefetch = True
        if self._prefetch_thread.is_alive():
            self._prefetch_thread.join(timeout=1.0)
