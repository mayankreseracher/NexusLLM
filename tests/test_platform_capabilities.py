import unittest
import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent
python_dir = repo_root / "python"
for p in [str(python_dir), str(repo_root), str(repo_root.parent)]:
    if p not in sys.path:
        sys.path.insert(0, p)

try:
    from nexus_llm.bindings import NexusNativeEngine
except ImportError:
    from nexus_engine.python.nexus_llm.bindings import NexusNativeEngine

class TestPlatformCapabilities(unittest.TestCase):
    def test_platform_capabilities_detection(self):
        engine = NexusNativeEngine(tier0_mb=64, tier1_mb=512)
        caps = engine.get_platform_capabilities()
        self.assertIsNotNone(caps)
        self.assertIsInstance(caps.os_name, str)
        self.assertGreater(len(caps.os_name), 0)
        self.assertIsInstance(caps.cpu_arch, str)
        self.assertGreater(len(caps.cpu_arch), 0)
        self.assertIsInstance(caps.vector_isa, str)
        self.assertIsInstance(caps.is_unified_memory, bool)
        self.assertIsInstance(caps.has_transparent_hugepages, bool)
        self.assertGreater(caps.hardware_concurrency, 0)
        self.assertGreater(caps.theoretical_bandwidth_gbps, 0.0)
        self.assertIsInstance(caps.performance_tier_desc, str)
        print(f"\n[Test] Detected Platform: {caps.os_name} ({caps.cpu_arch})")
        print(f"[Test] Vector ISA: {caps.vector_isa} | UMA: {caps.is_unified_memory} | THP: {caps.has_transparent_hugepages}")
        print(f"[Test] Concurrency: {caps.hardware_concurrency} threads | Peak Bandwidth: {caps.theoretical_bandwidth_gbps} GB/s")
        print(f"[Test] Tier Description: {caps.performance_tier_desc}")
        engine.close()

if __name__ == "__main__":
    unittest.main()
