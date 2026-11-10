import unittest
import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent
python_dir = repo_root / "python"
for p in [str(python_dir), str(repo_root), str(repo_root.parent)]:
    if p not in sys.path:
        sys.path.insert(0, p)

try:
    from nexus_llm.bindings import NexusNativeEngine, find_nexus_library
except ImportError:
    from nexus_engine.python.nexus_llm.bindings import NexusNativeEngine, find_nexus_library

class TestDakTmaOffload(unittest.TestCase):
    def test_native_abi_dak_tma_verification(self):
        """Verifies DAK TMA hardware descriptor setup, SmemBudget multi-head sizing,
        host-locality scheduling, and mathematical equivalence via native C++ ABI."""
        engine = NexusNativeEngine()
        res = engine.verify_dak_tma_offload()
        self.assertEqual(res, 0, f"DAK TMA Native verification failed with code: {res}")
        engine.close()

if __name__ == "__main__":
    unittest.main()
