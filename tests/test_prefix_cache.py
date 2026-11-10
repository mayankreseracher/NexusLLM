import unittest
import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent
python_dir = repo_root / "python"
for p in [str(python_dir), str(repo_root), str(repo_root.parent)]:
    if p not in sys.path:
        sys.path.insert(0, p)

try:
    from nexus_llm.prefix_cache import RadixPrefixCache
except ImportError:
    from nexus_engine.python.nexus_llm.prefix_cache import RadixPrefixCache

class TestPrefixCache(unittest.TestCase):
    def setUp(self):
        self.cache = RadixPrefixCache()

    def test_prefix_caching_and_matching(self):
        # Insert a shared system prompt
        system_prompt = [101, 2054, 2003, 1037, 102]
        block_ids = [0, 1, 2]
        self.cache.insert_prefix(system_prompt, block_ids)

        # Match exact prefix
        matched_tokens, hit_blocks = self.cache.match_prefix(system_prompt)
        self.assertEqual(hit_blocks, block_ids)
        self.assertEqual(matched_tokens, len(system_prompt))

        # Match longer sequence sharing the same prefix
        longer_seq = [101, 2054, 2003, 1037, 102, 999, 888]
        matched_tokens2, hit_blocks2 = self.cache.match_prefix(longer_seq)
        self.assertEqual(hit_blocks2, block_ids)
        self.assertEqual(matched_tokens2, len(system_prompt))

        # Completely different prefix should miss
        different_seq = [500, 600, 700]
        matched_tokens3, hit_blocks3 = self.cache.match_prefix(different_seq)
        self.assertEqual(hit_blocks3, [])
        self.assertEqual(matched_tokens3, 0)

        # Verify hit rate calculation
        self.assertGreater(self.cache.get_hit_rate(), 0.5)

    def test_partition_abstraction_and_bucketed_lru(self):
        import ctypes
        try:
            from nexus_llm.bindings import find_nexus_library
        except ImportError:
            from nexus_engine.python.nexus_llm.bindings import find_nexus_library

        lib_path = find_nexus_library()
        lib = ctypes.CDLL(str(lib_path))
        lib.nexus_verify_partition_and_bucketed_lru.restype = ctypes.c_int32
        lib.nexus_verify_partition_and_bucketed_lru.argtypes = []

        res = lib.nexus_verify_partition_and_bucketed_lru()
        self.assertEqual(res, 0, f"nexus_verify_partition_and_bucketed_lru failed with code {res}")

    def test_python_bindings_partition_and_spin_sparse_verification(self):
        try:
            from nexus_llm.bindings import NexusNativeEngine
        except ImportError:
            from nexus_engine.python.nexus_llm.bindings import NexusNativeEngine

        engine = NexusNativeEngine()
        part_res = engine.verify_partition_and_bucketed_lru()
        self.assertEqual(part_res, 0, f"Engine partition verification failed: {part_res}")

        spin_diff = engine.verify_spin_sparse_attention()
        self.assertGreaterEqual(spin_diff, 0.0)
        self.assertLess(spin_diff, 0.50, f"Spin sparse attention diff exceeded threshold: {spin_diff}")
        engine.close()

if __name__ == "__main__":
    unittest.main()
