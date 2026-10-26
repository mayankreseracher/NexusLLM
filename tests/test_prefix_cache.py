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

if __name__ == "__main__":
    unittest.main()
