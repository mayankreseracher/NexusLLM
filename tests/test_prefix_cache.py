import unittest
from nexus_engine.python.nexus_llm.prefix_cache import RadixPrefixCache

class TestPrefixCache(unittest.TestCase):
    def setUp(self):
        self.cache = RadixPrefixCache()

    def test_prefix_matching_and_reuse(self):
        system_prompt = [1, 2, 3, 4, 5, 6, 7, 8]
        blocks = [10, 11]

        # Initially empty
        matched, blks = self.cache.match_prefix(system_prompt)
        self.assertEqual(matched, 0)
        self.assertEqual(blks, [])

        # Insert prefix
        self.cache.insert_prefix(system_prompt, blocks)

        # Query with exact prompt
        matched, blks = self.cache.match_prefix(system_prompt)
        self.assertEqual(matched, 8)
        self.assertEqual(blks, [10, 11])

        # Query with prompt that shares system prompt prefix + extra user tokens
        user_prompt = system_prompt + [9, 10, 11]
        matched, blks = self.cache.match_prefix(user_prompt)
        self.assertEqual(matched, 8)
        self.assertEqual(blks, [10, 11])

        # Hit rate should be 2/3 = 66.7%
        self.assertAlmostEqual(self.cache.get_hit_rate(), 2.0 / 3.0)

if __name__ == "__main__":
    unittest.main()
