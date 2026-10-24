import unittest
from nexus_engine.python.nexus_llm.continuous_batcher import ContinuousBatchEngine
from nexus_engine.python.nexus_llm.prefix_cache import RadixPrefixCache

class TestContinuousBatch(unittest.TestCase):
    def setUp(self):
        self.p_cache = RadixPrefixCache()
        self.batcher = ContinuousBatchEngine(prefix_cache=self.p_cache, speculative_draft_len=3)

    def test_continuous_scheduling_lifecycle(self):
        prompt = [101, 102, 103, 104]
        req_id = self.batcher.add_request(prompt, max_new_tokens=10)

        # First step triggers prefill
        step_out = self.batcher.step()
        self.assertIn(req_id, step_out)

        # Subsequent steps generate speculative tokens until finished
        steps = 0
        while req_id not in self.batcher.finished_requests and steps < 20:
            self.batcher.step()
            steps += 1

        self.assertIn(req_id, self.batcher.finished_requests)
        fin = self.batcher.finished_requests[req_id]
        self.assertGreaterEqual(len(fin.generated_tokens), 10)

        # Verify speculative speedup metrics
        stats = self.batcher.get_throughput_stats()
        self.assertGreater(stats["speculative_speedup_factor"], 1.0)

if __name__ == "__main__":
    unittest.main()
