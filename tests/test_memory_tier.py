import unittest
import time
import numpy as np
from nexus_engine.python.nexus_llm.memory_manager import DynamicMemoryTierManager

class TestMemoryTier(unittest.TestCase):
    def setUp(self):
        self.mgr = DynamicMemoryTierManager(tier0_capacity_mb=64, tier1_capacity_mb=1024)

    def tearDown(self):
        self.mgr.close()

    def test_parameter_registration_and_activation(self):
        # Register a layer weight in Tier 1 Host RAM
        layer_name = "model.layers.0.mlp.up_proj.weight"
        self.mgr.register_tensor(layer_name, {
            "byte_size": 8 * 1024 * 1024, # 8 MB
            "shape": [1024, 2048],
            "dtype": "FP32"
        })

        stats = self.mgr.get_stats()
        self.assertEqual(stats["tier1_inactive_mb"], 8.0)
        self.assertEqual(stats["tier0_allocated_mb"], 0.0)

        # Activate into Tier 0
        buf = self.mgr.activate_tensor(layer_name)
        self.assertIsNotNone(buf)
        self.assertEqual(buf.shape, (1024, 2048))

        stats = self.mgr.get_stats()
        self.assertEqual(stats["tier0_allocated_mb"], 8.0)
        self.assertEqual(stats["tier1_inactive_mb"], 0.0)
        self.assertEqual(stats["prefetch_misses"], 1)

        # Second access should hit cache
        buf2 = self.mgr.activate_tensor(layer_name)
        stats2 = self.mgr.get_stats()
        self.assertEqual(stats2["prefetch_hits"], 1)

        # Deactivate back to Tier 1 Host RAM
        self.mgr.deactivate_tensor(layer_name)
        stats3 = self.mgr.get_stats()
        self.assertEqual(stats3["tier0_allocated_mb"], 0.0)
        self.assertEqual(stats3["tier1_inactive_mb"], 8.0)

    def test_asynchronous_prefetch(self):
        layer_name = "model.layers.1.mlp.up_proj.weight"
        self.mgr.register_tensor(layer_name, {
            "byte_size": 4 * 1024 * 1024,
            "shape": [1024, 1024],
            "dtype": "FP32"
        })
        self.mgr.prefetch_async(layer_name)
        time.sleep(0.05) # Allow worker thread to process

        stats = self.mgr.get_stats()
        self.assertEqual(stats["tier0_allocated_mb"], 4.0)

if __name__ == "__main__":
    unittest.main()
