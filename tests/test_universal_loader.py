import unittest
import tempfile
import struct
import json
from pathlib import Path
from nexus_engine.python.nexus_llm.universal_loader import UniversalModelLoader
from nexus_engine.python.nexus_llm.memory_manager import DynamicMemoryTierManager

class TestUniversalLoader(unittest.TestCase):
    def setUp(self):
        self.mem_mgr = DynamicMemoryTierManager(tier0_capacity_mb=64, tier1_capacity_mb=512)
        self.loader = UniversalModelLoader(self.mem_mgr)

    def tearDown(self):
        self.mem_mgr.close()

    def test_safetensors_loader(self):
        with tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False) as f:
            t_path = f.name
            # Write valid safetensors binary format
            header = {
                "weight1": {
                    "dtype": "F16",
                    "shape": [64, 64],
                    "data_offsets": [0, 64 * 64 * 2]
                }
            }
            header_json = json.dumps(header).encode('utf-8')
            f.write(struct.pack("<Q", len(header_json)))
            f.write(header_json)
            # Write tensor binary data
            f.write(b"\x00" * (64 * 64 * 2))

        try:
            ok = self.loader.load(t_path)
            self.assertTrue(ok)
            self.assertEqual(self.loader.detected_format, "safetensors")
            self.assertIn("weight1", self.loader.tensors)
        finally:
            Path(t_path).unlink(missing_ok=True)

    def test_gguf_loader(self):
        with tempfile.NamedTemporaryFile(suffix=".gguf", delete=False) as f:
            t_path = f.name
            # Write GGUF magic bytes
            f.write(b"GGUF")
            f.write(struct.pack("<III", 3, 2, 0)) # version 3, 2 tensors, 0 metadata
            f.write(b"\x00" * 64)

        try:
            ok = self.loader.load(t_path)
            self.assertTrue(ok)
            self.assertEqual(self.loader.detected_format, "gguf")
        finally:
            Path(t_path).unlink(missing_ok=True)

    def test_pytorch_bin_loader(self):
        with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
            t_path = f.name
            f.write(b"PK\x03\x04mock_torch_checkpoint")

        try:
            ok = self.loader.load(t_path)
            self.assertTrue(ok)
            self.assertEqual(self.loader.detected_format, "pytorch_bin")
        finally:
            Path(t_path).unlink(missing_ok=True)

if __name__ == "__main__":
    unittest.main()
