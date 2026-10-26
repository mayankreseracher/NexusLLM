import unittest
import os
import tempfile
import struct
import json
import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent
python_dir = repo_root / "python"
for p in [str(python_dir), str(repo_root), str(repo_root.parent)]:
    if p not in sys.path:
        sys.path.insert(0, p)

try:
    from nexus_llm.universal_loader import UniversalModelLoader
    from nexus_llm.memory_manager import DynamicMemoryTierManager
except ImportError:
    from nexus_engine.python.nexus_llm.universal_loader import UniversalModelLoader
    from nexus_engine.python.nexus_llm.memory_manager import DynamicMemoryTierManager

class TestUniversalLoader(unittest.TestCase):
    def setUp(self):
        self.mem_mgr = DynamicMemoryTierManager(tier0_capacity_mb=64, tier1_capacity_mb=512)
        self.loader = UniversalModelLoader(memory_tier_manager=self.mem_mgr)

    def tearDown(self):
        self.mem_mgr.close()

    def test_safetensors_parsing(self):
        with tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False) as f:
            temp_path = f.name
            header = {
                "weight1": {
                    "dtype": "F32",
                    "shape": [2, 2],
                    "data_offsets": [0, 16]
                },
                "__metadata__": {"format": "pt"}
            }
            header_json = json.dumps(header).encode("utf-8")
            header_size = len(header_json)
            f.write(struct.pack("<Q", header_size))
            f.write(header_json)
            f.write(b"\x00" * 16) # 16 bytes tensor data

        ok = self.loader.load(temp_path)
        os.remove(temp_path)
        self.assertTrue(ok)
        self.assertEqual(self.loader.detected_format, "safetensors")
        self.assertEqual(len(self.loader.tensors), 1)

    def test_gguf_parsing(self):
        with tempfile.NamedTemporaryFile(suffix=".gguf", delete=False) as f:
            temp_path = f.name
            f.write(b"GGUF")
            f.write(struct.pack("<I", 3)) # Version 3
            f.write(struct.pack("<Q", 2)) # 2 tensors
            f.write(struct.pack("<Q", 0)) # 0 metadata key-values

        ok = self.loader.load(temp_path)
        os.remove(temp_path)
        self.assertTrue(ok)
        self.assertEqual(self.loader.detected_format, "gguf")
        self.assertEqual(len(self.loader.tensors), 2)

if __name__ == "__main__":
    unittest.main()
