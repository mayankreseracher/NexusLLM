import unittest
import os
import struct
import tempfile
import ctypes
from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent
dll_path = repo_root / "nexus_engine.dll"

def write_gguf_v3_file(filepath: str, hidden_size: int = 64, num_layers: int = 2, vocab_size: int = 256, extra_tensors: list = None, exclude_tensors: list = None):
    """Generates a valid binary GGUF v3 file with real quantized and float weights."""
    intermediate_size = hidden_size * 2
    head_dim = hidden_size // 2
    num_heads = 2
    num_kv_heads = 2

    # Metadata KV entries
    metadata = [
        ("general.architecture", 8, "llama"),
        ("general.alignment", 4, 32),
        ("llama.context_length", 4, 512),
        ("llama.embedding_length", 4, hidden_size),
        ("llama.feed_forward_length", 4, intermediate_size),
        ("llama.block_count", 4, num_layers),
        ("llama.attention.head_count", 4, num_heads),
        ("llama.attention.head_count_kv", 4, num_kv_heads),
        ("llama.rope.freq_base", 6, 10000.0),
        ("llama.attention.layer_norm_rms_epsilon", 6, 1e-5),
        ("tokenizer.ggml.tokens", 9, [f"tok_{i}" for i in range(vocab_size)]),
    ]

    # Build tensor definitions
    # GGML types: 0 = FP32, 1 = FP16, 2 = Q4_0, 8 = Q8_0
    tensors = []

    # 1. token_embd.weight (vocab_size, hidden_size) in FP32
    emb_data = b"\x00" * (vocab_size * hidden_size * 4)
    tensors.append(("token_embd.weight", [hidden_size, vocab_size], 0, emb_data))

    # 2. Layer tensors
    for l in range(num_layers):
        pfx = f"blk.{l}."
        # attn_norm.weight (hidden_size) in FP32
        norm_data = struct.pack(f"<{hidden_size}f", *([1.0] * hidden_size))
        tensors.append((f"{pfx}attn_norm.weight", [hidden_size], 0, norm_data))

        # attn_q.weight (hidden_size, hidden_size) in Q8_0: 34 bytes per 32 values
        # (2 bytes FP16 delta + 32 bytes int8)
        num_q8_blocks = (hidden_size * hidden_size) // 32
        q8_block = struct.pack("<H", 0x3C00) + (b"\x01" * 32) # delta=1.0 in FP16, values=1
        q8_data = q8_block * num_q8_blocks
        tensors.append((f"{pfx}attn_q.weight", [hidden_size, hidden_size], 8, q8_data))

        # attn_k.weight (num_kv_heads * head_dim, hidden_size) in Q8_0
        tensors.append((f"{pfx}attn_k.weight", [hidden_size, hidden_size], 8, q8_data))

        # attn_v.weight (num_kv_heads * head_dim, hidden_size) in Q8_0
        tensors.append((f"{pfx}attn_v.weight", [hidden_size, hidden_size], 8, q8_data))

        # attn_output.weight (hidden_size, hidden_size) in Q8_0
        tensors.append((f"{pfx}attn_output.weight", [hidden_size, hidden_size], 8, q8_data))

        # ffn_norm.weight (hidden_size) in FP32
        tensors.append((f"{pfx}ffn_norm.weight", [hidden_size], 0, norm_data))

        # ffn_gate.weight (intermediate_size, hidden_size) in Q4_0: 18 bytes per 32 values
        # (2 bytes FP16 delta + 16 bytes uint8 nibbles)
        num_q4_blocks = (intermediate_size * hidden_size) // 32
        q4_block = struct.pack("<H", 0x3C00) + (b"\x11" * 16) # delta=1.0 in FP16, nibbles=1
        q4_data = q4_block * num_q4_blocks
        tensors.append((f"{pfx}ffn_gate.weight", [hidden_size, intermediate_size], 2, q4_data))

        # ffn_up.weight (intermediate_size, hidden_size) in Q4_0
        tensors.append((f"{pfx}ffn_up.weight", [hidden_size, intermediate_size], 2, q4_data))

        # ffn_down.weight (hidden_size, intermediate_size) in Q4_0
        tensors.append((f"{pfx}ffn_down.weight", [intermediate_size, hidden_size], 2, q4_data))

    # 3. Final norm & LM Head
    norm_data = struct.pack(f"<{hidden_size}f", *([1.0] * hidden_size))
    tensors.append(("output_norm.weight", [hidden_size], 0, norm_data))

    head_data = b"\x00" * (vocab_size * hidden_size * 4)
    tensors.append(("output.weight", [hidden_size, vocab_size], 0, head_data))

    if extra_tensors:
        tensors.extend(extra_tensors)

    if exclude_tensors:
        tensors = [t for t in tensors if t[0] not in exclude_tensors]

    with open(filepath, "wb") as f:
        # Header
        f.write(b"GGUF")
        f.write(struct.pack("<I", 3)) # Version 3
        f.write(struct.pack("<Q", len(tensors)))
        f.write(struct.pack("<Q", len(metadata)))

        # Write metadata KV pairs
        for key, val_type, val in metadata:
            k_bytes = key.encode("utf-8")
            f.write(struct.pack("<Q", len(k_bytes)))
            f.write(k_bytes)
            f.write(struct.pack("<I", val_type))

            if val_type == 4: # UINT32
                f.write(struct.pack("<I", int(val)))
            elif val_type == 6: # FLOAT32
                f.write(struct.pack("<f", float(val)))
            elif val_type == 8: # STRING
                s_bytes = str(val).encode("utf-8")
                f.write(struct.pack("<Q", len(s_bytes)))
                f.write(s_bytes)
            elif val_type == 9: # ARRAY
                elem_type = 8 # Array of strings
                f.write(struct.pack("<I", elem_type))
                f.write(struct.pack("<Q", len(val)))
                for s in val:
                    s_bytes = s.encode("utf-8")
                    f.write(struct.pack("<Q", len(s_bytes)))
                    f.write(s_bytes)

        # Precompute tensor offsets
        tensor_offsets = []
        cur_offset = 0
        for name, dims, ggml_t, raw_data in tensors:
            tensor_offsets.append(cur_offset)
            cur_offset += len(raw_data)

        # Write tensor info headers
        for idx, (name, dims, ggml_t, raw_data) in enumerate(tensors):
            n_bytes = name.encode("utf-8")
            f.write(struct.pack("<Q", len(n_bytes)))
            f.write(n_bytes)
            f.write(struct.pack("<I", len(dims)))
            for d in dims:
                f.write(struct.pack("<Q", d))
            f.write(struct.pack("<I", ggml_t))
            f.write(struct.pack("<Q", tensor_offsets[idx]))

        # Align to 32 bytes for tensor data base
        pos = f.tell()
        alignment = 32
        pad = (alignment - (pos % alignment)) % alignment
        if pad > 0:
            f.write(b"\x00" * pad)

        # Write tensor binary data
        for name, dims, ggml_t, raw_data in tensors:
            f.write(raw_data)

class TestGGUFBinaryMMap(unittest.TestCase):
    def setUp(self):
        self.temp_file = tempfile.NamedTemporaryFile(suffix=".gguf", delete=False)
        self.temp_path = self.temp_file.name
        self.temp_file.close()

        # Generate GGUF file
        write_gguf_v3_file(self.temp_path, hidden_size=64, num_layers=2, vocab_size=256)

        # Load native library
        self.lib = ctypes.CDLL(str(dll_path))

        # Declare C API signatures
        self.lib.nexus_create_engine.restype = ctypes.c_void_p
        self.lib.nexus_create_engine.argtypes = [ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t]

        self.lib.nexus_destroy_engine.restype = None
        self.lib.nexus_destroy_engine.argtypes = [ctypes.c_void_p]

        self.lib.nexus_load_model_auto.restype = ctypes.c_int32
        self.lib.nexus_load_model_auto.argtypes = [ctypes.c_void_p, ctypes.c_char_p]

        self.lib.nexus_get_loaded_tensor_count.restype = ctypes.c_int32
        self.lib.nexus_get_loaded_tensor_count.argtypes = [ctypes.c_void_p]

        self.lib.nexus_get_model_architecture.restype = ctypes.c_char_p
        self.lib.nexus_get_model_architecture.argtypes = [ctypes.c_void_p]

        self.lib.nexus_submit_request.restype = ctypes.c_int32
        self.lib.nexus_submit_request.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.POINTER(ctypes.c_int32), ctypes.c_size_t, ctypes.c_size_t]

        self.lib.nexus_step_continuous_batch.restype = ctypes.c_int32
        self.lib.nexus_step_continuous_batch.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_char_p), ctypes.POINTER(ctypes.c_int32), ctypes.c_size_t]

        self.lib.nexus_get_active_buffer.restype = ctypes.c_void_p
        self.lib.nexus_get_active_buffer.argtypes = [ctypes.c_void_p, ctypes.c_char_p]

        self.lib.nexus_forward_token.restype = ctypes.c_int32
        self.lib.nexus_forward_token.argtypes = [ctypes.c_void_p, ctypes.c_int32, ctypes.c_size_t, ctypes.c_float]

    def tearDown(self):
        if os.path.exists(self.temp_path):
            try:
                os.remove(self.temp_path)
            except Exception:
                pass

    def test_gguf_binary_mmap_and_inference(self):
        engine = self.lib.nexus_create_engine(64 * 1024 * 1024, 256 * 1024 * 1024, 128)
        self.assertIsNotNone(engine)

        # 1. Load GGUF file via zero-copy mmap
        status = self.lib.nexus_load_model_auto(engine, self.temp_path.encode("utf-8"))
        self.assertEqual(status, 0, f"Expected SUCCESS (0), got {status}")

        # 2. Verify architecture and loaded tensor count
        arch = self.lib.nexus_get_model_architecture(engine).decode("utf-8")
        self.assertEqual(arch, "llama")

        t_count = self.lib.nexus_get_loaded_tensor_count(engine)
        self.assertGreater(t_count, 10, f"Expected >10 tensors, got {t_count}")

        # 3. Submit inference requests and step forward pass
        prompt = (ctypes.c_int32 * 4)(1, 10, 20, 30)
        ret = self.lib.nexus_submit_request(engine, b"req-test-1", prompt, 4, 8)
        self.assertEqual(ret, 0)

        out_tokens = (ctypes.c_int32 * 16)()
        stepped = self.lib.nexus_step_continuous_batch(engine, None, out_tokens, 16)
        self.assertGreaterEqual(stepped, 1, "Expected at least 1 token generated")

        self.lib.nexus_destroy_engine(engine)

    def test_canonical_tensor_aliases(self):
        """Verifies that GGUF blk.{l}.* tensors are mapped and accessible via canonical keys layers.{l}.*"""
        engine = self.lib.nexus_create_engine(64 * 1024 * 1024, 256 * 1024 * 1024, 128)
        self.assertIsNotNone(engine)

        status = self.lib.nexus_load_model_auto(engine, self.temp_path.encode("utf-8"))
        self.assertEqual(status, 0)

        canonical_keys = [
            b"token_embd",
            b"lm_head",
            b"output_norm",
            b"layers.0.wq",
            b"layers.0.wk",
            b"layers.0.wv",
            b"layers.0.wo",
            b"layers.0.ffn_gate",
            b"layers.0.ffn_up",
            b"layers.0.ffn_down",
            b"layers.0.attn_norm",
            b"layers.0.ffn_norm",
            b"layers.1.wq",
            b"layers.1.wk",
            b"layers.1.wv",
            b"layers.1.wo",
            b"layers.1.ffn_gate",
            b"layers.1.ffn_up",
            b"layers.1.ffn_down",
            b"layers.1.attn_norm",
            b"layers.1.ffn_norm",
        ]
        for key in canonical_keys:
            buf = self.lib.nexus_get_active_buffer(engine, key)
            self.assertIsNotNone(buf, f"Canonical key '{key.decode()}' was not registered in tier manager!")

        self.lib.nexus_destroy_engine(engine)

    def test_mtp_tensor_registration(self):
        """Verifies that MTP / Medusa auxiliary heads are registered and alias-mapped."""
        temp_mtp = tempfile.NamedTemporaryFile(suffix=".gguf", delete=False)
        temp_mtp_path = temp_mtp.name
        temp_mtp.close()

        try:
            write_gguf_v3_file(temp_mtp_path, hidden_size=64, num_layers=2, vocab_size=256, extra_tensors=[
                ("mtp_head.0.weight", [64, 256], 0, b"\x00" * (256 * 64 * 4)),
                ("blk.0.mtp_head.1.weight", [64, 256], 0, b"\x00" * (256 * 64 * 4))
            ])

            engine = self.lib.nexus_create_engine(64 * 1024 * 1024, 256 * 1024 * 1024, 128)
            status = self.lib.nexus_load_model_auto(engine, temp_mtp_path.encode("utf-8"))
            self.assertEqual(status, 0)

            # Both mtp.0.weight and mtp_head.0.weight should be reachable
            b1 = self.lib.nexus_get_active_buffer(engine, b"mtp.0.weight")
            b2 = self.lib.nexus_get_active_buffer(engine, b"mtp_head.0.weight")
            self.assertIsNotNone(b1, "mtp.0.weight was not reachable")
            self.assertIsNotNone(b2, "mtp_head.0.weight was not reachable")

            self.lib.nexus_destroy_engine(engine)
        finally:
            if os.path.exists(temp_mtp_path):
                os.remove(temp_mtp_path)

    def test_missing_tensor_hard_fail(self):
        """Verifies that if a required weight is missing in a loaded model, forward pass fails hard with -4 (NEXUS_ERR_MISSING_TENSOR)."""
        temp_broken = tempfile.NamedTemporaryFile(suffix=".gguf", delete=False)
        temp_broken_path = temp_broken.name
        temp_broken.close()
        engine = None

        try:
            write_gguf_v3_file(temp_broken_path, hidden_size=64, num_layers=2, vocab_size=256, exclude_tensors=["blk.0.attn_q.weight"])

            engine = self.lib.nexus_create_engine(64 * 1024 * 1024, 256 * 1024 * 1024, 128)
            status = self.lib.nexus_load_model_auto(engine, temp_broken_path.encode("utf-8"))
            self.assertEqual(status, 0)

            ret = self.lib.nexus_forward_token(engine, 10, 0, 0.0)
            self.assertEqual(ret, -4, f"Expected NEXUS_ERR_MISSING_TENSOR (-4), got {ret}")
        finally:
            if engine:
                self.lib.nexus_destroy_engine(engine)
            if os.path.exists(temp_broken_path):
                try:
                    os.remove(temp_broken_path)
                except Exception:
                    pass

if __name__ == "__main__":
    unittest.main()
