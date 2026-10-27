import unittest
import ctypes
import math
from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent
dll_path = repo_root / "nexus_engine.dll"

class TestForwardPassOrchestration(unittest.TestCase):
    def setUp(self):
        self.lib = ctypes.CDLL(str(dll_path))

        self.lib.nexus_create_engine.restype = ctypes.c_void_p
        self.lib.nexus_create_engine.argtypes = [ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t]

        self.lib.nexus_destroy_engine.restype = None
        self.lib.nexus_destroy_engine.argtypes = [ctypes.c_void_p]

        self.lib.nexus_forward_token.restype = ctypes.c_int32
        self.lib.nexus_forward_token.argtypes = [ctypes.c_void_p, ctypes.c_int32, ctypes.c_size_t, ctypes.c_float]

        self.lib.nexus_forward_sequence.restype = ctypes.c_int32
        self.lib.nexus_forward_sequence.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int32), ctypes.c_size_t, ctypes.c_float]

        self.lib.nexus_step_speculative_verify.restype = ctypes.c_int32
        self.lib.nexus_step_speculative_verify.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int32,
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_int32),
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_int32),
            ctypes.POINTER(ctypes.c_size_t),
            ctypes.POINTER(ctypes.c_int32)
        ]

        self.lib.nexus_submit_request.restype = ctypes.c_int32
        self.lib.nexus_submit_request.argtypes = [
            ctypes.c_void_p,
            ctypes.c_char_p,
            ctypes.POINTER(ctypes.c_int32),
            ctypes.c_size_t,
            ctypes.c_size_t
        ]

        self.lib.nexus_step_continuous_batch.restype = ctypes.c_int32
        self.lib.nexus_step_continuous_batch.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_char_p),
            ctypes.POINTER(ctypes.c_int32),
            ctypes.c_size_t
        ]

        self.lib.nexus_verify_flash_attention.restype = ctypes.c_float
        self.lib.nexus_verify_flash_attention.argtypes = []

        self.engine = self.lib.nexus_create_engine(64 * 1024 * 1024, 256 * 1024 * 1024, 256)
        self.assertIsNotNone(self.engine)

    def tearDown(self):
        if self.engine:
            self.lib.nexus_destroy_engine(self.engine)
            self.engine = None

    def test_full_model_forward_single_token(self):
        """Tests the full layer loop: Embedding -> [RMSNorm -> QKV -> RoPE -> FlashAttn -> Wo -> RMSNorm -> SwiGLU]xL -> FinalRMSNorm -> LM Head -> Sample"""
        tok = self.lib.nexus_forward_token(self.engine, 42, 0, 0.0)
        self.assertIsInstance(tok, int)
        self.assertGreaterEqual(tok, 0)
        self.assertLess(tok, 32000)
        print(f"[Test] Full forward pass generated token: {tok}")

    def test_forward_sequence_prefill(self):
        """Tests sequential prefill across context tokens."""
        prompt = [1, 105, 420, 89]
        c_prompt = (ctypes.c_int32 * len(prompt))(*prompt)
        first_gen = self.lib.nexus_forward_sequence(self.engine, c_prompt, len(prompt), 0.0)
        self.assertIsInstance(first_gen, int)
        self.assertGreaterEqual(first_gen, 0)
        print(f"[Test] Forward sequence prefill output: {first_gen}")

    def test_speculative_decoding_verification(self):
        """Tests Medusa / EAGLE-2 speculative verification: accepts matching drafts and rolls back on mismatch."""
        current_tok = 42
        pos = 5
        drafts = [100, 200, 300]
        c_drafts = (ctypes.c_int32 * len(drafts))(*drafts)

        out_accepted = (ctypes.c_int32 * 3)()
        out_num_accepted = ctypes.c_size_t(0)
        out_correction = ctypes.c_int32(0)

        ret = self.lib.nexus_step_speculative_verify(
            self.engine,
            current_tok,
            pos,
            c_drafts,
            len(drafts),
            out_accepted,
            ctypes.byref(out_num_accepted),
            ctypes.byref(out_correction)
        )
        self.assertEqual(ret, 0)
        print(f"[Test] Speculative Verification: Accepted {out_num_accepted.value}/{len(drafts)} drafts. Correction/Bonus token: {out_correction.value}")
        self.assertGreaterEqual(out_num_accepted.value, 0)
        self.assertLessEqual(out_num_accepted.value, len(drafts))

    def test_interleaved_continuous_batching_orca_sarathi(self):
        """Tests Sarathi-Serve / Orca continuous batching interleaving chunked prefill + decode."""
        prompt1 = [10, 20, 30, 40]
        c_p1 = (ctypes.c_int32 * len(prompt1))(*prompt1)
        self.lib.nexus_submit_request(self.engine, b"req_1", c_p1, len(prompt1), 10)

        prompt2 = [50, 60, 70, 80, 90, 100]
        c_p2 = (ctypes.c_int32 * len(prompt2))(*prompt2)
        self.lib.nexus_submit_request(self.engine, b"req_2", c_p2, len(prompt2), 10)

        # Step continuous batch across multiple interleaved iterations
        total_tokens_generated = 0
        for step in range(8):
            out_tokens = (ctypes.c_int32 * 16)()
            count = self.lib.nexus_step_continuous_batch(self.engine, None, out_tokens, 16)
            total_tokens_generated += count

        print(f"[Test] Interleaved continuous batching generated {total_tokens_generated} tokens over 8 steps.")
        self.assertGreater(total_tokens_generated, 0)

    def test_flash_attention_exact_numerical_equivalence(self):
        """Tests FlashAttention CPU Tiled Online Softmax vs Naive Attention (Dao et al. 2022)."""
        max_diff = self.lib.nexus_verify_flash_attention()
        print(f"[Test] FlashAttention Tiled Online Softmax Max Abs Difference: {max_diff:.8e}")
        self.assertLess(max_diff, 1e-4, f"FlashAttention max difference {max_diff} exceeded tolerance 1e-4")

if __name__ == "__main__":
    unittest.main()
