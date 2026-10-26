import unittest
import ctypes
from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent
dll_path = repo_root / "nexus_engine.dll"

class NexusDeviceInfo(ctypes.Structure):
    _fields_ = [
        ("device_type", ctypes.c_uint32),
        ("device_name", ctypes.c_char * 64),
        ("vendor", ctypes.c_char * 64),
        ("architecture", ctypes.c_char * 32),
        ("is_unified_memory", ctypes.c_uint32),
        ("supports_fp16", ctypes.c_uint32),
        ("supports_int4", ctypes.c_uint32),
        ("driver_version", ctypes.c_char * 64),
        ("description", ctypes.c_char * 128),
    ]

class TestHardwareBackends(unittest.TestCase):
    def setUp(self):
        self.lib = ctypes.CDLL(str(dll_path))

        self.lib.nexus_create_engine.restype = ctypes.c_void_p
        self.lib.nexus_create_engine.argtypes = [ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t]

        self.lib.nexus_destroy_engine.restype = None
        self.lib.nexus_destroy_engine.argtypes = [ctypes.c_void_p]

        self.lib.nexus_get_available_device_count.restype = ctypes.c_uint32
        self.lib.nexus_get_available_device_count.argtypes = []

        self.lib.nexus_get_device_info.restype = ctypes.c_int32
        self.lib.nexus_get_device_info.argtypes = [ctypes.c_uint32, ctypes.POINTER(NexusDeviceInfo)]

        self.lib.nexus_set_compute_device.restype = ctypes.c_int32
        self.lib.nexus_set_compute_device.argtypes = [ctypes.c_void_p, ctypes.c_char_p]

        self.lib.nexus_get_active_device_name.restype = ctypes.c_char_p
        self.lib.nexus_get_active_device_name.argtypes = [ctypes.c_void_p]

        self.lib.nexus_prefetch_layer_index.restype = ctypes.c_int32
        self.lib.nexus_prefetch_layer_index.argtypes = [ctypes.c_void_p, ctypes.c_uint32]

        self.lib.nexus_deactivate_layer_index.restype = ctypes.c_int32
        self.lib.nexus_deactivate_layer_index.argtypes = [ctypes.c_void_p, ctypes.c_uint32]

    def test_device_discovery(self):
        count = self.lib.nexus_get_available_device_count()
        self.assertGreaterEqual(count, 1, "Expected at least 1 compute device (CPU)")

        devices = []
        for i in range(count):
            dinfo = NexusDeviceInfo()
            ret = self.lib.nexus_get_device_info(i, ctypes.byref(dinfo))
            self.assertEqual(ret, 0)
            dev_name = dinfo.device_name.decode("utf-8")
            vendor = dinfo.vendor.decode("utf-8")
            devices.append((dev_name, vendor))
            print(f"[Device {i}] {dev_name} ({vendor})")

        self.assertTrue(any("CPU" in d[0] or "Host" in d[0] for d in devices))

    def test_backend_selection_and_prefetch(self):
        engine = self.lib.nexus_create_engine(64 * 1024 * 1024, 256 * 1024 * 1024, 128)
        self.assertIsNotNone(engine)

        # Switch to CPU
        ret = self.lib.nexus_set_compute_device(engine, b"cpu")
        self.assertEqual(ret, 0)
        active_name = self.lib.nexus_get_active_device_name(engine).decode("utf-8")
        self.assertTrue("Host" in active_name or "CPU" in active_name)

        # Switch to CUDA
        ret = self.lib.nexus_set_compute_device(engine, b"cuda")
        self.assertEqual(ret, 0)
        active_name = self.lib.nexus_get_active_device_name(engine).decode("utf-8")
        self.assertTrue("CUDA" in active_name)

        # Switch to Metal (Apple Silicon with CuMetal bridge)
        ret = self.lib.nexus_set_compute_device(engine, b"metal")
        self.assertEqual(ret, 0)
        active_name = self.lib.nexus_get_active_device_name(engine).decode("utf-8")
        self.assertTrue("Metal" in active_name)

        # Switch to AMD ROCm/HIP
        ret = self.lib.nexus_set_compute_device(engine, b"rocm")
        self.assertEqual(ret, 0)
        active_name = self.lib.nexus_get_active_device_name(engine).decode("utf-8")
        self.assertTrue("AMD" in active_name or "HIP" in active_name or "ROCm" in active_name)

        # Switch to Intel Arc SYCL
        ret = self.lib.nexus_set_compute_device(engine, b"sycl")
        self.assertEqual(ret, 0)
        active_name = self.lib.nexus_get_active_device_name(engine).decode("utf-8")
        self.assertTrue("Intel" in active_name or "Arc" in active_name or "Level Zero" in active_name)

        # Switch to Universal OpenCL
        ret = self.lib.nexus_set_compute_device(engine, b"opencl")
        self.assertEqual(ret, 0)
        active_name = self.lib.nexus_get_active_device_name(engine).decode("utf-8")
        self.assertTrue("OpenCL" in active_name)

        # Test Phase 4 Layer Prefetch & Deactivation
        p_ret = self.lib.nexus_prefetch_layer_index(engine, 1)
        self.assertEqual(p_ret, 0)
        d_ret = self.lib.nexus_deactivate_layer_index(engine, 0)
        self.assertEqual(d_ret, 0)

        self.lib.nexus_destroy_engine(engine)

if __name__ == "__main__":
    unittest.main()
