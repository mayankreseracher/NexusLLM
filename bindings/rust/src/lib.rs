//! Rust bindings for NexusLLM Inference Engine
//! Direct zero-overhead FFI to libnexus_engine

use std::ffi::{CStr, CString};
use std::os::raw::{c_char, c_int, c_void};

#[repr(C)]
#[derive(Debug, Default, Clone, Copy)]
pub struct NexusLocalMemoryStats {
    pub tier0_allocated_bytes: usize,
    pub tier0_capacity_bytes: usize,
    pub tier1_host_ram_bytes: usize,
    pub tier1_inactive_bytes: usize,
    pub bytes_transferred_tier1_to_tier0: usize,
    pub prefetch_hits: usize,
    pub prefetch_misses: usize,
    pub pcie_bandwidth_gbps: f64,
    pub prefix_cache_hit_rate: f64,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct NexusPlatformInfoRaw {
    pub os_type: u32,
    pub os_name: [c_char; 64],
    pub cpu_arch: [c_char; 32],
    pub vector_isa: u32,
    pub vector_isa_name: [c_char; 32],
    pub is_unified_memory: u32,
    pub has_transparent_hugepages: u32,
    pub hardware_concurrency: u32,
    pub theoretical_bandwidth_gbps: f64,
    pub performance_tier_desc: [c_char; 128],
}

impl Default for NexusPlatformInfoRaw {
    fn default() -> Self {
        Self {
            os_type: 0,
            os_name: [0; 64],
            cpu_arch: [0; 32],
            vector_isa: 0,
            vector_isa_name: [0; 32],
            is_unified_memory: 0,
            has_transparent_hugepages: 0,
            hardware_concurrency: 1,
            theoretical_bandwidth_gbps: 0.0,
            performance_tier_desc: [0; 128],
        }
    }
}

#[derive(Debug, Clone)]
pub struct PlatformCapabilities {
    pub os_name: String,
    pub cpu_arch: String,
    pub vector_isa_name: String,
    pub is_unified_memory: bool,
    pub has_transparent_hugepages: bool,
    pub hardware_concurrency: u32,
    pub theoretical_bandwidth_gbps: f64,
    pub performance_tier_desc: String,
}

extern "C" {
    fn nexus_create_engine(tier0_bytes: usize, tier1_bytes: usize, max_kv_blocks: usize) -> *mut c_void;
    fn nexus_destroy_engine(engine: *mut c_void);
    fn nexus_get_platform_info(out_info: *mut NexusPlatformInfoRaw);
    fn nexus_load_model_auto(engine: *mut c_void, filepath: *const c_char) -> i32;
    fn nexus_prefetch_layer(engine: *mut c_void, layer_name: *const c_char) -> i32;
    fn nexus_deactivate_layer(engine: *mut c_void, layer_name: *const c_char) -> i32;
    fn nexus_submit_request(
        engine: *mut c_void,
        request_id: *const c_char,
        prompt_tokens: *const i32,
        prompt_len: usize,
        max_new_tokens: usize,
    ) -> i32;
    fn nexus_step_continuous_batch(
        engine: *mut c_void,
        out_finished_ids: *mut *mut c_char,
        out_tokens: *mut i32,
        max_out: usize,
    ) -> i32;
    fn nexus_get_memory_stats(engine: *mut c_void, out_stats: *mut NexusLocalMemoryStats);
}

pub struct NexusEngine {
    handle: *mut c_void,
}

impl NexusEngine {
    /// Creates a new NexusEngine with specified Tier 0 (Active Accelerator buffer)
    /// and Tier 1 (Host System RAM for inactive parameter paging).
    pub fn new(tier0_mb: usize, tier1_mb: usize, max_kv_blocks: usize) -> Result<Self, &'static str> {
        let t0_bytes = tier0_mb * 1024 * 1024;
        let t1_bytes = tier1_mb * 1024 * 1024;
        let handle = unsafe { nexus_create_engine(t0_bytes, t1_bytes, max_kv_blocks) };
        if handle.is_null() {
            Err("Failed to create NexusEngine instance")
        } else {
            Ok(Self { handle })
        }
    }

    /// Retrieve detected platform capabilities across macOS, Linux, and Windows
    pub fn get_platform_capabilities() -> PlatformCapabilities {
        let mut raw = NexusPlatformInfoRaw::default();
        unsafe { nexus_get_platform_info(&mut raw) };

        let to_str = |buf: &[c_char]| -> String {
            unsafe {
                CStr::from_ptr(buf.as_ptr())
                    .to_string_lossy()
                    .into_owned()
            }
        };

        PlatformCapabilities {
            os_name: to_str(&raw.os_name),
            cpu_arch: to_str(&raw.cpu_arch),
            vector_isa_name: to_str(&raw.vector_isa_name),
            is_unified_memory: raw.is_unified_memory != 0,
            has_transparent_hugepages: raw.has_transparent_hugepages != 0,
            hardware_concurrency: raw.hardware_concurrency,
            theoretical_bandwidth_gbps: raw.theoretical_bandwidth_gbps,
            performance_tier_desc: to_str(&raw.performance_tier_desc),
        }
    }

    /// Load any model format (SafeTensors, GGUF, PyTorch, ONNX) into Host RAM
    pub fn load_model(&self, path: &str) -> Result<(), i32> {
        let c_path = CString::new(path).map_err(|_| -1)?;
        let res = unsafe { nexus_load_model_auto(self.handle, c_path.as_ptr()) };
        if res == 0 { Ok(()) } else { Err(res) }
    }

    /// Prefetch upcoming layer into active buffer
    pub fn prefetch_layer(&self, layer_name: &str) -> Result<(), i32> {
        let c_name = CString::new(layer_name).map_err(|_| -1)?;
        let res = unsafe { nexus_prefetch_layer(self.handle, c_name.as_ptr()) };
        if res == 0 { Ok(()) } else { Err(res) }
    }

    /// Submit a token sequence for continuous batch inference
    pub fn submit_request(&self, request_id: &str, prompt: &[i32], max_new_tokens: usize) -> Result<(), i32> {
        let c_id = CString::new(request_id).map_err(|_| -1)?;
        let res = unsafe {
            nexus_submit_request(
                self.handle,
                c_id.as_ptr(),
                prompt.as_ptr(),
                prompt.len(),
                max_new_tokens,
            )
        };
        if res == 0 { Ok(()) } else { Err(res) }
    }

    /// Step the continuous iteration-level scheduling batch
    pub fn step_continuous_batch(&self, max_out: usize) -> Vec<i32> {
        let mut out_tokens = vec![0i32; max_out];
        let count = unsafe {
            nexus_step_continuous_batch(
                self.handle,
                std::ptr::null_mut(),
                out_tokens.as_mut_ptr(),
                max_out,
            )
        };
        if count > 0 {
            out_tokens.truncate(count as usize);
            out_tokens
        } else {
            vec![]
        }
    }

    /// Retrieve local memory buffer statistics (strictly offline)
    pub fn get_memory_stats(&self) -> NexusLocalMemoryStats {
        let mut stats = NexusLocalMemoryStats::default();
        unsafe { nexus_get_memory_stats(self.handle, &mut stats) };
        stats
    }
}

impl Drop for NexusEngine {
    fn drop(&mut self) {
        if !self.handle.is_null() {
            unsafe { nexus_destroy_engine(self.handle) };
            self.handle = std::ptr::null_mut();
        }
    }
}

unsafe impl Send for NexusEngine {}
unsafe impl Sync for NexusEngine {}
