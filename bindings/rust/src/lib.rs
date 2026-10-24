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

extern "C" {
    fn nexus_create_engine(tier0_bytes: usize, tier1_bytes: usize, max_kv_blocks: usize) -> *mut c_void;
    fn nexus_destroy_engine(engine: *mut c_void);
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
