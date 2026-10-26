#pragma once

#include "nexus_common.hpp"
#include "memory_tier.hpp"
#include <fstream>
#include <map>

namespace nexus {

struct ModelMetadata {
    std::string architecture = "llama";
    uint32_t vocab_size = 32000;
    uint32_t hidden_size = 4096;
    uint32_t intermediate_size = 11008;
    uint32_t num_hidden_layers = 32;
    uint32_t num_attention_heads = 32;
    uint32_t num_key_value_heads = 32;
    uint32_t max_position_embeddings = 4096;
    float rms_norm_eps = 1e-5f;
    float rope_theta = 10000.0f;
    bool is_moe = false;
    uint32_t num_experts = 0;
    uint32_t num_experts_per_tok = 0;
    std::map<std::string, std::string> custom_attributes;
};

class UniversalModelLoader {
public:
    UniversalModelLoader(std::shared_ptr<MemoryTierManager> tier_mgr);
    ~UniversalModelLoader();

    // Auto-detect format from file extension and header inspection
    NexusStatus load_model(const std::string& model_path);

    // Format specific parsers
    NexusStatus load_safetensors(const std::string& filepath);
    NexusStatus load_gguf(const std::string& filepath);
    NexusStatus load_pytorch_bin(const std::string& filepath);
    NexusStatus load_onnx(const std::string& filepath);

    const ModelMetadata& get_metadata() const { return metadata_; }
    ModelFormat get_detected_format() const { return detected_format_; }
    size_t get_loaded_tensor_count() const { return loaded_tensors_.size(); }

private:
    std::shared_ptr<MemoryTierManager> tier_mgr_;
    ModelMetadata metadata_;
    ModelFormat detected_format_ = ModelFormat::SAFETENSORS;
    std::vector<std::string> loaded_tensors_;

    void* mmap_base_ = nullptr;
    size_t mmap_size_ = 0;
#if defined(NEXUS_PLATFORM_WINDOWS)
    void* win_file_handle_ = nullptr;
    void* win_map_handle_ = nullptr;
#else
    int posix_fd_ = -1;
#endif

    void unmap_file();
    void register_aliases(const std::string& name, DataType dt, const TensorShape& shape, const void* ptr);
};

} // namespace nexus
