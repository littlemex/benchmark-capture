"""pytest configuration for vLLM-Neuron benchmarks."""

import pytest

from benchmark_capture.utils import VLLMConfigHelper


@pytest.fixture(scope="session")
def model_path():
    """Path to the model for vLLM-Neuron."""
    return "{{ model_path }}"


@pytest.fixture(scope="session")
def vllm_config():
    """
    vLLM configuration with hardware-aware optimizations.

    Uses VLLMConfigHelper to automatically detect hardware (Neuron or GPU)
    and apply appropriate configuration:
    - On Neuron: Includes override_neuron_config with performance optimizations
    - On GPU: Excludes Neuron-specific settings

    Returns:
        Dictionary with vLLM configuration
    """
    vllm_params = {
        # Standard vLLM settings
        "tensor_parallel_size": 2,  # Number of devices (NeuronCores or GPUs)
        "max_num_seqs": 4,  # Batch size
        "block_size": 128,  # KV cache block size
        "max_model_len": 2048,  # Maximum sequence length

        # Neuron-specific optimization (nested in additional_config)
        # These settings are automatically excluded on GPU
        "additional_config": {
            "override_neuron_config": {
                "skip_warmup": True,
                "pa_num_blocks": 512,
                "pa_block_size": 32,
                "enable_bucketing": True,
                # Prefix caching
                "is_prefix_caching": True,
                "is_block_kv_layout": True,
                # Flash decoding
                "flash_decoding_enabled": True,
                # KV cache quantization
                "kv_cache_quant": True,
                # Sequence parallelism
                "sequence_parallel_enabled": True,
                # On-device sampling
                "on_device_sampling_config": {
                    "do_sample": True,
                    "top_k": 1,
                    "dynamic": True,
                },
                # Fused kernels
                "attn_kernel_enabled": True,
                "fused_qkv": True,
                "qkv_kernel_enabled": True,
                "mlp_kernel_enabled": True,
            }
        }
    }

    return VLLMConfigHelper(vllm_params).build()


@pytest.fixture(scope="session")
def sample_prompts():
    """Sample prompts for benchmarking."""
    return [
        "What is machine learning?",
        "Explain quantum computing in simple terms.",
        "How does neural network training work?",
        "What are the benefits of cloud computing?",
    ]
