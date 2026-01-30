"""pytest configuration for vLLM-Neuron benchmarks."""

import pytest


@pytest.fixture(scope="session")
def model_path():
    """Path to the model for vLLM-Neuron."""
    return "{{ model_path }}"


@pytest.fixture(scope="session")
def vllm_config():
    """vLLM-Neuron configuration."""
    return {
        # Neuron-specific settings
        "tensor_parallel_degree": 2,  # Number of NeuronCores
        "max_num_seqs": 4,  # Batch size
        "block_size": 128,  # KV cache block size
        "max_model_len": 2048,  # Maximum sequence length
        # Device setting
        "device": "neuron",
    }


@pytest.fixture(scope="session")
def sample_prompts():
    """Sample prompts for benchmarking."""
    return [
        "What is machine learning?",
        "Explain quantum computing in simple terms.",
        "How does neural network training work?",
        "What are the benefits of cloud computing?",
    ]
