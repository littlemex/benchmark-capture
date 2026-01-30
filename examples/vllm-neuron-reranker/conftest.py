"""pytest configuration for vLLM-Neuron Reranker benchmarks.

This configuration is designed to be generic and work with various Reranker models.
Customize the config.yaml file for your specific model.
"""

import pytest
import yaml
from pathlib import Path


@pytest.fixture(scope="session")
def config():
    """Load configuration from config.yaml."""
    config_path = Path(__file__).parent / "config.yaml"
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


@pytest.fixture(scope="session")
def model_path(config):
    """Path to the reranker model."""
    path = config['model']['path']
    # Allow environment variable override
    import os
    return os.environ.get('RERANKER_MODEL_PATH', path)


@pytest.fixture(scope="session")
def vllm_config(config):
    """vLLM-Neuron configuration (vLLM 0.13+ compatible)."""
    return {
        "tensor_parallel_size": config['vllm']['tensor_parallel_size'],
        "max_num_seqs": config['vllm']['max_num_seqs'],
        "block_size": config['vllm']['block_size'],
        "max_model_len": config['vllm']['max_model_len'],
        "max_num_batched_tokens": config['vllm']['max_num_batched_tokens'],
        "num_gpu_blocks_override": config['vllm']['num_gpu_blocks_override'],
        "enable_prefix_caching": config['vllm']['enable_prefix_caching'],
        "dtype": config['vllm']['dtype'],
    }


@pytest.fixture(scope="session")
def reranker_config(config):
    """Reranker-specific configuration."""
    return config['reranker']


@pytest.fixture(scope="session")
def benchmark_config(config):
    """Benchmark configuration."""
    return config['benchmark']


@pytest.fixture(scope="session")
def reranker_prompts(reranker_config):
    """Prompt templates for reranker."""
    return {
        "prefix": reranker_config['prefix'],
        "suffix": reranker_config['suffix'],
        "instruction": reranker_config['instruction'],
    }


@pytest.fixture(scope="session")
def token_ids(reranker_config):
    """Token IDs for true/false responses."""
    return {
        "true": reranker_config['token_true'],
        "false": reranker_config['token_false'],
    }
