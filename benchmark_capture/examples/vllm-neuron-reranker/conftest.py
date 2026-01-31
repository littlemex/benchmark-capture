"""pytest configuration for vLLM-Neuron Reranker benchmarks.

This configuration is designed to be generic and work with various Reranker models.
Customize the config.yaml file for your specific model.
"""

import pytest
import yaml
from pathlib import Path

from benchmark_capture.utils import VLLMConfigHelper


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
    """
    vLLM-Neuron configuration with hardware-aware optimization.

    Uses VLLMConfigHelper to automatically detect hardware and apply
    appropriate configuration:
    - On Neuron: includes additional_config with override_neuron_config
    - On GPU: excludes Neuron-specific settings
    """
    vllm_params = {
        # Standard vLLM parameters
        "tensor_parallel_size": config['vllm']['tensor_parallel_size'],
        "max_num_seqs": config['vllm']['max_num_seqs'],
        "block_size": config['vllm']['block_size'],
        "max_model_len": config['vllm']['max_model_len'],
        "max_num_batched_tokens": config['vllm']['max_num_batched_tokens'],
        "num_gpu_blocks_override": config['vllm']['num_gpu_blocks_override'],
        "enable_prefix_caching": config['vllm']['enable_prefix_caching'],
        "dtype": config['vllm']['dtype'],

        # Neuron-specific optimization (nested in additional_config)
        # pa_num_blocks must match num_gpu_blocks_override
        "additional_config": {
            "override_neuron_config": {
                "skip_warmup": True,
                "pa_num_blocks": config['vllm']['num_gpu_blocks_override'],
                "pa_block_size": 32,
                "enable_bucketing": True,
            }
        }
    }

    # Hardware-aware configuration builder
    return VLLMConfigHelper(vllm_params).build()


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


@pytest.fixture(scope="session")
def profiler_config(config):
    """Profiler configuration for cache management."""
    return config.get('profiler', {
        'clear_cache_before': False,
        'clear_cache_after': False,
    })
