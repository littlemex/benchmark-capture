"""
vLLM-Neuron Configuration Sweep Test

Tests different Neuron configurations to profile performance:
- PA block sizes: 256, 512, 1024
- Prefix caching: enabled/disabled
- Flash decoding: enabled/disabled

Each configuration generates separate profiling files for comparison.

Run with:
    pytest test_config_sweep.py --benchmark-only -v
"""

import gc
import logging

import pytest
import vllm
from benchmark_capture import profile
from benchmark_capture.utils import VLLMConfigHelper

logger = logging.getLogger(__name__)


@pytest.mark.benchmark(group="pa_blocks_sweep")
@pytest.mark.vllm
@pytest.mark.neuron
@pytest.mark.parametrize("pa_num_blocks", [256, 512, 1024])
@profile("neuron", perfetto=True)
def test_pa_blocks_sweep(benchmark, model_path, pa_num_blocks):
    """
    PA block数をスイープしてプロファイリング。

    各pa_num_blocksで別々のプロファイルファイルが生成されることを確認。
    """
    logger.info(f"Testing with pa_num_blocks={pa_num_blocks}")

    # VLLMConfigHelperで設定をビルド
    config = VLLMConfigHelper({
        "tensor_parallel_size": 2,
        "max_num_seqs": 4,
        "max_model_len": 2048,
        "block_size": 32,
        "num_gpu_blocks_override": pa_num_blocks,  # Must match pa_num_blocks
        "additional_config": {
            "override_neuron_config": {
                "skip_warmup": True,
                "pa_num_blocks": pa_num_blocks,  # スイープ対象
                "pa_block_size": 32,
                "enable_bucketing": True,
            }
        }
    }).build()

    def setup():
        logger.info(f"Initializing LLM with pa_num_blocks={pa_num_blocks}")
        llm = vllm.LLM(model=model_path, **config)
        return llm

    def run_inference(llm):
        prompts = ["Test prompt for configuration sweep"]
        outputs = llm.generate(prompts)
        return outputs

    def teardown(llm):
        del llm
        gc.collect()

    # Setup
    llm = setup()

    try:
        # Benchmark
        outputs = benchmark(run_inference, llm)

        # Metrics
        benchmark.extra_info["pa_num_blocks"] = pa_num_blocks
        benchmark.extra_info["tensor_parallel_size"] = 2

        logger.info(f"✓ Completed with pa_num_blocks={pa_num_blocks}")

        assert len(outputs) == 1

    finally:
        teardown(llm)


@pytest.mark.benchmark(group="optimization_sweep")
@pytest.mark.vllm
@pytest.mark.neuron
@pytest.mark.parametrize("neuron_config", [
    {"is_prefix_caching": False, "flash_decoding_enabled": False},
    {"is_prefix_caching": True, "flash_decoding_enabled": False},
    {"is_prefix_caching": False, "flash_decoding_enabled": True},
    {"is_prefix_caching": True, "flash_decoding_enabled": True},
], ids=["baseline", "prefix_cache", "flash_decode", "both"])
@profile("neuron", perfetto=True)
def test_optimization_sweep(benchmark, model_path, neuron_config):
    """
    Neuron最適化の組み合わせをスイープ。

    各最適化の組み合わせで別々のプロファイルファイルが生成される。
    """
    config_name = f"prefix={neuron_config.get('is_prefix_caching', False)}_flash={neuron_config.get('flash_decoding_enabled', False)}"
    logger.info(f"Testing configuration: {config_name}")

    # Base config
    base_config = {
        "skip_warmup": True,
        "pa_num_blocks": 512,
        "pa_block_size": 32,
        "enable_bucketing": True,
    }

    # Merge with optimization config
    full_neuron_config = {**base_config, **neuron_config}

    # VLLMConfigHelperで設定をビルド
    config = VLLMConfigHelper({
        "tensor_parallel_size": 2,
        "max_num_seqs": 4,
        "max_model_len": 2048,
        "block_size": 32,
        "additional_config": {
            "override_neuron_config": full_neuron_config
        }
    }).build()

    def setup():
        logger.info(f"Initializing LLM with config: {config_name}")
        llm = vllm.LLM(model=model_path, **config)
        return llm

    def run_inference(llm):
        prompts = ["Test prompt for optimization sweep"]
        outputs = llm.generate(prompts)
        return outputs

    def teardown(llm):
        del llm
        gc.collect()

    # Setup
    llm = setup()

    try:
        # Benchmark
        outputs = benchmark(run_inference, llm)

        # Metrics
        benchmark.extra_info["is_prefix_caching"] = neuron_config.get("is_prefix_caching", False)
        benchmark.extra_info["flash_decoding_enabled"] = neuron_config.get("flash_decoding_enabled", False)

        logger.info(f"✓ Completed configuration: {config_name}")

        assert len(outputs) == 1

    finally:
        teardown(llm)
