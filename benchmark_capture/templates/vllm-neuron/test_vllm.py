"""
vLLM-Neuron benchmark tests for {{ project_name }}.

Run with:
    pytest --benchmark-only -v
"""

import gc

import pytest
import vllm
from benchmark_capture import profile


@pytest.mark.benchmark(group="inference")
@pytest.mark.vllm
@pytest.mark.neuron
@profile()  # Auto-detect Neuron hardware
def test_vllm_neuron_inference(benchmark, model_path, vllm_config, sample_prompts):
    """vLLM-Neuron inference benchmark."""

    def setup():
        llm = vllm.LLM(model=model_path, **vllm_config)
        return llm

    def run_inference(llm):
        outputs = llm.generate(sample_prompts)
        return outputs

    def teardown(llm):
        del llm
        gc.collect()

    # Setup
    llm = setup()

    try:
        # Benchmark
        outputs = benchmark(run_inference, llm)

        # Custom metrics
        num_outputs = len(outputs)
        duration = benchmark.stats.stats.mean
        benchmark.extra_info["throughput_qps"] = num_outputs / duration
        benchmark.extra_info["num_prompts"] = len(sample_prompts)
        benchmark.extra_info["tensor_parallel_degree"] = vllm_config["tensor_parallel_degree"]

        assert len(outputs) == len(sample_prompts)

    finally:
        teardown(llm)


@pytest.mark.benchmark(group="tensor_parallel")
@pytest.mark.vllm
@pytest.mark.neuron
@pytest.mark.parametrize("tp_degree", [2, 4, 8])
@profile("neuron")
def test_vllm_neuron_tensor_parallel(benchmark, model_path, vllm_config, tp_degree):
    """vLLM-Neuron tensor parallelism sweep."""

    def setup():
        config = vllm_config.copy()
        config["tensor_parallel_degree"] = tp_degree
        llm = vllm.LLM(model=model_path, **config)
        prompts = ["Test prompt for tensor parallel benchmark"]
        return llm, prompts

    def run_inference(llm, prompts):
        return llm.generate(prompts)

    def teardown(llm):
        del llm
        gc.collect()

    # Setup
    llm, prompts = setup()

    try:
        # Benchmark
        outputs = benchmark(run_inference, llm, prompts)

        # Metrics
        benchmark.extra_info["tensor_parallel_degree"] = tp_degree
        benchmark.extra_info["throughput_qps"] = len(outputs) / benchmark.stats.stats.mean

        assert len(outputs) == len(prompts)

    finally:
        teardown(llm)


@pytest.mark.benchmark(group="batch_size")
@pytest.mark.vllm
@pytest.mark.neuron
@pytest.mark.parametrize("batch_size", [1, 2, 4, 8])
@profile("neuron")
def test_vllm_neuron_batch_sizes(benchmark, model_path, vllm_config, batch_size):
    """vLLM-Neuron batch size sweep."""

    def setup():
        config = vllm_config.copy()
        config["max_num_seqs"] = batch_size
        llm = vllm.LLM(model=model_path, **config)
        prompts = ["Test prompt"] * batch_size
        return llm, prompts

    def run_inference(llm, prompts):
        return llm.generate(prompts)

    def teardown(llm):
        del llm
        gc.collect()

    # Setup
    llm, prompts = setup()

    try:
        # Benchmark
        outputs = benchmark(run_inference, llm, prompts)

        # Metrics
        benchmark.extra_info["batch_size"] = batch_size
        benchmark.extra_info["throughput_qps"] = len(outputs) / benchmark.stats.stats.mean
        benchmark.extra_info["tensor_parallel_degree"] = vllm_config["tensor_parallel_degree"]

        assert len(outputs) == batch_size

    finally:
        teardown(llm)
