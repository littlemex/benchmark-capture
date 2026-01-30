# {{ project_name }}

vLLM-Neuron benchmark project for AWS Inferentia/Trainium created with benchmark-capture.

## Prerequisites

### Hardware
- AWS Inferentia (inf2 instances) or Trainium (trn1 instances)
- Recommended: inf2.8xlarge or larger

### Software
```bash
# Install AWS Neuron SDK
# See: https://awsdocs-neuron.readthedocs-hosted.com/en/latest/

# Install vLLM with Neuron support
pip install vllm-neuron

# Install benchmarking tools
pip install pytest pytest-benchmark benchmark-capture
```

## Configuration

Edit `conftest.py` to configure:
- `model_path`: Path to your vLLM model (default: `{{ model_path }}`)
- `vllm_config`: vLLM-Neuron configuration parameters
  - `tensor_parallel_degree`: Number of NeuronCores to use (2, 4, 8)
  - `max_num_seqs`: Maximum batch size
  - `max_model_len`: Maximum sequence length
  - `block_size`: KV cache block size

## Running Benchmarks

```bash
# Run all benchmarks
pytest --benchmark-only

# Run specific group
pytest --benchmark-only -m neuron

# Tensor parallel sweep
pytest test_vllm.py::test_vllm_neuron_tensor_parallel --benchmark-only

# Batch size sweep
pytest test_vllm.py::test_vllm_neuron_batch_sizes --benchmark-only

# Save results
pytest --benchmark-only --benchmark-json=results.json
```

## Profiler Configuration

Current profiler: `{{ profiler }}`

Profiles are saved to: `{{ profiler_output_dir }}`

Profile files (.ntff) can be analyzed with:
```bash
# View profile
neuron-profile view {{ profiler_output_dir }}/<profile>.ntff

# Convert to JSON
neuron-profile view -n {{ profiler_output_dir }}/<profile>.ntff -o profile.json
```

## Neuron-Specific Notes

### Compilation Cache
- First run will compile the model (slow)
- Subsequent runs use cached graphs (fast)
- Cache location: `/var/tmp/neuron-compile-cache/`

### Tensor Parallelism
- Must match number of available NeuronCores
- inf2.xlarge: 2 cores
- inf2.8xlarge: 2 cores
- inf2.24xlarge: 12 cores
- inf2.48xlarge: 24 cores

### Memory Considerations
- Each NeuronCore has limited memory
- Adjust `max_model_len` if running out of memory
- Use smaller `block_size` for larger models

## Custom Benchmarks

Add your benchmarks to `test_vllm.py`:

```python
@pytest.mark.benchmark
@pytest.mark.neuron
@profile()
def test_my_vllm_neuron_benchmark(benchmark, model_path, vllm_config):
    # Setup
    llm = vllm.LLM(model=model_path, **vllm_config)

    # Benchmark
    result = benchmark(llm.generate, ["My prompt"])

    # Cleanup
    del llm
```

## Directory Structure

```
{{ project_name }}/
├── benchmark.toml      # Neuron profiler configuration
├── pytest.ini          # pytest settings
├── conftest.py         # vLLM-Neuron fixtures
├── test_vllm.py        # Benchmark tests
└── README.md           # This file
```

## Troubleshooting

### Model compilation fails
- Check model compatibility with vLLM-Neuron
- Ensure sufficient NeuronCore memory
- Verify tensor_parallel_degree matches available cores

### Out of memory
- Reduce `max_model_len`
- Reduce `max_num_seqs`
- Use smaller `block_size`

### Slow compilation
- First run compiles model (expected)
- Check `/var/tmp/neuron-compile-cache/` for cached graphs
- Increase timeout in benchmark.toml if needed

## Documentation

- [vLLM-Neuron Documentation](https://github.com/vllm-project/vllm)
- [AWS Neuron SDK](https://awsdocs-neuron.readthedocs-hosted.com/)
- [benchmark-capture](https://github.com/yourusername/benchmark-capture)
- [pytest-benchmark](https://pytest-benchmark.readthedocs.io/)

## Tips

1. **Warm-up**: First benchmark run includes compilation time
2. **Caching**: Reuse same model/config for consistent results
3. **Profiling**: Check `{{ profiler_output_dir }}` for detailed profiles
4. **Monitoring**: Use `neuron-top` to monitor NeuronCore utilization
5. **Batch Size**: Larger batches = higher throughput but higher latency
