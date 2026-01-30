# {{ project_name }}

vLLM benchmark project created with benchmark-capture.

## Setup

```bash
# Install dependencies
pip install pytest pytest-benchmark benchmark-capture vllm
```

## Configuration

Edit `conftest.py` to configure:
- `model_path`: Path to your vLLM model (default: `{{ model_path }}`)
- `vllm_config`: vLLM configuration parameters

## Running Benchmarks

```bash
# Run all benchmarks
pytest --benchmark-only

# Run specific group
pytest --benchmark-only -m vllm

# Save results
pytest --benchmark-only --benchmark-json=results.json

# Batch size sweep only
pytest test_vllm.py::test_vllm_batch_sizes --benchmark-only
```

## Profiler Configuration

Current profiler: `{{ profiler }}`

Profiles are saved to: `{{ profiler_output_dir }}`

### Change Profiler

```bash
# Use environment variable
BENCHMARK_PROFILER=neuron pytest --benchmark-only

# Or edit benchmark.toml
```

## Custom Benchmarks

Add your benchmarks to `test_vllm.py`:

```python
@pytest.mark.benchmark
@profile()
def test_my_vllm_benchmark(benchmark, model_path, vllm_config):
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
├── benchmark.toml      # Profiler configuration
├── pytest.ini          # pytest settings
├── conftest.py         # Fixtures (configure model here)
├── test_vllm.py        # vLLM benchmarks
└── README.md           # This file
```

## Tips

1. **Model Path**: Update `model_path` fixture in `conftest.py`
2. **Batch Sizes**: Adjust parametrize values in tests
3. **Save Results**: Use `--benchmark-json` for analysis
4. **Profile Data**: Check `{{ profiler_output_dir }}` for profiles

## Documentation

- [vLLM Documentation](https://docs.vllm.ai/)
- [benchmark-capture](https://github.com/yourusername/benchmark-capture)
- [pytest-benchmark](https://pytest-benchmark.readthedocs.io/)
