# {{ project_name }}

Benchmark project created with benchmark-capture.

## Setup

```bash
# Install dependencies
pip install pytest pytest-benchmark benchmark-capture
```

## Running Benchmarks

```bash
# Run all benchmarks
pytest --benchmark-only

# Run specific test
pytest test_example.py::test_simple_computation --benchmark-only

# Save results to JSON
pytest --benchmark-only --benchmark-json=results.json

# Compare with baseline
pytest --benchmark-only --benchmark-compare=baseline.json
```

## Configuration

- **benchmark.toml** - Profiler configuration
- **pytest.ini** - pytest-benchmark settings

Current profiler: `{{ profiler }}`

## Profiler Control

```bash
# Use auto-detection (default)
pytest --benchmark-only

# Force specific profiler
BENCHMARK_PROFILER=neuron pytest --benchmark-only
BENCHMARK_PROFILER=noop pytest --benchmark-only
```

## Adding Custom Metrics

```python
@profile()
def test_my_benchmark(benchmark):
    result = benchmark(my_function, args)

    # Add custom metrics
    benchmark.extra_info["throughput"] = len(result) / benchmark.stats.stats.mean
    benchmark.extra_info["custom_metric"] = calculate_metric(result)
```

## Directory Structure

```
{{ project_name }}/
├── benchmark.toml      # Profiler configuration
├── pytest.ini          # pytest settings
├── conftest.py         # pytest fixtures
├── test_example.py     # Example tests
└── README.md           # This file
```

## Next Steps

1. Edit `test_example.py` with your benchmarks
2. Add fixtures to `conftest.py` as needed
3. Run `pytest --benchmark-only`
4. Analyze results and iterate

## Documentation

- [benchmark-capture](https://github.com/yourusername/benchmark-capture)
- [pytest-benchmark](https://pytest-benchmark.readthedocs.io/)
