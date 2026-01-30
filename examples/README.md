# benchmark-capture Examples

This directory contains example benchmarks demonstrating benchmark-capture usage.

## Project Initialization

Start a new benchmark project using templates:

```bash
# Install with init support
pip install benchmark-capture[init]

# List available templates
benchmark-capture-init --list

# Create vLLM-Neuron project (AWS Inferentia)
benchmark-capture-init ./vllm-neuron-benchmarks --template vllm-neuron

# Create basic project
benchmark-capture-init ./my-benchmarks --template basic
```

**Available templates:**
- **basic** - Simple benchmark setup
- **vllm** - vLLM benchmarks (GPU/CPU)
- **vllm-neuron** - vLLM-Neuron benchmarks (AWS Inferentia)
- **minimal** - Minimal configuration only

## Quick Start

```bash
# Run all examples
pytest examples/ --benchmark-only -v

# Run specific example
pytest examples/test_simple.py --benchmark-only

# Save results to JSON
pytest examples/test_simple.py --benchmark-only --benchmark-json=results.json

# Compare with baseline
pytest examples/test_simple.py --benchmark-only --benchmark-compare=baseline.json
```

## Examples

### 1. test_simple.py
Basic examples demonstrating:
- Auto-detection (`@profile()`)
- Explicit profiler (`@profile("noop")`)
- Parametrized benchmarks
- Custom output directories
- Custom metrics in `benchmark.extra_info`

```bash
pytest examples/test_simple.py --benchmark-only -v
```

### 2. test_hardware_detection.py
Hardware detection diagnostic tool:
- Shows what profiler would be auto-detected
- Explains detection logic
- Provides usage examples

```bash
python3 examples/test_hardware_detection.py
```

### 3. test_with_neuron.py
AWS Neuron profiling examples (requires Inferentia):
- Auto-detected Neuron profiling
- Explicit Neuron profiler with custom options
- Custom timeout and output directory

```bash
pytest examples/test_with_neuron.py --benchmark-only -v
```

## Profiler Selection

### Auto-Detection (Recommended)
```python
@profile()  # Detects hardware automatically
def test_benchmark(benchmark):
    result = benchmark(compute_function)
```

### Explicit Selection
```python
@profile("neuron")     # Force AWS Neuron
@profile("nsight")     # Force NVIDIA NSight
@profile("noop")       # Disable profiling
def test_benchmark(benchmark):
    result = benchmark(compute_function)
```

### Custom Options
```python
@profile("neuron", output_dir="/custom/path", timeout=1200)
def test_benchmark(benchmark):
    result = benchmark(compute_function)
```

## Environment Variables

Override profiler selection:
```bash
# Force specific profiler
export BENCHMARK_PROFILER=noop
pytest examples/ --benchmark-only

# Use default (auto-detection)
unset BENCHMARK_PROFILER
pytest examples/ --benchmark-only
```

## Configuration File

Create `benchmark.toml` in your project root:
```toml
[profiler]
backend = "auto"  # or "neuron", "nsight", "noop"

[profiler.neuron]
output_dir = "/tmp/neuron_profiles"
timeout = 600

[profiler.nsight]
output_dir = "/tmp/nsight_profiles"
cuda_api_trace = true
```

## Custom Metrics

Add custom metrics to benchmarks:
```python
@profile()
def test_throughput(benchmark):
    result = benchmark(process_items, items)

    # Add custom metrics
    benchmark.extra_info["throughput"] = len(items) / benchmark.stats.stats.mean
    benchmark.extra_info["items_processed"] = len(items)
```

## Output

Profiling metadata is saved to:
- Default: `/tmp/profiles/metadata.json`
- Custom: `<output_dir>/metadata.json`

Metadata includes:
- Function name
- Profiler type
- Profile file paths
- Configuration options

## pytest-benchmark Features

All pytest-benchmark features work:
```bash
# Compare results
pytest examples/ --benchmark-compare=baseline.json

# Generate histogram
pytest examples/ --benchmark-histogram

# Save results
pytest examples/ --benchmark-save=experiment_v1

# Autosave results
pytest examples/ --benchmark-autosave

# Only run benchmarks
pytest examples/ --benchmark-only

# Skip benchmarks
pytest examples/ --benchmark-skip
```

## Tips

1. **Use auto-detection** for portable code
2. **Add custom metrics** via `benchmark.extra_info`
3. **Save results** with `--benchmark-json` for analysis
4. **Use parametrize** for parameter sweeps
5. **Disable profiling** with `@profile("noop")` for quick tests
