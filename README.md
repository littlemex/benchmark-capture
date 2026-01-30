# benchmark-capture

Lightweight profiling decorator for pytest-benchmark with automatic hardware detection.

**Primary use case:** vLLM and vLLM-Neuron benchmarking on AWS Inferentia/Trainium and NVIDIA GPUs. While the core library is general-purpose and works with any Python code, the templates and examples focus on LLM inference workloads.

## Features

- 🚀 **Zero configuration** - Auto-detects hardware (AWS Neuron, NVIDIA GPU, CPU)
- 🎯 **pytest-benchmark integration** - Works seamlessly with pytest-benchmark
- 🔧 **Minimal code pollution** - Single decorator line
- 🌍 **Portable** - Same code works on different hardware
- 📦 **Lightweight** - No heavy dependencies
- 🧠 **vLLM-optimized** - Templates and examples for vLLM/vLLM-Neuron inference

## Use Cases

### Primary Target
- **vLLM-Neuron** on AWS Inferentia/Trainium (inf2, trn1 instances)
- **vLLM** on NVIDIA GPUs (A100, H100)
- LLM inference benchmarking with profiling

### Also Works With
- Any PyTorch model
- General Python function benchmarking
- Custom ML inference workloads
- Hardware-agnostic benchmarking

The core `@profile()` decorator is general-purpose, but templates and documentation focus on vLLM inference workloads.

## Installation

### Using uv (Recommended - 100x faster)

```bash
# Install uv if not already installed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment
uv venv

# Install from TestPyPI
uv pip install \
  --index-url https://test.pypi.org/simple/ \
  benchmark-capture

# With CLI support
uv pip install click jinja2
```

### Using pip

```bash
pip install benchmark-capture
```

With project initialization support:
```bash
pip install benchmark-capture[init]
```

For development:
```bash
git clone https://github.com/yourusername/benchmark-capture.git
cd benchmark-capture
pip install -e ".[dev]"
```

## Quick Start

### vLLM-Neuron Example (AWS Inferentia/Trainium)

**IMPORTANT**: vLLM-Neuron requires AWS Neuron SDK virtual environment.

```bash
# 1. Activate AWS Neuron vLLM environment
source /opt/aws_neuronx_venv_pytorch_inference_vllm_0_13/bin/activate

# 2. Install benchmark-capture (if not already installed)
pip install --index-url https://test.pypi.org/simple/ benchmark-capture
pip install pytest pytest-benchmark
```

**Note**: The path `/opt/aws_neuronx_venv_pytorch_inference_vllm_0_13` may vary. Check `/opt/` for available Neuron environments.

Write your test:

```python
import pytest
import vllm
from benchmark_capture import profile

@pytest.mark.benchmark
@profile()  # Auto-detects Neuron hardware and profiles accordingly
def test_vllm_neuron_inference(benchmark, model_path):
    # vLLM-Neuron configuration
    llm = vllm.LLM(
        model=model_path,
        device="neuron",
        tensor_parallel_degree=2
    )

    result = benchmark(llm.generate, ["test prompt"])
    benchmark.extra_info["throughput"] = len(result) / benchmark.stats.stats.mean
```

Run with pytest-benchmark:
```bash
pytest test_vllm.py --benchmark-only --benchmark-json=results.json
```

### vLLM Example (NVIDIA GPU)

For GPU benchmarking, no special environment setup needed:

```python
import pytest
import vllm
from benchmark_capture import profile

@pytest.mark.benchmark
@profile()  # Auto-detects CUDA and uses NSight profiler
def test_vllm_gpu_inference(benchmark):
    llm = vllm.LLM(model="model_path", tensor_parallel_size=1)
    result = benchmark(llm.generate, ["test prompt"])
    benchmark.extra_info["throughput"] = len(result) / benchmark.stats.stats.mean
```

### General Python Function

The decorator works with any Python code:

```python
@profile()
def test_general_computation(benchmark):
    result = benchmark(my_function, arg1, arg2)
```

## Project Initialization

Scaffold a new benchmark project with vLLM-focused templates:

### Using uv with AWS Neuron

For AWS Inferentia/Trainium:

```bash
# 1. Activate Neuron environment first
source /opt/aws_neuronx_venv_pytorch_inference_vllm_0_13/bin/activate

# 2. Install dependencies in Neuron venv
pip install --index-url https://test.pypi.org/simple/ benchmark-capture
pip install click jinja2 pytest pytest-benchmark

# 3. Create vLLM-Neuron project
benchmark-capture-init ./vllm-benchmarks --template vllm-neuron

# 4. Run benchmarks
cd vllm-benchmarks
pytest --benchmark-only
```

### Using uv (General)

For local development or GPU:

```bash
# Install dependencies
uv pip install \
  --index-url https://test.pypi.org/simple/ \
  benchmark-capture
uv pip install click jinja2 pytest pytest-benchmark

# Create vLLM project (for NVIDIA GPU/CPU)
benchmark-capture-init ./vllm-benchmarks --template vllm

# Create basic project (general-purpose)
benchmark-capture-init ./benchmarks --template basic

# List available templates
benchmark-capture-init --list
```

### Using pip

```bash
# Install with init support
pip install benchmark-capture[init]

# Create project
benchmark-capture-init ./vllm-benchmarks --template vllm-neuron
```

**Available templates:**
- **vllm-neuron** - vLLM-Neuron benchmarks (AWS Inferentia) - **Recommended**
- **vllm** - vLLM benchmarks (GPU/CPU)
- **basic** - Simple benchmark setup (general-purpose)
- **minimal** - Minimal configuration only

Each template creates a ready-to-use project with:
- `benchmark.toml` - Profiler configuration
- `pytest.ini` - pytest settings
- `conftest.py` - Test fixtures
- `test_*.py` - Example benchmarks
- `README.md` - Template-specific documentation

See `examples/` directory for real-world usage examples.

## Supported Profilers

### Auto-Detection

The `@profile()` decorator automatically detects available hardware:

1. **AWS Neuron** - If `torch_neuronx` or `/opt/aws/neuron/` detected
2. **NVIDIA GPU** - If CUDA or `nvidia-smi` detected
3. **No-op** - Fallback (no profiling overhead)

### Manual Override

Force a specific profiler:

```python
@profile("neuron")      # Force AWS Neuron profiling
@profile("nsight")      # Force NVIDIA NSight Systems
@profile("noop")        # Disable profiling
```

### Configuration Priority

```
1. Decorator parameter:  @profile("neuron")
2. Environment variable: BENCHMARK_PROFILER=neuron
3. Config file:          benchmark.toml
4. Auto-detection:       Hardware detection
5. Fallback:             noop (no profiling)
```

## Configuration File (Optional)

Create `benchmark.toml` in your project root:

```toml
[profiler]
backend = "auto"  # Options: "auto", "neuron", "nsight", "noop"

[profiler.neuron]
output_dir = "/tmp/neuron_profiles"
timeout = 600

[profiler.nsight]
output_dir = "/tmp/nsight_profiles"
cuda_api_trace = true
```

## Advanced Usage

### Custom Output Directory

```python
@profile("neuron", output_dir="/custom/path")
def test_custom_output(benchmark):
    ...
```

### Profiler-Specific Options

```python
@profile("neuron", timeout=1200, framework_profile=True)
def test_with_options(benchmark):
    ...
```

### Disable Profiling for Quick Tests

```python
@profile("noop")  # No profiling overhead
def test_quick(benchmark):
    ...
```

### Environment Variable Override

```bash
# Disable profiling in CI/CD
export BENCHMARK_PROFILER=noop
pytest tests/
```

## AWS Neuron Environment Setup

**CRITICAL**: When using vLLM-Neuron on AWS Inferentia/Trainium, you **MUST** activate the Neuron SDK virtual environment before running benchmarks.

### Why is this required?

- vLLM-Neuron depends on AWS Neuron SDK (`torch_neuronx`, `neuronx-cc`)
- These packages are pre-installed in `/opt/aws_neuronx_venv_*` on Neuron instances
- The Neuron SDK cannot be installed via standard `pip install`

### Setup Steps

```bash
# 1. Check available Neuron environments
ls /opt/ | grep neuronx_venv

# Common paths:
# - /opt/aws_neuronx_venv_pytorch_inference_vllm_0_13  (vLLM 0.13)
# - /opt/aws_neuronx_venv_pytorch_2_9                 (PyTorch 2.9)

# 2. Activate the vLLM-Neuron environment
source /opt/aws_neuronx_venv_pytorch_inference_vllm_0_13/bin/activate

# 3. Verify activation
which python3  # Should show /opt/aws_neuronx_venv_*/bin/python3
python3 -c "import torch_neuronx; print('✓ Neuron SDK available')"

# 4. Install benchmark-capture (AFTER activation)
pip install --index-url https://test.pypi.org/simple/ benchmark-capture
pip install pytest pytest-benchmark

# 5. Run benchmarks
pytest --benchmark-only
```

### Alternative: Shell Script Wrapper

Create a wrapper script to ensure Neuron environment is activated:

```bash
#!/bin/bash
# run_benchmark.sh

set -e

NEURON_VENV="/opt/aws_neuronx_venv_pytorch_inference_vllm_0_13"

if [ ! -f "$NEURON_VENV/bin/activate" ]; then
    echo "Error: Neuron venv not found at $NEURON_VENV"
    exit 1
fi

echo "Activating Neuron environment..."
source "$NEURON_VENV/bin/activate"

echo "Running benchmarks..."
pytest "$@"
```

Usage:
```bash
chmod +x run_benchmark.sh
./run_benchmark.sh --benchmark-only -v
```

### Troubleshooting

**Error: `ModuleNotFoundError: No module named 'torch_neuronx'`**
- Solution: Activate Neuron venv before running tests

**Error: `vllm` not found**
- Solution: Use `/opt/aws_neuronx_venv_pytorch_inference_vllm_*` (includes vLLM)

**Wrong venv path**
- Check available environments: `ls /opt/ | grep neuronx_venv`
- Use the vLLM-specific environment (contains `vllm` in the name)

## Hardware-Agnostic Code

Write once, run anywhere:

```python
# Same code works on AWS Inferentia and NVIDIA GPU
@profile()  # Auto-detects hardware
def test_inference(benchmark):
    llm = vllm.LLM(model="model")  # vLLM adapts automatically
    result = benchmark(llm.generate, prompts)
```

**On AWS Inferentia:**
- Detects Neuron → Sets `NEURON_PROFILE` → Saves `.ntff` files

**On NVIDIA GPU:**
- Detects CUDA → Sets `NSYS_*` → Saves `.nsys-rep` files

**On CPU:**
- No profiler → No overhead → Tests run normally

## Integration with pytest-benchmark

benchmark-capture enhances pytest-benchmark without replacing it:

```python
@pytest.mark.benchmark(group="inference")
@profile()
def test_inference(benchmark):
    result = benchmark.pedantic(
        llm.generate,
        args=(prompts,),
        iterations=10,
        rounds=5
    )

    # Add custom metrics
    benchmark.extra_info["throughput_qps"] = len(result) / benchmark.stats.stats.mean
```

All pytest-benchmark features work:
```bash
pytest --benchmark-only
pytest --benchmark-compare=baseline.json
pytest --benchmark-histogram
pytest --benchmark-save=experiment_v003
```

## Metadata Output

Each profiling run saves metadata:

```json
{
  "function": "test_inference",
  "profiler": "NeuronProfiler",
  "output_dir": "/tmp/neuron_profiles",
  "profiler_type": "neuron",
  "profile_files": ["profile.ntff"],
  "timeout": 600
}
```

## Development

### Setup Development Environment

Using uv (recommended):
```bash
# Clone repository
git clone https://github.com/yourusername/benchmark-capture.git
cd benchmark-capture

# Create virtual environment
uv venv

# Install in editable mode with dev dependencies
uv pip install -e ".[dev]"
```

Using pip:
```bash
pip install -e ".[dev]"
```

### Run Tests

```bash
# All tests
pytest

# With coverage
pytest --cov=benchmark_capture --cov-report=html

# Specific test
pytest tests/test_decorators.py::test_auto_detect
```

## License

Apache 2.0 License

## Contributing

Contributions welcome! Please see CONTRIBUTING.md for guidelines.

## Related Projects

- [vLLM](https://github.com/vllm-project/vllm) - High-performance LLM inference (primary target)
- [pytest-benchmark](https://github.com/ionelmc/pytest-benchmark) - Benchmarking framework (foundation)
- [benchmark-analyze](https://github.com/yourusername/benchmark-analyze) - Analysis companion library
- [AWS Neuron SDK](https://awsdocs-neuron.readthedocs-hosted.com/) - AWS Inferentia/Trainium profiling

## Background

This library was developed from experience benchmarking vLLM-Neuron workloads on AWS Inferentia. While the core is general-purpose, it's optimized for LLM inference profiling across different hardware platforms.
