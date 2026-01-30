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

# Install from TestPyPI (latest: v0.2.0)
uv pip install \
  --index-url https://test.pypi.org/simple/ \
  --extra-index-url https://pypi.org/simple/ \
  'benchmark-capture[init]==0.2.0'
```

### Using pip

```bash
# Install from TestPyPI (latest: v0.2.0)
pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ 'benchmark-capture==0.2.0'
```

With project initialization support:
```bash
pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ 'benchmark-capture[init]==0.2.0'
```

For development:
```bash
git clone https://github.com/littlemex/benchmark-capture.git
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
pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ 'benchmark-capture==0.2.0'
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

Scaffold a new benchmark project from templates or deploy ready-to-use examples.

### Using uv with AWS Neuron

For AWS Inferentia/Trainium:

```bash
# 1. Activate Neuron environment first
source /opt/aws_neuronx_venv_pytorch_inference_vllm_0_13/bin/activate

# 2. Install dependencies in Neuron venv
pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ 'benchmark-capture[init]==0.2.0'
pip install pytest pytest-benchmark

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
  --extra-index-url https://pypi.org/simple/ \
  'benchmark-capture[init]==0.2.0'
uv pip install pytest pytest-benchmark

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
pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ 'benchmark-capture[init]==0.2.0'

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

### Deploy Ready-to-Use Examples

For quick start with real-world benchmarks, deploy complete examples:

```bash
# List available examples
benchmark-capture-init --list-examples

# Deploy vLLM-Neuron Reranker example (includes CSV data!)
benchmark-capture-init ./my-reranker --example vllm-neuron-reranker

# Follow example-specific setup
cd my-reranker
# 1. Activate Neuron venv
# 2. Edit config.yaml (set MODEL_PATH)
# 3. Run: pytest test_reranker.py --benchmark-only -v
```

**Available examples:**
- **vllm-neuron-reranker** - Complete Qwen3-Reranker benchmark
  - Includes sample CSV data (10 queries, 20 candidates each)
  - Configuration-driven (config.yaml)
  - **Perfetto mode enabled by default** for NTFF generation
  - Production-ready code
  - Just download model from Hugging Face and run!

**Example vs Template:**
- **Example** = Complete, runnable benchmark (data included)
- **Template** = Starting structure (you add data/model)

See `examples/` directory for browsing examples before deploying.

### Complete End-to-End Example: vLLM-Neuron Reranker

This walkthrough demonstrates the complete workflow from installation to running benchmarks on AWS Inferentia2.

**Prerequisites:**
- AWS Inferentia2 instance (inf2.xlarge or larger)
- Hugging Face account with access to model

**Step 1: Install benchmark-capture**

```bash
# Activate AWS Neuron vLLM environment
source /opt/aws_neuronx_venv_pytorch_inference_vllm_0_13/bin/activate

# Install from TestPyPI with init extras
pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ 'benchmark-capture[init]'
```

**Step 2: Download Model from Hugging Face**

```bash
# Install huggingface-cli if not available
pip install 'huggingface_hub[cli]'

# Login with your Hugging Face token
# Get your token from: https://huggingface.co/settings/tokens
huggingface-cli login
# Paste your token when prompted

# Create models directory
mkdir -p ~/models

# Download Qwen3-Reranker model
# Note: Check Hugging Face for the correct model repository name
huggingface-cli download \
  Alibaba-NLP/gte-Qwen3-0.6B-reranker \
  --local-dir ~/models/Qwen3-0.6B-Reranker \
  --local-dir-use-symlinks False

# Verify download
ls ~/models/Qwen3-0.6B-Reranker/
# You should see: config.json, model.safetensors, tokenizer.json, etc.
```

**Step 3: Deploy Example**

```bash
# List available examples
benchmark-capture-init --list-examples

# Deploy the reranker example to your home directory
benchmark-capture-init ~/reranker-benchmark --example vllm-neuron-reranker
cd ~/reranker-benchmark

# Verify files are deployed
ls -la
# You should see: config.yaml, test_reranker.py, input_sample.csv, etc.
```

**Step 4: Configure Model Path**

The config.yaml contains a placeholder `{{ MODEL_PATH }}` that needs to be replaced with your actual model path.

**Option A: Edit manually with your preferred editor**
```bash
# Open config.yaml in your editor
vim config.yaml  # or nano, emacs, etc.

# Find the line:
#   path: "{{ MODEL_PATH }}"
# Replace with your actual path:
#   path: "/home/coder/models/Qwen3-0.6B-Reranker"
```

**Option B: Use sed for automatic replacement**
```bash
# Replace placeholder with actual path using sed
sed -i 's|{{ MODEL_PATH }}|/home/coder/models/Qwen3-0.6B-Reranker|g' config.yaml

# Verify the change
grep "path:" config.yaml
# Should show: path: "/home/coder/models/Qwen3-0.6B-Reranker"
```

**Verify configuration:**
```bash
# Check the model path is correctly set
cat config.yaml | grep -A 2 "^model:"
# Output should be:
# model:
#   # Path to the reranker model
#   path: "/home/coder/models/Qwen3-0.6B-Reranker"
```

**Step 5: Install Dependencies**

```bash
pip install -r requirements.txt
```

**Step 6: Run Benchmark**

```bash
# Run with real-time logging (recommended for long-running benchmarks)
pytest test_reranker.py --benchmark-only --benchmark-json=results.json -v

# For quieter output (only show warnings and results)
pytest test_reranker.py --benchmark-only --benchmark-json=results.json -v --log-cli-level=WARNING
```

**What You'll See:**
- Model initialization and compilation (10-15 minutes first run)
- Real-time progress updates during benchmark
- Per-query latency metrics
- Final benchmark results with statistics

**Example Output:**
```
============================= test session starts ==============================
test_reranker.py::test_vllm_neuron_reranker
-------------------------------- live log call ---------------------------------
INFO     test_reranker:test_reranker.py:51 Loaded 10 queries from input_sample.csv
INFO     test_reranker:test_reranker.py:56 Initializing vLLM-Neuron reranker...
INFO     test_reranker:test_reranker.py:57 Model: /path/to/Qwen3-0.6B-Reranker
...
PASSED

----------------------- benchmark: 1 tests ------------------------
Name (time in s)                     Min      Max     Mean   StdDev
--------------------------------------------------------------------
test_vllm_neuron_reranker         3.042    3.062    3.052    0.010
--------------------------------------------------------------------
```

**Results:**
- `results.json` - Detailed benchmark data with statistics
  ```bash
  # View summary
  cat results.json | python3 -c "
  import json, sys
  b = json.load(sys.stdin)['benchmarks'][0]
  print(f\"Mean: {b['stats']['mean']:.4f}s\")
  print(f\"Throughput: {b['extra_info']['throughput_qps']:.2f} QPS\")
  print(f\"Latency/query: {b['extra_info']['latency_per_query_ms']:.2f}ms\")
  "
  ```

- **Perfetto Mode Output** - NTFF files in session directory (default in reranker example)
  ```bash
  # Check session directory was created (Perfetto mode)
  ls -la ./benchmarks/i-*_pid_*/
  # Expected: neff_*.ntff files

  # Verify Perfetto mode is enabled
  cat ./benchmarks/metadata.json | jq '.perfetto_mode'
  # Expected: true

  # View session directory path
  cat ./benchmarks/metadata.json | jq '.session_dir'
  # Expected: "./benchmarks/i-<instance>_pid_<number>"

  # List all NTFF files
  cat ./benchmarks/metadata.json | jq '.ntff_files'
  # Expected: ["i-.../neff_001.ntff", "i-.../neff_002.ntff", ...]
  ```

**Neuron Cache Management:**

First run will compile the model (10-15 minutes). Subsequent runs reuse the cache. To clear cache:

```yaml
# In config.yaml, set:
profiler:
  clear_cache_before: true  # Recompile on next run
  clear_cache_after: false  # Keep cache for future runs
```

**Note:** Cache clearing is useful when:
- Model configuration changed (batch size, max length, etc.)
- Neuron SDK version updated
- Testing clean compilation performance

### Troubleshooting

**Problem: Model not found or 404 error during download**
```bash
# Error: Repository Not Found for url: https://huggingface.co/...

# Solution 1: Check the exact model name on Hugging Face
# Search for the model on https://huggingface.co/models
# Copy the exact repository name (e.g., "Alibaba-NLP/gte-Qwen3-0.6B-reranker")

# Solution 2: Verify you're logged in with correct token
huggingface-cli whoami
huggingface-cli login --token YOUR_TOKEN

# Solution 3: Check if the model requires access request
# Some models need you to request access on the Hugging Face website first
```

**Problem: Config path not updated ({{ MODEL_PATH }} still showing)**
```bash
# Verify the path replacement worked
grep "{{ MODEL_PATH }}" config.yaml
# If this returns a match, the placeholder is still there

# Fix: Manually edit or re-run sed
sed -i 's|{{ MODEL_PATH }}|/home/coder/models/Qwen3-0.6B-Reranker|g' config.yaml

# Important: Use your actual model path, not the example path
# Check where you downloaded the model:
ls ~/models/
```

**Problem: Permission denied when installing benchmark-capture**
```bash
# Error: Permission denied: '/opt/aws_neuronx_venv_pytorch_inference_vllm_0_13/bin/...'

# Solution: The package is already installed but an older version
# Check current version:
python3 -c "import benchmark_capture; print(benchmark_capture.__version__)"

# For testing, use local development install:
cd /path/to/benchmark-capture
pip install -e .
```

**Problem: Model compilation takes too long**
```bash
# First compilation can take 10-15 minutes - this is normal!
# The model is being compiled to Neuron format.

# Check compilation progress:
# Look for log messages like:
# INFO Neuron:model_builder.py:... Generating HLOs for the following models...

# Subsequent runs will be much faster (seconds) as they reuse cached compilation
```

**Problem: ImportError or ModuleNotFoundError**
```bash
# Make sure you're in the Neuron venv
source /opt/aws_neuronx_venv_pytorch_inference_vllm_0_13/bin/activate

# Verify required packages
pip list | grep -E "pytest|vllm|benchmark"

# Install missing dependencies
pip install -r requirements.txt
```

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

# Compilation cache management (Neuron only)
# Clear cache before benchmarks (useful when config changes)
clear_cache_before = false
# Clear cache after benchmarks (useful for CI/CD to save disk space)
clear_cache_after = false

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

### Perfetto Mode for Advanced Analysis

Enable Perfetto-compatible NTFF generation for visualization in Perfetto UI:

```python
@profile("neuron", perfetto=True, output_dir="/tmp/profiles")
def test_perfetto_profile(benchmark):
    result = benchmark(llm.generate, prompts)
    # Generates session directory: /tmp/profiles/i-<instance>_pid_<number>/
    # Contains NTFF files suitable for Perfetto analysis
```

**Standard vs Perfetto Mode:**

| Feature | Standard Mode | Perfetto Mode |
|---------|--------------|---------------|
| Decorator | `@profile("neuron")` | `@profile("neuron", perfetto=True)` |
| Environment Variables | `NEURON_PROFILE` | `NEURON_RT_INSPECT_*` |
| Output Structure | `/tmp/profiles/*.ntff` | `/tmp/profiles/i-*_pid_*/` |
| Metadata Fields | `profile_files: [...]` | `session_dir: "...", ntff_files: [...]` |
| Use Case | neuron-profile CLI | Perfetto UI / External tools |

**Detailed Comparison:**

- **Standard mode** (`perfetto=False`, default):
  - Sets `NEURON_PROFILE` environment variable
  - Generates `.ntff` files directly in output directory
  - Compatible with `neuron-profile` CLI tools
  - Metadata contains `profile_files` list

- **Perfetto mode** (`perfetto=True`):
  - Sets `NEURON_RT_INSPECT_*` environment variables
  - Generates session directory with NTFF files
  - Session directory pattern: `i-<instance>_pid_<number>/`
  - Files ready for Perfetto conversion (use external tools like `neuron-workflow`)
  - Metadata contains `session_dir` and `ntff_files` list

**Verifying Perfetto Mode Output:**

After running your benchmark with `perfetto=True`, verify the output:

```bash
# 1. Check session directory was created
ls -la /tmp/profiles/
# Expected: i-<instance_id>_pid_<process_id>/ directory

# 2. List NTFF files in session directory
ls -la /tmp/profiles/i-*_pid_*/
# Expected: neff_*.ntff files

# 3. Check metadata.json
cat /tmp/profiles/metadata.json | jq
# Expected fields:
# - "perfetto_mode": true
# - "session_dir": "/tmp/profiles/i-..."
# - "ntff_files": ["i-.../neff_001.ntff", ...]
```

**Example metadata.json for Perfetto mode:**

```json
{
  "function": "test_inference",
  "profiler": "NeuronProfiler",
  "perfetto_mode": true,
  "session_dir": "/tmp/profiles/i-0abc123_pid_45678",
  "ntff_files": [
    "i-0abc123_pid_45678/neff_001.ntff",
    "i-0abc123_pid_45678/neff_002.ntff"
  ],
  "timeout": 600,
  "framework_profile": false
}
```

**Example metadata.json for Standard mode:**

```json
{
  "function": "test_inference",
  "profiler": "NeuronProfiler",
  "perfetto_mode": false,
  "profile_files": [
    "/tmp/profiles/profile_001.ntff",
    "/tmp/profiles/profile_002.ntff"
  ],
  "timeout": 600,
  "framework_profile": false
}
```

**Converting NTFF to Perfetto Format:**

```bash
# Using neuron-profile CLI
neuron-profile convert /tmp/profiles/i-*_pid_*/neff_*.ntff -o perfetto.json

# Using neuron-workflow (for advanced analysis)
# Requires neuron-workflow library installation
# See: https://github.com/aws-neuron/neuron-workflow
```

**Troubleshooting:**

- **No session directory created**: Ensure your code actually runs on Neuron hardware. The session directory is only created when Neuron runtime is invoked.
- **Empty ntff_files in metadata**: The benchmark may not have executed long enough. Try increasing benchmark rounds or warmup rounds.
- **Permission denied**: Check that the output_dir is writable by your user.

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

## Real-Time Output & Logging

**For long-running benchmarks** (e.g., vLLM-Neuron compilation, large model inference), real-time progress output is essential. This library follows OSS best practices for pytest output management.

### Best Practice: Use Python Logging

**Instead of `print()`:**
```python
import logging

logger = logging.getLogger(__name__)

@profile()
def test_long_benchmark(benchmark):
    logger.info("Starting model initialization...")
    model = load_model()  # Long operation
    logger.info("Model loaded successfully")

    logger.info("Running inference...")
    result = benchmark(model.predict, data)
    logger.info(f"Completed: {result}")
```

**Why logging instead of print?**
- ✅ Works with pytest's capture system
- ✅ Real-time output via `log_cli = true`
- ✅ Configurable log levels
- ✅ Automatic timestamps
- ✅ Can write to both console and file

### Configuration (pyproject.toml)

```toml
[tool.pytest.ini_options]
# Enable real-time logging to console
log_cli = true
log_cli_level = "INFO"
log_cli_format = "%(asctime)s [%(levelname)s] %(message)s"
log_cli_date_format = "%H:%M:%S"

# Optional: Also write to file
log_file = "pytest.log"
log_file_level = "DEBUG"
```

### Run with Different Log Levels

```bash
# Default (INFO level)
pytest test_benchmark.py --benchmark-only

# Verbose (DEBUG level)
pytest test_benchmark.py --benchmark-only --log-cli-level=DEBUG

# Quiet (only WARNING and ERROR)
pytest test_benchmark.py --benchmark-only --log-cli-level=WARNING

# Disable output capture (for print statements)
pytest test_benchmark.py --benchmark-only -s
```

### Common Pattern for Long Benchmarks

```python
import logging

logger = logging.getLogger(__name__)

@pytest.mark.benchmark
@profile()
def test_vllm_neuron_inference(benchmark, model_path, vllm_config):
    """Long-running vLLM-Neuron benchmark with progress updates."""

    def setup():
        logger.info("Initializing vLLM-Neuron (this may take 10-15 minutes)...")
        logger.info(f"Model: {model_path}")
        logger.info(f"Config: {vllm_config}")

        llm = vllm.LLM(model=model_path, **vllm_config)
        logger.info("vLLM-Neuron initialized successfully")
        return llm

    def run_benchmark(llm):
        logger.info("Starting inference benchmark...")
        result = llm.generate(prompts)
        logger.info(f"Completed {len(result)} generations")
        return result

    llm = setup()
    benchmark(run_benchmark, llm)
```

**Output during execution:**
```
12:34:56 [INFO] Initializing vLLM-Neuron (this may take 10-15 minutes)...
12:34:56 [INFO] Model: /path/to/model
12:34:56 [INFO] Config: {'block_size': 32, ...}
12:45:23 [INFO] vLLM-Neuron initialized successfully
12:45:23 [INFO] Starting inference benchmark...
12:45:45 [INFO] Completed 10 generations
```

### Why Not `sys.stdout.flush()`?

Common misconception: Using `sys.stdout.flush()` or `PYTHONUNBUFFERED=1`.

**These don't work** because:
- pytest's capture mechanisms intercept output before flush
- The solution is to use pytest's built-in logging system

### Progress Bars (tqdm)

For loops with progress tracking:

```python
from tqdm import tqdm
import sys

for item in tqdm(items, file=sys.stderr, desc="Processing"):
    # work
    pass

# Or run with: pytest -s (disables capture)
```

### References

This approach is used by major OSS projects:
- **PyTorch**: Uses `--capture=sys` with extensive pytest.ini configuration
- **vLLM**: Uses Python logging with `VLLM_CONFIGURE_LOGGING` env var
- **pytest-benchmark**: Provides `--benchmark-verbose` flag for detailed output
- **HuggingFace Transformers**: Uses logging module throughout benchmark code

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

### Compilation Cache Management

AWS Neuron SDK caches compiled models to speed up subsequent runs. However, the cache must be cleared when:
- Model configuration changes (batch size, sequence length, tensor parallel size)
- Neuron SDK version changes
- Testing clean compilation performance

**Cache Location:**
- Default: `/var/tmp/neuron-compile-cache/`
- Configured via: `NEURON_COMPILE_CACHE_URL` environment variable
- vLLM artifacts: `NEURON_COMPILED_ARTIFACTS` environment variable

**Automatic Cache Clearing:**

Configure in `benchmark.toml`:
```toml
[profiler.neuron]
# Clear cache before running benchmarks
clear_cache_before = true  # Default: false
# Clear cache after benchmarks (useful for CI/CD)
clear_cache_after = false   # Default: false
```

**Manual Cache Clearing:**

```python
from benchmark_capture.cache import clear_neuron_cache

# Clear cache manually
result = clear_neuron_cache()
print(f"Cleared {result['cache_size_mb']:.2f} MB from {result['cache_dir']}")
```

**Command Line:**
```bash
# Manual clearing (requires appropriate permissions)
sudo rm -rf /var/tmp/neuron-compile-cache/*

# Or clear vLLM artifacts
rm -rf $NEURON_COMPILED_ARTIFACTS/*
```

**Important Notes:**
- First run after clearing will recompile (10-15 minutes for large models)
- Subsequent runs use cached graphs (much faster)
- Cache size can grow to several GB
- Clear cache in CI/CD to save disk space
- Permission errors require `sudo` for system-wide cache

**Cache Status Check:**

```python
from benchmark_capture.cache import check_cache_status

status = check_cache_status()
print(f"Cache: {status['cache_size_mb']:.2f} MB")
print(f"Cached models: {status['cached_models_count']}")
```

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
git clone https://github.com/littlemex/benchmark-capture.git
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
