# vLLM-Neuron Reranker Benchmark Example

Generic Reranker model benchmark implementation for vLLM-Neuron. Works with Qwen3-Reranker and other reranker models.

## Overview

This example demonstrates how to benchmark Reranker models in vLLM-Neuron environment.

**Key Features:**
- Customizable prompt templates (prefix/suffix)
- Batch processing for memory efficiency
- Per-query latency metrics
- CSV data input support
- Configuration-driven design

## Requirements

- AWS Inferentia2 (inf2.xlarge or larger)
- vLLM 0.13+ with Neuron support
- pytest-benchmark
- benchmark-capture

## File Structure

```
examples/vllm-neuron-reranker/
├── README.md              # This file (complete guide)
├── config.yaml            # Benchmark configuration
├── conftest.py            # pytest fixtures (uses VLLMConfigHelper)
├── test_reranker.py       # Basic reranker test
├── test_config_sweep.py   # Configuration sweep tests
├── input_sample.csv       # Sample data (10 queries)
├── requirements.txt       # Dependencies
└── run_benchmark.sh       # Convenience script for running benchmarks
```

## Features

### VLLMConfigHelper - Hardware-Aware Configuration

This example uses `VLLMConfigHelper` to automatically adapt vLLM configuration to the detected hardware:

- **Neuron environment**: Includes `additional_config` with Neuron optimizations
- **GPU environment**: Excludes Neuron-specific settings automatically
- **Configuration sweep friendly**: Easy to sweep parameters with pytest

### Profiling Support

Each benchmark execution generates profiling data:

- **Perfetto mode enabled**: NTFF files for visualization in Perfetto UI
- **Configuration sweep**: Separate profile files for each configuration
- **Session directory**: `./benchmarks/i-<instance>_pid_<number>/`

## Quick Start

### Option A: Using the Convenience Script (Recommended)

The easiest way to run the benchmark is using the provided `run_benchmark.sh` script:

```bash
# 1. Navigate to the example directory
cd /path/to/benchmark-capture/benchmark_capture/examples/vllm-neuron-reranker

# 2. Edit run_benchmark.sh to set your venv path (if needed)
# Open the script and update VENV_PATH variable at the top

# 3. Run the benchmark
./run_benchmark.sh
```

The script will:
- Activate the Neuron environment
- Set up PYTHONPATH automatically
- Verify imports
- Run pytest with optimal settings for real-time compilation logs
- Display a summary of generated files

**IMPORTANT**: If your Neuron venv is not at the default path (`/opt/aws_neuronx_venv_pytorch_inference_vllm_0_13`), edit `run_benchmark.sh` and update the `VENV_PATH` variable at the top of the file.

### Option B: Manual Setup

If you prefer manual control or need to customize the setup:

### 1. Customize Configuration

Edit `config.yaml` to specify your model and parameters:

```yaml
model:
  path: "/path/to/your/reranker/model"

vllm:
  tensor_parallel_size: 2
  max_num_seqs: 4
  block_size: 32
  max_model_len: 2048
  max_num_batched_tokens: 256

reranker:
  input_file: "input_sample.csv"
  search_num: 20           # Number of candidates per query
  batch_size: 8            # Batch size for processing
  max_length: 1500         # Maximum prompt length
```

### 2. Adjust Prompt Templates

Modify the `reranker_prompts` fixture in `conftest.py` to customize prompt format:

```python
@pytest.fixture(scope="session")
def reranker_prompts():
    return {
        "prefix": "<|im_start|>system\n...",
        "suffix": "<|im_end|>\n...",
        "instruction": "Your custom instruction"
    }
```

### 3. Environment Setup (CRITICAL)

**IMPORTANT**: The vLLM-Neuron environment is system-managed and does not allow standard `pip install`. You **MUST** use `PYTHONPATH` for development.

#### Why PYTHONPATH is Required

The vLLM-Neuron environment (`/opt/aws_neuronx_venv_*`) is externally-managed:

```bash
pip install -e /path/to/benchmark-capture
# ERROR: externally-managed-environment

pip install --user -e /path/to/benchmark-capture
# ERROR: User site-packages are not visible in this virtualenv
```

**Solution**: Use `PYTHONPATH` to include benchmark-capture without installation.

#### Setup Steps

```bash
# 1. Activate vLLM-Neuron environment
source /opt/aws_neuronx_venv_pytorch_inference_vllm_0_13/bin/activate

# 2. Set PYTHONPATH (REQUIRED for development)
export PYTHONPATH=/path/to/benchmark-capture:$PYTHONPATH

# Example (replace with your actual path):
export PYTHONPATH=/path/to/benchmark-capture:$PYTHONPATH

# 3. Set model path (optional - overrides config.yaml)
export RERANKER_MODEL_PATH="/path/to/your/Qwen3-Reranker-0.6B"

# 4. Verify import works
python3 -c "from benchmark_capture.utils import VLLMConfigHelper; print('✓ Setup OK')"
```

**Expected output**: `✓ Setup OK`

#### Common Issues

**ModuleNotFoundError: No module named 'benchmark_capture'**
```bash
# Cause: PYTHONPATH not set
# Solution:
export PYTHONPATH=/path/to/benchmark-capture:$PYTHONPATH
python3 -c "import benchmark_capture; print('OK')"
```

**pytest: error: unrecognized arguments: --cov**
```bash
# Cause: pytest-cov not installed in vLLM environment
# Solution: Disable pytest config
pytest test_reranker.py --benchmark-only -v -o addopts=""
```

**Permission denied when installing**
```bash
# Cause: System venv restrictions
# Solution: Use PYTHONPATH instead of pip install
```

### 4. Run Benchmark

#### Basic Benchmark

```bash
# Activate environment and set paths (see step 3 above)
source /opt/aws_neuronx_venv_pytorch_inference_vllm_0_13/bin/activate
export PYTHONPATH=/path/to/benchmark-capture:$PYTHONPATH
export RERANKER_MODEL_PATH="/path/to/model"

# Run benchmark (default - with real-time logging)
pytest test_reranker.py --benchmark-only --benchmark-json=results.json -v

# For quieter output (only final results)
pytest test_reranker.py --benchmark-only --benchmark-json=results.json -v --log-cli-level=WARNING

# Without pytest config (if pytest-cov not installed)
pytest test_reranker.py --benchmark-only -v -o addopts=""
```

#### Configuration Sweep

After verifying with quick test, run full configuration sweeps:

**PA Blocks Sweep** (3 configurations: 256, 512, 1024):
```bash
pytest test_config_sweep.py::test_pa_blocks_sweep --benchmark-only -v -o addopts=""
```

**Optimization Sweep** (4 configurations: prefix caching × flash decoding):
```bash
pytest test_config_sweep.py::test_optimization_sweep --benchmark-only -v -o addopts=""
```

**All Sweeps**:
```bash
pytest test_config_sweep.py --benchmark-only -v -o addopts=""
```

**Expected execution time:**
- First configuration: 10-15 minutes (compilation)
- Subsequent configurations: ~1 minute each (cached)
- Total for PA blocks sweep: ~15-20 minutes
- Total for optimization sweep: ~15-20 minutes

**Important Notes:**
- **Each configuration generates a separate profile file** for performance comparison
- First run compiles the model (10-15 minutes)
- Subsequent runs use cached compilation artifacts (much faster)
- Use `-o addopts=""` to disable pytest-cov if not installed

**Real-time Progress Output:**

The benchmark uses Python logging for real-time progress updates:
- Model initialization status
- Token configuration
- First query verification
- Completion status

For quieter output, use `--log-cli-level=WARNING`.

## VLLMConfigHelper Usage

### Basic Usage

The example uses `VLLMConfigHelper` from `conftest.py`:

```python
from benchmark_capture.utils import VLLMConfigHelper

@pytest.fixture(scope="session")
def vllm_config(config):
    """vLLM-Neuron configuration with hardware-aware optimization."""
    vllm_params = {
        # Standard vLLM parameters
        "tensor_parallel_size": config['vllm']['tensor_parallel_size'],
        "max_num_seqs": config['vllm']['max_num_seqs'],
        "block_size": config['vllm']['block_size'],
        "max_model_len": config['vllm']['max_model_len'],
        # ... other parameters ...

        # Neuron-specific optimization (nested in additional_config)
        "additional_config": {
            "override_neuron_config": {
                "skip_warmup": True,
                "pa_num_blocks": 512,
                "pa_block_size": 32,
                "enable_bucketing": True,
            }
        }
    }

    # Hardware-aware configuration builder
    return VLLMConfigHelper(vllm_params).build()
```

### Configuration Sweep Pattern

Sweep Neuron configurations in your tests:

```python
@pytest.mark.parametrize("pa_num_blocks", [256, 512, 1024])
@profile("neuron", perfetto=True)
def test_pa_blocks_sweep(benchmark, model_path, pa_num_blocks):
    """Each configuration gets a separate profile file."""
    config = VLLMConfigHelper({
        "tensor_parallel_size": 2,
        "additional_config": {
            "override_neuron_config": {
                "pa_num_blocks": pa_num_blocks,  # Sweep target
            }
        }
    }).build()

    llm = vllm.LLM(model=model_path, **config)
    # Benchmark code...
```

## Customization

### Model-Specific Settings

#### Qwen3-Reranker
- Token IDs: `yes`, `no`
- Suffix typo: `assitant` (intentional for compatibility)

#### Other Reranker Models
Adjust the following in `conftest.py`:
- Token ID extraction (`token_true_id`, `token_false_id`)
- Prompt templates
- `SamplingParams` configuration

### Data Format

CSV file should have the following format:

```csv
query,answer_0,answer_1,...,answer_29
"Query 1","Candidate 1","Candidate 2",...
"Query 2","Candidate 1","Candidate 2",...
```

### 5. Quick Test (Recommended First Step)

Before running full benchmarks, verify your setup with a quick test:

```bash
# Navigate to example directory
cd /path/to/benchmark-capture/benchmark_capture/examples/vllm-neuron-reranker

# Run single configuration test
pytest test_config_sweep.py::test_pa_blocks_sweep[512] \
    --benchmark-only \
    -v \
    -o addopts="" \
    --benchmark-min-rounds=1
```

**What this does:**
- Tests a single configuration (pa_num_blocks=512)
- Minimal rounds for faster execution
- First run: 10-15 minutes (Neuron compilation)
- Subsequent runs: ~1 minute (uses cached compilation)

**Expected output:**
```
test_config_sweep.py::test_pa_blocks_sweep[512] PASSED

vLLM Configuration (Hardware: Neuron)
{
  "tensor_parallel_size": 2,
  "num_gpu_blocks_override": 512,
  "additional_config": {
    "override_neuron_config": {
      "pa_num_blocks": 512,
      ...
    }
  }
}

✓ Completed with pa_num_blocks=512
```

**If you see this**, your environment is correctly set up! Proceed to full benchmarks.

### 6. Verify Profiling Files

After benchmark execution, check that profiling files are generated:

```bash
# Navigate to example directory
cd /path/to/benchmark-capture/benchmark_capture/examples/vllm-neuron-reranker

# Check profiling directory
ls -la ./benchmarks/

# Expected structure:
# ./benchmarks/
# ├── metadata.json
# └── i-xxxxx_pid_yyy/
#     ├── neuron_profile_0.ntff
#     ├── neuron_profile_1.ntff
#     └── ...

# Count profile files (should match number of configurations tested)
find ./benchmarks -name "*.ntff" -type f | wc -l

# View profile in Perfetto UI
# 1. Open https://ui.perfetto.dev/
# 2. Drag and drop .ntff files
```

**Important**: For configuration sweeps, verify that **separate profile files are generated for each configuration**.

Example:
```bash
# PA blocks sweep (3 configs) → 3+ profile files
# Optimization sweep (4 configs) → 4+ profile files
```

## Generated Files Explained

### Directory Structure

After running a benchmark, you'll see the following structure:

```
benchmark-capture/benchmark_capture/examples/vllm-neuron-reranker/
├── profile_output/                          # Profiling data directory
│   ├── metadata.json                        # Profiling session metadata
│   └── i-<instance-id>_pid_<process-id>/   # Session directory
│       ├── <timestamp>_instid_0_vnc_0.ntff # NTFF file for NeuronCore 0
│       ├── <timestamp>_instid_0_vnc_1.ntff # NTFF file for NeuronCore 1
│       └── ...                              # Additional NTFF files
└── results.json                             # pytest-benchmark results
```

### File Details

#### 1. `profile_output/metadata.json`

Contains profiling configuration and file locations:

```json
{
    "function": "test_vllm_neuron_reranker",
    "profiler": "NeuronProfiler",
    "output_dir": "profile_output",
    "profiler_type": "neuron",
    "timeout": 600,
    "framework_profile": false,
    "clear_cache_before": false,
    "clear_cache_after": false,
    "perfetto_mode": true,
    "session_dir": "profile_output/i-0049acfde6046f237_pid_16695",
    "ntff_files": [
        "i-0049acfde6046f237_pid_16695/948584188481322_instid_0_vnc_0.ntff",
        "i-0049acfde6046f237_pid_16695/948584188481322_instid_0_vnc_1.ntff",
        "..."
    ]
}
```

**Key fields:**
- `perfetto_mode: true`: Confirms NTFF files are in Perfetto-compatible format
- `session_dir`: Path to the session directory with NTFF files
- `ntff_files`: List of all generated NTFF files

#### 2. Session Directory (`i-<instance-id>_pid_<process-id>/`)

Named with:
- **Instance ID**: EC2 instance identifier (e.g., `i-0049acfde6046f237`)
- **Process ID**: Python process ID (e.g., `pid_16695`)

This naming helps identify which instance and process generated the profile data, useful when running benchmarks across multiple instances.

#### 3. NTFF Files (`*.ntff`)

**Neuron Trace File Format** - Contains detailed performance traces from NeuronCores.

**File naming pattern**: `<timestamp>_instid_<instance>_vnc_<core>.ntff`
- **timestamp**: Unix timestamp when trace was captured
- **instid**: Internal instance identifier
- **vnc**: Virtual NeuronCore number (0, 1, 2, ...)

**Typical file sizes:**
- **15-25 MB per file** for inference workloads
- Larger for longer-running operations or more complex models

**What's inside:**
- DMA transfers (data movement between host and NeuronCores)
- Compute operations (matrix multiplications, activations)
- Kernel execution timing
- Memory allocation/deallocation
- Synchronization events

**Number of files:**
- **One NTFF file per NeuronCore per execution**
- For `tensor_parallel_size=2`: Expect 2-4 NTFF files
- Multiple timestamps indicate multiple inference runs or compilation phases

**Example from a real run:**
```bash
$ ls -lh profile_output/i-0049acfde6046f237_pid_16695/
-rw-r--r-- 1 coder coder 22M Jan 31 13:44 948584188481322_instid_0_vnc_0.ntff
-rw-r--r-- 1 coder coder 22M Jan 31 13:44 948584188481322_instid_0_vnc_1.ntff
-rw-r--r-- 1 coder coder 17M Jan 31 13:45 90860660587470_instid_0_vnc_0.ntff
-rw-r--r-- 1 coder coder 17M Jan 31 13:45 90860660587470_instid_0_vnc_1.ntff
```

**Interpretation:**
- 2 NeuronCores used (vnc_0 and vnc_1)
- 2 distinct execution phases (2 different timestamps)
- First phase: 22 MB per core (likely compilation or first inference)
- Second phase: 17 MB per core (subsequent inferences)

#### 4. `results.json`

pytest-benchmark output with performance metrics:

```json
{
    "benchmarks": [
        {
            "name": "test_vllm_neuron_reranker",
            "group": "reranker",
            "stats": {
                "min": 3.0749,
                "max": 3.0996,
                "mean": 3.0881,
                "median": 3.0888,
                "stddev": 0.0095
            },
            "rounds": 5
        }
    ],
    "machine_info": {
        "node": "ip-172-31-41-208",
        "processor": "x86_64",
        "system": "Linux"
    }
}
```

**Key metrics:**
- **mean**: Average execution time (seconds)
- **min/max**: Performance range
- **stddev**: Consistency of performance

### Using NTFF Files with Perfetto UI

1. **Open Perfetto UI**: https://ui.perfetto.dev/
2. **Drag and drop** one or more `.ntff` files
3. **Analyze**:
   - Timeline view: See when operations occurred
   - Flamegraph: Identify hotspots
   - Track view: Examine individual operations
   - Statistics: Aggregate performance data

**Tips for analysis:**
- Load all NTFF files from a session to see complete picture across cores
- Compare NTFF files from different timestamps to understand compilation vs inference
- Use search to find specific operations (e.g., "MatMul", "DMA")

### Typical Workflow

```bash
# 1. Run benchmark
./run_benchmark.sh

# 2. Check metadata
cat profile_output/metadata.json | jq .

# 3. Count NTFF files (should match expected cores × execution phases)
find profile_output -name "*.ntff" | wc -l

# 4. Check file sizes (15-25 MB is normal)
du -h profile_output/i-*_pid_*/*.ntff

# 5. Upload to Perfetto UI for visualization
# Open https://ui.perfetto.dev/ and drag-drop the NTFF files
```

## Benchmark Results

After execution, you'll see metrics like:

```
✅ Benchmark Results
================================================================================

📊 Overall Performance:
   Total time (mean): 3070.600 ms
   Min: 3046.139 ms
   Max: 3098.857 ms
   Median: 3068.363 ms

📈 Per-Query Metrics:
   Latency per query: 307.060 ms (0.3071 s)
   Throughput (QPS): 3.2567 queries/second

🔢 Configuration:
   Total queries: 10
   Candidates per query: 20
   Total pairs: 200
   Batch size: 8
   Block size: 32
   Tensor parallel size: 2
================================================================================
```

## Troubleshooting

### CRITICAL: vLLM Import Timing Issue

**Problem:**
NTFF files are not generated even though `perfetto_mode: true` in metadata.json.

**Root Cause:**
If `vllm` is imported at module level (top of file), it loads the Neuron runtime **before** the `@profile` decorator sets the `NEURON_RT_INSPECT_*` environment variables. This prevents profiling from working.

**Incorrect (Don't do this):**
```python
import vllm  # ❌ TOO EARLY - Neuron runtime loads before profiling setup
from vllm import SamplingParams

@profile("neuron", perfetto=True)
def test_benchmark(benchmark):
    llm = vllm.LLM(...)  # Too late - runtime already initialized
```

**Correct (Do this):**
```python
# NO vllm import at module level

@profile("neuron", perfetto=True)
def test_benchmark(benchmark):
    # ✅ Import AFTER profiling decorator activates
    import vllm
    from vllm import SamplingParams

    llm = vllm.LLM(...)  # Now profiling will work
```

**Why This Matters:**
1. `@profile` decorator runs **before** the test function executes
2. It sets `NEURON_RT_INSPECT_ENABLE=1` and other env vars
3. When `vllm` imports, it initializes the Neuron runtime
4. **The Neuron runtime only reads profiling env vars at initialization**
5. If vLLM imports too early, it misses the profiling configuration

**Verification:**
```bash
# After running benchmark
ls profile_output/i-*_pid_*/*.ntff

# Should see NTFF files
# If directory exists but no .ntff files → import timing issue
```

### ModuleNotFoundError: No module named 'benchmark_capture'

**Problem:**
```
ImportError: No module named 'benchmark_capture'
```

**Solution:**
```bash
# Set PYTHONPATH
export PYTHONPATH=/path/to/benchmark-capture:$PYTHONPATH

# Verify
python3 -c "import benchmark_capture; print('OK')"
```

**Root cause**: The vLLM-Neuron environment is externally-managed and does not allow `pip install`. PYTHONPATH is required for development.

### pytest: error: unrecognized arguments: --cov

**Problem:**
```
pytest: error: unrecognized arguments: --cov=benchmark_capture
```

**Solution:**
```bash
# Option 1: Disable pytest config (recommended)
pytest test_config_sweep.py --benchmark-only -v -o addopts=""

# Option 2: Install pytest-cov
pip install pytest-cov
```

**Root cause**: pyproject.toml configures pytest-cov but it's not installed in the vLLM environment.

### No profiling files generated

**Problem:**
```bash
find ./benchmarks -name "*.ntff" -type f
# No output
```

**Check:**

1. Verify Perfetto mode is enabled in test:
   ```python
   @profile("neuron", perfetto=True)  # perfetto=True is required
   ```

2. Check metadata.json:
   ```bash
   cat ./benchmarks/metadata.json | grep perfetto
   # Should show: "perfetto_mode": true
   ```

3. Verify Neuron environment:
   ```bash
   ls -la /opt/aws/neuron/
   # Should exist
   ```

4. Check test actually ran on Neuron hardware (session directory only created when Neuron runtime is invoked)

### vLLM Engine Error: pa_num_blocks mismatch

**Problem:**
```
ValueError: When setting pa_num_blocks (512) in override_neuron_config,
you must also set --num-gpu-blocks-override to the same value
```

**Solution:**

Ensure `num_gpu_blocks_override` matches `pa_num_blocks`:

```python
config = VLLMConfigHelper({
    "tensor_parallel_size": 2,
    "num_gpu_blocks_override": 512,  # Must match pa_num_blocks
    "additional_config": {
        "override_neuron_config": {
            "pa_num_blocks": 512,  # Must match num_gpu_blocks_override
            ...
        }
    }
}).build()
```

**Root cause**: vLLM and NxDI must have consistent block counts.

### Memory Errors

Increase `block_size` or decrease `batch_size`:

```yaml
vllm:
  block_size: 128  # 32 → 128
reranker:
  batch_size: 4    # 8 → 4
```

### Long Compilation Time

First run takes 10-15 minutes. Compiled artifacts are cached for subsequent runs.

**To clear cache:**
Edit `config.yaml`:
```yaml
profiler:
  clear_cache_before: true  # Clear cache before benchmark
```

Or manually:
```bash
sudo rm -rf /var/tmp/neuron-compile-cache/*
```

### Permission denied when installing

**Problem:**
```
ERROR: Could not install packages due to an OSError: [Errno 13] Permission denied
```

**Solution:**

Use PYTHONPATH instead of pip install:
```bash
export PYTHONPATH=/path/to/benchmark-capture:$PYTHONPATH
```

**Root cause**: System venv is externally-managed.

### AttributeError: 'int' object has no attribute 'stats'

Access statistics from `benchmark.stats` directly, not from the return value of `benchmark.pedantic()`.

## Environment Variables

Override model path:

```bash
export RERANKER_MODEL_PATH="/path/to/model"
pytest test_reranker.py --benchmark-only -v
```

## Configuration Options

### vLLM Settings

- `tensor_parallel_size`: Number of NeuronCores to use
- `block_size`: KV cache block size (32 for Zenn best case, 128 for stability)
- `max_num_seqs`: Batch size
- `max_num_batched_tokens`: Performance optimization

### Reranker Settings

- `search_num`: Number of candidates to process per query
- `batch_size`: Processing batch size (affects memory usage)
- `max_length`: Maximum prompt length (affects tokenization)

### Benchmark Settings

- `rounds`: Number of benchmark rounds
- `warmup_rounds`: Number of warmup rounds before measurement
- `num_test_queries`: Number of queries to use for testing

## Testing & Verification Checklist

Before running full benchmarks, verify:

- [ ] vLLM-Neuron environment activated
- [ ] PYTHONPATH set to benchmark-capture directory
- [ ] Model path set (via RERANKER_MODEL_PATH or config.yaml)
- [ ] Import test succeeds: `python3 -c "from benchmark_capture.utils import VLLMConfigHelper; print('OK')"`
- [ ] Quick test passes: `pytest test_config_sweep.py::test_pa_blocks_sweep[512] --benchmark-only -v -o addopts="" --benchmark-min-rounds=1`
- [ ] At least 1 .ntff file generated after quick test

**Complete test sequence:**

```bash
# 1. Environment setup
source /opt/aws_neuronx_venv_pytorch_inference_vllm_0_13/bin/activate
export PYTHONPATH=/path/to/benchmark-capture:$PYTHONPATH
export RERANKER_MODEL_PATH="/path/to/model"

# 2. Import verification
python3 -c "from benchmark_capture.utils import VLLMConfigHelper; print('✓ Setup OK')"

# 3. Quick test (single config)
cd /path/to/benchmark-capture/benchmark_capture/examples/vllm-neuron-reranker
pytest test_config_sweep.py::test_pa_blocks_sweep[512] \
    --benchmark-only -v -o addopts="" --benchmark-min-rounds=1

# 4. Verify profiling files
find ./benchmarks -name "*.ntff" -type f
# Expected: At least 1 file

# 5. Run full sweep (if quick test succeeds)
pytest test_config_sweep.py::test_pa_blocks_sweep --benchmark-only -v -o addopts=""

# 6. Verify sweep files
find ./benchmarks -name "*.ntff" -type f | wc -l
# Expected: 3 or more files
```


## References

- Zenn article: https://zenn.dev/kotaro666/scraps/c9bec0ac1fef5d
- vLLM Documentation: https://docs.vllm.ai/
- pytest-benchmark: https://pytest-benchmark.readthedocs.io/
- VLLMConfigHelper: See `conftest.py` for usage example

## License

This example follows the same license as benchmark-capture library.
