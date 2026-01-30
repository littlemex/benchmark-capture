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
├── README.md              # This file
├── config.yaml           # Benchmark configuration
├── conftest.py           # pytest fixtures
├── test_reranker.py      # Test implementation
├── input_sample.csv      # Sample data (10 queries)
└── requirements.txt      # Dependencies
```

## Quick Start

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

### 3. Run Benchmark

```bash
# Activate vLLM-Neuron environment
source /opt/aws_neuronx_venv_pytorch_inference_vllm_0_13/bin/activate

# Run benchmark (default - with real-time logging)
pytest test_reranker.py --benchmark-only --benchmark-json=results.json -v

# For quieter output (only final results)
pytest test_reranker.py --benchmark-only --benchmark-json=results.json -v --log-cli-level=WARNING
```

**Real-time Progress Output:**
The benchmark uses Python logging for real-time progress updates. You'll see:
- Model initialization status
- Token configuration
- First query verification
- Completion status

This is especially useful for long-running benchmarks on Inferentia2.

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

## References

- Original implementation: `/home/coder/data-science/investigations/inf2-vllm-performance/`
- Zenn article: https://zenn.dev/kotaro666/scraps/c9bec0ac1fef5d
- vLLM Documentation: https://docs.vllm.ai/
- pytest-benchmark: https://pytest-benchmark.readthedocs.io/

## License

This example follows the same license as benchmark-capture library.
