# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.2] - 2026-01-30

### Added

- **Perfetto Mode for Advanced Analysis**
  - New `perfetto=True` option for `@profile()` decorator
  - Generates Perfetto-compatible NTFF files in session directories
  - Automatic `NEURON_RT_INSPECT_*` environment variable configuration
  - Session directory pattern: `i-<instance>_pid_<number>/`
  - Enhanced metadata with `perfetto_mode`, `session_dir`, and `ntff_files` fields
  - Enables visualization in Perfetto UI and analysis with neuron-workflow

- **vLLM-Neuron Reranker Example**
  - Perfetto mode enabled by default in reranker example
  - Output verification commands in docstring
  - Ready for production profiling workflows

- **Documentation**
  - Comprehensive Perfetto mode section in README
  - Comparison table: Standard vs Perfetto mode
  - Output verification examples with expected results
  - Metadata field differences explained
  - Troubleshooting guide for common issues
  - NTFF conversion instructions

### Changed

- NeuronProfiler: Split `setup()` into `_setup_standard_mode()` and `_setup_perfetto_mode()`
- Metadata structure: Now includes mode-specific fields for better clarity
- Environment variable cleanup: Handles both standard and Perfetto mode variables

### Technical Details

- 100% backward compatibility - default behavior unchanged
- Test coverage: 89% for neuron.py, 73% overall
- All 126 tests passing (7 new Perfetto mode tests added)
- No new external dependencies required

## [0.2.1] - 2026-01-30

### Initial Release

This is the first public release of benchmark-capture, a lightweight profiling decorator for pytest-benchmark with automatic hardware detection.

### Added

- **Core Features**
  - `@profile()` decorator with automatic hardware detection (AWS Neuron, NVIDIA GPU, CPU)
  - pytest-benchmark integration for standardized benchmarking
  - Zero-configuration profiling with graceful fallback

- **Profiler Support**
  - AWS Neuron profiler for Inferentia/Trainium instances
  - NVIDIA NSight Systems profiler for GPU benchmarking
  - No-op profiler for CPU/unsupported hardware
  - Neuron compilation cache management (`clear_cache_before`/`clear_cache_after`)

- **Project Initialization**
  - `benchmark-capture-init` CLI tool for scaffolding projects
  - Template system: `basic`, `vllm`, `vllm-neuron`, `minimal`
  - Example deployment via `--example` flag
  - Complete vLLM-Neuron Reranker example with CSV data (10 queries, 200 pairs)

- **Real-time Logging**
  - Python logging integration for long-running benchmarks
  - Configurable log levels via pytest (`--log-cli-level`)
  - Real-time progress updates during model compilation and inference

- **Documentation**
  - Comprehensive README with end-to-end workflows
  - Model download from Hugging Face instructions
  - Two methods for config.yaml configuration (manual editing + sed)
  - Troubleshooting section for common issues
  - Step-by-step verification commands
  - Example output and result analysis

- **Quality Assurance**
  - 120 tests with comprehensive coverage
  - Type hints with mypy configuration
  - Code formatting with black and isort
  - Apache 2.0 license

### Package Information

- **Repository**: https://github.com/littlemex/benchmark-capture
- **TestPyPI**: https://test.pypi.org/project/benchmark-capture/0.2.1/
- **Python Support**: 3.9, 3.10, 3.11, 3.12
- **License**: Apache-2.0

### Installation

```bash
pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ 'benchmark-capture[init]==0.2.1'
```

### Primary Use Cases

- vLLM and vLLM-Neuron benchmarking on AWS Inferentia/Trainium
- LLM inference profiling on NVIDIA GPUs
- General Python function benchmarking with hardware-aware profiling

[Unreleased]: https://github.com/littlemex/benchmark-capture/compare/v0.2.2...HEAD
[0.2.2]: https://github.com/littlemex/benchmark-capture/compare/v0.2.1...v0.2.2
[0.2.1]: https://github.com/littlemex/benchmark-capture/releases/tag/v0.2.1
