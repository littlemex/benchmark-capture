#!/bin/bash
#
# vLLM-Neuron Reranker Benchmark Runner
#
# This script sets up the environment and runs the benchmark with profiling enabled.
#
# IMPORTANT: Update the venv path below to match your environment.
# The default path is for AWS Neuron AMI. Adjust as needed for your setup.
#

set -e  # Exit on error

# ============================================================================
# Configuration - CUSTOMIZE THESE PATHS FOR YOUR ENVIRONMENT
# ============================================================================

# Path to your Neuron venv (CHANGE THIS if using a different environment)
VENV_PATH="/opt/aws_neuronx_venv_pytorch_inference_vllm_0_13"

# Path to benchmark-capture package (assumes this script is in examples/)
BENCHMARK_CAPTURE_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"

# ============================================================================
# Environment Setup
# ============================================================================

echo "================================================================"
echo "vLLM-Neuron Reranker Benchmark with Profiling"
echo "================================================================"
echo ""

# Activate Neuron venv
if [ ! -d "$VENV_PATH" ]; then
    echo "ERROR: Neuron venv not found at: $VENV_PATH"
    echo "Please update VENV_PATH in this script to match your environment."
    exit 1
fi

echo "Activating Neuron environment..."
source "$VENV_PATH/bin/activate"

# Set PYTHONPATH for benchmark-capture
export PYTHONPATH="$BENCHMARK_CAPTURE_PATH:$PYTHONPATH"

# Verify imports
echo "Verifying benchmark-capture import..."
python3 -c "from benchmark_capture.utils import VLLMConfigHelper; print('✓ Import successful')"

echo ""
echo "Environment setup complete!"
echo "  - Neuron venv: $VENV_PATH"
echo "  - benchmark-capture: $BENCHMARK_CAPTURE_PATH"
echo ""

# ============================================================================
# Run Benchmark
# ============================================================================

echo "Running pytest..."
echo ""

# Run pytest with profiling enabled
# -s: Disable stdout capture (shows compilation logs in real-time)
# --log-cli-level=INFO: Show log messages in real-time
# PYTHONUNBUFFERED=1: Disable Python output buffering
PYTHONUNBUFFERED=1 pytest test_reranker.py \
    --benchmark-only \
    --benchmark-json=results.json \
    -v \
    -s \
    --log-cli-level=INFO

# ============================================================================
# Results Summary
# ============================================================================

echo ""
echo "================================================================"
echo "Benchmark Complete!"
echo "================================================================"
echo ""
echo "Generated files:"
echo "  - results.json: Benchmark performance metrics"
echo "  - profile_output/: Neuron profiling data (NTFF files)"
echo ""
echo "To view profiling data:"
echo "  1. Check session directory: ls -la profile_output/i-*_pid_*/"
echo "  2. Verify metadata: cat profile_output/metadata.json | jq"
echo "  3. Upload NTFF files to Perfetto UI: https://ui.perfetto.dev/"
echo ""
echo "For detailed usage, see README.md"
echo ""
