#!/bin/bash
#
# vLLM-Neuron Benchmark Runner (Template)
#
# This script sets up the environment and runs the benchmark with profiling enabled.
#
# IMPORTANT: This is a REFERENCE template. Customize these paths for your environment:
# - VENV_PATH: Path to your Neuron venv (default is for AWS Neuron AMI)
# - BENCHMARK_CAPTURE_PATH: Path to benchmark-capture package
# - Source activation path may vary depending on your installation
#

set -e  # Exit on error

# ============================================================================
# Configuration - CUSTOMIZE THESE PATHS FOR YOUR ENVIRONMENT
# ============================================================================

# Path to your Neuron venv (CHANGE THIS if using a different environment)
# Default: AWS Neuron AMI path
# Other common paths:
#   - /opt/aws_neuronx_venv_pytorch/bin/activate
#   - ~/neuron-venv/bin/activate
#   - Custom venv path
VENV_PATH="/opt/aws_neuronx_venv_pytorch_inference_vllm_0_13"

# Path to benchmark-capture package
# If using this as a template, this script assumes it's in the project root
# Adjust as needed for your directory structure
BENCHMARK_CAPTURE_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

# ============================================================================
# Environment Setup
# ============================================================================

echo "================================================================"
echo "vLLM-Neuron Benchmark with Profiling"
echo "================================================================"
echo ""

# Activate Neuron venv
if [ ! -d "$VENV_PATH" ]; then
    echo "ERROR: Neuron venv not found at: $VENV_PATH"
    echo ""
    echo "Please update VENV_PATH in this script to match your environment."
    echo ""
    echo "Common paths:"
    echo "  - /opt/aws_neuronx_venv_pytorch_inference_vllm_0_13 (AWS Neuron AMI default)"
    echo "  - /opt/aws_neuronx_venv_pytorch/bin/activate"
    echo "  - ~/neuron-venv/bin/activate"
    echo ""
    exit 1
fi

echo "Activating Neuron environment..."
source "$VENV_PATH/bin/activate"

# Set PYTHONPATH for benchmark-capture
export PYTHONPATH="$BENCHMARK_CAPTURE_PATH:$PYTHONPATH"

# Verify imports
echo "Verifying benchmark-capture import..."
if ! python3 -c "from benchmark_capture import profile; print('✓ Import successful')" 2>/dev/null; then
    echo "ERROR: Failed to import benchmark-capture"
    echo ""
    echo "Please verify:"
    echo "  1. BENCHMARK_CAPTURE_PATH is correct: $BENCHMARK_CAPTURE_PATH"
    echo "  2. benchmark-capture is installed or PYTHONPATH is set correctly"
    echo ""
    exit 1
fi

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
#
# Adjust test file name as needed for your project
PYTHONUNBUFFERED=1 pytest test_vllm.py \
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
