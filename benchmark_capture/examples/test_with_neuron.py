"""
Example demonstrating Neuron profiling with benchmark-capture.

This example will use the Neuron profiler (auto-detected on AWS Inferentia).
Run with:
    pytest examples/test_with_neuron.py --benchmark-only -v
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest

from benchmark_capture import profile


@pytest.mark.benchmark
@profile()  # Auto-detect: will use Neuron on Inferentia
def test_with_auto_neuron(benchmark):
    """Benchmark with auto-detected Neuron profiler."""

    def compute():
        # Simple computation
        result = sum(i**2 for i in range(1000))
        return result

    result = benchmark(compute)

    # Add custom metrics
    benchmark.extra_info["result"] = result
    benchmark.extra_info["computation"] = "sum of squares"

    assert result == 332833500


@pytest.mark.benchmark
@profile("neuron", output_dir="/tmp/neuron_profile_test", timeout=300)
def test_explicit_neuron_with_options(benchmark):
    """Benchmark with explicit Neuron profiler and custom options."""

    def compute():
        # More complex computation
        result = 0
        for i in range(5000):
            result += i * (i + 1) // 2
        return result

    result = benchmark(compute)
    benchmark.extra_info["iterations"] = 5000
    benchmark.extra_info["result"] = result

    # Expected: sum of (i * (i+1) / 2) for i in 0..4999
    # This is approximately (n^3)/6 for large n
    assert result > 0  # Basic sanity check


if __name__ == "__main__":
    print("Run with: pytest examples/test_with_neuron.py --benchmark-only -v")
