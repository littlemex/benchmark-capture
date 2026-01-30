"""
Simple example demonstrating benchmark-capture with pytest-benchmark.

Run with:
    pytest examples/test_simple.py --benchmark-only -v
"""

import time

import pytest

# Add parent directory to path to import benchmark_capture
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from benchmark_capture import profile


@pytest.mark.benchmark
@profile()  # Auto-detect hardware (will use noop on this system)
def test_simple_computation(benchmark):
    """Simple benchmark with auto-detection."""

    def compute():
        # Simulate some computation
        total = 0
        for i in range(1000):
            total += i
        return total

    result = benchmark(compute)
    assert result == 499500


@pytest.mark.benchmark(group="explicit")
@profile("noop")  # Explicitly use noop profiler
def test_explicit_noop(benchmark):
    """Benchmark with explicit noop profiler."""

    def compute():
        time.sleep(0.001)  # 1ms sleep
        return "done"

    result = benchmark(compute)
    assert result == "done"


@pytest.mark.benchmark(group="with_params")
@pytest.mark.parametrize("size", [100, 500, 1000])
@profile("noop", output_dir="/tmp/benchmark_capture_test")
def test_parametrized(benchmark, size):
    """Parametrized benchmark with custom output directory."""

    def compute(n):
        return sum(range(n))

    result = benchmark(compute, size)

    # Add custom metrics to benchmark.extra_info
    benchmark.extra_info["size"] = size
    benchmark.extra_info["result"] = result
    benchmark.extra_info["avg_time_per_item"] = benchmark.stats.stats.mean / size

    expected = (size * (size - 1)) // 2
    assert result == expected


if __name__ == "__main__":
    # Run with: python examples/test_simple.py
    print("Run with: pytest examples/test_simple.py --benchmark-only -v")
