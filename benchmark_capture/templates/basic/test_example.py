"""
Example benchmark tests for {{ project_name }}.

Run with:
    pytest --benchmark-only
"""

import pytest
from benchmark_capture import profile


@pytest.mark.benchmark
@profile()  # Auto-detect profiler
def test_simple_computation(benchmark, sample_data):
    """Simple benchmark example with auto-detection."""

    def compute(data):
        return sum(x * x for x in data)

    result = benchmark(compute, sample_data)

    # Add custom metrics
    benchmark.extra_info["data_size"] = len(sample_data)
    benchmark.extra_info["result"] = result

    assert result > 0


@pytest.mark.benchmark
@profile("{{ profiler }}")  # Use configured profiler
@pytest.mark.parametrize("size", [10, 100, 1000])
def test_parametrized(benchmark, size):
    """Parametrized benchmark example."""

    def compute(n):
        return sum(range(n))

    result = benchmark(compute, size)

    benchmark.extra_info["size"] = size
    benchmark.extra_info["result"] = result

    assert result == (size * (size - 1)) // 2
