"""Example demonstrating Perfetto mode for AWS Neuron profiling.

This example shows how to use the perfetto=True option to generate
NTFF files in session directories suitable for Perfetto analysis.
"""

import pytest
from benchmark_capture import profile


@pytest.mark.benchmark
@profile("neuron", perfetto=True, output_dir="/tmp/perfetto_example")
def test_perfetto_mode(benchmark):
    """Example using Perfetto mode for NTFF generation."""

    def sample_workload():
        """Simulated workload (replace with actual inference)."""
        import time

        time.sleep(0.1)
        return {"result": "success"}

    result = benchmark(sample_workload)
    assert result["result"] == "success"


@pytest.mark.benchmark
@profile("neuron", output_dir="/tmp/standard_example")
def test_standard_mode(benchmark):
    """Example using standard mode for comparison."""

    def sample_workload():
        """Simulated workload (replace with actual inference)."""
        import time

        time.sleep(0.1)
        return {"result": "success"}

    result = benchmark(sample_workload)
    assert result["result"] == "success"


if __name__ == "__main__":
    """Run examples with pytest."""
    pytest.main([__file__, "-v", "-s"])
