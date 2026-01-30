"""Pytest configuration and fixtures."""

import os
import tempfile
from pathlib import Path
from typing import Generator

import pytest


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def clean_env() -> Generator[None, None, None]:
    """Clean environment variables before and after tests."""
    # Backup original environment
    original_env = os.environ.copy()

    # Clean benchmark-related env vars
    for key in list(os.environ.keys()):
        if key.startswith("BENCHMARK_") or key.startswith("NEURON_") or key.startswith("NSYS_"):
            del os.environ[key]

    yield

    # Restore original environment
    os.environ.clear()
    os.environ.update(original_env)


@pytest.fixture
def mock_neuron_env(clean_env: None) -> Generator[None, None, None]:
    """Mock AWS Neuron environment."""
    os.environ["NEURON_RT_VISIBLE_CORES"] = "0-15"
    yield
    os.environ.pop("NEURON_RT_VISIBLE_CORES", None)


@pytest.fixture
def sample_config_file(temp_dir: Path) -> Path:
    """Create a sample benchmark.toml config file."""
    config_content = """
[profiler]
backend = "neuron"

[profiler.neuron]
output_dir = "/custom/neuron/path"
timeout = 1200

[profiler.nsight]
output_dir = "/custom/nsight/path"
cuda_api_trace = false
"""
    config_file = temp_dir / "benchmark.toml"
    config_file.write_text(config_content)
    return config_file
