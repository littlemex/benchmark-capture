"""pytest fixtures for {{ project_name }}."""

import pytest

try:
    import vllm
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False


@pytest.fixture(scope="session")
def model_path():
    """vLLM model path."""
    return "{{ model_path }}"


@pytest.fixture(scope="session")
def vllm_config():
    """Default vLLM configuration."""
    return {
        "tensor_parallel_size": 2,
        "max_model_len": 2048,
        "block_size": 32,
        "disable_log_stats": True,
    }


@pytest.fixture
def sample_prompts():
    """Sample prompts for testing."""
    return [
        "Hello, how are you?",
        "Tell me a story.",
        "What is the meaning of life?",
    ]


# Skip all tests if vLLM not available
if not VLLM_AVAILABLE:
    pytest.skip("vLLM not available", allow_module_level=True)
