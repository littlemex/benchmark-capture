"""pytest fixtures for {{ project_name }}."""

import pytest


@pytest.fixture
def sample_data():
    """Sample data fixture."""
    return list(range(100))
