"""Profiler implementations for different hardware platforms."""

from .base import Profiler
from .detector import detect_hardware, is_cuda_available, is_neuron_available
from .neuron import NeuronProfiler
from .noop import NoOpProfiler
from .nsight import NSightProfiler

__all__ = [
    "Profiler",
    "NeuronProfiler",
    "NSightProfiler",
    "NoOpProfiler",
    "detect_hardware",
    "is_neuron_available",
    "is_cuda_available",
    "get_profiler",
]


def get_profiler(profiler_name: str, **options) -> Profiler:
    """
    Get profiler instance by name.

    Args:
        profiler_name: Profiler name ("neuron", "nsight", "noop")
        **options: Additional profiler options

    Returns:
        Profiler instance

    Raises:
        ValueError: If profiler_name is unknown
    """
    profilers = {
        "neuron": NeuronProfiler,
        "nsight": NSightProfiler,
        "noop": NoOpProfiler,
    }

    if profiler_name not in profilers:
        raise ValueError(
            f"Unknown profiler: {profiler_name}. " f"Available: {', '.join(profilers.keys())}"
        )

    return profilers[profiler_name](**options)
