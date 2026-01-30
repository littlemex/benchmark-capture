"""
benchmark-capture: Lightweight profiling decorator for pytest-benchmark.

Provides automatic hardware detection and profiling for benchmarks:
- AWS Neuron (Inferentia)
- NVIDIA GPU (NSight Systems)
- Auto-detection with graceful fallback
"""

__version__ = "0.2.1"

from .decorators import profile

__all__ = ["profile", "__version__"]
