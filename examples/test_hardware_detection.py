"""
Example demonstrating hardware auto-detection.

This script shows what profiler would be detected on the current system.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from benchmark_capture.profilers.detector import (
    detect_hardware,
    is_cuda_available,
    is_neuron_available,
)


def main():
    """Show hardware detection results."""
    print("=" * 60)
    print("Hardware Detection Report")
    print("=" * 60)

    # Check individual components
    neuron = is_neuron_available()
    cuda = is_cuda_available()

    print(f"\n1. Individual Checks:")
    print(f"   AWS Neuron Available: {neuron}")
    print(f"   NVIDIA CUDA Available: {cuda}")

    # Auto-detection
    detected = detect_hardware()
    print(f"\n2. Auto-Detected Profiler: {detected}")

    # Explanation
    print(f"\n3. Detection Logic:")
    if neuron:
        print("   ✓ AWS Neuron SDK detected (priority 1)")
        print("   → Will use 'neuron' profiler by default")
    elif cuda:
        print("   ✓ NVIDIA CUDA detected (priority 2)")
        print("   → Will use 'nsight' profiler by default")
    else:
        print("   ✓ No specialized hardware detected")
        print("   → Will use 'noop' profiler (no profiling overhead)")

    # Usage examples
    print(f"\n4. Usage Examples:")
    print(f"   # Auto-detect (recommended):")
    print(f"   @profile()")
    print(f"   def test_benchmark(benchmark):")
    print(f"       ...")
    print()
    print(f"   # Force specific profiler:")
    print(f"   @profile('{detected}')")
    print(f"   def test_benchmark(benchmark):")
    print(f"       ...")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
