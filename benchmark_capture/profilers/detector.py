"""Hardware detection for automatic profiler selection."""

import os
import shutil


def detect_hardware() -> str:
    """
    Auto-detect available profiling hardware.

    Detection priority:
    1. AWS Neuron (torch_neuronx, /opt/aws/neuron/)
    2. NVIDIA GPU (CUDA, nvidia-smi)
    3. Fallback to no-op

    Returns:
        Profiler name: "neuron", "nsight", or "noop"
    """
    if is_neuron_available():
        return "neuron"

    if is_cuda_available():
        return "nsight"

    return "noop"


def is_neuron_available() -> bool:
    """
    Check if AWS Neuron SDK is available.

    Checks:
    1. torch_neuronx importable
    2. /opt/aws/neuron/ directory exists
    3. NEURON_RT_VISIBLE_CORES environment variable set

    Returns:
        True if Neuron is available, False otherwise
    """
    # Try importing torch_neuronx
    try:
        import torch_neuronx  # noqa: F401

        return True
    except ImportError:
        pass

    # Check for Neuron SDK installation
    if os.path.exists("/opt/aws/neuron/"):
        return True

    # Check environment variable
    if os.environ.get("NEURON_RT_VISIBLE_CORES"):
        return True

    return False


def is_cuda_available() -> bool:
    """
    Check if NVIDIA CUDA is available.

    Checks:
    1. torch.cuda.is_available()
    2. nvidia-smi command available
    3. nsys command available (NSight Systems)

    Returns:
        True if CUDA is available, False otherwise
    """
    # Try PyTorch CUDA check
    try:
        import torch

        if torch.cuda.is_available():
            return True
    except ImportError:
        pass

    # Check for nvidia-smi command
    if shutil.which("nvidia-smi"):
        return True

    # Check for NSight Systems
    if shutil.which("nsys"):
        return True

    return False
