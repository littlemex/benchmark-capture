"""NVIDIA NSight Systems profiler implementation."""

import os
from typing import Any, Dict

from .base import Profiler


class NSightProfiler(Profiler):
    """
    NVIDIA NSight Systems profiler implementation.

    Sets environment variables for NSight profiling:
    - NSYS_OUTPUT_FILE: Output file path for profile data
    - NSYS_CUDA_API_TRACE: Enable CUDA API tracing

    Profile data is saved as .nsys-rep files which can be analyzed with:
    - nsys analyze (CLI)
    - nsight-sys (GUI)

    Note: Actual profiling requires wrapping the application with 'nsys profile'
    This profiler only sets up the environment for potential profiling.
    For full profiling, consider using: nsys profile pytest ...
    """

    def setup(self, function_name: str) -> None:
        """
        Setup NSight profiling environment.

        Args:
            function_name: Name of the function being profiled
        """
        self.function_name = function_name

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Set NSight environment variables
        profile_file = self.output_dir / f"{function_name}.nsys-rep"
        os.environ["NSYS_OUTPUT_FILE"] = str(profile_file)

        # Optional: CUDA API tracing
        if self.options.get("cuda_api_trace", True):
            os.environ["NSYS_CUDA_API_TRACE"] = "1"

        # Optional: Sampling rate
        if "sampling_rate" in self.options:
            os.environ["NSYS_SAMPLING_RATE"] = str(self.options["sampling_rate"])

    def teardown(self) -> None:
        """Clean up NSight profiling environment variables."""
        os.environ.pop("NSYS_OUTPUT_FILE", None)
        os.environ.pop("NSYS_CUDA_API_TRACE", None)
        os.environ.pop("NSYS_SAMPLING_RATE", None)

    def get_metadata(self) -> Dict[str, Any]:
        """
        Get NSight profiling metadata.

        Returns:
            Dictionary with NSight-specific metadata
        """
        # Find all .nsys-rep files in output directory
        profile_files = [str(f) for f in self.output_dir.glob("*.nsys-rep")]
        profile_files.extend([str(f) for f in self.output_dir.glob("*.qdrep")])

        return {
            "profiler_type": "nsight",
            "profile_files": profile_files,
            "cuda_api_trace": self.options.get("cuda_api_trace", True),
            "sampling_rate": self.options.get("sampling_rate"),
        }
