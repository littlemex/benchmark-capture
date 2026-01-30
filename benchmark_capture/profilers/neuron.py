"""AWS Neuron profiler implementation."""

import os
from typing import Any, Dict

from .base import Profiler


class NeuronProfiler(Profiler):
    """
    AWS Neuron profiler implementation.

    Sets environment variables for Neuron profiling:
    - NEURON_PROFILE: Output directory for profile data
    - NEURON_RT_EXEC_TIMEOUT: Execution timeout

    Profile data is saved as .ntff files which can be analyzed with:
    - neuron-profile view
    - neuron-profile convert (to Perfetto, JSON, etc.)
    """

    def setup(self, function_name: str) -> None:
        """
        Setup Neuron profiling environment.

        Args:
            function_name: Name of the function being profiled
        """
        self.function_name = function_name

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Set Neuron profiling environment variables
        os.environ["NEURON_PROFILE"] = str(self.output_dir)
        os.environ["NEURON_RT_EXEC_TIMEOUT"] = str(self.options.get("timeout", 600))

        # Optional: Framework profiling
        if self.options.get("framework_profile", False):
            os.environ["NEURON_FRAMEWORK_DEBUG"] = "1"

    def teardown(self) -> None:
        """Clean up Neuron profiling environment variables."""
        os.environ.pop("NEURON_PROFILE", None)
        os.environ.pop("NEURON_RT_EXEC_TIMEOUT", None)
        os.environ.pop("NEURON_FRAMEWORK_DEBUG", None)

    def get_metadata(self) -> Dict[str, Any]:
        """
        Get Neuron profiling metadata.

        Returns:
            Dictionary with Neuron-specific metadata
        """
        # Find all .ntff files in output directory
        profile_files = [str(f) for f in self.output_dir.glob("*.ntff")]

        return {
            "profiler_type": "neuron",
            "profile_files": profile_files,
            "timeout": self.options.get("timeout", 600),
            "framework_profile": self.options.get("framework_profile", False),
        }
