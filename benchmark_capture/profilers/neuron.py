"""AWS Neuron profiler implementation."""

import logging
import os
from typing import Any, Dict

from ..cache import CacheClearError, clear_neuron_cache
from .base import Profiler

logger = logging.getLogger(__name__)

# Default timeout for Neuron execution (seconds)
DEFAULT_NEURON_TIMEOUT = 600


class NeuronProfiler(Profiler):
    """
    AWS Neuron profiler implementation.

    Supports two profiling modes:

    Standard mode (perfetto=False, default):
    - Sets NEURON_PROFILE environment variable
    - Generates .ntff files directly in output directory
    - Compatible with neuron-profile CLI tools

    Perfetto mode (perfetto=True):
    - Sets NEURON_RT_INSPECT_* environment variables
    - Generates session directory with NTFF files
    - Session directory: i-<instance>_pid_<number>/
    - Files suitable for Perfetto analysis (conversion handled externally)

    Args:
        output_dir: Base directory for profile output
        perfetto: Enable Perfetto-compatible NTFF generation (default: False)
        timeout: Execution timeout in seconds (default: 600)
        framework_profile: Enable framework-level profiling (default: False)
        clear_cache_before: Clear Neuron cache before profiling (default: False)
        clear_cache_after: Clear Neuron cache after profiling (default: False)
    """

    def setup(self, function_name: str) -> None:
        """
        Setup Neuron profiling environment.

        Args:
            function_name: Name of the function being profiled
        """
        self.function_name = function_name

        # Clear compilation cache if requested
        if self.options.get("clear_cache_before", False):
            logger.info("Clearing Neuron compilation cache before benchmark...")
            try:
                result = clear_neuron_cache(
                    cache_dir=None,  # Use default from env
                    clear_artifacts=True,
                    dry_run=False,
                )
                if result["cache_cleared"]:
                    logger.info(
                        f"Cache cleared: {result['cache_dir']} "
                        f"({result['cache_size_mb']:.2f} MB)"
                    )
                if result["artifacts_cleared"]:
                    logger.info(
                        f"Artifacts cleared: {result['artifacts_dir']} "
                        f"({result['artifacts_size_mb']:.2f} MB)"
                    )
            except CacheClearError as e:
                logger.error(f"Failed to clear cache: {e}")
                raise

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Choose profiling mode based on perfetto option
        if self.options.get("perfetto", False):
            self._setup_perfetto_mode()
        else:
            self._setup_standard_mode()

    def _setup_standard_mode(self) -> None:
        """Setup standard NEURON_PROFILE mode (backward compatible)."""
        os.environ["NEURON_PROFILE"] = str(self.output_dir)
        os.environ["NEURON_RT_EXEC_TIMEOUT"] = str(
            self.options.get("timeout", DEFAULT_NEURON_TIMEOUT)
        )

        if self.options.get("framework_profile", False):
            os.environ["NEURON_FRAMEWORK_DEBUG"] = "1"

    def _setup_perfetto_mode(self) -> None:
        """Setup Perfetto profiling mode with NEURON_RT_INSPECT_* variables."""
        os.environ["NEURON_RT_INSPECT_ENABLE"] = "1"
        os.environ["NEURON_RT_INSPECT_SYSTEM_PROFILE"] = "1"
        os.environ["NEURON_RT_INSPECT_DEVICE_PROFILE"] = "1"
        os.environ["NEURON_RT_INSPECT_OUTPUT_DIR"] = str(self.output_dir)
        os.environ["NEURON_RT_EXEC_TIMEOUT"] = str(
            self.options.get("timeout", DEFAULT_NEURON_TIMEOUT)
        )

        logger.info(f"Perfetto profiling enabled: {self.output_dir}")

    def teardown(self) -> None:
        """Clean up Neuron profiling environment variables."""
        # Clean up environment variables (both modes)
        for var in [
            "NEURON_PROFILE",
            "NEURON_RT_EXEC_TIMEOUT",
            "NEURON_FRAMEWORK_DEBUG",
            "NEURON_RT_INSPECT_ENABLE",
            "NEURON_RT_INSPECT_SYSTEM_PROFILE",
            "NEURON_RT_INSPECT_DEVICE_PROFILE",
            "NEURON_RT_INSPECT_OUTPUT_DIR",
        ]:
            os.environ.pop(var, None)

        # Clear compilation cache after benchmark if requested
        if self.options.get("clear_cache_after", False):
            logger.info("Clearing Neuron compilation cache after benchmark...")
            try:
                result = clear_neuron_cache(
                    cache_dir=None,  # Use default from env
                    clear_artifacts=True,
                    dry_run=False,
                )
                if result["cache_cleared"]:
                    logger.info(
                        f"Cache cleared: {result['cache_dir']} "
                        f"({result['cache_size_mb']:.2f} MB)"
                    )
                if result["artifacts_cleared"]:
                    logger.info(
                        f"Artifacts cleared: {result['artifacts_dir']} "
                        f"({result['artifacts_size_mb']:.2f} MB)"
                    )
            except CacheClearError as e:
                logger.warning(f"Failed to clear cache after benchmark: {e}")

    def get_metadata(self) -> Dict[str, Any]:
        """
        Get Neuron profiling metadata.

        Returns:
            Dictionary with Neuron-specific metadata
        """
        metadata = {
            "profiler_type": "neuron",
            "timeout": self.options.get("timeout", DEFAULT_NEURON_TIMEOUT),
            "framework_profile": self.options.get("framework_profile", False),
            "clear_cache_before": self.options.get("clear_cache_before", False),
            "clear_cache_after": self.options.get("clear_cache_after", False),
            "perfetto_mode": self.options.get("perfetto", False),
        }

        # Find session directories or profile files based on mode
        if self.options.get("perfetto", False):
            # Perfetto mode: Look for session directories
            session_dirs = list(self.output_dir.glob("i-*_pid_*"))
            if session_dirs:
                # Use the newest directory by modification time
                session_dir = sorted(
                    session_dirs, key=lambda p: p.stat().st_mtime, reverse=True
                )[0]
                metadata["session_dir"] = str(session_dir)

                # List NTFF files in session directory
                ntff_files = [
                    str(f.relative_to(self.output_dir)) for f in session_dir.glob("*.ntff")
                ]
                metadata["ntff_files"] = ntff_files
            else:
                metadata["session_dir"] = None
                metadata["ntff_files"] = []
        else:
            # Standard mode: List .ntff files directly
            profile_files = [str(f) for f in self.output_dir.glob("*.ntff")]
            metadata["profile_files"] = profile_files

        return metadata
