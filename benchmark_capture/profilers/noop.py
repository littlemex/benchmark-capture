"""No-op profiler that does nothing (fallback)."""

from typing import Any, Dict

from .base import Profiler


class NoOpProfiler(Profiler):
    """
    No-op profiler that does nothing.

    Used as fallback when no profiling hardware is detected,
    or when profiling is explicitly disabled.

    This ensures benchmarks run normally without profiling overhead.
    """

    def setup(self, function_name: str) -> None:
        """
        Setup (no-op).

        Args:
            function_name: Name of the function being profiled
        """
        self.function_name = function_name
        # Do nothing

    def teardown(self) -> None:
        """Teardown (no-op)."""
        # Do nothing
        pass

    def get_metadata(self) -> Dict[str, Any]:
        """
        Get metadata (empty for no-op).

        Returns:
            Dictionary with minimal metadata
        """
        return {
            "profiler_type": "noop",
            "note": "No profiling performed",
        }

    def save_metadata(self) -> None:
        """Override to skip metadata saving for no-op profiler."""
        # Don't create files for no-op profiler
        pass
