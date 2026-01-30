"""Base class for all profilers."""

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional


class Profiler(ABC):
    """
    Abstract base class for profilers.

    All profilers must implement:
    - setup(): Configure profiling environment
    - teardown(): Clean up profiling environment
    - get_metadata(): Return profiler-specific metadata
    """

    def __init__(self, output_dir: Optional[str] = None, **options: Any) -> None:
        """
        Initialize profiler.

        Args:
            output_dir: Output directory for profile data (default: /tmp/profiles)
            **options: Profiler-specific options
        """
        self.output_dir = Path(output_dir) if output_dir else Path("/tmp/profiles")
        self.options = options
        self.function_name: Optional[str] = None

    @abstractmethod
    def setup(self, function_name: str) -> None:
        """
        Setup profiling environment.

        Args:
            function_name: Name of the function being profiled
        """
        pass

    @abstractmethod
    def teardown(self) -> None:
        """Clean up profiling environment."""
        pass

    @abstractmethod
    def get_metadata(self) -> Dict[str, Any]:
        """
        Get profiling metadata.

        Returns:
            Dictionary with profiler-specific metadata
        """
        pass

    def save_metadata(self) -> None:
        """Save metadata to JSON file in output directory."""
        metadata = {
            "function": self.function_name,
            "profiler": self.__class__.__name__,
            "output_dir": str(self.output_dir),
            **self.get_metadata(),
        }

        metadata_file = self.output_dir / "metadata.json"
        with open(metadata_file, "w") as f:
            json.dump(metadata, f, indent=2)
