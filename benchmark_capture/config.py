"""Configuration file loading and management."""

from pathlib import Path
from typing import Any, Dict, Optional


def load_config(config_file: Optional[Path] = None) -> Dict[str, Any]:
    """
    Load configuration from TOML file.

    Search order:
    1. Provided config_file parameter
    2. benchmark.toml in current directory
    3. benchmark.toml in parent directories (walk up)
    4. Return empty dict if not found

    Args:
        config_file: Optional path to config file

    Returns:
        Configuration dictionary
    """
    # Try provided config file
    if config_file and config_file.exists():
        return _load_toml(config_file)

    # Try finding benchmark.toml in current or parent directories
    config_file = _find_config_file()
    if config_file:
        return _load_toml(config_file)

    # Return empty config if not found
    return {}


def _find_config_file() -> Optional[Path]:
    """
    Find benchmark.toml by walking up directory tree.

    Returns:
        Path to config file or None if not found
    """
    current = Path.cwd()
    for parent in [current] + list(current.parents):
        config_file = parent / "benchmark.toml"
        if config_file.exists():
            return config_file
    return None


def _load_toml(config_file: Path) -> Dict[str, Any]:
    """
    Load TOML configuration file.

    Args:
        config_file: Path to TOML file

    Returns:
        Configuration dictionary
    """
    try:
        # Try tomli (Python 3.11+)
        try:
            import tomllib
        except ImportError:
            # Fallback to tomli for Python 3.9-3.10
            try:
                import tomli as tomllib  # type: ignore
            except ImportError:
                # If neither available, skip config file loading
                return {}

        with open(config_file, "rb") as f:
            return tomllib.load(f)
    except Exception:
        # If loading fails, return empty config
        return {}
