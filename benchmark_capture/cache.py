"""Cache management utilities for benchmark-capture."""

import logging
import os
import shutil
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Constants for cache management
BYTES_PER_MB = 1024**2
DEFAULT_NEURON_CACHE_PATH = "/var/tmp/neuron-compile-cache"


class CacheClearError(Exception):
    """Raised when cache clearing fails due to permissions or other issues."""

    pass


def get_neuron_cache_dir() -> Path:
    """
    Get Neuron compilation cache directory.

    Priority:
    1. NEURON_COMPILE_CACHE_URL environment variable
    2. Default: /var/tmp/neuron-compile-cache

    Returns:
        Path to Neuron compilation cache directory
    """
    cache_url = os.getenv("NEURON_COMPILE_CACHE_URL", DEFAULT_NEURON_CACHE_PATH)

    # Handle S3 URLs (not supported for local clearing)
    if cache_url.startswith("s3://"):
        raise CacheClearError(
            f"S3-backed cache ({cache_url}) cannot be cleared locally. "
            "Use AWS CLI: aws s3 rm {cache_url} --recursive"
        )

    return Path(cache_url)


def get_neuron_artifacts_dir() -> Optional[Path]:
    """
    Get vLLM-Neuron compiled artifacts directory.

    Returns:
        Path to artifacts directory if NEURON_COMPILED_ARTIFACTS is set, None otherwise
    """
    artifacts_path = os.getenv("NEURON_COMPILED_ARTIFACTS")
    if artifacts_path:
        return Path(artifacts_path)
    return None


def clear_neuron_cache(
    cache_dir: Optional[Path] = None, clear_artifacts: bool = True, dry_run: bool = False
) -> dict:
    """
    Clear Neuron compilation cache.

    Args:
        cache_dir: Custom cache directory (default: from NEURON_COMPILE_CACHE_URL)
        clear_artifacts: Also clear NEURON_COMPILED_ARTIFACTS if set (default: True)
        dry_run: If True, only report what would be deleted without actually deleting

    Returns:
        Dictionary with clearing results:
        {
            "cache_cleared": bool,
            "cache_dir": str,
            "cache_size_mb": float,
            "artifacts_cleared": bool,
            "artifacts_dir": str or None,
            "artifacts_size_mb": float or None,
        }

    Raises:
        CacheClearError: If clearing fails due to permissions or other errors
    """
    result = {
        "cache_cleared": False,
        "cache_dir": None,
        "cache_size_mb": 0.0,
        "artifacts_cleared": False,
        "artifacts_dir": None,
        "artifacts_size_mb": None,
    }

    # Get cache directory
    if cache_dir is None:
        cache_dir = get_neuron_cache_dir()

    result["cache_dir"] = str(cache_dir)

    # Clear compilation cache
    if cache_dir.exists():
        try:
            # Calculate size before deletion
            cache_size = sum(f.stat().st_size for f in cache_dir.rglob("*") if f.is_file())
            result["cache_size_mb"] = cache_size / BYTES_PER_MB

            if dry_run:
                logger.info(
                    f"[DRY RUN] Would clear Neuron cache: {cache_dir} "
                    f"({result['cache_size_mb']:.2f} MB)"
                )
            else:
                logger.info(
                    f"Clearing Neuron compilation cache: {cache_dir} "
                    f"({result['cache_size_mb']:.2f} MB)"
                )
                shutil.rmtree(cache_dir)
                cache_dir.mkdir(parents=True, exist_ok=True)
                logger.info(f"✓ Cache cleared: {cache_dir}")

            result["cache_cleared"] = not dry_run

        except PermissionError as e:
            raise CacheClearError(
                f"Permission denied when clearing cache: {cache_dir}\n"
                f"Try running with appropriate permissions or use:\n"
                f"  sudo rm -rf {cache_dir}/*"
            ) from e
        except Exception as e:
            raise CacheClearError(f"Failed to clear cache {cache_dir}: {e}") from e
    else:
        logger.info(f"Cache directory does not exist: {cache_dir}")
        result["cache_cleared"] = True  # Nothing to clear

    # Clear vLLM-Neuron compiled artifacts
    if clear_artifacts:
        artifacts_dir = get_neuron_artifacts_dir()
        if artifacts_dir:
            result["artifacts_dir"] = str(artifacts_dir)

            if artifacts_dir.exists():
                try:
                    # Calculate size before deletion
                    artifacts_size = sum(
                        f.stat().st_size for f in artifacts_dir.rglob("*") if f.is_file()
                    )
                    result["artifacts_size_mb"] = artifacts_size / BYTES_PER_MB

                    if dry_run:
                        logger.info(
                            f"[DRY RUN] Would clear vLLM artifacts: {artifacts_dir} "
                            f"({result['artifacts_size_mb']:.2f} MB)"
                        )
                    else:
                        logger.info(
                            f"Clearing vLLM-Neuron artifacts: {artifacts_dir} "
                            f"({result['artifacts_size_mb']:.2f} MB)"
                        )
                        shutil.rmtree(artifacts_dir)
                        artifacts_dir.mkdir(parents=True, exist_ok=True)
                        logger.info(f"✓ Artifacts cleared: {artifacts_dir}")

                    result["artifacts_cleared"] = not dry_run

                except PermissionError as e:
                    raise CacheClearError(
                        f"Permission denied when clearing artifacts: {artifacts_dir}\n"
                        f"Try running with appropriate permissions."
                    ) from e
                except Exception as e:
                    raise CacheClearError(
                        f"Failed to clear artifacts {artifacts_dir}: {e}"
                    ) from e
            else:
                logger.info(f"Artifacts directory does not exist: {artifacts_dir}")
                result["artifacts_cleared"] = True  # Nothing to clear

    return result


def check_cache_status() -> dict:
    """
    Check Neuron cache status without modifying anything.

    Returns:
        Dictionary with cache information:
        {
            "cache_dir": str,
            "cache_exists": bool,
            "cache_size_mb": float,
            "cached_models_count": int,
            "artifacts_dir": str or None,
            "artifacts_exists": bool,
            "artifacts_size_mb": float or None,
        }
    """
    cache_dir = get_neuron_cache_dir()

    result = {
        "cache_dir": str(cache_dir),
        "cache_exists": cache_dir.exists(),
        "cache_size_mb": 0.0,
        "cached_models_count": 0,
        "artifacts_dir": None,
        "artifacts_exists": False,
        "artifacts_size_mb": None,
    }

    # Check compilation cache
    if cache_dir.exists():
        try:
            neff_files = list(cache_dir.rglob("*.neff"))
            result["cached_models_count"] = len(neff_files)
            cache_size = sum(f.stat().st_size for f in cache_dir.rglob("*") if f.is_file())
            result["cache_size_mb"] = cache_size / BYTES_PER_MB
        except PermissionError:
            logger.warning(f"Permission denied when checking cache: {cache_dir}")

    # Check artifacts
    artifacts_dir = get_neuron_artifacts_dir()
    if artifacts_dir:
        result["artifacts_dir"] = str(artifacts_dir)
        result["artifacts_exists"] = artifacts_dir.exists()

        if artifacts_dir.exists():
            try:
                artifacts_size = sum(
                    f.stat().st_size for f in artifacts_dir.rglob("*") if f.is_file()
                )
                result["artifacts_size_mb"] = artifacts_size / BYTES_PER_MB
            except PermissionError:
                logger.warning(f"Permission denied when checking artifacts: {artifacts_dir}")

    return result
