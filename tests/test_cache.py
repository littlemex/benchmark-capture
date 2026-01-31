"""Tests for cache management utilities."""

import os
from pathlib import Path
from unittest.mock import patch

import pytest

from benchmark_capture.cache import (
    CacheClearError,
    check_cache_status,
    clear_neuron_cache,
    get_neuron_artifacts_dir,
    get_neuron_cache_dir,
)


class TestGetCacheDirectories:
    """Tests for getting cache directory paths."""

    def test_get_neuron_cache_dir_default(self, clean_env: None) -> None:
        """Test default Neuron cache directory."""
        cache_dir = get_neuron_cache_dir()
        assert cache_dir == Path("/var/tmp/neuron-compile-cache")

    def test_get_neuron_cache_dir_from_env(self, clean_env: None) -> None:
        """Test Neuron cache directory from environment variable."""
        os.environ["NEURON_COMPILE_CACHE_URL"] = "/custom/cache"
        cache_dir = get_neuron_cache_dir()
        assert cache_dir == Path("/custom/cache")

    def test_get_neuron_cache_dir_s3_raises_error(self, clean_env: None) -> None:
        """Test S3 cache URL raises error."""
        os.environ["NEURON_COMPILE_CACHE_URL"] = "s3://bucket/cache"
        with pytest.raises(CacheClearError, match="S3-backed cache"):
            get_neuron_cache_dir()

    def test_get_neuron_artifacts_dir_not_set(self, clean_env: None) -> None:
        """Test artifacts directory when env var not set."""
        artifacts_dir = get_neuron_artifacts_dir()
        assert artifacts_dir is None

    def test_get_neuron_artifacts_dir_from_env(self, clean_env: None) -> None:
        """Test artifacts directory from environment variable."""
        os.environ["NEURON_COMPILED_ARTIFACTS"] = "/custom/artifacts"
        artifacts_dir = get_neuron_artifacts_dir()
        assert artifacts_dir == Path("/custom/artifacts")


class TestClearNeuronCache:
    """Tests for clearing Neuron cache."""

    def test_clear_cache_directory_not_exists(self, temp_dir: Path, clean_env: None) -> None:
        """Test clearing cache when directory doesn't exist."""
        cache_dir = temp_dir / "cache"
        result = clear_neuron_cache(cache_dir=cache_dir, clear_artifacts=False)

        assert result["cache_cleared"] is True
        assert result["cache_dir"] == str(cache_dir)
        assert result["cache_size_mb"] == 0.0
        assert result["artifacts_cleared"] is False

    def test_clear_cache_directory_exists(self, temp_dir: Path, clean_env: None) -> None:
        """Test clearing cache when directory exists with files."""
        cache_dir = temp_dir / "cache"
        cache_dir.mkdir()

        # Create some dummy cache files
        (cache_dir / "model1.neff").write_text("dummy content 1")
        (cache_dir / "model2.neff").write_text("dummy content 2")

        result = clear_neuron_cache(cache_dir=cache_dir, clear_artifacts=False)

        assert result["cache_cleared"] is True
        assert result["cache_dir"] == str(cache_dir)
        assert result["cache_size_mb"] > 0
        # Directory should be recreated empty
        assert cache_dir.exists()
        assert len(list(cache_dir.iterdir())) == 0

    def test_clear_cache_with_artifacts(self, temp_dir: Path, clean_env: None) -> None:
        """Test clearing both cache and artifacts."""
        cache_dir = temp_dir / "cache"
        artifacts_dir = temp_dir / "artifacts"

        cache_dir.mkdir()
        artifacts_dir.mkdir()

        (cache_dir / "model.neff").write_text("cache")
        (artifacts_dir / "compiled.pt").write_text("artifacts")

        os.environ["NEURON_COMPILED_ARTIFACTS"] = str(artifacts_dir)

        result = clear_neuron_cache(cache_dir=cache_dir, clear_artifacts=True)

        assert result["cache_cleared"] is True
        assert result["artifacts_cleared"] is True
        assert result["artifacts_dir"] == str(artifacts_dir)
        assert result["artifacts_size_mb"] > 0

        # Both should be empty
        assert len(list(cache_dir.iterdir())) == 0
        assert len(list(artifacts_dir.iterdir())) == 0

    def test_clear_cache_dry_run(self, temp_dir: Path, clean_env: None) -> None:
        """Test dry run mode doesn't actually delete."""
        cache_dir = temp_dir / "cache"
        cache_dir.mkdir()
        (cache_dir / "model.neff").write_text("content")

        result = clear_neuron_cache(cache_dir=cache_dir, clear_artifacts=False, dry_run=True)

        assert result["cache_cleared"] is False  # Not actually cleared
        assert result["cache_size_mb"] > 0
        # File should still exist
        assert (cache_dir / "model.neff").exists()

    def test_clear_cache_permission_error(self, temp_dir: Path, clean_env: None) -> None:
        """Test permission error handling."""
        cache_dir = temp_dir / "cache"
        cache_dir.mkdir()

        with patch("shutil.rmtree") as mock_rmtree:
            mock_rmtree.side_effect = PermissionError("Access denied")

            with pytest.raises(CacheClearError, match="Permission denied"):
                clear_neuron_cache(cache_dir=cache_dir, clear_artifacts=False)

    def test_clear_cache_generic_exception(self, temp_dir: Path, clean_env: None) -> None:
        """Test generic exception handling for cache."""
        cache_dir = temp_dir / "cache"
        cache_dir.mkdir()
        (cache_dir / "model.neff").write_text("content")

        with patch("shutil.rmtree") as mock_rmtree:
            mock_rmtree.side_effect = RuntimeError("Unexpected error")

            with pytest.raises(CacheClearError, match="Failed to clear cache"):
                clear_neuron_cache(cache_dir=cache_dir, clear_artifacts=False)

    def test_clear_artifacts_permission_error(self, temp_dir: Path, clean_env: None) -> None:
        """Test permission error handling for artifacts."""
        cache_dir = temp_dir / "cache"
        artifacts_dir = temp_dir / "artifacts"

        cache_dir.mkdir()
        artifacts_dir.mkdir()
        (artifacts_dir / "compiled.pt").write_text("artifact")

        os.environ["NEURON_COMPILED_ARTIFACTS"] = str(artifacts_dir)

        with patch("shutil.rmtree") as mock_rmtree:
            # First call succeeds (cache), second call fails (artifacts)
            mock_rmtree.side_effect = [None, PermissionError("Access denied")]

            with pytest.raises(CacheClearError, match="Permission denied when clearing artifacts"):
                clear_neuron_cache(cache_dir=cache_dir, clear_artifacts=True)

    def test_clear_artifacts_generic_exception(self, temp_dir: Path, clean_env: None) -> None:
        """Test generic exception handling for artifacts."""
        cache_dir = temp_dir / "cache"
        artifacts_dir = temp_dir / "artifacts"

        cache_dir.mkdir()
        artifacts_dir.mkdir()
        (artifacts_dir / "compiled.pt").write_text("artifact")

        os.environ["NEURON_COMPILED_ARTIFACTS"] = str(artifacts_dir)

        with patch("shutil.rmtree") as mock_rmtree:
            # First call succeeds (cache), second call fails (artifacts)
            mock_rmtree.side_effect = [None, RuntimeError("Unexpected error")]

            with pytest.raises(CacheClearError, match="Failed to clear artifacts"):
                clear_neuron_cache(cache_dir=cache_dir, clear_artifacts=True)

    def test_clear_artifacts_not_exists(self, temp_dir: Path, clean_env: None) -> None:
        """Test clearing when artifacts directory doesn't exist."""
        cache_dir = temp_dir / "cache"
        artifacts_dir = temp_dir / "artifacts"

        cache_dir.mkdir()
        # artifacts_dir does not exist

        os.environ["NEURON_COMPILED_ARTIFACTS"] = str(artifacts_dir)

        result = clear_neuron_cache(cache_dir=cache_dir, clear_artifacts=True)

        assert result["cache_cleared"] is True
        assert result["artifacts_cleared"] is True  # Nothing to clear
        assert result["artifacts_dir"] == str(artifacts_dir)


class TestCheckCacheStatus:
    """Tests for checking cache status."""

    def test_check_cache_not_exists(self, temp_dir: Path, clean_env: None) -> None:
        """Test checking status when cache doesn't exist."""
        os.environ["NEURON_COMPILE_CACHE_URL"] = str(temp_dir / "cache")

        status = check_cache_status()

        assert status["cache_exists"] is False
        assert status["cache_size_mb"] == 0.0
        assert status["cached_models_count"] == 0

    def test_check_cache_exists_with_files(self, temp_dir: Path, clean_env: None) -> None:
        """Test checking status when cache exists with files."""
        cache_dir = temp_dir / "cache"
        cache_dir.mkdir()
        (cache_dir / "model1.neff").write_text("content1")
        (cache_dir / "model2.neff").write_text("content2")

        os.environ["NEURON_COMPILE_CACHE_URL"] = str(cache_dir)

        status = check_cache_status()

        assert status["cache_exists"] is True
        assert status["cache_size_mb"] > 0
        assert status["cached_models_count"] == 2

    def test_check_cache_with_artifacts(self, temp_dir: Path, clean_env: None) -> None:
        """Test checking status with artifacts."""
        cache_dir = temp_dir / "cache"
        artifacts_dir = temp_dir / "artifacts"

        cache_dir.mkdir()
        artifacts_dir.mkdir()
        (artifacts_dir / "compiled.pt").write_text("artifact")

        os.environ["NEURON_COMPILE_CACHE_URL"] = str(cache_dir)
        os.environ["NEURON_COMPILED_ARTIFACTS"] = str(artifacts_dir)

        status = check_cache_status()

        assert status["cache_exists"] is True
        assert status["artifacts_exists"] is True
        assert status["artifacts_dir"] == str(artifacts_dir)
        assert status["artifacts_size_mb"] > 0

    def test_check_cache_permission_error(self, temp_dir: Path, clean_env: None) -> None:
        """Test checking status with permission error on cache."""
        cache_dir = temp_dir / "cache"
        cache_dir.mkdir()
        (cache_dir / "model.neff").write_text("content")

        os.environ["NEURON_COMPILE_CACHE_URL"] = str(cache_dir)

        # Mock Path.rglob to raise PermissionError
        with patch.object(Path, "rglob") as mock_rglob:
            mock_rglob.side_effect = PermissionError("Access denied")

            status = check_cache_status()

            # Should still return status but with zero values
            assert status["cache_exists"] is True
            assert status["cache_size_mb"] == 0.0
            assert status["cached_models_count"] == 0

    def test_check_artifacts_permission_error(self, temp_dir: Path, clean_env: None) -> None:
        """Test checking status with permission error on artifacts."""
        cache_dir = temp_dir / "cache"
        artifacts_dir = temp_dir / "artifacts"

        cache_dir.mkdir()
        artifacts_dir.mkdir()
        (artifacts_dir / "compiled.pt").write_text("artifact")

        os.environ["NEURON_COMPILE_CACHE_URL"] = str(cache_dir)
        os.environ["NEURON_COMPILED_ARTIFACTS"] = str(artifacts_dir)

        # Mock to allow cache check but fail on artifacts
        original_rglob = Path.rglob

        def selective_rglob(self, pattern):
            if "artifacts" in str(self):
                raise PermissionError("Access denied")
            return original_rglob(self, pattern)

        with patch.object(Path, "rglob", selective_rglob):
            status = check_cache_status()

            # Cache should work, artifacts should have None for size
            assert status["cache_exists"] is True
            assert status["artifacts_exists"] is True
            assert status["artifacts_size_mb"] is None
