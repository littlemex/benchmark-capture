"""Tests for profiler implementations."""

import json
import os
from pathlib import Path

import pytest

from benchmark_capture.profilers import (
    NeuronProfiler,
    NoOpProfiler,
    NSightProfiler,
    get_profiler,
)


class TestNeuronProfiler:
    """Tests for NeuronProfiler."""

    def test_setup_creates_directory(self, temp_dir: Path, clean_env: None) -> None:
        """Test setup creates output directory."""
        profiler = NeuronProfiler(output_dir=str(temp_dir / "neuron"))
        profiler.setup("test_func")

        assert (temp_dir / "neuron").exists()
        assert profiler.function_name == "test_func"

    def test_setup_sets_env_vars(self, temp_dir: Path, clean_env: None) -> None:
        """Test setup sets Neuron environment variables."""
        profiler = NeuronProfiler(output_dir=str(temp_dir / "neuron"), timeout=1200)
        profiler.setup("test_func")

        assert os.environ["NEURON_PROFILE"] == str(temp_dir / "neuron")
        assert os.environ["NEURON_RT_EXEC_TIMEOUT"] == "1200"

    def test_setup_framework_profile(self, temp_dir: Path, clean_env: None) -> None:
        """Test framework profiling option."""
        profiler = NeuronProfiler(output_dir=str(temp_dir), framework_profile=True)
        profiler.setup("test_func")

        assert os.environ.get("NEURON_FRAMEWORK_DEBUG") == "1"

    def test_teardown_cleans_env_vars(self, temp_dir: Path, clean_env: None) -> None:
        """Test teardown removes environment variables."""
        profiler = NeuronProfiler(output_dir=str(temp_dir))
        profiler.setup("test_func")
        profiler.teardown()

        assert "NEURON_PROFILE" not in os.environ
        assert "NEURON_RT_EXEC_TIMEOUT" not in os.environ

    def test_get_metadata(self, temp_dir: Path, clean_env: None) -> None:
        """Test metadata generation."""
        profiler = NeuronProfiler(output_dir=str(temp_dir))
        profiler.setup("test_func")

        # Create a fake .ntff file
        (temp_dir / "profile.ntff").touch()

        metadata = profiler.get_metadata()
        assert metadata["profiler_type"] == "neuron"
        assert len(metadata["profile_files"]) == 1
        assert metadata["timeout"] == 600

    def test_save_metadata(self, temp_dir: Path, clean_env: None) -> None:
        """Test metadata is saved to JSON file."""
        profiler = NeuronProfiler(output_dir=str(temp_dir))
        profiler.setup("test_func")
        profiler.save_metadata()

        metadata_file = temp_dir / "metadata.json"
        assert metadata_file.exists()

        with open(metadata_file) as f:
            data = json.load(f)
            assert data["function"] == "test_func"
            assert data["profiler"] == "NeuronProfiler"


class TestNSightProfiler:
    """Tests for NSightProfiler."""

    def test_setup_creates_directory(self, temp_dir: Path, clean_env: None) -> None:
        """Test setup creates output directory."""
        profiler = NSightProfiler(output_dir=str(temp_dir / "nsight"))
        profiler.setup("test_func")

        assert (temp_dir / "nsight").exists()

    def test_setup_sets_env_vars(self, temp_dir: Path, clean_env: None) -> None:
        """Test setup sets NSight environment variables."""
        profiler = NSightProfiler(output_dir=str(temp_dir))
        profiler.setup("test_func")

        assert "NSYS_OUTPUT_FILE" in os.environ
        assert os.environ["NSYS_CUDA_API_TRACE"] == "1"

    def test_teardown_cleans_env_vars(self, temp_dir: Path, clean_env: None) -> None:
        """Test teardown removes environment variables."""
        profiler = NSightProfiler(output_dir=str(temp_dir))
        profiler.setup("test_func")
        profiler.teardown()

        assert "NSYS_OUTPUT_FILE" not in os.environ
        assert "NSYS_CUDA_API_TRACE" not in os.environ

    def test_get_metadata(self, temp_dir: Path, clean_env: None) -> None:
        """Test metadata generation."""
        profiler = NSightProfiler(output_dir=str(temp_dir))
        profiler.setup("test_func")

        metadata = profiler.get_metadata()
        assert metadata["profiler_type"] == "nsight"
        assert metadata["cuda_api_trace"] is True


class TestNoOpProfiler:
    """Tests for NoOpProfiler."""

    def test_setup_does_nothing(self, temp_dir: Path, clean_env: None) -> None:
        """Test setup does nothing."""
        profiler = NoOpProfiler(output_dir=str(temp_dir))
        profiler.setup("test_func")
        # Should not raise any exceptions

    def test_teardown_does_nothing(self, temp_dir: Path, clean_env: None) -> None:
        """Test teardown does nothing."""
        profiler = NoOpProfiler(output_dir=str(temp_dir))
        profiler.setup("test_func")
        profiler.teardown()
        # Should not raise any exceptions

    def test_save_metadata_does_nothing(self, temp_dir: Path, clean_env: None) -> None:
        """Test save_metadata does not create files."""
        profiler = NoOpProfiler(output_dir=str(temp_dir))
        profiler.setup("test_func")
        profiler.save_metadata()

        # No metadata.json should be created
        assert not (temp_dir / "metadata.json").exists()


class TestGetProfiler:
    """Tests for get_profiler factory function."""

    def test_get_neuron_profiler(self) -> None:
        """Test getting NeuronProfiler."""
        profiler = get_profiler("neuron")
        assert isinstance(profiler, NeuronProfiler)

    def test_get_nsight_profiler(self) -> None:
        """Test getting NSightProfiler."""
        profiler = get_profiler("nsight")
        assert isinstance(profiler, NSightProfiler)

    def test_get_noop_profiler(self) -> None:
        """Test getting NoOpProfiler."""
        profiler = get_profiler("noop")
        assert isinstance(profiler, NoOpProfiler)

    def test_invalid_profiler_raises_error(self) -> None:
        """Test invalid profiler name raises ValueError."""
        with pytest.raises(ValueError, match="Unknown profiler"):
            get_profiler("invalid")

    def test_profiler_options_passed(self, temp_dir: Path) -> None:
        """Test profiler options are passed correctly."""
        profiler = get_profiler("neuron", output_dir=str(temp_dir), timeout=1200)
        assert profiler.output_dir == temp_dir
        assert profiler.options["timeout"] == 1200


class TestNeuronProfilerCacheManagement:
    """Tests for Neuron profiler cache management."""

    def test_clear_cache_before_setup(self, temp_dir: Path, clean_env: None) -> None:
        """Test cache clearing before benchmark."""
        cache_dir = temp_dir / "cache"
        cache_dir.mkdir()
        (cache_dir / "model.neff").write_text("cached model")

        os.environ["NEURON_COMPILE_CACHE_URL"] = str(cache_dir)

        profiler = NeuronProfiler(
            output_dir=str(temp_dir / "output"), clear_cache_before=True
        )
        profiler.setup("test_func")

        # Cache should be cleared
        assert cache_dir.exists()
        assert len(list(cache_dir.iterdir())) == 0

    def test_clear_cache_after_teardown(self, temp_dir: Path, clean_env: None) -> None:
        """Test cache clearing after benchmark."""
        cache_dir = temp_dir / "cache"
        cache_dir.mkdir()
        (cache_dir / "model.neff").write_text("cached model")

        os.environ["NEURON_COMPILE_CACHE_URL"] = str(cache_dir)

        profiler = NeuronProfiler(
            output_dir=str(temp_dir / "output"), clear_cache_after=True
        )
        profiler.setup("test_func")

        # Recreate file (simulating compilation during benchmark)
        (cache_dir / "model.neff").write_text("new cached model")

        profiler.teardown()

        # Cache should be cleared after teardown
        assert cache_dir.exists()
        assert len(list(cache_dir.iterdir())) == 0

    def test_no_cache_clearing_by_default(self, temp_dir: Path, clean_env: None) -> None:
        """Test cache is not cleared by default."""
        cache_dir = temp_dir / "cache"
        cache_dir.mkdir()
        (cache_dir / "model.neff").write_text("cached model")

        os.environ["NEURON_COMPILE_CACHE_URL"] = str(cache_dir)

        profiler = NeuronProfiler(output_dir=str(temp_dir / "output"))
        profiler.setup("test_func")
        profiler.teardown()

        # Cache should still exist
        assert (cache_dir / "model.neff").exists()

    def test_metadata_includes_cache_settings(self, temp_dir: Path, clean_env: None) -> None:
        """Test metadata includes cache clearing settings."""
        profiler = NeuronProfiler(
            output_dir=str(temp_dir), clear_cache_before=True, clear_cache_after=True
        )
        profiler.setup("test_func")

        metadata = profiler.get_metadata()
        assert metadata["clear_cache_before"] is True
        assert metadata["clear_cache_after"] is True
