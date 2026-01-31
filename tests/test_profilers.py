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

    def test_clear_cache_before_raises_error(self, temp_dir: Path, clean_env: None) -> None:
        """Test cache clearing error in setup is propagated."""
        from unittest.mock import patch
        from benchmark_capture.cache import CacheClearError

        cache_dir = temp_dir / "cache"
        cache_dir.mkdir()
        os.environ["NEURON_COMPILE_CACHE_URL"] = str(cache_dir)

        profiler = NeuronProfiler(output_dir=str(temp_dir / "output"), clear_cache_before=True)

        # Patch where it's used, not where it's defined
        with patch("benchmark_capture.profilers.neuron.clear_neuron_cache") as mock_clear:
            mock_clear.side_effect = CacheClearError("Permission denied")

            with pytest.raises(CacheClearError, match="Permission denied"):
                profiler.setup("test_func")

    def test_clear_cache_after_handles_error_gracefully(
        self, temp_dir: Path, clean_env: None
    ) -> None:
        """Test cache clearing error in teardown is logged as warning."""
        from unittest.mock import patch
        from benchmark_capture.cache import CacheClearError

        cache_dir = temp_dir / "cache"
        cache_dir.mkdir()
        os.environ["NEURON_COMPILE_CACHE_URL"] = str(cache_dir)

        profiler = NeuronProfiler(output_dir=str(temp_dir / "output"), clear_cache_after=True)
        profiler.setup("test_func")

        # Patch where it's used, not where it's defined
        with patch("benchmark_capture.profilers.neuron.clear_neuron_cache") as mock_clear:
            mock_clear.side_effect = CacheClearError("Permission denied")

            # Should not raise, just log warning
            profiler.teardown()

            # Verify clear was attempted
            mock_clear.assert_called_once()


class TestNeuronProfilerPerfettoMode:
    """Tests for NeuronProfiler Perfetto mode."""

    def test_perfetto_mode_sets_inspect_env_vars(self, temp_dir: Path, clean_env: None) -> None:
        """Test that perfetto=True sets NEURON_RT_INSPECT_* variables."""
        profiler = NeuronProfiler(output_dir=str(temp_dir), perfetto=True)
        profiler.setup("test_func")

        assert os.environ.get("NEURON_RT_INSPECT_ENABLE") == "1"
        assert os.environ.get("NEURON_RT_INSPECT_SYSTEM_PROFILE") == "1"
        assert os.environ.get("NEURON_RT_INSPECT_DEVICE_PROFILE") == "1"
        assert os.environ.get("NEURON_RT_INSPECT_OUTPUT_DIR") == str(temp_dir)

        # Standard mode vars should NOT be set
        assert "NEURON_PROFILE" not in os.environ

    def test_standard_mode_unchanged(self, temp_dir: Path, clean_env: None) -> None:
        """Test that perfetto=False uses standard NEURON_PROFILE."""
        profiler = NeuronProfiler(output_dir=str(temp_dir))
        profiler.setup("test_func")

        assert os.environ.get("NEURON_PROFILE") == str(temp_dir)

        # Perfetto vars should NOT be set
        assert "NEURON_RT_INSPECT_ENABLE" not in os.environ

    def test_metadata_includes_session_dir_in_perfetto_mode(self, temp_dir: Path, clean_env: None) -> None:
        """Test that metadata includes session_dir for perfetto mode."""
        # Create mock session directory
        session_dir = temp_dir / "i-instance_pid_1234"
        session_dir.mkdir()

        # Create mock NTFF file
        (session_dir / "test.ntff").write_text("mock ntff data")

        profiler = NeuronProfiler(output_dir=str(temp_dir), perfetto=True)
        profiler.setup("test_func")

        metadata = profiler.get_metadata()

        assert metadata["perfetto_mode"] is True
        assert "session_dir" in metadata
        assert "i-instance_pid_1234" in metadata["session_dir"]
        assert "ntff_files" in metadata
        assert len(metadata["ntff_files"]) > 0

    def test_teardown_cleans_both_modes_env_vars(self, temp_dir: Path, clean_env: None) -> None:
        """Test that teardown cleans up all environment variables."""
        profiler = NeuronProfiler(output_dir=str(temp_dir), perfetto=True)
        profiler.setup("test_func")

        # Verify vars are set
        assert "NEURON_RT_INSPECT_ENABLE" in os.environ

        profiler.teardown()

        # Verify all vars are cleaned
        assert "NEURON_RT_INSPECT_ENABLE" not in os.environ
        assert "NEURON_RT_INSPECT_SYSTEM_PROFILE" not in os.environ
        assert "NEURON_RT_INSPECT_DEVICE_PROFILE" not in os.environ
        assert "NEURON_RT_INSPECT_OUTPUT_DIR" not in os.environ
        assert "NEURON_PROFILE" not in os.environ

    def test_standard_mode_metadata_includes_profile_files(self, temp_dir: Path, clean_env: None) -> None:
        """Test that standard mode metadata includes profile_files."""
        profiler = NeuronProfiler(output_dir=str(temp_dir))
        profiler.setup("test_func")

        # Create mock NTFF files
        (temp_dir / "profile1.ntff").write_text("mock data")
        (temp_dir / "profile2.ntff").write_text("mock data")

        metadata = profiler.get_metadata()

        assert metadata["perfetto_mode"] is False
        assert "profile_files" in metadata
        assert len(metadata["profile_files"]) == 2
        assert "session_dir" not in metadata

    def test_perfetto_mode_selects_newest_session_dir(self, temp_dir: Path, clean_env: None) -> None:
        """Test that perfetto mode selects the newest session directory."""
        # Create multiple session directories
        old_session = temp_dir / "i-instance_pid_1000"
        old_session.mkdir()
        (old_session / "old.ntff").write_text("old")

        # Make newer session
        import time
        time.sleep(0.01)
        new_session = temp_dir / "i-instance_pid_2000"
        new_session.mkdir()
        (new_session / "new.ntff").write_text("new")

        profiler = NeuronProfiler(output_dir=str(temp_dir), perfetto=True)
        profiler.setup("test_func")

        metadata = profiler.get_metadata()

        # Should select the newer session
        assert "i-instance_pid_2000" in metadata["session_dir"]
        assert "new.ntff" in metadata["ntff_files"][0]

    def test_perfetto_mode_handles_no_session_dirs(self, temp_dir: Path, clean_env: None) -> None:
        """Test that perfetto mode handles missing session directories gracefully."""
        profiler = NeuronProfiler(output_dir=str(temp_dir), perfetto=True)
        profiler.setup("test_func")

        metadata = profiler.get_metadata()

        assert metadata["perfetto_mode"] is True
        assert metadata["session_dir"] is None
        assert metadata["ntff_files"] == []
