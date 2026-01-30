"""Tests for profiling decorator."""

import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from benchmark_capture import profile
from benchmark_capture.decorators import _resolve_profiler_name


class TestResolveProfilerName:
    """Tests for profiler name resolution logic."""

    def test_explicit_decorator_parameter(self, clean_env: None) -> None:
        """Test explicit decorator parameter takes priority."""
        assert _resolve_profiler_name("neuron") == "neuron"
        assert _resolve_profiler_name("nsight") == "nsight"
        assert _resolve_profiler_name("noop") == "noop"

    def test_environment_variable_override(self, clean_env: None) -> None:
        """Test environment variable overrides auto-detection."""
        os.environ["BENCHMARK_PROFILER"] = "neuron"

        with patch("benchmark_capture.decorators.detect_hardware") as mock_detect:
            mock_detect.return_value = "nsight"
            assert _resolve_profiler_name("auto") == "neuron"

    def test_config_file_override(
        self, clean_env: None, sample_config_file: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test config file overrides auto-detection."""
        monkeypatch.chdir(sample_config_file.parent)

        with patch("benchmark_capture.decorators.detect_hardware") as mock_detect:
            mock_detect.return_value = "nsight"
            assert _resolve_profiler_name("auto") == "neuron"

    def test_auto_detection(self, clean_env: None) -> None:
        """Test falls back to auto-detection."""
        with patch("benchmark_capture.decorators.detect_hardware") as mock_detect:
            mock_detect.return_value = "nsight"
            assert _resolve_profiler_name("auto") == "nsight"

    def test_fallback_to_noop(self, clean_env: None) -> None:
        """Test fallback to noop if detection returns None."""
        with patch("benchmark_capture.decorators.detect_hardware") as mock_detect:
            mock_detect.return_value = None
            assert _resolve_profiler_name("auto") == "noop"

    def test_priority_order(
        self, clean_env: None, sample_config_file: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test complete priority order."""
        monkeypatch.chdir(sample_config_file.parent)
        os.environ["BENCHMARK_PROFILER"] = "nsight"

        with patch("benchmark_capture.decorators.detect_hardware") as mock_detect:
            mock_detect.return_value = "noop"

            # 1. Explicit parameter wins
            assert _resolve_profiler_name("noop") == "noop"

            # 2. Env var wins over config and detection
            assert _resolve_profiler_name("auto") == "nsight"

            # 3. Remove env var, config wins
            del os.environ["BENCHMARK_PROFILER"]
            assert _resolve_profiler_name("auto") == "neuron"

    def test_config_loading_exception_handling(self, clean_env: None) -> None:
        """Test handles exceptions during config loading gracefully."""
        with patch("benchmark_capture.decorators.load_config") as mock_load:
            # Mock load_config to raise an exception
            mock_load.side_effect = Exception("Config loading failed")

            with patch("benchmark_capture.decorators.detect_hardware") as mock_detect:
                mock_detect.return_value = "nsight"
                # Should fall back to auto-detection, not crash
                result = _resolve_profiler_name("auto")
                assert result == "nsight"

    def test_config_with_auto_backend(
        self, clean_env: None, temp_dir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test config file with 'auto' backend falls through to detection."""
        # Create config with backend = "auto"
        config_file = temp_dir / "benchmark.toml"
        config_file.write_text('[profiler]\nbackend = "auto"')

        monkeypatch.chdir(temp_dir)

        with patch("benchmark_capture.decorators.detect_hardware") as mock_detect:
            mock_detect.return_value = "nsight"
            # Should use auto-detection, not the 'auto' value
            result = _resolve_profiler_name("auto")
            assert result == "nsight"


class TestProfileDecorator:
    """Tests for @profile decorator."""

    def test_decorator_with_auto_detection(self, temp_dir: Path, clean_env: None) -> None:
        """Test decorator with auto-detection."""

        @profile()
        def sample_function() -> str:
            return "result"

        with patch("benchmark_capture.decorators.detect_hardware") as mock_detect:
            mock_detect.return_value = "noop"
            result = sample_function()

        assert result == "result"

    def test_decorator_with_explicit_profiler(self, temp_dir: Path, clean_env: None) -> None:
        """Test decorator with explicit profiler."""

        @profile("neuron", output_dir=str(temp_dir))
        def sample_function() -> str:
            return "result"

        result = sample_function()
        assert result == "result"

        # Check metadata was saved
        metadata_file = temp_dir / "metadata.json"
        assert metadata_file.exists()

    def test_decorator_with_noop(self, temp_dir: Path, clean_env: None) -> None:
        """Test decorator with noop profiler."""

        @profile("noop", output_dir=str(temp_dir))
        def sample_function() -> str:
            return "result"

        result = sample_function()
        assert result == "result"

        # noop should not create metadata file
        metadata_file = temp_dir / "metadata.json"
        assert not metadata_file.exists()

    def test_decorator_preserves_function_attributes(self, clean_env: None) -> None:
        """Test decorator preserves function name and docstring."""

        @profile("noop")
        def sample_function() -> str:
            """Sample docstring."""
            return "result"

        assert sample_function.__name__ == "sample_function"
        assert sample_function.__doc__ == "Sample docstring."

    def test_decorator_with_profiler_options(self, temp_dir: Path, clean_env: None) -> None:
        """Test decorator passes options to profiler."""

        @profile("neuron", output_dir=str(temp_dir), timeout=1200, framework_profile=True)
        def sample_function() -> str:
            return "result"

        result = sample_function()
        assert result == "result"

        # Verify env vars were set
        # Note: They'll be cleaned up after function returns
        # We'd need to check during execution or via metadata

    def test_decorator_teardown_on_exception(self, temp_dir: Path, clean_env: None) -> None:
        """Test decorator calls teardown even if function raises."""

        @profile("neuron", output_dir=str(temp_dir))
        def failing_function() -> None:
            raise ValueError("Test error")

        with pytest.raises(ValueError, match="Test error"):
            failing_function()

        # Teardown should still have been called
        # Env vars should be cleaned up
        assert "NEURON_PROFILE" not in os.environ

    def test_decorator_with_arguments(self, temp_dir: Path, clean_env: None) -> None:
        """Test decorator works with functions that take arguments."""

        @profile("noop")
        def add(a: int, b: int) -> int:
            return a + b

        result = add(2, 3)
        assert result == 5

    def test_decorator_with_kwargs(self, temp_dir: Path, clean_env: None) -> None:
        """Test decorator works with keyword arguments."""

        @profile("noop")
        def greet(name: str, greeting: str = "Hello") -> str:
            return f"{greeting}, {name}!"

        result = greet("World", greeting="Hi")
        assert result == "Hi, World!"


class TestIntegrationWithPytestBenchmark:
    """Tests for integration with pytest-benchmark."""

    def test_with_mock_benchmark_fixture(self, temp_dir: Path, clean_env: None) -> None:
        """Test decorator works with mock benchmark fixture."""
        # Mock benchmark fixture
        mock_benchmark = MagicMock()
        mock_benchmark.stats.stats.mean = 0.5

        @profile("noop")
        def test_function(benchmark: MagicMock) -> str:
            result = benchmark(lambda: "computed")
            return result

        _ = test_function(mock_benchmark)
        assert mock_benchmark.called
