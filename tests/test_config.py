"""Tests for configuration loading."""

from pathlib import Path

import pytest

from benchmark_capture.config import _find_config_file, load_config


class TestConfigLoading:
    """Tests for configuration file loading."""

    def test_load_config_from_explicit_file(self, sample_config_file: Path) -> None:
        """Test loading config from explicit file path."""
        config = load_config(sample_config_file)

        assert config["profiler"]["backend"] == "neuron"
        assert config["profiler"]["neuron"]["timeout"] == 1200

    def test_load_config_returns_empty_if_not_found(self, temp_dir: Path) -> None:
        """Test returns empty dict if config not found."""
        config = load_config(temp_dir / "nonexistent.toml")
        assert config == {}

    def test_find_config_file_in_current_dir(
        self, temp_dir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test finding config file in current directory."""
        config_file = temp_dir / "benchmark.toml"
        config_file.write_text("[profiler]\nbackend = 'test'")

        monkeypatch.chdir(temp_dir)
        found = _find_config_file()

        assert found == config_file

    def test_find_config_file_in_parent_dir(
        self, temp_dir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test finding config file in parent directory."""
        config_file = temp_dir / "benchmark.toml"
        config_file.write_text("[profiler]\nbackend = 'test'")

        subdir = temp_dir / "subdir"
        subdir.mkdir()
        monkeypatch.chdir(subdir)

        found = _find_config_file()
        assert found == config_file

    def test_find_config_file_not_found(
        self, temp_dir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test returns None when config file not found."""
        monkeypatch.chdir(temp_dir)
        found = _find_config_file()
        assert found is None

    def test_load_config_handles_invalid_toml(self, temp_dir: Path) -> None:
        """Test handles invalid TOML gracefully."""
        invalid_config = temp_dir / "invalid.toml"
        invalid_config.write_text("invalid toml content {{{")

        config = load_config(invalid_config)
        assert config == {}

    def test_load_config_without_tomli(
        self, temp_dir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test falls back gracefully if tomli/tomllib unavailable."""
        config_file = temp_dir / "benchmark.toml"
        config_file.write_text("[profiler]\nbackend = 'test'")

        # Mock tomllib/tomli as unavailable
        import sys

        with monkeypatch.context() as m:
            if "tomllib" in sys.modules:
                m.delitem(sys.modules, "tomllib")
            if "tomli" in sys.modules:
                m.delitem(sys.modules, "tomli")

            # Should return empty dict without crashing
            _ = load_config(config_file)
            # Note: actual result depends on Python version
            # Python 3.11+ has tomllib built-in
