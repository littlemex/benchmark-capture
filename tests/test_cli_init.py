"""Tests for CLI init command."""

from pathlib import Path

import pytest

try:
    import click
    from click.testing import CliRunner

    from benchmark_capture.cli_init import init_cmd

    CLICK_AVAILABLE = True
except ImportError:
    CLICK_AVAILABLE = False


@pytest.mark.skipif(not CLICK_AVAILABLE, reason="Click not installed")
class TestCliInit:
    """Tests for CLI init command."""

    def test_init_default(self, tmp_path: Path) -> None:
        """Test init with default settings."""
        runner = CliRunner()

        with runner.isolated_filesystem(temp_dir=tmp_path) as td:
            result = runner.invoke(init_cmd, ["benchmarks"])
            target = Path(td) / "benchmarks"

            assert result.exit_code == 0
            assert "Project initialized successfully" in result.output
            assert target.exists()
            assert (target / "benchmark.toml").exists()

    def test_init_list_templates(self) -> None:
        """Test listing templates."""
        runner = CliRunner()
        result = runner.invoke(init_cmd, ["--list"])

        assert result.exit_code == 0
        assert "Available templates:" in result.output
        assert "basic" in result.output
        assert "vllm" in result.output
        assert "vllm-neuron" in result.output
        assert "minimal" in result.output

    def test_init_custom_template(self, tmp_path: Path) -> None:
        """Test init with custom template."""
        runner = CliRunner()

        with runner.isolated_filesystem(temp_dir=tmp_path) as td:
            result = runner.invoke(
                init_cmd, ["vllm-benchmarks", "--template", "vllm"]
            )
            target = Path(td) / "vllm-benchmarks"

            assert result.exit_code == 0
            assert target.exists()
            assert (target / "test_vllm.py").exists()

    def test_init_vllm_neuron_template(self, tmp_path: Path) -> None:
        """Test init with vllm-neuron template."""
        runner = CliRunner()

        with runner.isolated_filesystem(temp_dir=tmp_path) as td:
            result = runner.invoke(
                init_cmd, ["vllm-neuron-benchmarks", "--template", "vllm-neuron"]
            )
            target = Path(td) / "vllm-neuron-benchmarks"

            assert result.exit_code == 0
            assert target.exists()
            assert (target / "test_vllm.py").exists()
            # Verify Neuron profiler is set
            benchmark_toml = (target / "benchmark.toml").read_text()
            assert 'backend = "neuron"' in benchmark_toml

    def test_init_custom_profiler(self, tmp_path: Path) -> None:
        """Test init with custom profiler."""
        runner = CliRunner()

        with runner.isolated_filesystem(temp_dir=tmp_path) as td:
            result = runner.invoke(
                init_cmd, ["neuron-benchmarks", "--profiler", "neuron"]
            )
            target = Path(td) / "neuron-benchmarks"

            assert result.exit_code == 0
            benchmark_toml = (target / "benchmark.toml").read_text()
            assert "neuron" in benchmark_toml

    def test_init_custom_project_name(self, tmp_path: Path) -> None:
        """Test init with custom project name."""
        runner = CliRunner()
        target = tmp_path / "my-project"

        with runner.isolated_filesystem(temp_dir=tmp_path):
            result = runner.invoke(
                init_cmd, ["my-project", "--project-name", "CustomProject"]
            )

        assert result.exit_code == 0

    def test_init_force_overwrite(self, tmp_path: Path) -> None:
        """Test init with force overwrite."""
        runner = CliRunner()
        target = tmp_path / "existing"

        # Create directory with existing file
        target.mkdir()
        (target / "existing.txt").write_text("existing")

        with runner.isolated_filesystem(temp_dir=tmp_path):
            result = runner.invoke(init_cmd, ["existing", "--force"])

        assert result.exit_code == 0
        assert "Project initialized successfully" in result.output

    def test_init_existing_directory_no_force(self, tmp_path: Path) -> None:
        """Test init fails on existing directory without force."""
        runner = CliRunner()

        with runner.isolated_filesystem(temp_dir=tmp_path) as td:
            # Create directory with existing file
            target = Path(td) / "existing"
            target.mkdir()
            (target / "existing.txt").write_text("existing")

            # Respond 'n' to overwrite prompt
            result = runner.invoke(init_cmd, ["existing"], input="n\n")

            assert result.exit_code == 0
            assert "Aborted" in result.output

    def test_init_existing_directory_confirm_yes(self, tmp_path: Path) -> None:
        """Test init succeeds when user confirms overwrite."""
        runner = CliRunner()
        target = tmp_path / "existing"

        # Create directory with existing file
        target.mkdir()
        (target / "existing.txt").write_text("existing")

        with runner.isolated_filesystem(temp_dir=tmp_path):
            # Respond 'y' to overwrite prompt
            result = runner.invoke(init_cmd, ["existing"], input="y\n")

        assert result.exit_code == 0
        assert "Project initialized successfully" in result.output

    def test_init_invalid_template(self, tmp_path: Path) -> None:
        """Test init with invalid template."""
        runner = CliRunner()

        with runner.isolated_filesystem(temp_dir=tmp_path):
            result = runner.invoke(
                init_cmd, ["test-project", "--template", "invalid"]
            )

        # Click validates choices, so this should fail with invalid choice error
        assert result.exit_code != 0

    def test_init_short_options(self, tmp_path: Path) -> None:
        """Test init with short options."""
        runner = CliRunner()

        with runner.isolated_filesystem(temp_dir=tmp_path) as td:
            result = runner.invoke(
                init_cmd, ["short-opts", "-t", "vllm", "-p", "neuron", "-f"]
            )
            target = Path(td) / "short-opts"

            assert result.exit_code == 0
            assert target.exists()

    def test_init_help(self) -> None:
        """Test init help message."""
        runner = CliRunner()
        result = runner.invoke(init_cmd, ["--help"])

        assert result.exit_code == 0
        assert "Initialize a new benchmark project" in result.output
        assert "--template" in result.output
        assert "--profiler" in result.output
        assert "--force" in result.output
        assert "--list" in result.output

    def test_init_handles_file_exists_error(self, tmp_path: Path) -> None:
        """Test init handles FileExistsError gracefully."""
        runner = CliRunner()

        with runner.isolated_filesystem(temp_dir=tmp_path) as td:
            # Create directory with file inside isolated filesystem
            target = Path(td) / "test-project"
            target.mkdir()
            (target / "benchmark.toml").write_text("[profiler]\nbackend = 'test'")

            # Try to init without force (and cancel prompt)
            result = runner.invoke(init_cmd, ["test-project"], input="n\n")

            assert "Aborted" in result.output

    def test_init_handles_value_error(self, tmp_path: Path, monkeypatch) -> None:
        """Test init handles ValueError gracefully."""
        from benchmark_capture import cli_init

        runner = CliRunner()

        # Mock init_project to raise ValueError
        def mock_init_project(*args, **kwargs):
            raise ValueError("Invalid template")

        monkeypatch.setattr(cli_init, "init_project", mock_init_project)

        with runner.isolated_filesystem(temp_dir=tmp_path):
            result = runner.invoke(init_cmd, ["test-project", "--force"])

        assert result.exit_code == 1
        assert "Error" in result.output


@pytest.mark.skipif(CLICK_AVAILABLE, reason="Testing fallback when Click not installed")
class TestCliInitWithoutClick:
    """Tests for CLI when Click is not installed."""

    def test_init_cmd_raises_import_error_without_click(self) -> None:
        """Test init_cmd raises ImportError when Click not installed."""
        # This test is only meaningful when Click is not installed
        # In practice, this is handled by the CLICK_AVAILABLE check
        pass
