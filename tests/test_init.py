"""Tests for project initialization."""

from pathlib import Path
from typing import Any, Dict

import pytest

from benchmark_capture.init import (
    ProjectTemplate,
    TemplateConfig,
    init_project,
    list_templates,
)


class TestTemplateConfig:
    """Tests for TemplateConfig."""

    def test_default_config(self) -> None:
        """Test default configuration."""
        config = TemplateConfig()
        assert config.project_name == "my-benchmarks"
        assert config.profiler == "auto"
        assert config.profiler_output_dir == "/tmp/profiles"
        assert config.timeout == 600
        assert config.model_path == "./models"
        assert config.benchmark_min_rounds == 5
        assert config.benchmark_warmup is True

    def test_custom_config(self) -> None:
        """Test custom configuration."""
        config = TemplateConfig(
            project_name="test-project",
            profiler="neuron",
            timeout=300,
            custom_key="custom_value",
        )
        assert config.project_name == "test-project"
        assert config.profiler == "neuron"
        assert config.timeout == 300
        assert config.extra["custom_key"] == "custom_value"

    def test_to_dict(self) -> None:
        """Test conversion to dictionary."""
        config = TemplateConfig(
            project_name="test", profiler="nsight", extra_var="extra_value"
        )
        config_dict = config.to_dict()

        assert config_dict["project_name"] == "test"
        assert config_dict["profiler"] == "nsight"
        assert config_dict["extra_var"] == "extra_value"


class TestProjectTemplate:
    """Tests for ProjectTemplate."""

    def test_valid_template(self) -> None:
        """Test creating template with valid name."""
        template = ProjectTemplate("basic")
        assert template.name == "basic"
        assert template.template_dir.name == "basic"

    def test_invalid_template(self) -> None:
        """Test creating template with invalid name."""
        with pytest.raises(ValueError, match="Unknown template: invalid"):
            ProjectTemplate("invalid")

    def test_from_name(self) -> None:
        """Test creating template from name."""
        template = ProjectTemplate.from_name("vllm")
        assert template.name == "vllm"

    def test_list_templates(self) -> None:
        """Test listing available templates."""
        templates = ProjectTemplate.list_templates()
        assert len(templates) == 4
        assert any(t["name"] == "basic" for t in templates)
        assert any(t["name"] == "vllm" for t in templates)
        assert any(t["name"] == "vllm-neuron" for t in templates)
        assert any(t["name"] == "minimal" for t in templates)

    def test_create_basic_template(self, tmp_path: Path) -> None:
        """Test creating basic template."""
        template = ProjectTemplate("basic")
        target = tmp_path / "test-project"

        created = template.create(str(target))

        assert len(created) > 0
        assert target.exists()
        assert (target / "benchmark.toml").exists()
        assert (target / "pytest.ini").exists()

    def test_create_vllm_template(self, tmp_path: Path) -> None:
        """Test creating vLLM template."""
        template = ProjectTemplate("vllm")
        target = tmp_path / "vllm-project"

        created = template.create(str(target))

        assert len(created) > 0
        assert (target / "test_vllm.py").exists()
        assert (target / "conftest.py").exists()

    def test_create_minimal_template(self, tmp_path: Path) -> None:
        """Test creating minimal template."""
        template = ProjectTemplate("minimal")
        target = tmp_path / "minimal-project"

        created = template.create(str(target))

        assert len(created) > 0
        assert (target / "benchmark.toml").exists()

    def test_create_vllm_neuron_template(self, tmp_path: Path) -> None:
        """Test creating vllm-neuron template."""
        template = ProjectTemplate("vllm-neuron")
        target = tmp_path / "vllm-neuron-project"

        created = template.create(str(target))

        assert len(created) > 0
        assert (target / "test_vllm.py").exists()
        assert (target / "conftest.py").exists()
        # Check Neuron profiler is set
        benchmark_toml = (target / "benchmark.toml").read_text()
        assert "neuron" in benchmark_toml

    def test_create_with_config(self, tmp_path: Path) -> None:
        """Test creating template with custom config."""
        template = ProjectTemplate("basic")
        target = tmp_path / "configured-project"
        config = TemplateConfig(project_name="my-custom-project", profiler="neuron")

        created = template.create(str(target), config=config)

        assert len(created) > 0

        # Check if config values were rendered in templates
        benchmark_toml = (target / "benchmark.toml").read_text()
        assert "neuron" in benchmark_toml

    def test_create_overwrite_false(self, tmp_path: Path) -> None:
        """Test create fails when directory exists and overwrite=False."""
        template = ProjectTemplate("basic")
        target = tmp_path / "existing"
        target.mkdir()
        (target / "existing_file.txt").write_text("existing content")

        with pytest.raises(FileExistsError, match="already exists"):
            template.create(str(target), overwrite=False)

    def test_create_overwrite_true(self, tmp_path: Path) -> None:
        """Test create succeeds when overwrite=True."""
        template = ProjectTemplate("basic")
        target = tmp_path / "existing"
        target.mkdir()
        (target / "existing_file.txt").write_text("existing content")

        created = template.create(str(target), overwrite=True)

        assert len(created) > 0
        assert (target / "benchmark.toml").exists()


class TestInitProject:
    """Tests for init_project function."""

    def test_init_default(self, tmp_path: Path) -> None:
        """Test init with default settings."""
        target = tmp_path / "benchmarks"

        created = init_project(path=str(target))

        assert len(created) > 0
        assert target.exists()
        assert (target / "benchmark.toml").exists()

    def test_init_custom_template(self, tmp_path: Path) -> None:
        """Test init with custom template."""
        target = tmp_path / "vllm-benchmarks"

        created = init_project(path=str(target), template="vllm")

        assert (target / "test_vllm.py").exists()

    def test_init_custom_profiler(self, tmp_path: Path) -> None:
        """Test init with custom profiler."""
        target = tmp_path / "neuron-benchmarks"

        created = init_project(path=str(target), profiler="neuron")

        benchmark_toml = (target / "benchmark.toml").read_text()
        assert "neuron" in benchmark_toml

    def test_init_custom_project_name(self, tmp_path: Path) -> None:
        """Test init with custom project name."""
        target = tmp_path / "my-project"

        created = init_project(
            path=str(target), project_name="CustomProject", template="basic"
        )

        assert len(created) > 0

    def test_init_invalid_template(self, tmp_path: Path) -> None:
        """Test init with invalid template."""
        target = tmp_path / "invalid-project"

        with pytest.raises(ValueError, match="Unknown template"):
            init_project(path=str(target), template="invalid")

    def test_init_with_kwargs(self, tmp_path: Path) -> None:
        """Test init with additional kwargs."""
        target = tmp_path / "custom-project"

        created = init_project(
            path=str(target), template="vllm", model_path="/custom/path"
        )

        assert len(created) > 0


def test_list_templates_function(capsys: Any) -> None:
    """Test list_templates function."""
    list_templates()
    captured = capsys.readouterr()
    assert "Available templates:" in captured.out
    assert "basic" in captured.out
    assert "vllm" in captured.out
    assert "minimal" in captured.out
