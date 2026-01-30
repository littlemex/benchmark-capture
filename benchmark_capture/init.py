"""Project initialization functionality."""

import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

# Try to import jinja2 for template rendering
try:
    from jinja2 import Environment, FileSystemLoader

    JINJA2_AVAILABLE = True
except ImportError:
    JINJA2_AVAILABLE = False


class TemplateConfig:
    """Configuration for template rendering."""

    def __init__(
        self,
        project_name: str = "my-benchmarks",
        profiler: str = "auto",
        profiler_output_dir: str = "/tmp/profiles",
        timeout: int = 600,
        model_path: str = "./models",
        benchmark_min_rounds: int = 5,
        benchmark_warmup: bool = True,
        **kwargs: Any
    ):
        """
        Initialize template configuration.

        Args:
            project_name: Project name
            profiler: Profiler backend (auto, neuron, nsight, noop)
            profiler_output_dir: Output directory for profile data
            timeout: Profiling timeout in seconds
            model_path: Model path (for vLLM template)
            benchmark_min_rounds: Minimum benchmark rounds
            benchmark_warmup: Enable benchmark warmup
            **kwargs: Additional template variables
        """
        self.project_name = project_name
        self.profiler = profiler
        self.profiler_output_dir = profiler_output_dir
        self.timeout = timeout
        self.model_path = model_path
        self.benchmark_min_rounds = benchmark_min_rounds
        self.benchmark_warmup = benchmark_warmup
        self.extra = kwargs

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for template rendering."""
        return {
            "project_name": self.project_name,
            "profiler": self.profiler,
            "profiler_output_dir": self.profiler_output_dir,
            "timeout": self.timeout,
            "model_path": self.model_path,
            "benchmark_min_rounds": self.benchmark_min_rounds,
            "benchmark_warmup": self.benchmark_warmup,
            **self.extra,
        }


class ProjectTemplate:
    """Project template manager."""

    AVAILABLE_TEMPLATES = ["basic", "vllm", "vllm-neuron", "minimal"]

    def __init__(self, template_name: str):
        """
        Initialize project template.

        Args:
            template_name: Template name (basic, vllm, vllm-neuron, minimal)

        Raises:
            ValueError: If template name is unknown
        """
        if template_name not in self.AVAILABLE_TEMPLATES:
            raise ValueError(
                f"Unknown template: {template_name}. "
                f"Available: {', '.join(self.AVAILABLE_TEMPLATES)}"
            )
        self.name = template_name
        self.template_dir = Path(__file__).parent / "templates" / template_name

    @classmethod
    def from_name(cls, name: str) -> "ProjectTemplate":
        """Create template from name."""
        return cls(name)

    @classmethod
    def list_templates(cls) -> List[Dict[str, str]]:
        """List available templates with descriptions."""
        return [
            {"name": "basic", "description": "Simple benchmark setup"},
            {"name": "vllm", "description": "vLLM benchmarks (GPU/CPU)"},
            {"name": "vllm-neuron", "description": "vLLM-Neuron benchmarks (AWS Inferentia)"},
            {"name": "minimal", "description": "Minimal configuration only"},
        ]

    def create(
        self,
        path: str,
        config: Optional[TemplateConfig] = None,
        overwrite: bool = False,
    ) -> List[str]:
        """
        Create project from template.

        Args:
            path: Target directory path
            config: Template configuration
            overwrite: Overwrite existing files

        Returns:
            List of created file paths

        Raises:
            FileExistsError: If directory exists and overwrite=False
        """
        target_path = Path(path)

        # Check if directory exists
        if target_path.exists() and not overwrite:
            existing_files = list(target_path.glob("*"))
            if existing_files:
                raise FileExistsError(
                    f"Directory already exists and is not empty: {target_path}. "
                    "Use overwrite=True to force."
                )

        # Create directory
        target_path.mkdir(parents=True, exist_ok=True)

        # Default config
        if config is None:
            config = TemplateConfig()

        # Copy/render template files
        if JINJA2_AVAILABLE:
            created_files = self._render_templates(target_path, config)
        else:
            created_files = self._copy_templates(target_path)

        return created_files

    def _render_templates(
        self, target_path: Path, config: TemplateConfig
    ) -> List[str]:
        """Render templates with Jinja2."""
        # Setup Jinja2 environment
        templates_root = Path(__file__).parent / "templates"
        env = Environment(loader=FileSystemLoader(str(templates_root)))

        created = []
        template_vars = config.to_dict()

        # Render each template file
        for template_file in self.template_dir.glob("*"):
            if template_file.is_file() and not template_file.name.startswith("_"):
                # Load and render template
                template_path = f"{self.name}/{template_file.name}"
                template = env.get_template(template_path)
                rendered = template.render(**template_vars)

                # Write to target
                target_file = target_path / template_file.name
                target_file.write_text(rendered)
                created.append(str(target_file))

        return created

    def _copy_templates(self, target_path: Path) -> List[str]:
        """Copy templates without rendering (fallback if jinja2 unavailable)."""
        created = []

        for template_file in self.template_dir.glob("*"):
            if template_file.is_file() and not template_file.name.startswith("_"):
                target_file = target_path / template_file.name
                shutil.copy2(template_file, target_file)
                created.append(str(target_file))

        return created


def init_project(
    path: str = "./benchmarks",
    template: str = "basic",
    profiler: str = "auto",
    project_name: Optional[str] = None,
    overwrite: bool = False,
    **kwargs: Any
) -> List[str]:
    """
    Initialize a new benchmark project.

    Args:
        path: Target directory path
        template: Template name (basic, vllm, vllm-neuron, minimal)
        profiler: Profiler backend (auto, neuron, nsight, noop)
        project_name: Project name (default: directory name)
        overwrite: Overwrite existing files
        **kwargs: Additional template variables

    Returns:
        List of created file paths

    Raises:
        FileExistsError: If directory exists and overwrite=False
        ValueError: If template name is unknown

    Examples:
        # Basic initialization
        >>> init_project("./benchmarks")
        ['benchmarks/benchmark.toml', 'benchmarks/pytest.ini', ...]

        # vLLM template with custom settings
        >>> init_project(
        ...     path="./benchmarks",
        ...     template="vllm",
        ...     profiler="neuron",
        ...     model_path="./models/my-model"
        ... )
    """
    if project_name is None:
        project_name = Path(path).name

    config = TemplateConfig(project_name=project_name, profiler=profiler, **kwargs)

    template_obj = ProjectTemplate.from_name(template)
    return template_obj.create(path, config, overwrite)


def list_templates() -> None:
    """Print available templates."""
    templates = ProjectTemplate.list_templates()
    print("Available templates:")
    for tmpl in templates:
        print(f"  {tmpl['name']:10s} - {tmpl['description']}")
