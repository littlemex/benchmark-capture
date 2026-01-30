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


class ProjectExample:
    """Project example manager."""

    AVAILABLE_EXAMPLES = ["vllm-neuron-reranker"]

    def __init__(self, example_name: str):
        """
        Initialize project example.

        Args:
            example_name: Example name (vllm-neuron-reranker)

        Raises:
            ValueError: If example name is unknown
        """
        if example_name not in self.AVAILABLE_EXAMPLES:
            raise ValueError(
                f"Unknown example: {example_name}. "
                f"Available: {', '.join(self.AVAILABLE_EXAMPLES)}"
            )

        self.example_name = example_name
        self.base_path = Path(__file__).parent / "examples" / example_name

        if not self.base_path.exists():
            raise FileNotFoundError(f"Example directory not found: {self.base_path}")

    @classmethod
    def list_examples(cls) -> List[Dict[str, str]]:
        """
        List available examples.

        Returns:
            List of dictionaries with example metadata
        """
        examples = []
        base_examples_path = Path(__file__).parent / "examples"

        for example_name in cls.AVAILABLE_EXAMPLES:
            example_path = base_examples_path / example_name
            readme_path = example_path / "README.md"

            # Extract description from README
            description = "No description available"
            if readme_path.exists():
                with open(readme_path) as f:
                    lines = f.readlines()
                    # Look for first non-empty line after title
                    for i, line in enumerate(lines):
                        if line.startswith("#"):
                            # Get next non-empty line
                            for j in range(i + 1, min(i + 5, len(lines))):
                                desc_line = lines[j].strip()
                                if desc_line and not desc_line.startswith("#"):
                                    description = desc_line
                                    break
                            break

            examples.append({"name": example_name, "description": description})

        return examples

    def copy_to(self, target_path: Path, overwrite: bool = False) -> List[str]:
        """
        Copy example to target directory.

        Args:
            target_path: Target directory path
            overwrite: If True, overwrite existing files

        Returns:
            List of created file paths (relative to target)

        Raises:
            FileExistsError: If target exists and overwrite is False
        """
        # Check if target exists
        if target_path.exists() and not overwrite:
            existing_files = list(target_path.glob("*"))
            if existing_files:
                raise FileExistsError(f"Directory {target_path} already exists and is not empty")

        # Create target directory
        target_path.mkdir(parents=True, exist_ok=True)

        # Copy all files from example
        created_files = []

        for item in self.base_path.rglob("*"):
            if item.is_file():
                # Calculate relative path
                rel_path = item.relative_to(self.base_path)
                target_file = target_path / rel_path

                # Create parent directories
                target_file.parent.mkdir(parents=True, exist_ok=True)

                # Copy file
                shutil.copy2(item, target_file)
                created_files.append(str(target_path.name / rel_path))

        return sorted(created_files)


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


def init_example(
    path: str,
    example: str,
    overwrite: bool = False,
) -> List[str]:
    """
    Initialize a project from an example.

    Args:
        path: Target directory path
        example: Example name (vllm-neuron-reranker)
        overwrite: If True, overwrite existing files

    Returns:
        List of created file paths (relative to target)

    Raises:
        FileExistsError: If directory exists and overwrite is False
        ValueError: If example name is unknown

    Examples:
        >>> init_example(
        ...     path="./my-reranker-benchmark",
        ...     example="vllm-neuron-reranker"
        ... )
    """
    example_obj = ProjectExample(example)
    return example_obj.copy_to(Path(path), overwrite)


def list_examples() -> None:
    """Print available examples."""
    examples = ProjectExample.list_examples()
    print("Available examples:")
    for ex in examples:
        print(f"  {ex['name']:20s} - {ex['description']}")
