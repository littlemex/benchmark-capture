"""CLI command for project initialization."""

from pathlib import Path
from typing import Optional

try:
    import click

    CLICK_AVAILABLE = True
except ImportError:
    CLICK_AVAILABLE = False

from .init import ProjectExample, ProjectTemplate, init_example, init_project


def _init_cmd_impl(
    path: str,
    template: Optional[str],
    profiler: str,
    project_name: Optional[str],
    force: bool,
    list_templates: bool,
    example: Optional[str],
    list_examples: bool,
) -> None:
    """Implementation of init command."""
    # List examples
    if list_examples:
        examples = ProjectExample.list_examples()
        click.echo("Available examples:")
        for ex in examples:
            click.echo(f"  {ex['name']:20s} - {ex['description']}")
        return

    # List templates
    if list_templates:
        templates = ProjectTemplate.list_templates()
        click.echo("Available templates:")
        for tmpl in templates:
            click.echo(f"  {tmpl['name']:10s} - {tmpl['description']}")
        return

    # Check for mutual exclusivity
    if example and template:
        click.secho("Error: Cannot use both --template and --example", fg="red")
        raise SystemExit(1)

    # Require either template or example
    if not example and not template:
        template = "basic"  # Default to basic template

    # Confirm overwrite if directory exists
    target_path = Path(path)
    if target_path.exists() and not force:
        existing_files = list(target_path.glob("*"))
        if existing_files:
            if not click.confirm(
                f"Directory {path} exists and is not empty. Overwrite?", default=False
            ):
                click.echo("Aborted.")
                return

    # Create project from example or template
    try:
        if example:
            # Deploy from example
            created_files = init_example(path=path, example=example, overwrite=force)

            click.secho("✓ Example deployed successfully!", fg="green")
            click.echo(f"\nCreated files in {path}:")
            for file in created_files:
                click.echo(f"  - {file}")

            click.echo("\nNext steps:")
            click.echo(f"  1. cd {path}")

            if example == "vllm-neuron-reranker":
                click.echo("  2. Activate Neuron venv:")
                click.echo("     source /opt/aws_neuronx_venv_pytorch_inference_vllm_0_13/bin/activate")
                click.echo("  3. Edit config.yaml to set MODEL_PATH")
                click.echo("  4. pytest test_reranker.py --benchmark-only -v")
            else:
                click.echo("  2. Follow instructions in README.md")

        else:
            # Deploy from template
            created_files = init_project(
                path=path,
                template=template,
                profiler=profiler,
                project_name=project_name,
                overwrite=force,
            )

            click.secho("✓ Project initialized successfully!", fg="green")
            click.echo(f"\nCreated files in {path}:")
            for file in created_files:
                click.echo(f"  - {file}")

            click.echo("\nNext steps:")
            click.echo(f"  1. cd {path}")
            click.echo("  2. pip install pytest pytest-benchmark benchmark-capture")
            click.echo("  3. pytest --benchmark-only")

    except FileExistsError as e:
        click.secho(f"Error: {e}", fg="red")
        click.echo("Use --force to overwrite existing files.")
        raise SystemExit(1)
    except ValueError as e:
        click.secho(f"Error: {e}", fg="red")
        raise SystemExit(1)


if CLICK_AVAILABLE:

    @click.command()
    @click.argument("path", default="./benchmarks", type=click.Path())
    @click.option(
        "--template",
        "-t",
        default=None,
        type=click.Choice(["basic", "vllm", "vllm-neuron", "minimal"], case_sensitive=False),
        help="Template to use (basic, vllm, vllm-neuron, minimal)",
    )
    @click.option(
        "--example",
        "-e",
        default=None,
        type=click.Choice(["vllm-neuron-reranker"], case_sensitive=False),
        help="Example to deploy (vllm-neuron-reranker)",
    )
    @click.option(
        "--profiler",
        "-p",
        default="auto",
        type=click.Choice(["auto", "neuron", "nsight", "noop"], case_sensitive=False),
        help="Default profiler backend (for templates only)",
    )
    @click.option(
        "--project-name",
        "-n",
        default=None,
        help="Project name (for templates only, defaults to directory name)",
    )
    @click.option(
        "--force", "-f", is_flag=True, help="Overwrite existing files without prompt"
    )
    @click.option(
        "--list",
        "list_templates",
        is_flag=True,
        help="List available templates and exit",
    )
    @click.option(
        "--list-examples",
        is_flag=True,
        help="List available examples and exit",
    )
    def init_cmd(
        path: str,
        template: Optional[str],
        example: Optional[str],
        profiler: str,
        project_name: Optional[str],
        force: bool,
        list_templates: bool,
        list_examples: bool,
    ) -> None:
        """
        Initialize a new benchmark project.

        Creates a benchmark project from a template or deploys a ready-to-use example.

        Examples:

            # Basic initialization from template
            benchmark-capture-init ./benchmarks

            # Use vLLM-Neuron template
            benchmark-capture-init ./vllm-benchmarks --template vllm-neuron

            # Deploy vLLM-Neuron reranker example
            benchmark-capture-init ./my-reranker --example vllm-neuron-reranker

            # List available templates
            benchmark-capture-init --list

            # List available examples
            benchmark-capture-init --list-examples
        """
        _init_cmd_impl(path, template, profiler, project_name, force, list_templates, example, list_examples)

else:
    # Fallback if Click not available
    def init_cmd() -> None:
        """Init command requires click to be installed."""
        raise ImportError(
            "Click is required for CLI commands. Install with: "
            "pip install benchmark-capture[init]"
        )


if __name__ == "__main__":
    if CLICK_AVAILABLE:
        init_cmd()
    else:
        click.echo("Error: Click is not installed.")
        click.echo("Install with: pip install click")
        raise SystemExit(1)
