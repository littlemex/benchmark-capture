"""Main profiling decorator implementation."""

import os
from functools import wraps
from typing import Any, Callable, Optional

from .config import load_config
from .profilers import detect_hardware, get_profiler


def profile(
    profiler_name: str = "auto", output_dir: Optional[str] = None, **options: Any
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """
    Profile decorator for benchmarks with automatic hardware detection.

    Priority order for profiler selection:
    1. Decorator explicit parameter (if not "auto")
    2. Environment variable BENCHMARK_PROFILER
    3. Config file [profiler] backend setting
    4. Auto-detection (hardware detection)
    5. Fallback to "noop"

    Args:
        profiler_name: Profiler to use ("auto", "neuron", "nsight", "noop")
        output_dir: Output directory for profile data
        **options: Additional profiler-specific options

    Returns:
        Decorated function

    Examples:
        # Auto-detect hardware (recommended)
        @profile()
        def test_inference(benchmark):
            result = benchmark(llm.generate, prompts)

        # Force specific profiler
        @profile("neuron")
        def test_neuron_inference(benchmark):
            ...

        # Custom output directory
        @profile("auto", output_dir="/tmp/my_profiles")
        def test_with_custom_dir(benchmark):
            ...

        # Profiler-specific options
        @profile("neuron", timeout=1200, framework_profile=True)
        def test_with_options(benchmark):
            ...

        # Disable profiling
        @profile("noop")
        def test_no_profiling(benchmark):
            ...
    """

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Resolve profiler name using priority order
            actual_profiler = _resolve_profiler_name(profiler_name)

            # Get profiler instance
            profiler = get_profiler(actual_profiler, output_dir=output_dir, **options)

            # Setup profiling
            profiler.setup(func.__name__)

            try:
                # Run the function
                result = func(*args, **kwargs)
            finally:
                # Teardown profiling
                profiler.teardown()

                # Save metadata
                profiler.save_metadata()

            return result

        return wrapper

    return decorator


def _resolve_profiler_name(decorator_value: str) -> str:
    """
    Resolve profiler name using priority order.

    Priority:
    1. Decorator parameter (if not "auto")
    2. Environment variable BENCHMARK_PROFILER
    3. Config file setting
    4. Auto-detection
    5. Fallback to "noop"

    Args:
        decorator_value: Value from decorator parameter

    Returns:
        Resolved profiler name
    """
    # Priority 1: Explicit decorator parameter
    if decorator_value != "auto":
        return decorator_value

    # Priority 2: Environment variable
    env_profiler = os.environ.get("BENCHMARK_PROFILER")
    if env_profiler:
        return env_profiler

    # Priority 3: Config file
    try:
        config = load_config()
        if config and "profiler" in config and "backend" in config["profiler"]:
            config_profiler = config["profiler"]["backend"]
            if config_profiler != "auto":
                return config_profiler
    except Exception:
        # If config loading fails, continue to auto-detection
        pass

    # Priority 4: Auto-detection
    detected = detect_hardware()

    # Priority 5: Fallback
    return detected if detected else "noop"
