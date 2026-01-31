"""
vLLM configuration helper with hardware-aware optimization.

This module provides a thin wrapper for vLLM configuration that automatically
adapts to detected hardware (AWS Neuron or GPU). It does NOT manage or convert
Neuron-specific settings - users have full control over their configuration.

Responsibilities:
- Hardware detection (Neuron vs GPU)
- Exclude Neuron config on GPU environments
- Log configuration for visibility

NOT responsible for:
- Neuron config conversion or mapping
- Parameter validation
- Default value management

Example:
    >>> from benchmark_capture.utils import VLLMConfigHelper
    >>>
    >>> # User manages all configuration
    >>> config = {
    ...     "tensor_parallel_size": 2,
    ...     "max_num_seqs": 4,
    ...     "override_neuron_config": {
    ...         "pa_num_blocks": 512,
    ...         "is_prefix_caching": True,
    ...     }
    ... }
    >>>
    >>> helper = VLLMConfigHelper(config)
    >>> vllm_config = helper.build()
    >>>
    >>> # On Neuron: config includes override_neuron_config
    >>> # On GPU: override_neuron_config is excluded
"""

import json
import logging
from typing import Any, Dict

from benchmark_capture.profilers.detector import is_neuron_available

logger = logging.getLogger(__name__)


class VLLMConfigHelper:
    """
    Hardware-aware vLLM configuration helper.

    This class provides a thin wrapper around vLLM configuration that:
    1. Detects hardware (Neuron or GPU)
    2. Excludes Neuron-specific config on GPU
    3. Logs configuration for debugging

    Users retain full control over configuration - this helper only adapts
    the config to the detected hardware environment.

    Attributes:
        config: User-provided vLLM configuration
        is_neuron: Whether running on Neuron hardware

    Example:
        >>> # User provides complete configuration
        >>> config = {
        ...     "tensor_parallel_size": 2,
        ...     "override_neuron_config": {
        ...         "pa_num_blocks": 512,
        ...     }
        ... }
        >>>
        >>> helper = VLLMConfigHelper(config)
        >>> vllm_config = helper.build()
        >>>
        >>> import vllm
        >>> llm = vllm.LLM(model="meta-llama/Llama-3.1-8B", **vllm_config)
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """
        Initialize the config helper.

        Args:
            config: vLLM configuration dictionary. May include:
                    - Standard vLLM parameters
                    - override_neuron_config (direct)
                    - additional_config with override_neuron_config (nested)
        """
        # Immutable: create deep copy to prevent mutation
        self.config = self._deep_copy(config)
        self.is_neuron = is_neuron_available()

    def build(self) -> Dict[str, Any]:
        """
        Build hardware-appropriate vLLM configuration.

        On Neuron hardware: Returns config as-is
        On GPU hardware: Excludes override_neuron_config and additional_config
                        containing Neuron settings

        Returns:
            Configuration dictionary ready for vLLM.LLM(**config)

        Side effects:
            - Logs configuration to logger
            - Prints configuration to stdout
        """
        result = self._deep_copy(self.config)

        # GPU environment: exclude Neuron-specific configuration
        if not self.is_neuron:
            result = self._exclude_neuron_config(result)

        # Log and print configuration
        self._log_config(result)
        self._print_config(result)

        return result

    def _exclude_neuron_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Remove Neuron-specific configuration from config.

        Args:
            config: Configuration dictionary

        Returns:
            Configuration with Neuron settings excluded
        """
        result = {**config}

        # Remove direct override_neuron_config
        if "override_neuron_config" in result:
            result.pop("override_neuron_config")

        # Remove additional_config if it contains override_neuron_config
        if "additional_config" in result:
            if isinstance(result["additional_config"], dict):
                if "override_neuron_config" in result["additional_config"]:
                    result.pop("additional_config")

        return result

    def _deep_copy(self, obj: Any) -> Any:
        """
        Create a deep copy of the configuration.

        Args:
            obj: Object to copy

        Returns:
            Deep copy of the object
        """
        if isinstance(obj, dict):
            return {key: self._deep_copy(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._deep_copy(item) for item in obj]
        else:
            return obj

    def _log_config(self, config: Dict[str, Any]) -> None:
        """
        Log the configuration using Python logger.

        Args:
            config: Configuration to log
        """
        hardware = "Neuron" if self.is_neuron else "GPU"
        logger.info(f"vLLM Configuration (Hardware: {hardware})")
        logger.info(json.dumps(config, indent=2))

    def _print_config(self, config: Dict[str, Any]) -> None:
        """
        Print the configuration to stdout for visibility.

        Args:
            config: Configuration to print
        """
        hardware = "Neuron" if self.is_neuron else "GPU"

        print(f"\n{'=' * 80}")
        print("vLLM Configuration")
        print(f"{'=' * 80}")
        print(f"Hardware: {hardware}")
        print(f"{'-' * 80}")
        print(json.dumps(config, indent=2))
        print(f"{'=' * 80}\n")
