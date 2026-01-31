"""Tests for VLLMConfigHelper - hardware-aware vLLM configuration."""

from typing import Any, Dict
from unittest.mock import MagicMock, patch

import pytest

from benchmark_capture.utils.vllm_config import VLLMConfigHelper


class TestVLLMConfigHelperNeuron:
    """Tests for Neuron environment."""

    @patch("benchmark_capture.utils.vllm_config.is_neuron_available")
    def test_preserves_config_on_neuron(self, mock_neuron: MagicMock) -> None:
        """Neuron環境では設定がそのまま保持される"""
        mock_neuron.return_value = True

        config = {
            "tensor_parallel_size": 2,
            "max_num_seqs": 4,
            "override_neuron_config": {
                "pa_num_blocks": 512,
                "is_prefix_caching": True,
            }
        }

        helper = VLLMConfigHelper(config)
        result = helper.build()

        # 設定がそのまま返される
        assert result == config
        assert "override_neuron_config" in result
        assert result["override_neuron_config"]["pa_num_blocks"] == 512

    @patch("benchmark_capture.utils.vllm_config.is_neuron_available")
    def test_preserves_additional_config_on_neuron(self, mock_neuron: MagicMock) -> None:
        """additional_config形式もサポート"""
        mock_neuron.return_value = True

        config = {
            "tensor_parallel_size": 2,
            "additional_config": {
                "override_neuron_config": {
                    "pa_num_blocks": 512,
                }
            }
        }

        helper = VLLMConfigHelper(config)
        result = helper.build()

        assert result == config
        assert "additional_config" in result

    @patch("benchmark_capture.utils.vllm_config.is_neuron_available")
    def test_neuron_without_override(self, mock_neuron: MagicMock) -> None:
        """Neuron環境でもoverride_neuron_configなしでOK"""
        mock_neuron.return_value = True

        config = {
            "tensor_parallel_size": 2,
            "max_num_seqs": 4,
        }

        helper = VLLMConfigHelper(config)
        result = helper.build()

        assert result == config


class TestVLLMConfigHelperGPU:
    """Tests for GPU environment."""

    @patch("benchmark_capture.utils.vllm_config.is_neuron_available")
    def test_excludes_override_neuron_config_on_gpu(self, mock_neuron: MagicMock) -> None:
        """GPU環境ではoverride_neuron_configが除外される"""
        mock_neuron.return_value = False

        config = {
            "tensor_parallel_size": 2,
            "max_num_seqs": 4,
            "override_neuron_config": {
                "pa_num_blocks": 512,
            }
        }

        helper = VLLMConfigHelper(config)
        result = helper.build()

        # Neuron設定が除外される
        assert "override_neuron_config" not in result
        # 他の設定は保持される
        assert result["tensor_parallel_size"] == 2
        assert result["max_num_seqs"] == 4

    @patch("benchmark_capture.utils.vllm_config.is_neuron_available")
    def test_excludes_additional_config_neuron_on_gpu(self, mock_neuron: MagicMock) -> None:
        """GPU環境ではadditional_config内のNeuron設定も除外"""
        mock_neuron.return_value = False

        config = {
            "tensor_parallel_size": 2,
            "additional_config": {
                "override_neuron_config": {
                    "pa_num_blocks": 512,
                }
            }
        }

        helper = VLLMConfigHelper(config)
        result = helper.build()

        # additional_configごと除外
        assert "additional_config" not in result
        assert result["tensor_parallel_size"] == 2

    @patch("benchmark_capture.utils.vllm_config.is_neuron_available")
    def test_gpu_without_override(self, mock_neuron: MagicMock) -> None:
        """GPU環境でNeuron設定なしならそのまま"""
        mock_neuron.return_value = False

        config = {
            "tensor_parallel_size": 4,
            "max_num_seqs": 8,
        }

        helper = VLLMConfigHelper(config)
        result = helper.build()

        assert result == config


class TestVLLMConfigHelperImmutability:
    """Tests for immutability."""

    @patch("benchmark_capture.utils.vllm_config.is_neuron_available")
    def test_does_not_mutate_original_config(self, mock_neuron: MagicMock) -> None:
        """元の設定を変更しない"""
        mock_neuron.return_value = True

        original = {
            "tensor_parallel_size": 2,
            "override_neuron_config": {"pa_num_blocks": 512}
        }

        helper = VLLMConfigHelper(original)
        result = helper.build()

        # 返された設定を変更
        result["new_key"] = "new_value"
        result["override_neuron_config"]["new_param"] = True

        # 元の設定は変更されていない
        assert "new_key" not in original
        assert "new_param" not in original["override_neuron_config"]

    @patch("benchmark_capture.utils.vllm_config.is_neuron_available")
    def test_gpu_exclusion_does_not_mutate(self, mock_neuron: MagicMock) -> None:
        """GPU環境での除外も元の設定を変更しない"""
        mock_neuron.return_value = False

        original = {
            "tensor_parallel_size": 2,
            "override_neuron_config": {"pa_num_blocks": 512}
        }

        helper = VLLMConfigHelper(original)
        result = helper.build()

        # 元の設定にはまだoverride_neuron_configがある
        assert "override_neuron_config" in original
        # 返された設定にはない
        assert "override_neuron_config" not in result


class TestVLLMConfigHelperLogging:
    """Tests for logging functionality."""

    @patch("benchmark_capture.utils.vllm_config.is_neuron_available")
    def test_logs_config_to_stdout(
        self, mock_neuron: MagicMock, capsys: pytest.CaptureFixture
    ) -> None:
        """設定が標準出力に表示される"""
        mock_neuron.return_value = True

        config = {
            "tensor_parallel_size": 2,
            "override_neuron_config": {"pa_num_blocks": 512}
        }

        helper = VLLMConfigHelper(config)
        helper.build()

        captured = capsys.readouterr()
        output = captured.out

        # 基本情報が表示される
        assert "vLLM Configuration" in output
        assert "Neuron" in output
        assert "tensor_parallel_size" in output
        assert "512" in output

    @patch("benchmark_capture.utils.vllm_config.is_neuron_available")
    def test_logs_gpu_config(
        self, mock_neuron: MagicMock, capsys: pytest.CaptureFixture
    ) -> None:
        """GPU設定も正しくログ出力される"""
        mock_neuron.return_value = False

        config = {
            "tensor_parallel_size": 4,
            "override_neuron_config": {"pa_num_blocks": 512}
        }

        helper = VLLMConfigHelper(config)
        helper.build()

        captured = capsys.readouterr()
        output = captured.out

        assert "vLLM Configuration" in output
        assert "GPU" in output
        assert "tensor_parallel_size" in output
        # Neuron設定は表示されない
        assert "override_neuron_config" not in output


class TestVLLMConfigHelperHardwareDetection:
    """Tests for hardware detection."""

    @patch("benchmark_capture.utils.vllm_config.is_neuron_available")
    def test_detects_neuron_hardware(self, mock_neuron: MagicMock) -> None:
        """Neuronハードウェアを正しく検出"""
        mock_neuron.return_value = True

        helper = VLLMConfigHelper({})
        assert helper.is_neuron is True

    @patch("benchmark_capture.utils.vllm_config.is_neuron_available")
    def test_detects_gpu_hardware(self, mock_neuron: MagicMock) -> None:
        """GPUハードウェアを正しく検出"""
        mock_neuron.return_value = False

        helper = VLLMConfigHelper({})
        assert helper.is_neuron is False


class TestVLLMConfigHelperRealWorldCases:
    """Real-world usage patterns."""

    @patch("benchmark_capture.utils.vllm_config.is_neuron_available")
    def test_reranker_example_config(self, mock_neuron: MagicMock) -> None:
        """Rerankerサンプルの実際の設定をテスト"""
        mock_neuron.return_value = True

        config = {
            "tensor_parallel_size": 2,
            "max_num_seqs": 4,
            "max_num_batched_tokens": 256,
            "max_model_len": 2048,
            "block_size": 32,
            "num_gpu_blocks_override": 512,
            "enable_prefix_caching": False,
            "dtype": "bfloat16",
            "additional_config": {
                "override_neuron_config": {
                    "skip_warmup": True,
                    "pa_num_blocks": 512,
                    "pa_block_size": 32,
                    "enable_bucketing": True,
                }
            }
        }

        helper = VLLMConfigHelper(config)
        result = helper.build()

        # すべての設定が保持される
        assert result == config
        assert result["additional_config"]["override_neuron_config"]["pa_num_blocks"] == 512

    @patch("benchmark_capture.utils.vllm_config.is_neuron_available")
    def test_pytest_parametrize_pattern(self, mock_neuron: MagicMock) -> None:
        """pytestのparametrizeパターンをテスト"""
        mock_neuron.return_value = True

        # parametrizeで渡される値
        pa_num_blocks_values = [256, 512, 1024]

        for pa_num_blocks in pa_num_blocks_values:
            config = {
                "tensor_parallel_size": 2,
                "override_neuron_config": {
                    "pa_num_blocks": pa_num_blocks,
                    "is_prefix_caching": True,
                }
            }

            helper = VLLMConfigHelper(config)
            result = helper.build()

            assert result["override_neuron_config"]["pa_num_blocks"] == pa_num_blocks
