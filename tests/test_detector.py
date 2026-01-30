"""Tests for hardware detection."""

import os
from unittest.mock import MagicMock, patch

from benchmark_capture.profilers.detector import (
    detect_hardware,
    is_cuda_available,
    is_neuron_available,
)


class TestNeuronDetection:
    """Tests for Neuron availability detection."""

    def test_neuron_detected_via_import(self, clean_env: None) -> None:
        """Test Neuron detection via torch_neuronx import."""
        with patch.dict("sys.modules", {"torch_neuronx": MagicMock()}):
            assert is_neuron_available() is True

    def test_neuron_detected_via_directory(self, clean_env: None) -> None:
        """Test Neuron detection via /opt/aws/neuron/ directory."""
        with patch("os.path.exists") as mock_exists:
            mock_exists.return_value = True
            assert is_neuron_available() is True

    def test_neuron_detected_via_env_var(self, clean_env: None) -> None:
        """Test Neuron detection via NEURON_RT_VISIBLE_CORES."""
        os.environ["NEURON_RT_VISIBLE_CORES"] = "0-15"
        assert is_neuron_available() is True

    def test_neuron_not_available(self, clean_env: None) -> None:
        """Test Neuron not detected when unavailable."""
        with patch.dict("sys.modules", {"torch_neuronx": None}):
            with patch("os.path.exists") as mock_exists:
                mock_exists.return_value = False
                assert is_neuron_available() is False


class TestCUDADetection:
    """Tests for CUDA availability detection."""

    def test_cuda_detected_via_torch(self, clean_env: None) -> None:
        """Test CUDA detection via torch.cuda.is_available()."""
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = True
        with patch.dict("sys.modules", {"torch": mock_torch}):
            assert is_cuda_available() is True

    def test_cuda_detected_via_nvidia_smi(self, clean_env: None) -> None:
        """Test CUDA detection via nvidia-smi command."""
        with patch("shutil.which") as mock_which:
            mock_which.return_value = "/usr/bin/nvidia-smi"
            assert is_cuda_available() is True

    def test_cuda_detected_via_nsys(self, clean_env: None) -> None:
        """Test CUDA detection via nsys command."""
        with patch("shutil.which") as mock_which:

            def which_side_effect(cmd: str) -> str:
                if cmd == "nvidia-smi":
                    return ""
                elif cmd == "nsys":
                    return "/usr/bin/nsys"
                return ""

            mock_which.side_effect = which_side_effect
            assert is_cuda_available() is True

    def test_cuda_not_available(self, clean_env: None) -> None:
        """Test CUDA not detected when unavailable."""
        with patch.dict("sys.modules", {"torch": None}):
            with patch("shutil.which") as mock_which:
                mock_which.return_value = None
                assert is_cuda_available() is False


class TestHardwareDetection:
    """Tests for overall hardware detection."""

    def test_detect_neuron(self, clean_env: None) -> None:
        """Test auto-detection prefers Neuron over CUDA."""
        with patch("benchmark_capture.profilers.detector.is_neuron_available") as mock_neuron:
            with patch("benchmark_capture.profilers.detector.is_cuda_available") as mock_cuda:
                mock_neuron.return_value = True
                mock_cuda.return_value = True
                assert detect_hardware() == "neuron"

    def test_detect_cuda(self, clean_env: None) -> None:
        """Test auto-detection falls back to CUDA if Neuron unavailable."""
        with patch("benchmark_capture.profilers.detector.is_neuron_available") as mock_neuron:
            with patch("benchmark_capture.profilers.detector.is_cuda_available") as mock_cuda:
                mock_neuron.return_value = False
                mock_cuda.return_value = True
                assert detect_hardware() == "nsight"

    def test_detect_noop(self, clean_env: None) -> None:
        """Test auto-detection falls back to noop if nothing available."""
        with patch("benchmark_capture.profilers.detector.is_neuron_available") as mock_neuron:
            with patch("benchmark_capture.profilers.detector.is_cuda_available") as mock_cuda:
                mock_neuron.return_value = False
                mock_cuda.return_value = False
                assert detect_hardware() == "noop"
