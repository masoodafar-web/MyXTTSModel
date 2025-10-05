"""
Test GPU strategy selection functionality (single GPU only).
"""

import unittest
from unittest.mock import patch, MagicMock
import sys
import os

# Add the parent directory to sys.path to import myxtts
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Mock dependencies before importing any modules that depend on them
sys.modules['tensorflow'] = MagicMock()
sys.modules['tensorflow.config'] = MagicMock()
sys.modules['tensorflow.distribute'] = MagicMock()
sys.modules['librosa'] = MagicMock()
sys.modules['numpy'] = MagicMock()
sys.modules['tqdm'] = MagicMock()
sys.modules['wandb'] = MagicMock()

# Import after mocking
from myxtts.utils.commons import setup_gpu_strategy
from myxtts.config.config import XTTSConfig, TrainingConfig


class TestGPUStrategy(unittest.TestCase):
    """Test GPU strategy selection (single GPU or CPU only)."""

    def setUp(self):
        """Set up test fixtures."""
        # Reset tensorflow mocks for each test
        self.tf_mock = sys.modules['tensorflow']
        self.tf_mock.reset_mock()
        
        # Mock physical devices
        self.tf_mock.config.list_physical_devices.return_value = []
        
        # Mock strategies
        self.default_strategy = MagicMock()
        
        self.tf_mock.distribute.get_strategy.return_value = self.default_strategy
        
        # Mock device context
        self.tf_mock.device.return_value.__enter__ = MagicMock()
        self.tf_mock.device.return_value.__exit__ = MagicMock()

    def test_no_gpu_returns_default_strategy(self):
        """Test that when no GPUs are available, default strategy is returned."""
        # Arrange
        self.tf_mock.config.list_physical_devices.return_value = []
        
        # Act
        strategy = setup_gpu_strategy()
        
        # Assert
        self.assertEqual(strategy, self.default_strategy)
        self.tf_mock.distribute.get_strategy.assert_called_once()

    def test_single_gpu_returns_default_strategy(self):
        """Test that with one GPU, default strategy is returned for single GPU training."""
        # Arrange
        mock_gpu = MagicMock()
        self.tf_mock.config.list_physical_devices.return_value = [mock_gpu]
        
        # Act
        strategy = setup_gpu_strategy()
        
        # Assert
        self.assertEqual(strategy, self.default_strategy)
        self.tf_mock.distribute.get_strategy.assert_called_once()

    def test_multiple_gpus_returns_default_strategy(self):
        """Test that even with multiple GPUs, default strategy is returned (single GPU training only)."""
        # Arrange
        mock_gpu1 = MagicMock()
        mock_gpu2 = MagicMock()
        self.tf_mock.config.list_physical_devices.return_value = [mock_gpu1, mock_gpu2]
        
        # Act
        strategy = setup_gpu_strategy()
        
        # Assert
        self.assertEqual(strategy, self.default_strategy)
        self.tf_mock.distribute.get_strategy.assert_called_once()


if __name__ == '__main__':
    unittest.main()