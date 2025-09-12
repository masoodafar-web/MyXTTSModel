"""
Test GPU strategy selection functionality.
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
    """Test GPU strategy selection based on configuration."""

    def setUp(self):
        """Set up test fixtures."""
        # Reset tensorflow mocks for each test
        self.tf_mock = sys.modules['tensorflow']
        self.tf_mock.reset_mock()
        
        # Mock physical devices
        self.tf_mock.config.list_physical_devices.return_value = []
        
        # Mock strategies
        self.default_strategy = MagicMock()
        self.one_device_strategy = MagicMock()
        self.mirrored_strategy = MagicMock()
        
        self.tf_mock.distribute.get_strategy.return_value = self.default_strategy
        self.tf_mock.distribute.OneDeviceStrategy.return_value = self.one_device_strategy
        self.tf_mock.distribute.MirroredStrategy.return_value = self.mirrored_strategy
        
        # Mock device context
        self.tf_mock.device.return_value.__enter__ = MagicMock()
        self.tf_mock.device.return_value.__exit__ = MagicMock()

    def test_no_gpu_returns_default_strategy(self):
        """Test that when no GPUs are available, default strategy is returned."""
        # Arrange
        self.tf_mock.config.list_physical_devices.return_value = []
        
        # Act
        strategy = setup_gpu_strategy(enable_multi_gpu=False)
        
        # Assert
        self.assertEqual(strategy, self.default_strategy)
        self.tf_mock.distribute.get_strategy.assert_called_once()

    def test_single_gpu_returns_one_device_strategy(self):
        """Test that with one GPU, OneDeviceStrategy is returned."""
        # Arrange
        mock_gpu = MagicMock()
        self.tf_mock.config.list_physical_devices.return_value = [mock_gpu]
        
        # Act
        strategy = setup_gpu_strategy(enable_multi_gpu=False)
        
        # Assert
        self.assertEqual(strategy, self.one_device_strategy)
        self.tf_mock.distribute.OneDeviceStrategy.assert_called_with("/gpu:0")

    def test_multi_gpu_with_enable_false_returns_one_device_strategy(self):
        """Test that with multiple GPUs but multi_gpu=False, OneDeviceStrategy is returned."""
        # Arrange
        mock_gpu1 = MagicMock()
        mock_gpu2 = MagicMock()
        self.tf_mock.config.list_physical_devices.return_value = [mock_gpu1, mock_gpu2]
        
        # Act
        strategy = setup_gpu_strategy(enable_multi_gpu=False)
        
        # Assert
        self.assertEqual(strategy, self.one_device_strategy)
        self.tf_mock.distribute.OneDeviceStrategy.assert_called_with("/gpu:0")

    def test_multi_gpu_with_enable_true_returns_mirrored_strategy(self):
        """Test that with multiple GPUs and multi_gpu=True, MirroredStrategy is returned."""
        # Arrange
        mock_gpu1 = MagicMock()
        mock_gpu2 = MagicMock()
        self.tf_mock.config.list_physical_devices.return_value = [mock_gpu1, mock_gpu2]
        
        # Act
        strategy = setup_gpu_strategy(enable_multi_gpu=True)
        
        # Assert
        self.assertEqual(strategy, self.mirrored_strategy)
        self.tf_mock.distribute.MirroredStrategy.assert_called_once()

    def test_training_config_multi_gpu_default_false(self):
        """Test that TrainingConfig has multi_gpu=False by default."""
        config = TrainingConfig()
        self.assertFalse(config.multi_gpu)

    def test_training_config_multi_gpu_can_be_set_true(self):
        """Test that TrainingConfig multi_gpu can be set to True."""
        config = TrainingConfig(multi_gpu=True)
        self.assertTrue(config.multi_gpu)

    def test_xtts_config_preserves_multi_gpu_setting(self):
        """Test that XTTSConfig preserves multi_gpu setting."""
        training_config = TrainingConfig(multi_gpu=True)
        config = XTTSConfig(training=training_config)
        self.assertTrue(config.training.multi_gpu)


if __name__ == '__main__':
    unittest.main()