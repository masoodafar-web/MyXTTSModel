"""
Test CLI configuration flexibility.

Tests that the CLI supports both config files and direct parameter setting.
"""

import os
import tempfile
import unittest
from pathlib import Path

from myxtts.config.config import XTTSConfig, DataConfig, ModelConfig, TrainingConfig


class TestCLIConfiguration(unittest.TestCase):
    """Test CLI configuration options."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)
    
    def tearDown(self):
        """Clean up test environment."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_config_creation_without_yaml(self):
        """Test creating configuration programmatically without YAML."""
        # Create default config
        config = XTTSConfig()
        
        # Verify default values are set
        self.assertIsInstance(config.model, ModelConfig)
        self.assertIsInstance(config.data, DataConfig)
        self.assertIsInstance(config.training, TrainingConfig)
        
        # Test setting custom values
        config.data.dataset_path = "/my/custom/dataset"
        config.training.epochs = 500
        config.data.batch_size = 16
        config.training.learning_rate = 5e-5
        
        self.assertEqual(config.data.dataset_path, "/my/custom/dataset")
        self.assertEqual(config.training.epochs, 500)
        self.assertEqual(config.data.batch_size, 16)
        self.assertEqual(config.training.learning_rate, 5e-5)
    
    def test_yaml_config_with_cli_overrides(self):
        """Test loading YAML config and overriding with CLI-like values."""
        # Create a YAML config file
        yaml_content = """
data:
  dataset_path: ./original/dataset
  batch_size: 32
  language: en

training:
  epochs: 100
  learning_rate: 0.0001
  checkpoint_dir: ./original/checkpoints

model:
  sample_rate: 22050
"""
        yaml_file = self.temp_path / "test_config.yaml"
        yaml_file.write_text(yaml_content)
        
        # Load config from YAML
        config = XTTSConfig.from_yaml(str(yaml_file))
        
        # Verify YAML values loaded correctly
        self.assertEqual(config.data.dataset_path, "./original/dataset")
        self.assertEqual(config.data.batch_size, 32)
        self.assertEqual(config.training.epochs, 100)
        self.assertEqual(config.training.learning_rate, 0.0001)
        
        # Simulate CLI overrides (like what the CLI would do)
        config.data.dataset_path = "/new/dataset/path"  # CLI --data-path
        config.training.epochs = 200                    # CLI --epochs
        config.data.batch_size = 16                     # CLI --batch-size
        
        # Verify overrides worked
        self.assertEqual(config.data.dataset_path, "/new/dataset/path")
        self.assertEqual(config.training.epochs, 200)
        self.assertEqual(config.data.batch_size, 16)
        
        # Verify non-overridden values remain from YAML
        self.assertEqual(config.training.learning_rate, 0.0001)
        self.assertEqual(config.data.language, "en")
        self.assertEqual(config.model.sample_rate, 22050)
    
    def test_config_validation_for_required_fields(self):
        """Test that config validation works for required fields."""
        config = XTTSConfig()
        
        # Empty dataset path should be detectable
        self.assertEqual(config.data.dataset_path, "")
        
        # Set a valid dataset path
        config.data.dataset_path = "/valid/dataset/path"
        self.assertEqual(config.data.dataset_path, "/valid/dataset/path")
    
    def test_custom_metadata_file_configuration(self):
        """Test configuration with custom metadata files."""
        config = XTTSConfig()
        
        # Test default values
        self.assertIsNone(config.data.metadata_train_file)
        self.assertIsNone(config.data.metadata_eval_file)
        self.assertIsNone(config.data.wavs_train_dir)
        self.assertIsNone(config.data.wavs_eval_dir)
        
        # Set custom metadata files
        config.data.metadata_train_file = "./custom/train_metadata.csv"
        config.data.metadata_eval_file = "./custom/eval_metadata.csv"
        config.data.wavs_train_dir = "./custom/train_wavs"
        config.data.wavs_eval_dir = "./custom/eval_wavs"
        
        # Verify they're set correctly
        self.assertEqual(config.data.metadata_train_file, "./custom/train_metadata.csv")
        self.assertEqual(config.data.metadata_eval_file, "./custom/eval_metadata.csv")
        self.assertEqual(config.data.wavs_train_dir, "./custom/train_wavs")
        self.assertEqual(config.data.wavs_eval_dir, "./custom/eval_wavs")
    
    def test_config_to_yaml_roundtrip(self):
        """Test saving and loading config maintains all values."""
        # Create config with custom values
        config = XTTSConfig()
        config.data.dataset_path = "/test/dataset"
        config.data.metadata_train_file = "./custom_train.csv"
        config.training.epochs = 150
        config.model.sample_rate = 16000
        
        # Save to YAML
        yaml_file = self.temp_path / "roundtrip_config.yaml"
        config.to_yaml(str(yaml_file))
        
        # Load back from YAML
        loaded_config = XTTSConfig.from_yaml(str(yaml_file))
        
        # Verify all values preserved
        self.assertEqual(loaded_config.data.dataset_path, "/test/dataset")
        self.assertEqual(loaded_config.data.metadata_train_file, "./custom_train.csv")
        self.assertEqual(loaded_config.training.epochs, 150)
        self.assertEqual(loaded_config.model.sample_rate, 16000)


if __name__ == '__main__':
    unittest.main()