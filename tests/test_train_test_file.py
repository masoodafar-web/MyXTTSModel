"""Tests for trainTestFile.py script."""

import os
import sys
import tempfile
import unittest
from unittest.mock import patch

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the functions from trainTestFile
from trainTestFile import create_default_config, create_config_file


class TestTrainTestFile(unittest.TestCase):
    """Test trainTestFile.py functionality."""

    def test_create_default_config(self):
        """Test creating default configuration programmatically."""
        config = create_default_config(
            data_path="./test_data",
            language="es",
            batch_size=8,
            epochs=50,
            learning_rate=2e-4,
            sample_rate=16000,
            checkpoint_dir="./test_checkpoints"
        )
        
        # Check data configuration
        self.assertEqual(config.data.dataset_path, "./test_data")
        self.assertEqual(config.data.language, "es")
        self.assertEqual(config.data.batch_size, 8)
        self.assertEqual(config.data.sample_rate, 16000)
        
        # Check model configuration
        self.assertEqual(config.model.sample_rate, 16000)
        
        # Check training configuration
        self.assertEqual(config.training.epochs, 50)
        self.assertEqual(config.training.learning_rate, 2e-4)
        self.assertEqual(config.training.checkpoint_dir, "./test_checkpoints")
        
        # Check computed values (max with 1000 and 500 respectively)
        self.assertEqual(config.training.save_step, max(1000, 50 // 10))  # max(1000, epochs // 10)
        self.assertEqual(config.training.val_step, max(500, 50 // 20))     # max(500, epochs // 20)

    def test_create_config_file(self):
        """Test creating and saving configuration file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml_path = f.name
        
        try:
            # Create config file
            create_config_file(
                yaml_path,
                data_path="./test_data",
                batch_size=4,
                epochs=25
            )
            
            # Check file exists
            self.assertTrue(os.path.exists(yaml_path))
            
            # Read and verify content
            from myxtts.config.config import XTTSConfig
            loaded_config = XTTSConfig.from_yaml(yaml_path)
            
            self.assertEqual(loaded_config.data.dataset_path, "./test_data")
            self.assertEqual(loaded_config.data.batch_size, 4)
            self.assertEqual(loaded_config.training.epochs, 25)
            
        finally:
            if os.path.exists(yaml_path):
                os.unlink(yaml_path)

    def test_default_config_values(self):
        """Test that default configuration has expected values."""
        config = create_default_config()
        
        # Check defaults match expected values
        self.assertEqual(config.data.dataset_path, "./data/ljspeech")
        self.assertEqual(config.data.language, "en")
        self.assertEqual(config.data.batch_size, 16)
        self.assertEqual(config.training.epochs, 100)
        self.assertEqual(config.training.learning_rate, 1e-4)
        self.assertEqual(config.model.sample_rate, 22050)
        self.assertEqual(config.training.checkpoint_dir, "./checkpoints")


if __name__ == '__main__':
    unittest.main()