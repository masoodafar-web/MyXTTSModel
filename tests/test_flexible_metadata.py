"""Tests for flexible metadata and path handling functionality."""

import os
import sys
import tempfile
import unittest
from unittest.mock import patch, MagicMock
from pathlib import Path

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from myxtts.config.config import XTTSConfig, DataConfig
from trainTestFile import create_default_config


class TestFlexibleMetadata(unittest.TestCase):
    """Test flexible metadata and path handling functionality."""
    
    def test_config_with_custom_metadata_files(self):
        """Test creating config with custom metadata files."""
        config = create_default_config(
            metadata_train_file="./custom_train/metadata_train.csv",
            metadata_eval_file="./custom_eval/metadata_eval.csv",
            wavs_train_dir="./custom_train/wavs",
            wavs_eval_dir="./custom_eval/wavs"
        )
        
        # Check custom metadata files are set
        self.assertEqual(config.data.metadata_train_file, "./custom_train/metadata_train.csv")
        self.assertEqual(config.data.metadata_eval_file, "./custom_eval/metadata_eval.csv")
        
        # Check custom wav directories are set
        self.assertEqual(config.data.wavs_train_dir, "./custom_train/wavs")
        self.assertEqual(config.data.wavs_eval_dir, "./custom_eval/wavs")

    def test_config_with_only_metadata_files(self):
        """Test creating config with custom metadata files but no wav directories."""
        config = create_default_config(
            metadata_train_file="./custom_train/metadata_train.csv",
            metadata_eval_file="./custom_eval/metadata_eval.csv"
        )
        
        # Check custom metadata files are set
        self.assertEqual(config.data.metadata_train_file, "./custom_train/metadata_train.csv")
        self.assertEqual(config.data.metadata_eval_file, "./custom_eval/metadata_eval.csv")
        
        # Check wav directories are None (will use defaults based on metadata file locations)
        self.assertIsNone(config.data.wavs_train_dir)
        self.assertIsNone(config.data.wavs_eval_dir)

    def test_config_without_custom_files(self):
        """Test creating config without custom metadata files (default behavior)."""
        config = create_default_config()
        
        # Check that custom metadata files are None
        self.assertIsNone(config.data.metadata_train_file)
        self.assertIsNone(config.data.metadata_eval_file)
        
        # Check that custom wav directories are None
        self.assertIsNone(config.data.wavs_train_dir)
        self.assertIsNone(config.data.wavs_eval_dir)

    def test_yaml_config_with_custom_fields(self):
        """Test creating and saving YAML config with custom metadata fields."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml_path = f.name

        try:
            # Create config with custom metadata
            config = create_default_config(
                metadata_train_file="./train/metadata.csv",
                metadata_eval_file="./eval/metadata.csv",
                wavs_train_dir="./train/audio",
                wavs_eval_dir="./eval/audio"
            )
            
            # Save to YAML
            config.to_yaml(yaml_path)
            
            # Load back from YAML
            loaded_config = XTTSConfig.from_yaml(yaml_path)
            
            # Verify custom fields are preserved
            self.assertEqual(loaded_config.data.metadata_train_file, "./train/metadata.csv")
            self.assertEqual(loaded_config.data.metadata_eval_file, "./eval/metadata.csv")
            self.assertEqual(loaded_config.data.wavs_train_dir, "./train/audio")
            self.assertEqual(loaded_config.data.wavs_eval_dir, "./eval/audio")
            
        finally:
            if os.path.exists(yaml_path):
                os.unlink(yaml_path)

    def test_data_config_defaults(self):
        """Test DataConfig default values for new fields."""
        data_config = DataConfig()
        
        # Check that new fields default to None
        self.assertIsNone(data_config.metadata_train_file)
        self.assertIsNone(data_config.metadata_eval_file)
        self.assertIsNone(data_config.wavs_train_dir)
        self.assertIsNone(data_config.wavs_eval_dir)

    def test_scenario_detection_logic(self):
        """Test the logic for detecting custom vs default metadata scenarios."""
        # Scenario A: Custom metadata files provided
        config_custom = DataConfig(
            metadata_train_file="./train/metadata.csv",
            metadata_eval_file="./eval/metadata.csv"
        )
        
        # Check that both custom files are set
        self.assertIsNotNone(config_custom.metadata_train_file)
        self.assertIsNotNone(config_custom.metadata_eval_file)
        
        # Scenario B: Default behavior (single metadata file)
        config_default = DataConfig()
        
        # Check that custom files are None
        self.assertIsNone(config_default.metadata_train_file)
        self.assertIsNone(config_default.metadata_eval_file)


if __name__ == '__main__':
    unittest.main()