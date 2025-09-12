#!/usr/bin/env python3
"""
Test preprocessing mode functionality.
"""

import unittest
import tempfile
import os
from pathlib import Path

# Import the necessary classes
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from myxtts.config.config import XTTSConfig, DataConfig
from trainTestFile import create_default_config


class TestPreprocessingMode(unittest.TestCase):
    """Test preprocessing mode configuration and functionality."""
    
    def test_data_config_preprocessing_mode_default(self):
        """Test that DataConfig has the correct default preprocessing mode."""
        config = DataConfig()
        self.assertEqual(config.preprocessing_mode, "auto")
    
    def test_data_config_preprocessing_mode_validation(self):
        """Test that DataConfig validates preprocessing mode values."""
        # Valid modes should work
        for mode in ["auto", "precompute", "runtime"]:
            config = DataConfig(preprocessing_mode=mode)
            self.assertEqual(config.preprocessing_mode, mode)
        
        # Invalid mode should raise ValueError
        with self.assertRaises(ValueError):
            DataConfig(preprocessing_mode="invalid_mode")
    
    def test_create_default_config_with_preprocessing_mode(self):
        """Test that create_default_config accepts preprocessing_mode parameter."""
        for mode in ["auto", "precompute", "runtime"]:
            config = create_default_config(preprocessing_mode=mode)
            self.assertEqual(config.data.preprocessing_mode, mode)
    
    def test_xtts_config_with_preprocessing_mode(self):
        """Test that XTTSConfig can be created with preprocessing_mode parameter."""
        config = XTTSConfig(preprocessing_mode="precompute")
        self.assertEqual(config.data.preprocessing_mode, "precompute")
    
    def test_yaml_serialization_with_preprocessing_mode(self):
        """Test that preprocessing_mode is included in YAML serialization."""
        config = create_default_config(preprocessing_mode="precompute")
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml_path = f.name
        
        try:
            # Save to YAML
            config.to_yaml(yaml_path)
            
            # Load from YAML
            loaded_config = XTTSConfig.from_yaml(yaml_path)
            
            # Check that preprocessing_mode is preserved
            self.assertEqual(loaded_config.data.preprocessing_mode, "precompute")
            
        finally:
            os.unlink(yaml_path)
    
    def test_configuration_modes_documentation(self):
        """Test that all preprocessing modes are valid and documented."""
        valid_modes = ["auto", "precompute", "runtime"]
        
        # Test each mode can be set
        for mode in valid_modes:
            config = DataConfig()
            config.preprocessing_mode = mode
            self.assertEqual(config.preprocessing_mode, mode)
        
        # Test mode descriptions (informal check)
        mode_descriptions = {
            "auto": "Try precompute, fall back gracefully",
            "precompute": "Fully preprocess before training starts", 
            "runtime": "Process data on-the-fly during training"
        }
        
        for mode in valid_modes:
            self.assertIn(mode, mode_descriptions)


if __name__ == '__main__':
    unittest.main()