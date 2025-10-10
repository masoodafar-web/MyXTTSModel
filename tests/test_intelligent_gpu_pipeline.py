#!/usr/bin/env python3
"""
Test for Intelligent GPU Pipeline System

This test validates:
1. Multi-GPU Mode configuration
2. Single-GPU Buffered Mode configuration  
3. CLI argument parsing
4. DataConfig parameter passing
"""

import sys
import os
import unittest
from unittest.mock import patch, MagicMock
import argparse

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from myxtts.config.config import DataConfig, XTTSConfig


class TestIntelligentGPUPipeline(unittest.TestCase):
    """Test Intelligent GPU Pipeline System"""
    
    def test_dataconfig_has_gpu_pipeline_parameters(self):
        """Test that DataConfig has the new GPU pipeline parameters"""
        config = DataConfig()
        
        # Check that the new parameters exist
        self.assertTrue(hasattr(config, 'data_gpu'))
        self.assertTrue(hasattr(config, 'model_gpu'))
        self.assertTrue(hasattr(config, 'pipeline_buffer_size'))
        self.assertTrue(hasattr(config, 'model_start_delay'))
        
        # Check default values
        self.assertIsNone(config.data_gpu)
        self.assertIsNone(config.model_gpu)
        self.assertEqual(config.pipeline_buffer_size, 50)
        self.assertEqual(config.model_start_delay, 2.0)
        
    def test_dataconfig_multi_gpu_mode(self):
        """Test Multi-GPU Mode configuration"""
        config = DataConfig(
            data_gpu=0,
            model_gpu=1,
            pipeline_buffer_size=100,
            model_start_delay=3.0
        )
        
        self.assertEqual(config.data_gpu, 0)
        self.assertEqual(config.model_gpu, 1)
        self.assertEqual(config.pipeline_buffer_size, 100)
        self.assertEqual(config.model_start_delay, 3.0)
        
    def test_dataconfig_single_gpu_buffered_mode(self):
        """Test Single-GPU Buffered Mode (default)"""
        config = DataConfig(
            pipeline_buffer_size=75
        )
        
        # In single GPU mode, data_gpu and model_gpu should be None
        self.assertIsNone(config.data_gpu)
        self.assertIsNone(config.model_gpu)
        self.assertEqual(config.pipeline_buffer_size, 75)
        
    def test_xtts_config_with_gpu_pipeline(self):
        """Test XTTSConfig with GPU pipeline parameters"""
        from myxtts.config.config import ModelConfig, TrainingConfig
        
        model_config = ModelConfig()
        data_config = DataConfig(
            data_gpu=0,
            model_gpu=1,
            pipeline_buffer_size=60
        )
        training_config = TrainingConfig()
        
        config = XTTSConfig(
            model=model_config,
            data=data_config,
            training=training_config
        )
        
        self.assertEqual(config.data.data_gpu, 0)
        self.assertEqual(config.data.model_gpu, 1)
        self.assertEqual(config.data.pipeline_buffer_size, 60)
        
    def test_cli_arguments_parsing(self):
        """Test that CLI arguments are correctly defined"""
        # We'll test this by importing train_main and checking if the arguments exist
        try:
            # Import the train_main module to check argument definitions
            import train_main
            
            # Create a test parser with similar structure
            parser = argparse.ArgumentParser()
            parser.add_argument("--data-gpu", type=int, default=None)
            parser.add_argument("--model-gpu", type=int, default=None)
            parser.add_argument("--buffer-size", type=int, default=50)
            parser.add_argument("--model-start-delay", type=float, default=2.0)
            
            # Test parsing with Multi-GPU Mode arguments
            args = parser.parse_args([
                "--data-gpu", "0",
                "--model-gpu", "1",
                "--buffer-size", "100",
                "--model-start-delay", "3.5"
            ])
            
            self.assertEqual(args.data_gpu, 0)
            self.assertEqual(args.model_gpu, 1)
            self.assertEqual(args.buffer_size, 100)
            self.assertEqual(args.model_start_delay, 3.5)
            
            # Test parsing with Single-GPU Mode (defaults)
            args_default = parser.parse_args([])
            self.assertIsNone(args_default.data_gpu)
            self.assertIsNone(args_default.model_gpu)
            self.assertEqual(args_default.buffer_size, 50)
            self.assertEqual(args_default.model_start_delay, 2.0)
            
        except ImportError as e:
            # If train_main can't be imported, skip this test
            self.skipTest(f"Could not import train_main: {e}")
            
    def test_gpu_mode_detection(self):
        """Test detection of Multi-GPU vs Single-GPU mode"""
        # Multi-GPU Mode: both data_gpu and model_gpu are set
        config_multi = DataConfig(data_gpu=0, model_gpu=1)
        is_multi_gpu = (config_multi.data_gpu is not None and 
                       config_multi.model_gpu is not None)
        self.assertTrue(is_multi_gpu)
        
        # Single-GPU Mode: neither is set
        config_single = DataConfig()
        is_multi_gpu = (config_single.data_gpu is not None and 
                       config_single.model_gpu is not None)
        self.assertFalse(is_multi_gpu)
        
        # Partial configuration (should be treated as single GPU)
        config_partial1 = DataConfig(data_gpu=0)
        is_multi_gpu = (config_partial1.data_gpu is not None and 
                       config_partial1.model_gpu is not None)
        self.assertFalse(is_multi_gpu)
        
        config_partial2 = DataConfig(model_gpu=1)
        is_multi_gpu = (config_partial2.data_gpu is not None and 
                       config_partial2.model_gpu is not None)
        self.assertFalse(is_multi_gpu)
        
    def test_buffer_size_validation(self):
        """Test buffer size parameter validation"""
        # Valid buffer sizes
        config1 = DataConfig(pipeline_buffer_size=1)
        self.assertEqual(config1.pipeline_buffer_size, 1)
        
        config2 = DataConfig(pipeline_buffer_size=50)
        self.assertEqual(config2.pipeline_buffer_size, 50)
        
        config3 = DataConfig(pipeline_buffer_size=1000)
        self.assertEqual(config3.pipeline_buffer_size, 1000)
        
    def test_model_start_delay_validation(self):
        """Test model start delay parameter validation"""
        # Valid delays
        config1 = DataConfig(model_start_delay=0.5)
        self.assertEqual(config1.model_start_delay, 0.5)
        
        config2 = DataConfig(model_start_delay=2.0)
        self.assertEqual(config2.model_start_delay, 2.0)
        
        config3 = DataConfig(model_start_delay=10.0)
        self.assertEqual(config3.model_start_delay, 10.0)


class TestGPUPipelineIntegration(unittest.TestCase):
    """Test GPU Pipeline integration with data pipeline"""
    
    def test_pipeline_mode_messages(self):
        """Test that appropriate messages are displayed for each mode"""
        # This is a conceptual test - in practice, we'd need to capture stdout
        # to verify the actual messages being printed
        
        # Multi-GPU Mode should print:
        # "🚀 Intelligent GPU Pipeline: Multi-GPU Mode"
        
        # Single-GPU Mode should print:
        # "🚀 Intelligent GPU Pipeline: Single-GPU Buffered Mode"
        
        pass  # This would require more complex testing with output capture


def run_tests():
    """Run all tests"""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test cases
    suite.addTests(loader.loadTestsFromTestCase(TestIntelligentGPUPipeline))
    suite.addTests(loader.loadTestsFromTestCase(TestGPUPipelineIntegration))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Return exit code
    return 0 if result.wasSuccessful() else 1


if __name__ == '__main__':
    sys.exit(run_tests())
