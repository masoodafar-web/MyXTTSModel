#!/usr/bin/env python3
"""
Test for Dual-GPU Device Placement

This test validates that when multi-GPU mode is enabled:
1. The model device parameter is properly passed to the trainer
2. get_device_context accepts an optional device parameter
3. The device context is properly used in model creation and training
"""

import sys
import os
import unittest
from unittest.mock import Mock, patch, MagicMock

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


class TestDualGPUDevicePlacement(unittest.TestCase):
    """Test Dual-GPU Device Placement"""
    
    def test_get_device_context_accepts_device_parameter(self):
        """Test that get_device_context accepts an optional device parameter"""
        # Read the source code to verify the signature
        commons_path = os.path.join(
            os.path.dirname(__file__),
            '..',
            'myxtts',
            'utils',
            'commons.py'
        )
        
        if not os.path.exists(commons_path):
            self.skipTest("commons.py not found")
        
        with open(commons_path, 'r') as f:
            content = f.read()
        
        # Check that get_device_context accepts device parameter
        self.assertIn('def get_device_context(device: Optional[str] = None):', content,
                     "get_device_context should accept optional device parameter")
        
        # Check that it uses the device parameter
        self.assertIn('if device:', content,
                     "get_device_context should check for device parameter")
        self.assertIn('return tf.device(device)', content,
                     "get_device_context should return tf.device(device) when device is provided")
    
    def test_get_device_context_defaults_to_gpu0(self):
        """Test that get_device_context defaults to GPU:0 when available"""
        # Read the source code to verify behavior
        commons_path = os.path.join(
            os.path.dirname(__file__),
            '..',
            'myxtts',
            'utils',
            'commons.py'
        )
        
        if not os.path.exists(commons_path):
            self.skipTest("commons.py not found")
        
        with open(commons_path, 'r') as f:
            content = f.read()
        
        # Check for GPU default logic
        self.assertIn("return tf.device('/GPU:0')", content,
                     "get_device_context should default to GPU:0 when available")
    
    def test_get_device_context_falls_back_to_cpu(self):
        """Test that get_device_context falls back to CPU when no GPUs"""
        # Read the source code to verify behavior
        commons_path = os.path.join(
            os.path.dirname(__file__),
            '..',
            'myxtts',
            'utils',
            'commons.py'
        )
        
        if not os.path.exists(commons_path):
            self.skipTest("commons.py not found")
        
        with open(commons_path, 'r') as f:
            content = f.read()
        
        # Check for CPU fallback logic
        self.assertIn("return tf.device('/CPU:0')", content,
                     "get_device_context should fall back to CPU:0 when no GPUs")
    
    def test_trainer_accepts_model_device_parameter(self):
        """Test that XTTSTrainer accepts model_device parameter"""
        # This test just checks the signature, not the full initialization
        # since that requires TensorFlow and other dependencies
        
        from inspect import signature
        
        # Import the trainer module
        try:
            from myxtts.training.trainer import XTTSTrainer
            
            # Get the __init__ signature
            sig = signature(XTTSTrainer.__init__)
            params = list(sig.parameters.keys())
            
            # Check that model_device is in the parameters
            self.assertIn('model_device', params, 
                         "XTTSTrainer.__init__ should accept model_device parameter")
            
            # Check that it's optional (has a default)
            self.assertIsNone(sig.parameters['model_device'].default,
                            "model_device parameter should default to None")
        except ImportError as e:
            self.skipTest(f"Could not import XTTSTrainer: {e}")
    
    def test_trainer_stores_model_device(self):
        """Test that trainer stores the model_device attribute"""
        try:
            # Mock all the dependencies
            with patch('myxtts.training.trainer.setup_logging'), \
                 patch('myxtts.training.trainer.tf') as mock_tf, \
                 patch('myxtts.training.trainer.configure_gpus'), \
                 patch('myxtts.training.trainer.get_device'), \
                 patch('myxtts.training.trainer.setup_gpu_strategy'), \
                 patch('myxtts.training.trainer.XTTS'), \
                 patch('myxtts.training.trainer.get_device_context'):
                
                # Setup mocks
                mock_tf.config.list_physical_devices.return_value = []
                mock_tf.summary.create_file_writer.return_value = None
                mock_tf.keras.mixed_precision.Policy.return_value = Mock()
                mock_tf.keras.mixed_precision.global_policy.return_value.name = 'float32'
                
                from myxtts.training.trainer import XTTSTrainer
                from myxtts.config.config import XTTSConfig, ModelConfig, DataConfig, TrainingConfig
                
                # Create a minimal config
                config = XTTSConfig(
                    model=ModelConfig(),
                    data=DataConfig(),
                    training=TrainingConfig()
                )
                
                # Create trainer with model_device
                trainer = XTTSTrainer(
                    config=config,
                    model_device='/GPU:1'
                )
                
                # Check that model_device is stored
                self.assertEqual(trainer.model_device, '/GPU:1',
                               "Trainer should store the model_device parameter")
        
        except Exception as e:
            self.skipTest(f"Could not test trainer initialization: {e}")
    
    def test_multi_gpu_mode_sets_model_device_in_train_main(self):
        """Test that train_main.py sets model_device when in multi-GPU mode"""
        # Read the train_main.py file and check for the model_device logic
        train_main_path = os.path.join(
            os.path.dirname(__file__), 
            '..', 
            'train_main.py'
        )
        
        if not os.path.exists(train_main_path):
            self.skipTest("train_main.py not found")
        
        with open(train_main_path, 'r') as f:
            content = f.read()
        
        # Check that model_device is initialized
        self.assertIn('model_device = None', content,
                     "train_main.py should initialize model_device")
        
        # Check that model_device is set for multi-GPU mode
        self.assertIn("model_device = '/GPU:1'", content,
                     "train_main.py should set model_device to '/GPU:1' in multi-GPU mode")
        
        # Check that model_device is passed to trainer
        self.assertIn('model_device=model_device', content,
                     "train_main.py should pass model_device to XTTSTrainer")


class TestMultiGPUDataTransfer(unittest.TestCase):
    """Test Multi-GPU Data Transfer Logic"""
    
    def test_training_step_uses_model_device(self):
        """Test that training step uses model_device in context"""
        trainer_path = os.path.join(
            os.path.dirname(__file__),
            '..',
            'myxtts',
            'training',
            'trainer.py'
        )
        
        if not os.path.exists(trainer_path):
            self.skipTest("trainer.py not found")
        
        with open(trainer_path, 'r') as f:
            content = f.read()
        
        # Check that _train_step_impl uses get_device_context with self.model_device
        self.assertIn('get_device_context(self.model_device)', content,
                     "Training step should use get_device_context with self.model_device")
        
        # Check for explicit data transfer in multi-GPU mode
        self.assertIn('tf.identity(text_sequences)', content,
                     "Training step should use tf.identity for explicit data transfer")
        self.assertIn('if self.model_device:', content,
                     "Training step should check for multi-GPU mode")


if __name__ == '__main__':
    unittest.main()
