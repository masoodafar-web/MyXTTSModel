"""Tests for configuration module."""

import os
import tempfile
import unittest
from myxtts.config.config import XTTSConfig, ModelConfig, DataConfig, TrainingConfig


class TestConfig(unittest.TestCase):
    """Test configuration classes."""
    
    def test_model_config_defaults(self):
        """Test ModelConfig default values."""
        config = ModelConfig()
        
        self.assertEqual(config.text_encoder_dim, 512)
        self.assertEqual(config.n_mels, 80)
        self.assertEqual(config.sample_rate, 22050)
        self.assertTrue(config.use_voice_conditioning)
        self.assertIsInstance(config.languages, list)
        self.assertIn("en", config.languages)
        
        # Test NLLB tokenizer defaults
        self.assertEqual(config.text_vocab_size, 256_256)
        self.assertEqual(config.tokenizer_type, "nllb")
        self.assertEqual(config.tokenizer_model, "facebook/nllb-200-distilled-600M")
    
    def test_data_config_defaults(self):
        """Test DataConfig default values."""
        config = DataConfig()
        
        self.assertEqual(config.sample_rate, 22050)
        self.assertEqual(config.language, "en")
        self.assertEqual(config.batch_size, 32)
        self.assertTrue(config.normalize_audio)
        self.assertIsInstance(config.text_cleaners, list)
    
    def test_training_config_defaults(self):
        """Test TrainingConfig default values."""
        config = TrainingConfig()
        
        self.assertEqual(config.epochs, 1000)
        self.assertEqual(config.learning_rate, 1e-4)
        self.assertEqual(config.optimizer, "adamw")
        self.assertEqual(config.scheduler, "noam")
    
    def test_xtts_config(self):
        """Test XTTSConfig integration."""
        config = XTTSConfig()
        
        self.assertIsInstance(config.model, ModelConfig)
        self.assertIsInstance(config.data, DataConfig)
        self.assertIsInstance(config.training, TrainingConfig)
    
    def test_yaml_serialization(self):
        """Test YAML save/load functionality."""
        config = XTTSConfig()
        
        # Modify some values
        config.model.sample_rate = 16000
        config.data.batch_size = 16
        config.training.epochs = 500
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml_path = f.name
        
        try:
            config.to_yaml(yaml_path)
            
            # Load from file
            loaded_config = XTTSConfig.from_yaml(yaml_path)
            
            # Check values
            self.assertEqual(loaded_config.model.sample_rate, 16000)
            self.assertEqual(loaded_config.data.batch_size, 16)
            self.assertEqual(loaded_config.training.epochs, 500)
        
        finally:
            if os.path.exists(yaml_path):
                os.unlink(yaml_path)
    
    def test_to_dict(self):
        """Test dictionary conversion."""
        config = XTTSConfig()
        config_dict = config.to_dict()
        
        self.assertIn('model', config_dict)
        self.assertIn('data', config_dict)
        self.assertIn('training', config_dict)
        
        # Check nested structure
        self.assertIn('text_encoder_dim', config_dict['model'])
        self.assertIn('sample_rate', config_dict['data'])
        self.assertIn('learning_rate', config_dict['training'])


if __name__ == '__main__':
    unittest.main()