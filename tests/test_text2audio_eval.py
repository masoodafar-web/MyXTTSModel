"""Tests for text-to-audio evaluation callback feature."""

import os
import tempfile
import unittest
from unittest.mock import Mock, patch, MagicMock
import tensorflow as tf
import numpy as np

from myxtts.config.config import XTTSConfig, TrainingConfig
from myxtts.training.trainer import XTTSTrainer


class TestText2AudioEvalConfig(unittest.TestCase):
    """Test text-to-audio evaluation configuration."""
    
    def test_config_defaults(self):
        """Test that text-to-audio eval config has correct defaults."""
        config = TrainingConfig()
        
        # Check default values
        self.assertTrue(config.enable_text2audio_eval)
        self.assertEqual(config.text2audio_interval_steps, 200)
        self.assertEqual(config.text2audio_output_dir, "./eval_samples")
        self.assertIsNotNone(config.text2audio_texts)
        self.assertIsInstance(config.text2audio_texts, list)
        self.assertGreater(len(config.text2audio_texts), 0)
        self.assertIsNone(config.text2audio_speaker_id)
        self.assertTrue(config.text2audio_log_tensorboard)
    
    def test_config_custom_values(self):
        """Test setting custom text-to-audio eval config values."""
        custom_texts = ["Test text 1", "Test text 2"]
        config = TrainingConfig(
            enable_text2audio_eval=False,
            text2audio_interval_steps=500,
            text2audio_output_dir="/custom/path",
            text2audio_texts=custom_texts,
            text2audio_speaker_id=42,
            text2audio_log_tensorboard=False
        )
        
        self.assertFalse(config.enable_text2audio_eval)
        self.assertEqual(config.text2audio_interval_steps, 500)
        self.assertEqual(config.text2audio_output_dir, "/custom/path")
        self.assertEqual(config.text2audio_texts, custom_texts)
        self.assertEqual(config.text2audio_speaker_id, 42)
        self.assertFalse(config.text2audio_log_tensorboard)


class TestText2AudioEvalCallback(unittest.TestCase):
    """Test text-to-audio evaluation callback in trainer."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        
        # Create minimal config
        self.config = XTTSConfig()
        self.config.training.checkpoint_dir = self.temp_dir
        self.config.training.enable_text2audio_eval = True
        self.config.training.text2audio_interval_steps = 200
        self.config.training.text2audio_output_dir = os.path.join(self.temp_dir, "eval_samples")
        self.config.training.text2audio_texts = ["Test text for evaluation"]
        self.config.training.text2audio_log_tensorboard = False  # Disable for testing
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    @patch('myxtts.training.trainer.XTTS')
    def test_maybe_eval_text2audio_disabled(self, mock_model_class):
        """Test that callback does nothing when disabled."""
        self.config.training.enable_text2audio_eval = False
        
        # Create trainer with mocked model
        mock_model = MagicMock()
        mock_model_class.return_value = mock_model
        
        with patch('myxtts.training.trainer.setup_gpu_strategy'):
            with patch('myxtts.training.trainer.configure_gpus'):
                trainer = XTTSTrainer(self.config, model=mock_model)
                trainer.current_step = 200
                
                # Should not generate audio
                with patch.object(trainer, '_generate_eval_audio') as mock_generate:
                    trainer._maybe_eval_text2audio()
                    mock_generate.assert_not_called()
    
    @patch('myxtts.training.trainer.XTTS')
    def test_maybe_eval_text2audio_wrong_step(self, mock_model_class):
        """Test that callback does nothing when not at interval step."""
        # Create trainer with mocked model
        mock_model = MagicMock()
        mock_model_class.return_value = mock_model
        
        with patch('myxtts.training.trainer.setup_gpu_strategy'):
            with patch('myxtts.training.trainer.configure_gpus'):
                trainer = XTTSTrainer(self.config, model=mock_model)
                trainer.current_step = 199  # Not at interval
                
                # Should not generate audio
                with patch.object(trainer, '_generate_eval_audio') as mock_generate:
                    trainer._maybe_eval_text2audio()
                    mock_generate.assert_not_called()
    
    @patch('myxtts.training.trainer.XTTS')
    def test_maybe_eval_text2audio_correct_step(self, mock_model_class):
        """Test that callback generates audio at correct interval."""
        # Create trainer with mocked model
        mock_model = MagicMock()
        mock_model_class.return_value = mock_model
        
        with patch('myxtts.training.trainer.setup_gpu_strategy'):
            with patch('myxtts.training.trainer.configure_gpus'):
                trainer = XTTSTrainer(self.config, model=mock_model)
                trainer.current_step = 200  # At interval
                
                # Should generate audio
                with patch.object(trainer, '_generate_eval_audio') as mock_generate:
                    trainer._maybe_eval_text2audio()
                    mock_generate.assert_called_once()
    
    @patch('myxtts.training.trainer.XTTS')
    def test_maybe_eval_text2audio_multiple_intervals(self, mock_model_class):
        """Test that callback works at multiple interval steps."""
        # Create trainer with mocked model
        mock_model = MagicMock()
        mock_model_class.return_value = mock_model
        
        with patch('myxtts.training.trainer.setup_gpu_strategy'):
            with patch('myxtts.training.trainer.configure_gpus'):
                trainer = XTTSTrainer(self.config, model=mock_model)
                
                with patch.object(trainer, '_generate_eval_audio') as mock_generate:
                    # Test multiple steps
                    for step in [0, 100, 200, 400, 600]:
                        trainer.current_step = step
                        trainer._maybe_eval_text2audio()
                    
                    # Should be called at steps 200, 400, 600 (not 0 or 100)
                    self.assertEqual(mock_generate.call_count, 3)
    
    @patch('myxtts.training.trainer.XTTS')
    def test_generate_eval_audio_creates_directory(self, mock_model_class):
        """Test that evaluation creates output directory."""
        # Create trainer with mocked model
        mock_model = MagicMock()
        mock_model.trainable = True
        mock_model_class.return_value = mock_model
        
        with patch('myxtts.training.trainer.setup_gpu_strategy'):
            with patch('myxtts.training.trainer.configure_gpus'):
                trainer = XTTSTrainer(self.config, model=mock_model)
                trainer.current_step = 200
                
                # Mock the text and audio processors
                with patch('myxtts.training.trainer.TextProcessor') as mock_text_proc:
                    with patch('myxtts.training.trainer.AudioProcessor') as mock_audio_proc:
                        with patch('myxtts.training.trainer.sf') as mock_sf:
                            with patch('myxtts.training.trainer.get_device_context'):
                                # Setup mocks
                                mock_text_proc_instance = MagicMock()
                                mock_text_proc_instance.clean_text.return_value = "cleaned text"
                                mock_text_proc_instance.text_to_sequence.return_value = [1, 2, 3]
                                mock_text_proc.return_value = mock_text_proc_instance
                                
                                mock_audio_proc_instance = MagicMock()
                                mock_audio_proc_instance.mel_to_wav.return_value = np.zeros(1000)
                                mock_audio_proc.return_value = mock_audio_proc_instance
                                
                                # Mock model output
                                mock_model.generate = MagicMock()
                                mock_mel = tf.constant(np.zeros((1, 100, 80)), dtype=tf.float32)
                                mock_model.generate.return_value = {"mel_output": mock_mel}
                                
                                # Run evaluation
                                trainer._generate_eval_audio()
                                
                                # Check directory was created
                                expected_dir = os.path.join(self.temp_dir, "eval_samples", "step_200")
                                self.assertTrue(os.path.exists(expected_dir))


class TestText2AudioEvalIntegration(unittest.TestCase):
    """Integration tests for text-to-audio evaluation."""
    
    def test_config_yaml_roundtrip(self):
        """Test that text-to-audio config can be saved and loaded from YAML."""
        import tempfile
        import yaml
        
        # Create config with custom values
        config = XTTSConfig()
        config.training.enable_text2audio_eval = True
        config.training.text2audio_interval_steps = 300
        config.training.text2audio_texts = ["Custom test text"]
        
        # Save to YAML
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml_path = f.name
            config.to_yaml(yaml_path)
        
        try:
            # Load from YAML
            loaded_config = XTTSConfig.from_yaml(yaml_path)
            
            # Verify values
            self.assertTrue(loaded_config.training.enable_text2audio_eval)
            self.assertEqual(loaded_config.training.text2audio_interval_steps, 300)
            self.assertEqual(loaded_config.training.text2audio_texts, ["Custom test text"])
        finally:
            os.unlink(yaml_path)


if __name__ == '__main__':
    unittest.main()
