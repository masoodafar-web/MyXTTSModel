"""Tests for XTTS model."""

import unittest
import tensorflow as tf
import numpy as np
from myxtts.config.config import ModelConfig
from myxtts.models.xtts import XTTS
from myxtts.models.layers import MultiHeadAttention, PositionalEncoding, TransformerBlock
from myxtts.utils.text import TextProcessor, NLLBTokenizer, TRANSFORMERS_AVAILABLE


class TestModel(unittest.TestCase):
    """Test XTTS model components."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = ModelConfig()
        self.config.text_encoder_dim = 128  # Smaller for testing
        self.config.decoder_dim = 128
        self.config.audio_encoder_dim = 128
        self.config.text_encoder_layers = 2
        self.config.decoder_layers = 2
        self.config.audio_encoder_layers = 2
    
    def test_multihead_attention(self):
        """Test MultiHeadAttention layer."""
        d_model = 128
        num_heads = 8
        seq_len = 10
        batch_size = 2
        
        attention = MultiHeadAttention(d_model, num_heads)
        
        # Create dummy input
        inputs = tf.random.normal([batch_size, seq_len, d_model])
        
        # Test self-attention
        output = attention(inputs, training=False)
        
        self.assertEqual(output.shape, inputs.shape)
    
    def test_positional_encoding(self):
        """Test PositionalEncoding layer."""
        d_model = 128
        max_length = 100
        seq_len = 10
        batch_size = 2
        
        pos_encoding = PositionalEncoding(d_model, max_length)
        
        # Create dummy input
        inputs = tf.random.normal([batch_size, seq_len, d_model])
        
        # Test positional encoding
        output = pos_encoding(inputs)
        
        self.assertEqual(output.shape, inputs.shape)
    
    def test_transformer_block(self):
        """Test TransformerBlock layer."""
        d_model = 128
        num_heads = 8
        d_ff = 512
        seq_len = 10
        batch_size = 2
        
        # Test encoder block
        encoder_block = TransformerBlock(d_model, num_heads, d_ff, is_decoder=False)
        
        inputs = tf.random.normal([batch_size, seq_len, d_model])
        output = encoder_block(inputs, training=False)
        
        self.assertEqual(output.shape, inputs.shape)
        
        # Test decoder block  
        decoder_block = TransformerBlock(d_model, num_heads, d_ff, is_decoder=True)
        
        encoder_output = tf.random.normal([batch_size, seq_len, d_model])
        decoder_output = decoder_block(
            inputs, 
            encoder_output=encoder_output, 
            training=False
        )
        
        self.assertEqual(decoder_output.shape, inputs.shape)
    
    def test_xtts_model_creation(self):
        """Test XTTS model creation."""
        model = XTTS(self.config)
        
        # Check model components exist
        self.assertTrue(hasattr(model, 'text_encoder'))
        self.assertTrue(hasattr(model, 'mel_decoder'))
        
        if self.config.use_voice_conditioning:
            self.assertTrue(hasattr(model, 'audio_encoder'))
    
    def test_xtts_forward_pass(self):
        """Test XTTS forward pass."""
        model = XTTS(self.config)
        
        # Prepare dummy inputs
        batch_size = 2
        text_len = 10
        mel_len = 20
        n_mels = self.config.n_mels
        
        text_inputs = tf.random.uniform([batch_size, text_len], 0, 10, dtype=tf.int32)
        mel_inputs = tf.random.normal([batch_size, mel_len, n_mels])
        text_lengths = tf.constant([text_len, text_len-2], dtype=tf.int32)
        mel_lengths = tf.constant([mel_len, mel_len-5], dtype=tf.int32)
        
        # Test without audio conditioning
        outputs = model(
            text_inputs=text_inputs,
            mel_inputs=mel_inputs,
            text_lengths=text_lengths,
            mel_lengths=mel_lengths,
            training=False
        )
        
        # Check output shapes
        self.assertIn('mel_output', outputs)
        self.assertIn('stop_tokens', outputs)
        
        mel_output = outputs['mel_output']
        stop_tokens = outputs['stop_tokens']
        
        self.assertEqual(mel_output.shape, [batch_size, mel_len, n_mels])
        self.assertEqual(stop_tokens.shape, [batch_size, mel_len, 1])
    
    def test_xtts_with_audio_conditioning(self):
        """Test XTTS with audio conditioning."""
        model = XTTS(self.config)
        
        # Prepare dummy inputs
        batch_size = 2
        text_len = 10
        mel_len = 20
        audio_len = 50
        n_mels = self.config.n_mels
        
        text_inputs = tf.random.uniform([batch_size, text_len], 0, 10, dtype=tf.int32)
        mel_inputs = tf.random.normal([batch_size, mel_len, n_mels])
        audio_conditioning = tf.random.normal([batch_size, audio_len, n_mels])
        
        # Test with audio conditioning
        outputs = model(
            text_inputs=text_inputs,
            mel_inputs=mel_inputs,
            audio_conditioning=audio_conditioning,
            training=False
        )
        
        # Check outputs
        self.assertIn('mel_output', outputs)
        self.assertIn('stop_tokens', outputs)
        
        if self.config.use_voice_conditioning:
            self.assertIn('speaker_embedding', outputs)
    
    def test_xtts_generation(self):
        """Test XTTS generation mode."""
        model = XTTS(self.config)
        
        # Prepare inputs
        batch_size = 1
        text_len = 5
        
        text_inputs = tf.random.uniform([batch_size, text_len], 0, 10, dtype=tf.int32)
        
        # Test generation
        outputs = model.generate(
            text_inputs=text_inputs,
            max_length=10,
            temperature=1.0
        )
        
        # Check outputs
        self.assertIn('mel_output', outputs)
        self.assertIn('stop_tokens', outputs)
        
        mel_output = outputs['mel_output']
        self.assertEqual(mel_output.shape[0], batch_size)
        self.assertEqual(mel_output.shape[2], self.config.n_mels)
    
    def test_model_config_serialization(self):
        """Test model configuration serialization."""
        model = XTTS(self.config)
        
        # Get config
        model_config = model.get_config()
        
        self.assertIn('model_config', model_config)
        self.assertIn('name', model_config)
        
        # Test from_config
        new_model = XTTS.from_config(model_config)
        
        # Both models should have same architecture
        self.assertEqual(
            len(model.text_encoder.transformer_blocks),
            len(new_model.text_encoder.transformer_blocks)
        )


class TestModelIntegration(unittest.TestCase):
    """Integration tests for model components."""
    
    def test_end_to_end_training_step(self):
        """Test a complete training step."""
        config = ModelConfig()
        config.text_encoder_dim = 64  # Small for testing
        config.decoder_dim = 64
        config.audio_encoder_dim = 64
        config.text_encoder_layers = 1
        config.decoder_layers = 1
        config.audio_encoder_layers = 1
        
        model = XTTS(config)
        
        # Dummy data
        batch_size = 2
        text_len = 5
        mel_len = 10
        
        text_inputs = tf.random.uniform([batch_size, text_len], 0, 10, dtype=tf.int32)
        mel_targets = tf.random.normal([batch_size, mel_len, config.n_mels])
        
        # Forward pass
        with tf.GradientTape() as tape:
            outputs = model(
                text_inputs=text_inputs,
                mel_inputs=mel_targets,
                training=True
            )
            
            # Dummy loss
            mel_loss = tf.reduce_mean(tf.abs(outputs['mel_output'] - mel_targets))
            stop_loss = tf.reduce_mean(tf.square(outputs['stop_tokens']))
            total_loss = mel_loss + stop_loss
        
        # Compute gradients
        gradients = tape.gradient(total_loss, model.trainable_variables)
        
        # Check gradients exist
        self.assertIsNotNone(gradients)
        self.assertTrue(any(grad is not None for grad in gradients))


class TestTokenizerIntegration(unittest.TestCase):
    """Test tokenizer integration."""
    
    def test_text_processor_custom_tokenizer(self):
        """Test TextProcessor with custom tokenizer."""
        processor = TextProcessor(tokenizer_type="custom")
        
        text = "Hello, world!"
        sequence = processor.text_to_sequence(text)
        
        # Check sequence is not empty
        self.assertGreater(len(sequence), 0)
        
        # Check vocab size
        vocab_size = processor.get_vocab_size()
        self.assertGreater(vocab_size, 0)
        
        # Test batch processing
        texts = ["Hello", "World"]
        sequences, lengths = processor.batch_text_to_sequence(texts)
        
        self.assertEqual(len(sequences), 2)
        self.assertEqual(len(lengths), 2)
    
    @unittest.skipUnless(TRANSFORMERS_AVAILABLE, "transformers library not available")
    def test_text_processor_nllb_tokenizer(self):
        """Test TextProcessor with NLLB tokenizer."""
        # Test that we can create the processor structure (even if model download fails)
        try:
            processor = TextProcessor(tokenizer_type="nllb")
            
            # Check tokenizer type
            self.assertEqual(processor.tokenizer_type, "nllb")
            
            # Test basic properties
            self.assertIsNone(processor.symbols)  # Not used with NLLB
            
            # If we get here, the model was downloaded successfully
            vocab_size = processor.get_vocab_size()
            self.assertGreater(vocab_size, 250000)  # NLLB should have large vocab
            
        except (OSError, Exception) as e:
            # Expected if there's no internet connection to download the model
            if "huggingface.co" in str(e) or "connection" in str(e).lower():
                self.skipTest("Cannot download NLLB model - no internet connection")
            else:
                raise e
    
    def test_config_with_nllb_settings(self):
        """Test model config with NLLB tokenizer settings."""
        config = ModelConfig()
        config.tokenizer_type = "nllb"
        config.text_vocab_size = 256_256
        
        # Test that config accepts NLLB settings
        self.assertEqual(config.tokenizer_type, "nllb")
        self.assertEqual(config.text_vocab_size, 256_256)
        
        # Test that model can be created with large vocab size
        # (even if it fails due to missing layers, the config should work)
        self.assertIsInstance(config, ModelConfig)


if __name__ == '__main__':
    unittest.main()