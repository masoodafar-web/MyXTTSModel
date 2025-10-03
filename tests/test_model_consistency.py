"""Test to validate consistency between model components and configuration."""

import unittest
from myxtts.config.config import ModelConfig
from myxtts.models.xtts import TextEncoder, AudioEncoder, MelDecoder, XTTS


class TestModelConfigurationConsistency(unittest.TestCase):
    """Test that model components are properly synchronized with configuration."""
    
    def setUp(self):
        """Set up test configuration."""
        self.config = ModelConfig()
    
    def test_text_encoder_dimensions(self):
        """Verify TextEncoder uses correct dimensions from config."""
        encoder = TextEncoder(self.config)
        
        # Check that d_model matches config
        self.assertEqual(encoder.d_model, self.config.text_encoder_dim)
        self.assertEqual(encoder.d_model, 512, "TextEncoder d_model should be 512")
        
        # Check number of transformer blocks matches config
        self.assertEqual(len(encoder.transformer_blocks), self.config.text_encoder_layers)
        self.assertEqual(len(encoder.transformer_blocks), 8, "TextEncoder should have 8 layers")
        
        # Check each transformer block has correct dimensions
        for block in encoder.transformer_blocks:
            self.assertEqual(block.d_model, self.config.text_encoder_dim)
            self.assertEqual(block.num_heads, self.config.text_encoder_heads)
            self.assertEqual(block.num_heads, 8, "TextEncoder should have 8 heads")
    
    def test_audio_encoder_dimensions(self):
        """Verify AudioEncoder uses correct dimensions from config."""
        encoder = AudioEncoder(self.config)
        
        # Check that d_model matches config
        self.assertEqual(encoder.d_model, self.config.audio_encoder_dim)
        self.assertEqual(encoder.d_model, 768, "AudioEncoder d_model should be 768")
        
        # For original architecture (not pretrained)
        if not encoder.use_pretrained:
            # Check number of transformer blocks matches config
            self.assertEqual(len(encoder.transformer_blocks), self.config.audio_encoder_layers)
            self.assertEqual(len(encoder.transformer_blocks), 8, "AudioEncoder should have 8 layers")
            
            # Check each transformer block has correct dimensions
            for block in encoder.transformer_blocks:
                self.assertEqual(block.d_model, self.config.audio_encoder_dim)
                self.assertEqual(block.num_heads, self.config.audio_encoder_heads)
                self.assertEqual(block.num_heads, 12, "AudioEncoder should have 12 heads")
            
            # Check speaker embedding dimension
            self.assertEqual(
                encoder.speaker_projection.units, 
                self.config.speaker_embedding_dim
            )
            self.assertEqual(
                encoder.speaker_projection.units, 
                512, 
                "Speaker embedding should be 512-dimensional"
            )
    
    def test_mel_decoder_dimensions(self):
        """Verify MelDecoder uses correct dimensions from config."""
        decoder = MelDecoder(self.config)
        
        # Check that d_model matches config
        self.assertEqual(decoder.d_model, self.config.decoder_dim)
        self.assertEqual(decoder.d_model, 1536, "MelDecoder d_model should be 1536")
        
        # Check number of transformer blocks matches config
        self.assertEqual(len(decoder.transformer_blocks), self.config.decoder_layers)
        self.assertEqual(len(decoder.transformer_blocks), 16, "MelDecoder should have 16 layers")
        
        # Check each transformer block has correct dimensions
        for block in decoder.transformer_blocks:
            self.assertEqual(block.d_model, self.config.decoder_dim)
            self.assertEqual(block.num_heads, self.config.decoder_heads)
            self.assertEqual(block.num_heads, 24, "MelDecoder should have 24 heads")
        
        # Check output projection matches n_mels
        self.assertEqual(decoder.mel_projection.units, self.config.n_mels)
        self.assertEqual(decoder.mel_projection.units, 80, "Mel output should be 80-dimensional")
    
    def test_full_model_consistency(self):
        """Verify full XTTS model has consistent component dimensions."""
        model = XTTS(self.config)
        
        # Check text encoder
        self.assertEqual(model.text_encoder.d_model, self.config.text_encoder_dim)
        self.assertEqual(len(model.text_encoder.transformer_blocks), self.config.text_encoder_layers)
        
        # Check audio encoder (if voice conditioning is enabled)
        if self.config.use_voice_conditioning:
            self.assertEqual(model.audio_encoder.d_model, self.config.audio_encoder_dim)
            if not model.audio_encoder.use_pretrained:
                self.assertEqual(
                    len(model.audio_encoder.transformer_blocks), 
                    self.config.audio_encoder_layers
                )
        
        # Check mel decoder (if using autoregressive decoder)
        if hasattr(model, 'mel_decoder'):
            self.assertEqual(model.mel_decoder.d_model, self.config.decoder_dim)
            self.assertEqual(len(model.mel_decoder.transformer_blocks), self.config.decoder_layers)
    
    def test_configuration_values(self):
        """Verify configuration has expected synergistic values."""
        config = ModelConfig()
        
        # Text encoder configuration
        self.assertEqual(config.text_encoder_dim, 512)
        self.assertEqual(config.text_encoder_layers, 8)
        self.assertEqual(config.text_encoder_heads, 8)
        
        # Audio encoder configuration
        self.assertEqual(config.audio_encoder_dim, 768)
        self.assertEqual(config.audio_encoder_layers, 8)
        self.assertEqual(config.audio_encoder_heads, 12)
        
        # Decoder configuration
        self.assertEqual(config.decoder_dim, 1536)
        self.assertEqual(config.decoder_layers, 16)
        self.assertEqual(config.decoder_heads, 24)
        
        # Speaker embedding configuration
        self.assertEqual(config.speaker_embedding_dim, 512)
        
        # Mel spectrogram configuration
        self.assertEqual(config.n_mels, 80)
        self.assertEqual(config.sample_rate, 22050)
    
    def test_dimension_ratios(self):
        """Test that dimension ratios are synergistic."""
        config = ModelConfig()
        
        # Audio encoder should be larger than text encoder (more complex)
        self.assertGreater(config.audio_encoder_dim, config.text_encoder_dim)
        
        # Decoder should be largest (most complex modeling)
        self.assertGreater(config.decoder_dim, config.audio_encoder_dim)
        self.assertGreater(config.decoder_dim, config.text_encoder_dim)
        
        # More decoder layers for complex autoregressive modeling
        self.assertGreaterEqual(config.decoder_layers, config.text_encoder_layers)
        self.assertGreaterEqual(config.decoder_layers, config.audio_encoder_layers)
        
        # More attention heads in decoder for complex cross-attention
        self.assertGreater(config.decoder_heads, config.text_encoder_heads)
        self.assertGreater(config.decoder_heads, config.audio_encoder_heads)


if __name__ == '__main__':
    unittest.main()
