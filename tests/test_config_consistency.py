"""Test to validate consistency between configuration values without requiring TensorFlow."""

import unittest
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from myxtts.config.config import ModelConfig, DataConfig


class TestConfigurationConsistency(unittest.TestCase):
    """Test that configuration values are properly synchronized and synergistic."""
    
    def setUp(self):
        """Set up test configuration."""
        self.model_config = ModelConfig()
        self.data_config = DataConfig()
    
    def test_text_encoder_config_values(self):
        """Verify TextEncoder configuration values."""
        config = self.model_config
        
        # Expected values from synchronized documentation
        self.assertEqual(config.text_encoder_dim, 512, 
                        "TextEncoder dimension should be 512")
        self.assertEqual(config.text_encoder_layers, 8, 
                        "TextEncoder should have 8 layers (enhanced from 6)")
        self.assertEqual(config.text_encoder_heads, 8, 
                        "TextEncoder should have 8 attention heads")
        
        # Verify vocab size for NLLB-200
        self.assertEqual(config.text_vocab_size, 256_256,
                        "NLLB-200 vocab size should be 256,256")
    
    def test_audio_encoder_config_values(self):
        """Verify AudioEncoder configuration values."""
        config = self.model_config
        
        # Expected values from synchronized documentation
        self.assertEqual(config.audio_encoder_dim, 768, 
                        "AudioEncoder dimension should be 768 (enhanced from 512)")
        self.assertEqual(config.audio_encoder_layers, 8, 
                        "AudioEncoder should have 8 layers (enhanced from 6)")
        self.assertEqual(config.audio_encoder_heads, 12, 
                        "AudioEncoder should have 12 attention heads (enhanced from 8)")
    
    def test_decoder_config_values(self):
        """Verify MelDecoder configuration values."""
        config = self.model_config
        
        # Expected values from synchronized documentation
        self.assertEqual(config.decoder_dim, 1536, 
                        "MelDecoder dimension should be 1536 (enhanced from 1024)")
        self.assertEqual(config.decoder_layers, 16, 
                        "MelDecoder should have 16 layers (enhanced from 12)")
        self.assertEqual(config.decoder_heads, 24, 
                        "MelDecoder should have 24 attention heads (enhanced from 16)")
    
    def test_speaker_embedding_config(self):
        """Verify speaker embedding configuration."""
        config = self.model_config
        
        # Expected value from synchronized documentation
        self.assertEqual(config.speaker_embedding_dim, 512, 
                        "Speaker embedding should be 512-dimensional (enhanced from 256)")
    
    def test_mel_spectrogram_config(self):
        """Verify mel spectrogram configuration."""
        config = self.model_config
        
        self.assertEqual(config.n_mels, 80, 
                        "Mel spectrogram should have 80 mel bins")
        self.assertEqual(config.sample_rate, 22050, 
                        "Sample rate should be 22050 Hz")
        self.assertEqual(config.n_fft, 1024, 
                        "FFT size should be 1024")
        self.assertEqual(config.hop_length, 256, 
                        "Hop length should be 256")
        self.assertEqual(config.win_length, 1024, 
                        "Window length should be 1024")
    
    def test_dimension_synergy(self):
        """Test that dimension ratios are synergistic for model harmony."""
        config = self.model_config
        
        # Audio encoder should be larger than text encoder
        # (audio features are more complex than text)
        self.assertGreater(config.audio_encoder_dim, config.text_encoder_dim,
                          "Audio encoder should have larger dimension than text encoder")
        
        # Decoder should be largest (handles most complex modeling)
        self.assertGreater(config.decoder_dim, config.audio_encoder_dim,
                          "Decoder should have larger dimension than audio encoder")
        self.assertGreater(config.decoder_dim, config.text_encoder_dim,
                          "Decoder should have larger dimension than text encoder")
        
        # More decoder layers for complex autoregressive modeling
        self.assertGreaterEqual(config.decoder_layers, config.text_encoder_layers,
                               "Decoder should have at least as many layers as text encoder")
        self.assertGreaterEqual(config.decoder_layers, config.audio_encoder_layers,
                               "Decoder should have at least as many layers as audio encoder")
        
        # More attention heads in decoder for complex multi-modal attention
        self.assertGreater(config.decoder_heads, config.text_encoder_heads,
                          "Decoder should have more attention heads than text encoder")
        self.assertGreater(config.decoder_heads, config.audio_encoder_heads,
                          "Decoder should have more attention heads than audio encoder")
    
    def test_gst_config_values(self):
        """Verify Global Style Token configuration."""
        config = self.model_config
        
        if config.use_gst:
            self.assertEqual(config.gst_num_style_tokens, 10,
                            "GST should have 10 style tokens")
            self.assertEqual(config.gst_style_token_dim, 256,
                            "GST style tokens should be 256-dimensional")
            self.assertEqual(config.gst_style_embedding_dim, 256,
                            "GST style embedding should be 256-dimensional")
            self.assertEqual(config.gst_num_heads, 4,
                            "GST should use 4 attention heads")
            self.assertEqual(config.gst_reference_encoder_dim, 128,
                            "GST reference encoder should be 128-dimensional")
    
    def test_data_config_values(self):
        """Verify DataConfig values are synchronized."""
        config = self.data_config
        
        # Optimized batch size for GPU utilization
        self.assertEqual(config.batch_size, 56,
                        "Batch size should be 56 (optimized for better GPU utilization)")
        
        # Optimized worker count for CPU-GPU overlap
        self.assertEqual(config.num_workers, 18,
                        "Num workers should be 18 (optimized for better CPU-GPU overlap)")
        
        # Optimized prefetch buffer
        self.assertEqual(config.prefetch_buffer_size, 12,
                        "Prefetch buffer should be 12 (optimized for sustained GPU utilization)")
        
        # Sample rate should match model config
        self.assertEqual(config.sample_rate, 22050,
                        "Data sample rate should be 22050 Hz")
    
    def test_voice_conditioning_synergy(self):
        """Test voice conditioning features are harmonized."""
        config = self.model_config
        
        if config.use_voice_conditioning:
            # Speaker embedding should be large enough for quality voice cloning
            self.assertGreaterEqual(config.speaker_embedding_dim, 256,
                                   "Speaker embedding should be at least 256-dimensional")
            
            # Voice conditioning layers should exist
            self.assertGreater(config.voice_conditioning_layers, 0,
                             "Voice conditioning should have dedicated layers")
            
            # Voice similarity threshold should be reasonable
            self.assertGreater(config.voice_similarity_threshold, 0.0)
            self.assertLessEqual(config.voice_similarity_threshold, 1.0,
                               "Voice similarity threshold should be in [0, 1]")
    
    def test_feedforward_dimension_ratios(self):
        """Test that feedforward dimensions follow standard transformer ratios."""
        config = self.model_config
        
        # Standard transformer practice: feedforward dim is 4x model dim
        # This is implicitly set in the code, but we verify the base dimensions support it
        
        # Text encoder: 512 * 4 = 2048 (feedforward dim)
        expected_text_ff = config.text_encoder_dim * 4
        self.assertEqual(expected_text_ff, 2048,
                        "Text encoder feedforward should be 2048")
        
        # Audio encoder: 768 * 4 = 3072 (feedforward dim)
        expected_audio_ff = config.audio_encoder_dim * 4
        self.assertEqual(expected_audio_ff, 3072,
                        "Audio encoder feedforward should be 3072")
        
        # Decoder: 1536 * 4 = 6144 (feedforward dim)
        expected_decoder_ff = config.decoder_dim * 4
        self.assertEqual(expected_decoder_ff, 6144,
                        "Decoder feedforward should be 6144")
    
    def test_attention_head_divisibility(self):
        """Test that dimensions are divisible by number of attention heads."""
        config = self.model_config
        
        # Text encoder: 512 / 8 = 64 (head dimension)
        self.assertEqual(config.text_encoder_dim % config.text_encoder_heads, 0,
                        "Text encoder dim should be divisible by number of heads")
        self.assertEqual(config.text_encoder_dim // config.text_encoder_heads, 64,
                        "Text encoder head dimension should be 64")
        
        # Audio encoder: 768 / 12 = 64 (head dimension)
        self.assertEqual(config.audio_encoder_dim % config.audio_encoder_heads, 0,
                        "Audio encoder dim should be divisible by number of heads")
        self.assertEqual(config.audio_encoder_dim // config.audio_encoder_heads, 64,
                        "Audio encoder head dimension should be 64")
        
        # Decoder: 1536 / 24 = 64 (head dimension)
        self.assertEqual(config.decoder_dim % config.decoder_heads, 0,
                        "Decoder dim should be divisible by number of heads")
        self.assertEqual(config.decoder_dim // config.decoder_heads, 64,
                        "Decoder head dimension should be 64")
        
        # All components use same head dimension (64) for consistent attention patterns
        text_head_dim = config.text_encoder_dim // config.text_encoder_heads
        audio_head_dim = config.audio_encoder_dim // config.audio_encoder_heads
        decoder_head_dim = config.decoder_dim // config.decoder_heads
        
        self.assertEqual(text_head_dim, audio_head_dim,
                        "Text and audio encoders should use same head dimension")
        self.assertEqual(audio_head_dim, decoder_head_dim,
                        "Audio encoder and decoder should use same head dimension")
        self.assertEqual(text_head_dim, 64,
                        "All components should use 64-dimensional attention heads")


if __name__ == '__main__':
    # Run tests
    unittest.main(verbosity=2)
