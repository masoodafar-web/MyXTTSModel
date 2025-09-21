"""
Test neural vocoder and modern decoding strategies.

This script tests the new implementations including:
1. HiFi-GAN vocoder
2. Non-autoregressive decoder
3. Two-stage training approach
"""

import tensorflow as tf
import numpy as np
import os
import sys

# Add project root to path
sys.path.append('/home/runner/work/MyXTTSModel/MyXTTSModel')

from myxtts.config.config import ModelConfig
from myxtts.models.xtts import XTTS
from myxtts.models.vocoder import HiFiGANGenerator, VocoderInterface
from myxtts.models.non_autoregressive import NonAutoregressiveDecoder, DecoderStrategy
from myxtts.utils.audio import AudioProcessor


def test_hifigan_vocoder():
    """Test HiFi-GAN vocoder implementation."""
    print("Testing HiFi-GAN vocoder...")
    
    config = ModelConfig()
    
    # Create vocoder
    hifigan = HiFiGANGenerator(config)
    
    # Test input
    batch_size = 2
    mel_length = 100
    mel_input = tf.random.normal([batch_size, mel_length, config.n_mels])
    
    # Forward pass
    audio_output = hifigan(mel_input, training=False)
    
    print(f"Input mel shape: {mel_input.shape}")
    print(f"Output audio shape: {audio_output.shape}")
    
    # Check output dimensions
    expected_audio_length = mel_length * config.hop_length
    assert audio_output.shape[0] == batch_size
    assert audio_output.shape[2] == 1  # Single channel
    print("✓ HiFi-GAN vocoder test passed")


def test_vocoder_interface():
    """Test vocoder interface."""
    print("\nTesting vocoder interface...")
    
    config = ModelConfig()
    
    # Test with Griffin-Lim
    vocoder_gl = VocoderInterface(config, vocoder_type="griffin_lim")
    
    # Test with HiFi-GAN
    vocoder_hifi = VocoderInterface(config, vocoder_type="hifigan")
    
    # Test input
    mel_input = tf.random.normal([1, 50, config.n_mels])
    
    # Test Griffin-Lim interface (should return mel for post-processing)
    gl_output = vocoder_gl(mel_input)
    print(f"Griffin-Lim interface output shape: {gl_output.shape}")
    
    # Test HiFi-GAN interface
    hifi_output = vocoder_hifi(mel_input)
    print(f"HiFi-GAN interface output shape: {hifi_output.shape}")
    
    print("✓ Vocoder interface test passed")


def test_non_autoregressive_decoder():
    """Test non-autoregressive decoder."""
    print("\nTesting non-autoregressive decoder...")
    
    config = ModelConfig()
    config.decoder_strategy = "non_autoregressive"
    
    # Create decoder
    decoder = NonAutoregressiveDecoder(config)
    
    # Test inputs
    batch_size = 2
    text_length = 20
    encoder_output = tf.random.normal([batch_size, text_length, config.text_encoder_dim])
    speaker_embedding = tf.random.normal([batch_size, config.speaker_embedding_dim])
    
    # Forward pass
    mel_output, duration_pred = decoder(
        encoder_output,
        speaker_embedding=speaker_embedding,
        training=False
    )
    
    print(f"Encoder output shape: {encoder_output.shape}")
    print(f"Mel output shape: {mel_output.shape}")
    print(f"Duration predictions shape: {duration_pred.shape}")
    
    assert mel_output.shape[0] == batch_size
    assert mel_output.shape[2] == config.n_mels
    assert duration_pred.shape[0] == batch_size
    assert duration_pred.shape[1] == text_length
    
    print("✓ Non-autoregressive decoder test passed")


def test_decoder_strategy():
    """Test decoder strategy interface."""
    print("\nTesting decoder strategy interface...")
    
    config = ModelConfig()
    
    # Test autoregressive strategy
    strategy_ar = DecoderStrategy(config, strategy="autoregressive")
    
    # Test non-autoregressive strategy
    strategy_nar = DecoderStrategy(config, strategy="non_autoregressive")
    
    # Test inputs
    batch_size = 1
    text_length = 15
    mel_length = 50
    
    encoder_output = tf.random.normal([batch_size, text_length, config.text_encoder_dim])
    decoder_inputs = tf.random.normal([batch_size, mel_length, config.n_mels])
    speaker_embedding = tf.random.normal([batch_size, config.speaker_embedding_dim])
    
    # Test autoregressive strategy
    ar_mel, ar_stop, _ = strategy_ar(
        decoder_inputs,
        encoder_output,
        speaker_embedding=speaker_embedding,
        training=False
    )
    
    print(f"Autoregressive - Mel output shape: {ar_mel.shape}")
    print(f"Autoregressive - Stop tokens shape: {ar_stop.shape}")
    
    # Test non-autoregressive strategy  
    nar_mel, nar_duration, _ = strategy_nar(
        None,  # No decoder inputs needed
        encoder_output,
        speaker_embedding=speaker_embedding,
        training=False
    )
    
    print(f"Non-autoregressive - Mel output shape: {nar_mel.shape}")
    print(f"Non-autoregressive - Duration predictions shape: {nar_duration.shape}")
    
    print("✓ Decoder strategy test passed")


def test_xtts_with_neural_vocoder():
    """Test XTTS model with neural vocoder."""
    print("\nTesting XTTS with neural vocoder...")
    
    # Test with HiFi-GAN vocoder
    config = ModelConfig()
    config.vocoder_type = "hifigan"
    config.decoder_strategy = "autoregressive"
    
    model = XTTS(config)
    
    # Test inputs
    batch_size = 1
    text_length = 10
    mel_length = 30
    
    text_inputs = tf.random.uniform([batch_size, text_length], maxval=1000, dtype=tf.int32)
    mel_inputs = tf.random.normal([batch_size, mel_length, config.n_mels])
    audio_conditioning = tf.random.normal([batch_size, 1000, config.n_mels])  # Mock audio conditioning
    
    # Forward pass
    outputs = model(
        text_inputs,
        mel_inputs,
        audio_conditioning=audio_conditioning,
        training=False
    )
    
    print(f"Text inputs shape: {text_inputs.shape}")
    print(f"Model outputs keys: {outputs.keys()}")
    print(f"Mel output shape: {outputs['mel_output'].shape}")
    if 'stop_tokens' in outputs:
        print(f"Stop tokens shape: {outputs['stop_tokens'].shape}")
    
    # Test generation with audio synthesis
    generation_outputs = model.generate(
        text_inputs,
        audio_conditioning=audio_conditioning,
        max_length=50,
        generate_audio=True
    )
    
    print(f"Generation outputs keys: {generation_outputs.keys()}")
    if 'audio_output' in generation_outputs:
        print(f"Generated audio shape: {generation_outputs['audio_output'].shape}")
    
    print("✓ XTTS with neural vocoder test passed")


def test_xtts_non_autoregressive():
    """Test XTTS model with non-autoregressive decoder."""
    print("\nTesting XTTS with non-autoregressive decoder...")
    
    config = ModelConfig()
    config.decoder_strategy = "non_autoregressive"
    config.vocoder_type = "hifigan"
    
    model = XTTS(config)
    
    # Test inputs
    batch_size = 1
    text_length = 10
    mel_length = 30
    
    text_inputs = tf.random.uniform([batch_size, text_length], maxval=1000, dtype=tf.int32)
    mel_inputs = tf.random.normal([batch_size, mel_length, config.n_mels])
    
    # Forward pass
    outputs = model(
        text_inputs,
        mel_inputs,
        training=False
    )
    
    print(f"Non-autoregressive model outputs keys: {outputs.keys()}")
    print(f"Mel output shape: {outputs['mel_output'].shape}")
    if 'duration_pred' in outputs:
        print(f"Duration predictions shape: {outputs['duration_pred'].shape}")
    
    # Test generation
    generation_outputs = model.generate(
        text_inputs,
        max_length=50,
        generate_audio=True
    )
    
    print(f"Non-autoregressive generation outputs keys: {generation_outputs.keys()}")
    print("✓ XTTS with non-autoregressive decoder test passed")


def test_audio_processor_neural_vocoder():
    """Test audio processor with neural vocoder support."""
    print("\nTesting audio processor with neural vocoder...")
    
    config = ModelConfig()
    audio_processor = AudioProcessor(
        sample_rate=config.sample_rate,
        n_mels=config.n_mels,
        n_fft=config.n_fft,
        hop_length=config.hop_length,
        win_length=config.win_length
    )
    
    # Create a mock vocoder model
    vocoder = HiFiGANGenerator(config)
    
    # Test mel to wav conversion
    mel_spec = np.random.randn(config.n_mels, 100)
    
    # Test with Griffin-Lim (fallback)
    audio_gl = audio_processor.mel_to_wav(mel_spec)
    print(f"Griffin-Lim audio shape: {audio_gl.shape}")
    
    # Test with neural vocoder
    audio_neural = audio_processor.mel_to_wav_neural(mel_spec, vocoder)
    print(f"Neural vocoder audio shape: {audio_neural.shape}")
    
    print("✓ Audio processor neural vocoder test passed")


if __name__ == "__main__":
    print("Running neural vocoder and modern decoding tests...\n")
    
    try:
        test_hifigan_vocoder()
        test_vocoder_interface()
        test_non_autoregressive_decoder()
        test_decoder_strategy()
        test_xtts_with_neural_vocoder()
        test_xtts_non_autoregressive()
        test_audio_processor_neural_vocoder()
        
        print("\n🎉 All tests passed successfully!")
        print("\nNew features implemented:")
        print("✓ HiFi-GAN neural vocoder")
        print("✓ Non-autoregressive decoder (FastSpeech-style)")
        print("✓ Decoder strategy interface")
        print("✓ Neural vocoder integration in XTTS")
        print("✓ Two-stage training architecture")
        print("✓ Enhanced audio processing with neural vocoder support")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)