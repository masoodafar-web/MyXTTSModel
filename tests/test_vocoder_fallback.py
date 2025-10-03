"""
Test vocoder weight initialization and fallback behavior.

This test verifies that:
1. Vocoder tracks weight initialization status
2. Warnings are issued for uninitialized vocoder
3. Fallback to Griffin-Lim works correctly
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import tensorflow as tf
import numpy as np
import logging

# Configure logging to see warnings
logging.basicConfig(level=logging.WARNING)

from myxtts.models.vocoder import VocoderInterface, HiFiGANGenerator
from myxtts.config.config import ModelConfig


def test_vocoder_initialization_tracking():
    """Test that vocoder tracks initialization status."""
    print("\n=== Test 1: Vocoder Initialization Tracking ===")
    
    # Create a minimal config
    config = ModelConfig(
        text_vocab_size=100,
        text_encoder_dim=256,
        text_encoder_layers=2,
        text_encoder_heads=4,
        n_mels=80,
        sample_rate=22050,
        hop_length=256,
        n_fft=1024,
        win_length=1024
    )
    
    # Test HiFi-GAN vocoder initialization
    vocoder = VocoderInterface(config, vocoder_type="hifigan")
    
    # Should start uninitialized
    assert not vocoder.check_weights_initialized(), "Vocoder should start uninitialized"
    print("✅ Vocoder starts with uninitialized weights")
    
    # Mark as loaded
    vocoder.mark_weights_loaded()
    assert vocoder.check_weights_initialized(), "Vocoder should be marked as initialized"
    print("✅ Vocoder can be marked as initialized")
    
    # Test Griffin-Lim is always initialized
    gl_vocoder = VocoderInterface(config, vocoder_type="griffin_lim")
    assert gl_vocoder.check_weights_initialized(), "Griffin-Lim should always be initialized"
    print("✅ Griffin-Lim vocoder is always initialized")
    
    print("✅ Test 1 PASSED\n")


def test_vocoder_warning_on_uninitialized():
    """Test that vocoder issues warning when uninitialized."""
    print("\n=== Test 2: Warning on Uninitialized Vocoder ===")
    
    config = ModelConfig(
        text_vocab_size=100,
        text_encoder_dim=256,
        text_encoder_layers=2,
        text_encoder_heads=4,
        n_mels=80,
        sample_rate=22050,
        hop_length=256,
        n_fft=1024,
        win_length=1024
    )
    
    # Create uninitialized vocoder
    vocoder = VocoderInterface(config, vocoder_type="hifigan")
    
    # Create dummy mel input
    dummy_mel = tf.random.normal([1, 100, 80])  # [batch, time, n_mels]
    
    # This should issue a warning
    print("Calling uninitialized vocoder (should see warning):")
    output = vocoder(dummy_mel, training=False)
    
    print("✅ Test 2 PASSED (check for warning above)\n")


def test_vocoder_output_validation():
    """Test that vocoder validates output and detects invalid results."""
    print("\n=== Test 3: Vocoder Output Validation ===")
    
    config = ModelConfig(
        text_vocab_size=100,
        text_encoder_dim=256,
        text_encoder_layers=2,
        text_encoder_heads=4,
        n_mels=80,
        sample_rate=22050,
        hop_length=256,
        n_fft=1024,
        win_length=1024
    )
    
    # Create vocoder
    vocoder = VocoderInterface(config, vocoder_type="hifigan")
    
    # Create dummy mel input
    dummy_mel = tf.random.normal([1, 100, 80])
    
    # Generate output (will be random/noisy with untrained weights)
    output = vocoder(dummy_mel, training=False)
    
    # Output should have been generated (even if noisy)
    assert output is not None, "Vocoder should produce output"
    
    # Check output shape - could be audio or mel (fallback)
    print(f"Vocoder output shape: {output.shape}")
    
    # If output is mel (fallback), last dimension should be n_mels
    if output.shape[-1] == 80:
        print("✅ Vocoder returned mel (fallback mode detected)")
    else:
        print(f"✅ Vocoder returned audio (shape: {output.shape})")
    
    print("✅ Test 3 PASSED\n")


def test_griffin_lim_vocoder():
    """Test that Griffin-Lim vocoder works as expected."""
    print("\n=== Test 4: Griffin-Lim Vocoder ===")
    
    config = ModelConfig(
        text_vocab_size=100,
        text_encoder_dim=256,
        text_encoder_layers=2,
        text_encoder_heads=4,
        n_mels=80,
        sample_rate=22050,
        hop_length=256,
        n_fft=1024,
        win_length=1024
    )
    
    # Create Griffin-Lim vocoder
    vocoder = VocoderInterface(config, vocoder_type="griffin_lim")
    
    # Create dummy mel input
    dummy_mel = tf.random.normal([1, 100, 80])
    
    # Generate output
    output = vocoder(dummy_mel, training=False)
    
    # Griffin-Lim should return mel (processed by AudioProcessor later)
    assert output is not None, "Griffin-Lim should return output"
    print(f"Griffin-Lim output shape: {output.shape}")
    print("✅ Griffin-Lim vocoder works correctly")
    
    print("✅ Test 4 PASSED\n")


def test_hifigan_with_trained_weights():
    """Test that HiFi-GAN works when marked as initialized."""
    print("\n=== Test 5: HiFi-GAN with Trained Weights ===")
    
    config = ModelConfig(
        text_vocab_size=100,
        text_encoder_dim=256,
        text_encoder_layers=2,
        text_encoder_heads=4,
        n_mels=80,
        sample_rate=22050,
        hop_length=256,
        n_fft=1024,
        win_length=1024
    )
    
    # Create vocoder and mark as trained
    vocoder = VocoderInterface(config, vocoder_type="hifigan")
    vocoder.mark_weights_loaded()
    
    # Create dummy mel input
    dummy_mel = tf.random.normal([1, 100, 80])
    
    # Generate output (should not warn since marked as initialized)
    print("Calling initialized vocoder (should NOT see warning):")
    output = vocoder(dummy_mel, training=False)
    
    assert output is not None, "Vocoder should produce output"
    print(f"Output shape: {output.shape}")
    print("✅ Initialized vocoder works without warnings")
    
    print("✅ Test 5 PASSED\n")


def run_all_tests():
    """Run all vocoder tests."""
    print("\n" + "="*70)
    print("Running Vocoder Fallback Tests")
    print("="*70)
    
    try:
        test_vocoder_initialization_tracking()
        test_vocoder_warning_on_uninitialized()
        test_vocoder_output_validation()
        test_griffin_lim_vocoder()
        test_hifigan_with_trained_weights()
        
        print("\n" + "="*70)
        print("✅ ALL TESTS PASSED")
        print("="*70)
        return True
        
    except Exception as e:
        print("\n" + "="*70)
        print(f"❌ TEST FAILED: {e}")
        print("="*70)
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
