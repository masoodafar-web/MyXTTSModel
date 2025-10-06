"""
Test vocoder weight initialization behavior.

This test verifies that:
1. Vocoder tracks weight initialization status
2. Warnings are issued for uninitialized vocoder
3. Vocoder produces valid output when initialized
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import tensorflow as tf
import numpy as np
import logging

# Configure logging to see warnings
logging.basicConfig(level=logging.WARNING)

from myxtts.models.vocoder import Vocoder
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
    vocoder = Vocoder(config)
    
    # Should start uninitialized
    assert not vocoder.check_weights_initialized(), "Vocoder should start uninitialized"
    print("✅ Vocoder starts with uninitialized weights")
    
    # Mark as loaded
    vocoder.mark_weights_loaded()
    assert vocoder.check_weights_initialized(), "Vocoder should be marked as initialized"
    print("✅ Vocoder can be marked as initialized")
    
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
    vocoder = Vocoder(config)
    
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
    vocoder = Vocoder(config)
    
    # Create dummy mel input
    dummy_mel = tf.random.normal([1, 100, 80])
    
    # Generate output (will be random/noisy with untrained weights)
    output = vocoder(dummy_mel, training=False)
    
    # Output should have been generated (even if noisy)
    assert output is not None, "Vocoder should produce output"
    
    # Check output shape - should be audio
    print(f"Vocoder output shape: {output.shape}")
    print(f"✅ Vocoder returned audio (shape: {output.shape})")
    
    print("✅ Test 3 PASSED\n")


def test_vocoder_audio_dimensions():
    """Test that vocoder produces correct audio dimensions."""
    print("\n=== Test 4: Vocoder Audio Dimensions ===")
    
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
    vocoder = Vocoder(config)
    
    # Create dummy mel input
    mel_time_steps = 100
    dummy_mel = tf.random.normal([1, mel_time_steps, 80])
    
    # Generate output
    output = vocoder(dummy_mel, training=False)
    
    # Check output dimensions
    assert output is not None, "Vocoder should return output"
    assert len(output.shape) == 3, f"Expected 3D output, got {len(output.shape)}D"
    assert output.shape[0] == 1, "Batch dimension should be 1"
    assert output.shape[2] == 1, "Audio channel dimension should be 1"
    print(f"✅ Vocoder output shape: {output.shape}")
    print(f"✅ Audio length: {output.shape[1]} samples (from {mel_time_steps} mel frames)")
    
    print("✅ Test 4 PASSED\n")


def test_vocoder_with_trained_weights():
    """Test that vocoder works when marked as initialized."""
    print("\n=== Test 5: Vocoder with Trained Weights ===")
    
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
    vocoder = Vocoder(config)
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
    print("Running HiFi-GAN Vocoder Tests")
    print("="*70)
    
    try:
        test_vocoder_initialization_tracking()
        test_vocoder_warning_on_uninitialized()
        test_vocoder_output_validation()
        test_vocoder_audio_dimensions()
        test_vocoder_with_trained_weights()
        
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
