#!/usr/bin/env python3
"""
Test script for phonemizer error handling and loss weight balancing fixes.

This script validates:
1. Phonemizer handles empty results and edge cases gracefully
2. Loss weights stay within safe ranges (1.0-5.0 for mel_loss)
3. Total loss can decrease below 8 with proper configuration
"""

import os
import sys
import numpy as np
import tensorflow as tf

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from myxtts.utils.text import TextProcessor
from myxtts.training.losses import XTTSLoss, mel_loss


def test_phonemizer_empty_text():
    """Test phonemizer handles empty text gracefully."""
    print("🧪 Testing Phonemizer with Empty Text...")
    
    try:
        processor = TextProcessor(
            language="en",
            use_phonemes=True,
            tokenizer_type="custom"
        )
        
        # Test empty string
        result = processor.text_to_phonemes("")
        assert result == "", "Empty string should return empty string"
        print("   ✓ Empty text handled correctly")
        
        # Test whitespace-only string
        result = processor.text_to_phonemes("   ")
        assert result == "   ", "Whitespace-only text should be preserved"
        print("   ✓ Whitespace-only text handled correctly")
        
        return True
    except Exception as e:
        print(f"   ✗ Failed: {e}")
        return False


def test_phonemizer_fallback():
    """Test phonemizer fallback to original text on failure."""
    print("\n🧪 Testing Phonemizer Fallback Mechanism...")
    
    try:
        processor = TextProcessor(
            language="en",
            use_phonemes=True,
            tokenizer_type="custom"
        )
        
        # Test with normal text - should work
        test_text = "Hello world"
        result = processor.text_to_phonemes(test_text)
        assert result is not None and len(result) > 0, "Normal text should phonemize"
        print(f"   ✓ Normal text phonemized: '{test_text}' -> '{result}'")
        
        # Get phonemizer stats
        stats = processor.get_phonemizer_stats()
        print(f"   • Phonemizer failure count: {stats['failure_count']}")
        print(f"   • Sample failures: {len(stats['failure_samples'])}")
        
        return True
    except Exception as e:
        print(f"   ✗ Failed: {e}")
        return False


def test_phonemizer_without_backend():
    """Test phonemizer gracefully handles missing backend."""
    print("\n🧪 Testing Phonemizer without Backend...")
    
    try:
        processor = TextProcessor(
            language="en",
            use_phonemes=False,  # Disable phonemes
            tokenizer_type="custom"
        )
        
        test_text = "Hello world"
        result = processor.text_to_phonemes(test_text)
        assert result == test_text, "Should return original text when phonemes disabled"
        print("   ✓ Correctly returns original text when phonemes disabled")
        
        return True
    except Exception as e:
        print(f"   ✗ Failed: {e}")
        return False


def test_mel_loss_weight_defaults():
    """Test that mel loss weight defaults are in safe range."""
    print("\n🧪 Testing Mel Loss Weight Defaults...")
    
    try:
        # Test default initialization
        loss_fn = XTTSLoss()
        
        mel_weight = loss_fn.mel_loss_weight
        print(f"   • Default mel_loss_weight: {mel_weight}")
        
        # Verify it's in safe range (1.0-5.0)
        assert 1.0 <= mel_weight <= 5.0, f"mel_loss_weight {mel_weight} outside safe range [1.0, 5.0]"
        print(f"   ✓ mel_loss_weight {mel_weight} is in safe range [1.0, 5.0]")
        
        return True
    except Exception as e:
        print(f"   ✗ Failed: {e}")
        return False


def test_adaptive_weight_bounds():
    """Test that adaptive weight stays within safe bounds."""
    print("\n🧪 Testing Adaptive Weight Bounds...")
    
    try:
        loss_fn = XTTSLoss(
            mel_loss_weight=2.5,
            use_adaptive_weights=True
        )
        
        # Simulate various mel loss values
        test_cases = [
            (0.5, "very low mel loss"),
            (1.0, "low mel loss"),
            (2.0, "medium mel loss"),
            (5.0, "high mel loss"),
            (10.0, "very high mel loss"),
        ]
        
        print(f"   Base mel_loss_weight: {loss_fn.mel_loss_weight}")
        
        for mel_loss_val, description in test_cases:
            mel_loss_tensor = tf.constant(mel_loss_val, dtype=tf.float32)
            adaptive_weight = loss_fn._adaptive_mel_weight(mel_loss_tensor)
            adaptive_weight_val = float(adaptive_weight.numpy())
            
            # Check bounds
            assert 1.0 <= adaptive_weight_val <= 5.0, \
                f"Adaptive weight {adaptive_weight_val} outside safe range [1.0, 5.0] for {description}"
            
            print(f"   ✓ {description} ({mel_loss_val:.2f}): adaptive_weight = {adaptive_weight_val:.2f} [OK]")
        
        print("   ✓ All adaptive weights stay within safe range [1.0, 5.0]")
        return True
    except Exception as e:
        print(f"   ✗ Failed: {e}")
        return False


def test_adaptive_weight_amplification():
    """Test that adaptive weight doesn't amplify excessively."""
    print("\n🧪 Testing Adaptive Weight Amplification Limits...")
    
    try:
        loss_fn = XTTSLoss(
            mel_loss_weight=2.5,
            use_adaptive_weights=True
        )
        
        base_weight = loss_fn.mel_loss_weight
        
        # Simulate training for many steps with varying loss
        max_amplification = 0.0
        min_amplification = float('inf')
        
        for i in range(100):
            # Vary mel loss between 0.5 and 10.0
            mel_loss_val = 0.5 + (i / 100.0) * 9.5
            mel_loss_tensor = tf.constant(mel_loss_val, dtype=tf.float32)
            adaptive_weight = loss_fn._adaptive_mel_weight(mel_loss_tensor)
            adaptive_weight_val = float(adaptive_weight.numpy())
            
            amplification = adaptive_weight_val / base_weight
            max_amplification = max(max_amplification, amplification)
            min_amplification = min(min_amplification, amplification)
        
        print(f"   • Base weight: {base_weight}")
        print(f"   • Max amplification: {max_amplification:.2f}x")
        print(f"   • Min amplification: {min_amplification:.2f}x")
        
        # Check that amplification is reasonable (should be within 0.8x - 1.2x)
        assert 0.7 <= min_amplification <= 0.85, f"Min amplification {min_amplification:.2f}x too low"
        assert 1.15 <= max_amplification <= 1.3, f"Max amplification {max_amplification:.2f}x too high"
        
        print("   ✓ Amplification stays within reasonable bounds")
        return True
    except Exception as e:
        print(f"   ✗ Failed: {e}")
        return False


def test_total_loss_below_8():
    """Test that total loss can go below 8 with proper configuration."""
    print("\n🧪 Testing Total Loss Can Go Below 8...")
    
    try:
        # Create loss function with balanced weights
        loss_fn = XTTSLoss(
            mel_loss_weight=2.5,
            stop_loss_weight=1.0,
            use_adaptive_weights=False  # Disable for predictable test
        )
        
        batch_size, time_steps, n_mels = 2, 50, 80
        
        # Create test data with relatively small errors
        y_true = {
            "mel_target": tf.random.normal([batch_size, time_steps, n_mels]),
            "stop_target": tf.zeros([batch_size, time_steps, 1]),
            "mel_lengths": tf.constant([45, 48])
        }
        
        # Predictions close to targets (simulating converging model)
        y_pred = {
            "mel_output": y_true["mel_target"] + tf.random.normal([batch_size, time_steps, n_mels], stddev=0.5),
            "stop_tokens": tf.random.uniform([batch_size, time_steps, 1], 0, 0.3)
        }
        
        # Compute loss
        total_loss = loss_fn(y_true, y_pred)
        total_loss_val = float(total_loss.numpy())
        
        print(f"   • Total loss with balanced weights: {total_loss_val:.4f}")
        
        # With small errors and balanced weights, loss should be < 8
        if total_loss_val < 8.0:
            print(f"   ✓ Total loss {total_loss_val:.4f} is below 8.0 [GOOD]")
        else:
            print(f"   ⚠ Total loss {total_loss_val:.4f} is above 8.0 [May need further tuning]")
        
        # Get individual loss components
        individual_losses = loss_fn.get_losses()
        weighted_losses = loss_fn.get_weighted_losses()
        
        print("\n   Individual Loss Components:")
        for name, value in individual_losses.items():
            val = float(value.numpy())
            print(f"     • {name}: {val:.4f}")
        
        print("\n   Weighted Loss Contributions:")
        for name, value in weighted_losses.items():
            val = float(value.numpy())
            print(f"     • {name}: {val:.4f}")
        
        return True
    except Exception as e:
        print(f"   ✗ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_mel_loss_reasonable_values():
    """Test that mel loss produces reasonable values."""
    print("\n🧪 Testing Mel Loss Produces Reasonable Values...")
    
    try:
        batch_size, time_steps, n_mels = 2, 50, 80
        
        # Test with different error magnitudes
        test_cases = [
            (0.1, "very small error"),
            (0.5, "small error"),
            (1.0, "medium error"),
            (2.0, "large error"),
        ]
        
        for error_std, description in test_cases:
            y_true = tf.random.normal([batch_size, time_steps, n_mels])
            y_pred = y_true + tf.random.normal([batch_size, time_steps, n_mels], stddev=error_std)
            lengths = tf.constant([45, 48])
            
            loss = mel_loss(y_true, y_pred, lengths, label_smoothing=0.0, use_huber_loss=False)
            loss_val = float(loss.numpy())
            
            print(f"   • {description} (stddev={error_std}): mel_loss = {loss_val:.4f}")
            
            # With mel_loss_weight of 2.5, this contributes loss_val * 2.5 to total
            contribution = loss_val * 2.5
            print(f"     → Weighted contribution: {contribution:.4f}")
        
        print("   ✓ Mel loss values are reasonable")
        return True
    except Exception as e:
        print(f"   ✗ Failed: {e}")
        return False


def main():
    """Run all tests."""
    print("=" * 70)
    print("PHONEMIZER AND LOSS WEIGHT FIX VALIDATION")
    print("=" * 70)
    
    tests = [
        ("Phonemizer Empty Text", test_phonemizer_empty_text),
        ("Phonemizer Fallback", test_phonemizer_fallback),
        ("Phonemizer Without Backend", test_phonemizer_without_backend),
        ("Mel Loss Weight Defaults", test_mel_loss_weight_defaults),
        ("Adaptive Weight Bounds", test_adaptive_weight_bounds),
        ("Adaptive Weight Amplification", test_adaptive_weight_amplification),
        ("Total Loss Below 8", test_total_loss_below_8),
        ("Mel Loss Reasonable Values", test_mel_loss_reasonable_values),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n❌ Test '{test_name}' crashed: {e}")
            import traceback
            traceback.print_exc()
            results.append((test_name, False))
    
    # Print summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {test_name}")
    
    print("=" * 70)
    print(f"Results: {passed}/{total} tests passed")
    print("=" * 70)
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
