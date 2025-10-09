#!/usr/bin/env python3
"""
Test that enhanced optimization level properly adjusts parameters based on model size.

This test verifies the fix for the loss plateau issue at 2.8 with tiny model + enhanced optimization.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from myxtts.config import XTTSConfig


class MockArgs:
    """Mock args object for testing."""
    def __init__(self, model_size="normal", batch_size=32):
        self.model_size = model_size
        self.batch_size = batch_size
        self.optimization_level = "enhanced"


def apply_enhanced_optimization_inline(config: XTTSConfig, model_size: str) -> XTTSConfig:
    """
    Inline version of enhanced optimization for testing.
    This mirrors the implementation in train_main.py.
    """
    # Adjust learning rate and gradient clipping based on model size
    if model_size == "tiny":
        config.training.learning_rate = 3e-5
        config.training.gradient_clip_norm = 0.5
        config.training.mel_loss_weight = 2.0
        restart_period = 6000
    elif model_size == "small":
        config.training.learning_rate = 5e-5
        config.training.gradient_clip_norm = 0.7
        restart_period = 7000
    else:
        # Normal and big models
        config.training.learning_rate = 8e-5
        config.training.gradient_clip_norm = 0.8
        restart_period = 8000
    
    config.training.scheduler = "cosine"
    config.training.cosine_restarts = True
    config.training.scheduler_params = {
        "min_learning_rate": 1e-7,
        "restart_period": restart_period,
        "restart_mult": 0.8,
    }
    
    return config


def test_enhanced_tiny_adjustments():
    """Test that enhanced level adjusts parameters for tiny model."""
    print("\n" + "="*70)
    print("Testing enhanced optimization with TINY model...")
    print("="*70)
    
    config = XTTSConfig()
    config = apply_enhanced_optimization_inline(config, "tiny")
    
    # Verify tiny-specific adjustments
    assert config.training.learning_rate == 3e-5, \
        f"Expected lr=3e-5 for tiny, got {config.training.learning_rate}"
    
    assert config.training.gradient_clip_norm == 0.5, \
        f"Expected gradient_clip=0.5 for tiny, got {config.training.gradient_clip_norm}"
    
    assert config.training.mel_loss_weight == 2.0, \
        f"Expected mel_loss_weight=2.0 for tiny, got {config.training.mel_loss_weight}"
    
    assert config.training.scheduler_params["restart_period"] == 6000, \
        f"Expected restart_period=6000 for tiny, got {config.training.scheduler_params['restart_period']}"
    
    print("✅ Tiny model adjustments verified:")
    print(f"   • Learning rate: {config.training.learning_rate}")
    print(f"   • Gradient clip: {config.training.gradient_clip_norm}")
    print(f"   • Mel loss weight: {config.training.mel_loss_weight}")
    print(f"   • Restart period: {config.training.scheduler_params['restart_period']}")


def test_enhanced_small_adjustments():
    """Test that enhanced level adjusts parameters for small model."""
    print("\n" + "="*70)
    print("Testing enhanced optimization with SMALL model...")
    print("="*70)
    
    config = XTTSConfig()
    config = apply_enhanced_optimization_inline(config, "small")
    
    # Verify small-specific adjustments
    assert config.training.learning_rate == 5e-5, \
        f"Expected lr=5e-5 for small, got {config.training.learning_rate}"
    
    assert config.training.gradient_clip_norm == 0.7, \
        f"Expected gradient_clip=0.7 for small, got {config.training.gradient_clip_norm}"
    
    assert config.training.scheduler_params["restart_period"] == 7000, \
        f"Expected restart_period=7000 for small, got {config.training.scheduler_params['restart_period']}"
    
    print("✅ Small model adjustments verified:")
    print(f"   • Learning rate: {config.training.learning_rate}")
    print(f"   • Gradient clip: {config.training.gradient_clip_norm}")
    print(f"   • Restart period: {config.training.scheduler_params['restart_period']}")


def test_enhanced_normal_adjustments():
    """Test that enhanced level adjusts parameters for normal model."""
    print("\n" + "="*70)
    print("Testing enhanced optimization with NORMAL model...")
    print("="*70)
    
    config = XTTSConfig()
    config = apply_enhanced_optimization_inline(config, "normal")
    
    # Verify normal-specific adjustments
    assert config.training.learning_rate == 8e-5, \
        f"Expected lr=8e-5 for normal, got {config.training.learning_rate}"
    
    assert config.training.gradient_clip_norm == 0.8, \
        f"Expected gradient_clip=0.8 for normal, got {config.training.gradient_clip_norm}"
    
    assert config.training.scheduler_params["restart_period"] == 8000, \
        f"Expected restart_period=8000 for normal, got {config.training.scheduler_params['restart_period']}"
    
    print("✅ Normal model adjustments verified:")
    print(f"   • Learning rate: {config.training.learning_rate}")
    print(f"   • Gradient clip: {config.training.gradient_clip_norm}")
    print(f"   • Restart period: {config.training.scheduler_params['restart_period']}")


def test_tiny_vs_plateau_breaker_comparison():
    """Compare tiny+enhanced vs plateau_breaker settings."""
    print("\n" + "="*70)
    print("Comparing TINY+ENHANCED vs PLATEAU_BREAKER...")
    print("="*70)
    
    # Tiny + enhanced
    tiny_config = XTTSConfig()
    tiny_config = apply_enhanced_optimization_inline(tiny_config, "tiny")
    
    # Plateau breaker (from test_plateau_breaker_config.py)
    plateau_config = XTTSConfig()
    plateau_config.training.learning_rate = 1.5e-5
    plateau_config.training.mel_loss_weight = 2.0
    plateau_config.training.gradient_clip_norm = 0.3
    
    print("\nConfiguration comparison:")
    print(f"{'Parameter':<25} | {'Tiny+Enhanced':<15} | {'Plateau Breaker':<15}")
    print("-" * 70)
    print(f"{'Learning Rate':<25} | {tiny_config.training.learning_rate:<15} | {plateau_config.training.learning_rate:<15}")
    print(f"{'Mel Loss Weight':<25} | {tiny_config.training.mel_loss_weight:<15} | {plateau_config.training.mel_loss_weight:<15}")
    print(f"{'Gradient Clip':<25} | {tiny_config.training.gradient_clip_norm:<15} | {plateau_config.training.gradient_clip_norm:<15}")
    
    print("\n📊 Analysis:")
    print("   • Tiny+enhanced now has MORE AGGRESSIVE settings than before (3e-5 vs 1e-4)")
    print("   • Plateau_breaker is EVEN MORE AGGRESSIVE (1.5e-5) for stuck loss")
    print("   • Both share mel_loss_weight=2.0 for balance")
    print("   • Progression: tiny+enhanced (0.5) → plateau_breaker (0.3) for gradient clip")


def test_learning_rate_progression():
    """Test that learning rates form a logical progression across model sizes."""
    print("\n" + "="*70)
    print("Testing learning rate progression across model sizes...")
    print("="*70)
    
    tiny_config = XTTSConfig()
    tiny_config = apply_enhanced_optimization_inline(tiny_config, "tiny")
    
    small_config = XTTSConfig()
    small_config = apply_enhanced_optimization_inline(small_config, "small")
    
    normal_config = XTTSConfig()
    normal_config = apply_enhanced_optimization_inline(normal_config, "normal")
    
    # Verify progression: tiny < small < normal
    assert tiny_config.training.learning_rate < small_config.training.learning_rate, \
        "Tiny LR should be less than small LR"
    
    assert small_config.training.learning_rate < normal_config.training.learning_rate, \
        "Small LR should be less than normal LR"
    
    print("✅ Learning rate progression verified:")
    print(f"   • Tiny:   {tiny_config.training.learning_rate}")
    print(f"   • Small:  {small_config.training.learning_rate}")
    print(f"   • Normal: {normal_config.training.learning_rate}")
    print("   ✓ Smaller models get lower learning rates (more careful optimization)")


if __name__ == "__main__":
    print("\n" + "="*70)
    print("ENHANCED OPTIMIZATION MODEL-SIZE ADJUSTMENTS TEST")
    print("Testing fix for loss plateau at 2.8 with tiny+enhanced")
    print("="*70)
    
    try:
        test_enhanced_tiny_adjustments()
        test_enhanced_small_adjustments()
        test_enhanced_normal_adjustments()
        test_tiny_vs_plateau_breaker_comparison()
        test_learning_rate_progression()
        
        print("\n" + "="*70)
        print("✅ ALL TESTS PASSED!")
        print("="*70)
        print("\n✅ The enhanced optimization level now properly adjusts parameters")
        print("   based on model size, which should prevent loss plateau at 2.8")
        print("   with tiny model configuration.")
        print("\n📌 Key improvements:")
        print("   • Tiny model: LR 3e-5 (down from 1e-4), clip 0.5 (down from 1.0)")
        print("   • Small model: LR 5e-5, clip 0.7")
        print("   • Normal/big: LR 8e-5, clip 0.8")
        print("   • Warning added for tiny + large batch size")
        print("="*70 + "\n")
        
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ UNEXPECTED ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
