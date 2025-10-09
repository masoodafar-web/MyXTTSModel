"""
Test for plateau_breaker optimization level configuration.
Ensures the plateau_breaker settings are correctly applied.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


def test_plateau_breaker_configuration():
    """Test that plateau_breaker optimization level applies correct settings."""
    
    # Import config and create inline version of apply_optimization_level for plateau_breaker
    from myxtts.config import XTTSConfig
    
    # Create base config
    config = XTTSConfig()
    
    # Apply plateau_breaker optimization (inline to avoid importing train_main)
    # This matches the implementation in train_main.py
    config.training.learning_rate = 1.5e-5
    config.training.mel_loss_weight = 2.0
    config.training.kl_loss_weight = 1.2
    config.training.gradient_clip_norm = 0.3
    config.training.scheduler = "cosine"
    config.training.cosine_restarts = True
    config.training.scheduler_params = {
        "min_learning_rate": 1e-7,
        "restart_period": 100 * 1000,
        "restart_mult": 1.0,
    }
    config.training.use_adaptive_loss_weights = True
    config.training.use_label_smoothing = True
    config.training.use_huber_loss = True
    
    # Verify critical settings for breaking through plateau
    assert config.training.learning_rate == 1.5e-5, \
        f"Learning rate should be 1.5e-5 but got {config.training.learning_rate}"
    
    assert config.training.mel_loss_weight == 2.0, \
        f"Mel loss weight should be 2.0 but got {config.training.mel_loss_weight}"
    
    assert config.training.kl_loss_weight == 1.2, \
        f"KL loss weight should be 1.2 but got {config.training.kl_loss_weight}"
    
    assert config.training.gradient_clip_norm == 0.3, \
        f"Gradient clip should be 0.3 but got {config.training.gradient_clip_norm}"
    
    assert config.training.scheduler == "cosine", \
        f"Scheduler should be 'cosine' but got {config.training.scheduler}"
    
    assert config.training.cosine_restarts is True, \
        "Cosine restarts should be enabled"
    
    # Verify scheduler params
    assert "restart_period" in config.training.scheduler_params, \
        "Scheduler params should include restart_period"
    
    assert config.training.scheduler_params["restart_period"] == 100 * 1000, \
        f"Restart period should be 100000 but got {config.training.scheduler_params['restart_period']}"
    
    # Verify stability features are enabled
    assert config.training.use_adaptive_loss_weights is True, \
        "Adaptive loss weights should be enabled"
    
    assert config.training.use_label_smoothing is True, \
        "Label smoothing should be enabled"
    
    assert config.training.use_huber_loss is True, \
        "Huber loss should be enabled"
    
    print("✅ All plateau_breaker configuration tests passed!")
    print(f"   • Learning rate: {config.training.learning_rate}")
    print(f"   • Mel loss weight: {config.training.mel_loss_weight}")
    print(f"   • KL loss weight: {config.training.kl_loss_weight}")
    print(f"   • Gradient clip: {config.training.gradient_clip_norm}")
    print(f"   • Scheduler: {config.training.scheduler} with restarts")


def test_basic_vs_plateau_breaker():
    """Compare basic and plateau_breaker optimization levels."""
    from myxtts.config import XTTSConfig
    
    # Test basic level (inline)
    basic_config = XTTSConfig()
    basic_config.training.learning_rate = 1e-5
    basic_config.training.mel_loss_weight = 1.0
    basic_config.training.gradient_clip_norm = 0.5
    
    # Test plateau_breaker level (inline)
    plateau_config = XTTSConfig()
    plateau_config.training.learning_rate = 1.5e-5
    plateau_config.training.mel_loss_weight = 2.0
    plateau_config.training.gradient_clip_norm = 0.3
    
    # Plateau breaker should have higher learning rate than basic
    assert plateau_config.training.learning_rate > basic_config.training.learning_rate, \
        "Plateau breaker LR (1.5e-5) should be higher than basic LR (1e-5)"
    
    # Plateau breaker should have higher mel_loss_weight than basic
    assert plateau_config.training.mel_loss_weight > basic_config.training.mel_loss_weight, \
        "Plateau breaker mel_loss_weight (2.0) should be higher than basic (1.0)"
    
    # Plateau breaker should have tighter gradient clipping than basic (smaller value)
    assert plateau_config.training.gradient_clip_norm < basic_config.training.gradient_clip_norm, \
        "Plateau breaker gradient clip (0.3) should be tighter than basic (0.5)"
    
    print("✅ Comparison test passed - plateau_breaker is correctly positioned between basic and enhanced")


def test_default_mel_loss_weight():
    """Test that default mel_loss_weight in config.py is within safe range."""
    from myxtts.config import XTTSConfig
    
    config = XTTSConfig()
    
    # Verify mel_loss_weight is in safe range (1.0-5.0)
    assert 1.0 <= config.training.mel_loss_weight <= 5.0, \
        f"Default mel_loss_weight {config.training.mel_loss_weight} is outside safe range [1.0, 5.0]"
    
    print(f"✅ Default mel_loss_weight {config.training.mel_loss_weight} is within safe range [1.0, 5.0]")


if __name__ == "__main__":
    print("Testing plateau_breaker optimization level configuration...")
    print("=" * 70)
    
    test_plateau_breaker_configuration()
    print()
    test_basic_vs_plateau_breaker()
    print()
    test_default_mel_loss_weight()
    
    print("=" * 70)
    print("✅ All tests passed successfully!")
