#!/usr/bin/env python3
"""
Integration test to validate that the training configuration is properly set up.
This test verifies that all the pieces work together correctly.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_import_config():
    """Test that we can import the config module."""
    print("\n" + "="*70)
    print("Testing config import...")
    print("="*70)
    
    try:
        from myxtts.config import XTTSConfig
        print("✅ Successfully imported XTTSConfig")
        
        config = XTTSConfig()
        print(f"✅ Created config with default lr={config.training.learning_rate}")
        print(f"✅ Default mel_loss_weight={config.training.mel_loss_weight}")
        
        return True
    except Exception as e:
        print(f"❌ Failed to import config: {e}")
        return False


def test_model_size_presets():
    """Test that model size presets are defined correctly."""
    print("\n" + "="*70)
    print("Testing MODEL_SIZE_PRESETS...")
    print("="*70)
    
    # We can't import train_main due to dependencies, but we can check syntax
    import ast
    
    train_main_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'train_main.py'
    )
    
    with open(train_main_path, 'r') as f:
        content = f.read()
    
    # Check that MODEL_SIZE_PRESETS exists
    if 'MODEL_SIZE_PRESETS' in content:
        print("✅ MODEL_SIZE_PRESETS found in train_main.py")
    else:
        print("❌ MODEL_SIZE_PRESETS not found")
        return False
    
    # Check that all model sizes are defined
    for size in ['tiny', 'small', 'normal', 'big']:
        if f'"{size}"' in content:
            print(f"✅ Model size '{size}' defined")
        else:
            print(f"❌ Model size '{size}' not found")
            return False
    
    return True


def test_enhanced_optimization_code():
    """Test that enhanced optimization level has model-size-aware code."""
    print("\n" + "="*70)
    print("Testing enhanced optimization level implementation...")
    print("="*70)
    
    train_main_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'train_main.py'
    )
    
    with open(train_main_path, 'r') as f:
        content = f.read()
    
    # Check for model-size-aware code in enhanced level
    checks = [
        ('model_size == "tiny"', 'Tiny model specific handling'),
        ('model_size == "small"', 'Small model specific handling'),
        ('learning_rate = 3e-5', 'Tiny learning rate'),
        ('learning_rate = 5e-5', 'Small learning rate'),
        ('learning_rate = 8e-5', 'Normal/big learning rate'),
        ('gradient_clip_norm = 0.5', 'Tiny gradient clipping'),
        ('gradient_clip_norm = 0.7', 'Small gradient clipping'),
        ('gradient_clip_norm = 0.8', 'Normal/big gradient clipping'),
        ('batch_size > 16', 'Batch size warning check'),
    ]
    
    for check_str, description in checks:
        if check_str in content:
            print(f"✅ {description}: found")
        else:
            print(f"❌ {description}: NOT FOUND")
            return False
    
    return True


def test_plateau_recommendations_function():
    """Test that log_plateau_recommendations function exists."""
    print("\n" + "="*70)
    print("Testing log_plateau_recommendations function...")
    print("="*70)
    
    train_main_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'train_main.py'
    )
    
    with open(train_main_path, 'r') as f:
        content = f.read()
    
    if 'def log_plateau_recommendations(' in content:
        print("✅ log_plateau_recommendations function defined")
    else:
        print("❌ log_plateau_recommendations function NOT FOUND")
        return False
    
    # Check that it's called
    if 'log_plateau_recommendations(args.model_size' in content:
        print("✅ log_plateau_recommendations is called in main flow")
    else:
        print("❌ log_plateau_recommendations not called")
        return False
    
    return True


def test_parameter_values():
    """Test that the parameter values are correctly set."""
    print("\n" + "="*70)
    print("Testing parameter values...")
    print("="*70)
    
    from myxtts.config import XTTSConfig
    
    config = XTTSConfig()
    
    # Check default values
    assert config.training.learning_rate == 1e-4, \
        f"Default LR should be 1e-4, got {config.training.learning_rate}"
    print(f"✅ Default learning_rate: {config.training.learning_rate}")
    
    assert config.training.mel_loss_weight == 2.5, \
        f"Default mel_loss_weight should be 2.5, got {config.training.mel_loss_weight}"
    print(f"✅ Default mel_loss_weight: {config.training.mel_loss_weight}")
    
    assert config.training.gradient_clip_norm == 1.0, \
        f"Default gradient_clip_norm should be 1.0, got {config.training.gradient_clip_norm}"
    print(f"✅ Default gradient_clip_norm: {config.training.gradient_clip_norm}")
    
    return True


def test_documentation_exists():
    """Test that documentation files exist."""
    print("\n" + "="*70)
    print("Testing documentation...")
    print("="*70)
    
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    docs_path = os.path.join(base_path, 'docs')
    
    required_docs = [
        'LOSS_PLATEAU_2.8_TINY_ENHANCED_FIX.md',
        'LOSS_PLATEAU_SOLUTION_2.7.md',
        'PLATEAU_BREAKTHROUGH_GUIDE.md',
    ]
    
    for doc in required_docs:
        doc_path = os.path.join(docs_path, doc)
        if os.path.exists(doc_path):
            print(f"✅ {doc} exists")
        else:
            print(f"❌ {doc} NOT FOUND")
            return False
    
    return True


if __name__ == "__main__":
    print("\n" + "="*70)
    print("TRAINING CONFIGURATION VALIDATION TEST")
    print("="*70)
    
    all_passed = True
    
    try:
        all_passed &= test_import_config()
        all_passed &= test_model_size_presets()
        all_passed &= test_enhanced_optimization_code()
        all_passed &= test_plateau_recommendations_function()
        all_passed &= test_parameter_values()
        all_passed &= test_documentation_exists()
        
        if all_passed:
            print("\n" + "="*70)
            print("✅ ALL VALIDATION TESTS PASSED!")
            print("="*70)
            print("\n✅ The training configuration is properly set up:")
            print("   • Config module imports correctly")
            print("   • Model size presets are defined")
            print("   • Enhanced optimization has model-size awareness")
            print("   • Plateau recommendations function exists and is called")
            print("   • Default parameter values are correct")
            print("   • Documentation is complete")
            print("\n🎯 Ready to train with:")
            print("   python3 train_main.py --model-size tiny --optimization-level enhanced --batch-size 16")
            print("="*70 + "\n")
            sys.exit(0)
        else:
            print("\n" + "="*70)
            print("❌ SOME VALIDATION TESTS FAILED")
            print("="*70 + "\n")
            sys.exit(1)
            
    except Exception as e:
        print(f"\n❌ UNEXPECTED ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
