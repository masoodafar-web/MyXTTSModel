#!/usr/bin/env python3
"""
Simple test for phonemizer error handling fixes (no dependencies).
"""

import sys
import os

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_phonemizer_code_changes():
    """Test that phonemizer code has been updated correctly."""
    print("🧪 Testing Phonemizer Code Changes...")
    
    # Read the text.py file
    with open(os.path.join(os.path.dirname(os.path.dirname(__file__)), 
                          'myxtts/utils/text.py'), 'r') as f:
        content = f.read()
    
    # Check for improved error handling
    checks = [
        ("Empty text check", "if not text or not text.strip():"),
        ("Empty result check", "if not phoneme_result or len(phoneme_result) == 0:"),
        ("IndexError handling", "except IndexError as e:"),
        ("Failure logging", "_log_phonemizer_failure"),
        ("Phonemizer stats", "get_phonemizer_stats"),
        ("Logging import", "import logging"),
    ]
    
    passed = 0
    for check_name, check_string in checks:
        if check_string in content:
            print(f"   ✓ {check_name}: Found")
            passed += 1
        else:
            print(f"   ✗ {check_name}: NOT FOUND")
    
    print(f"\n   {passed}/{len(checks)} checks passed")
    return passed == len(checks)


def test_loss_weight_code_changes():
    """Test that loss weight code has been updated correctly."""
    print("\n🧪 Testing Loss Weight Code Changes...")
    
    # Read the losses.py file
    with open(os.path.join(os.path.dirname(os.path.dirname(__file__)), 
                          'myxtts/training/losses.py'), 'r') as f:
        content = f.read()
    
    # Check for balanced loss weights
    checks = [
        ("Reduced default mel_loss_weight", "mel_loss_weight: float = 2.5"),
        ("Tighter min bound", "0.8,  # Minimum 80%"),
        ("Tighter max bound", "1.2   # Maximum 120%"),
        ("Hard limit clipping", "tf.clip_by_value(adaptive_weight, 1.0, 5.0)"),
        ("Safe range comment", "safe range: 1.0-5.0"),
    ]
    
    passed = 0
    for check_name, check_string in checks:
        if check_string in content:
            print(f"   ✓ {check_name}: Found")
            passed += 1
        else:
            print(f"   ✗ {check_name}: NOT FOUND")
    
    print(f"\n   {passed}/{len(checks)} checks passed")
    return passed == len(checks)


def test_config_changes():
    """Test that config.yaml has been updated correctly."""
    print("\n🧪 Testing Config Changes...")
    
    # Read the config.yaml file
    with open(os.path.join(os.path.dirname(os.path.dirname(__file__)), 
                          'configs/config.yaml'), 'r') as f:
        content = f.read()
    
    # Check for balanced loss weights
    checks = [
        ("Reduced mel_loss_weight", "mel_loss_weight: 2.5"),
        ("Safe range comment", "Safe range per LOSS_FIX_GUIDE.md"),
    ]
    
    passed = 0
    for check_name, check_string in checks:
        if check_string in content:
            print(f"   ✓ {check_name}: Found")
            passed += 1
        else:
            print(f"   ✗ {check_name}: NOT FOUND")
    
    print(f"\n   {passed}/{len(checks)} checks passed")
    return passed == len(checks)


def test_train_main_changes():
    """Test that train_main.py has been updated correctly."""
    print("\n🧪 Testing train_main.py Changes...")
    
    # Read the train_main.py file
    with open(os.path.join(os.path.dirname(os.path.dirname(__file__)), 
                          'train_main.py'), 'r') as f:
        content = f.read()
    
    # Check for balanced loss weights
    checks = [
        ("Reduced mel_loss_weight", "mel_loss_weight=2.5"),
        ("Reduced kl_loss_weight", "kl_loss_weight=1.0"),
        ("Safe range comment", "Safe range per LOSS_FIX_GUIDE.md"),
        ("Loss below 8 comment", "loss < 8"),
    ]
    
    passed = 0
    for check_name, check_string in checks:
        if check_string in content:
            print(f"   ✓ {check_name}: Found")
            passed += 1
        else:
            print(f"   ✗ {check_name}: NOT FOUND")
    
    print(f"\n   {passed}/{len(checks)} checks passed")
    return passed == len(checks)


def main():
    """Run all tests."""
    print("=" * 70)
    print("PHONEMIZER AND LOSS WEIGHT FIX CODE VALIDATION")
    print("=" * 70)
    
    tests = [
        ("Phonemizer Code Changes", test_phonemizer_code_changes),
        ("Loss Weight Code Changes", test_loss_weight_code_changes),
        ("Config Changes", test_config_changes),
        ("train_main.py Changes", test_train_main_changes),
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
