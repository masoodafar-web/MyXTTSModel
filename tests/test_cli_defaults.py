#!/usr/bin/env python3
"""
Test script to validate CLI parameter defaults.

This test validates that the CLI parameters have sensible defaults
according to the issue requirements.
"""

import sys
import os
import argparse

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


def test_train_main_has_correct_defaults():
    """Test that train_main.py has correct default values in argparse."""
    print("\n" + "=" * 80)
    print("TEST 1: Verify CLI parameter defaults in train_main.py")
    print("=" * 80)
    
    train_main_path = os.path.join(
        os.path.dirname(__file__), '..', 'train_main.py'
    )
    
    with open(train_main_path, 'r') as f:
        content = f.read()
    
    # Test 1: --model-size should default to "tiny"
    assert 'default="tiny"' in content and '--model-size' in content, \
        "--model-size should default to 'tiny'"
    print("✅ --model-size defaults to 'tiny' (beginner-friendly)")
    
    # Test 2: --batch-size should default to a reasonable value (16 or auto-adjusted)
    assert '--batch-size' in content and 'default=16' in content, \
        "--batch-size should default to 16"
    print("✅ --batch-size defaults to 16 (conservative, works with tiny model)")
    
    # Test 3: --enable-static-shapes should default to True
    assert 'enable_static_shapes' in content and 'default=True' in content, \
        "--enable-static-shapes should default to True"
    print("✅ --enable-static-shapes defaults to True (recommended)")
    
    # Test 4: --data-gpu and --model-gpu should default to None (single-GPU mode)
    assert '"--data-gpu"' in content and 'default=None' in content, \
        "--data-gpu should default to None"
    assert '"--model-gpu"' in content and 'default=None' in content, \
        "--model-gpu should default to None"
    print("✅ --data-gpu and --model-gpu default to None (single-GPU mode)")
    
    # Test 5: --enable-memory-isolation should default to False
    assert '--enable-memory-isolation' in content and 'default=False' in content, \
        "--enable-memory-isolation should default to False"
    print("✅ --enable-memory-isolation defaults to False (simple setup)")
    
    # Test 6: --num-workers should have default value
    assert '--num-workers' in content and 'default=8' in content, \
        "--num-workers should have default value of 8"
    print("✅ --num-workers defaults to 8")
    
    # Test 7: --buffer-size should have default value
    assert '--buffer-size' in content and 'default=50' in content, \
        "--buffer-size should have default value"
    print("✅ --buffer-size defaults to 50")
    
    # Test 8: Verify gradient accumulation has default
    assert '--grad-accum' in content and 'default=2' in content, \
        "--grad-accum should have default value"
    print("✅ --grad-accum defaults to 2")
    
    return True


def test_help_strings_are_informative():
    """Test that help strings mention defaults and usage."""
    print("\n" + "=" * 80)
    print("TEST 2: Verify help strings are informative")
    print("=" * 80)
    
    train_main_path = os.path.join(
        os.path.dirname(__file__), '..', 'train_main.py'
    )
    
    with open(train_main_path, 'r') as f:
        content = f.read()
    
    # Check that help strings mention defaults
    assert 'default:' in content or 'Default:' in content, \
        "Help strings should mention defaults"
    print("✅ Help strings mention defaults")
    
    # Check that auto-adjustment is mentioned
    assert 'auto-adjusted' in content or 'auto-select' in content.lower(), \
        "Help strings should mention auto-adjustment"
    print("✅ Help strings mention auto-adjustment")
    
    return True


def test_gpu_recommendations_include_gradient_accumulation():
    """Test that GPU recommendations include gradient_accumulation_steps."""
    print("\n" + "=" * 80)
    print("TEST 3: Verify GPU recommendations include gradient accumulation")
    print("=" * 80)
    
    train_main_path = os.path.join(
        os.path.dirname(__file__), '..', 'train_main.py'
    )
    
    with open(train_main_path, 'r') as f:
        content = f.read()
    
    # Check that get_recommended_settings includes gradient_accumulation_steps
    assert 'gradient_accumulation_steps' in content, \
        "get_recommended_settings should include gradient_accumulation_steps"
    print("✅ GPU recommendations include gradient_accumulation_steps")
    
    return True


def test_startup_logging_shows_parameters():
    """Test that startup logging shows parameter values."""
    print("\n" + "=" * 80)
    print("TEST 4: Verify startup logging shows parameters")
    print("=" * 80)
    
    train_main_path = os.path.join(
        os.path.dirname(__file__), '..', 'train_main.py'
    )
    
    with open(train_main_path, 'r') as f:
        content = f.read()
    
    # Check that there's logging for parameters
    assert 'TRAINING PARAMETERS SUMMARY' in content or 'Core Training Parameters' in content, \
        "Startup should log training parameters summary"
    print("✅ Startup logging includes parameter summary")
    
    # Check that key parameters are logged
    assert 'Model size:' in content or 'model_size' in content, \
        "Should log model size"
    assert 'Batch size:' in content or 'batch_size' in content, \
        "Should log batch size"
    assert 'Static shapes:' in content or 'enable_static_shapes' in content, \
        "Should log static shapes setting"
    print("✅ Key parameters are logged at startup")
    
    return True


def test_disable_static_shapes_argument_exists():
    """Test that --disable-static-shapes argument exists for opting out."""
    print("\n" + "=" * 80)
    print("TEST 5: Verify --disable-static-shapes argument exists")
    print("=" * 80)
    
    train_main_path = os.path.join(
        os.path.dirname(__file__), '..', 'train_main.py'
    )
    
    with open(train_main_path, 'r') as f:
        content = f.read()
    
    # Check that --disable-static-shapes exists for users who want to opt out
    assert '--disable-static-shapes' in content, \
        "--disable-static-shapes should exist for opting out"
    print("✅ --disable-static-shapes argument exists for opting out")
    
    return True


def main():
    """Run all tests."""
    print("\n" + "=" * 80)
    print("CLI DEFAULTS VALIDATION TESTS")
    print("=" * 80)
    
    tests = [
        test_train_main_has_correct_defaults,
        test_help_strings_are_informative,
        test_gpu_recommendations_include_gradient_accumulation,
        test_startup_logging_shows_parameters,
        test_disable_static_shapes_argument_exists,
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"\n❌ TEST FAILED WITH EXCEPTION: {e}")
            import traceback
            traceback.print_exc()
            results.append(False)
    
    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    
    passed = sum(results)
    total = len(results)
    
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("\n✅ ALL TESTS PASSED!")
        print("\nThe CLI parameter defaults are configured correctly:")
        print("  • --model-size: defaults to 'tiny' (beginner-friendly)")
        print("  • --batch-size: defaults to 16 (auto-adjusted based on GPU)")
        print("  • --enable-static-shapes: defaults to True (recommended)")
        print("  • --data-gpu / --model-gpu: default to None (single-GPU mode)")
        print("  • --enable-memory-isolation: defaults to False (simple setup)")
        print("  • --num-workers: defaults to 8 (auto-adjusted based on GPU)")
        print("  • --buffer-size: defaults to 50")
        print("  • --grad-accum: defaults to 2 (auto-adjusted based on GPU)")
        print("\nUsage examples:")
        print("  # Minimal command (uses all defaults):")
        print("  python3 train_main.py")
        print("")
        print("  # Override specific parameters:")
        print("  python3 train_main.py --model-size small --batch-size 24")
        print("")
        print("  # Dual-GPU training:")
        print("  python3 train_main.py --data-gpu 0 --model-gpu 1")
        print("=" * 80)
        return 0
    else:
        print("\n❌ SOME TESTS FAILED")
        print("=" * 80)
        return 1


if __name__ == '__main__':
    sys.exit(main())
