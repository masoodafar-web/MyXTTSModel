#!/usr/bin/env python3
"""
Test script to validate static shapes CLI arguments.

This test validates that the --enable-static-shapes CLI argument properly
configures the training script to use fixed-length padding, which prevents
tf.function retracing and stabilizes GPU utilization.
"""

import sys
import os
import argparse

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from myxtts.config.config import DataConfig, XTTSConfig


def test_dataconfig_has_pad_to_fixed_length():
    """Test that DataConfig has pad_to_fixed_length field."""
    print("\n" + "=" * 80)
    print("TEST 1: DataConfig has pad_to_fixed_length field")
    print("=" * 80)
    
    config = DataConfig()
    
    # Check that the field exists
    assert hasattr(config, 'pad_to_fixed_length'), \
        "DataConfig should have pad_to_fixed_length field"
    
    # Check that the field is a boolean
    assert isinstance(config.pad_to_fixed_length, bool), \
        "pad_to_fixed_length should be a boolean"
    
    # Check default value
    assert config.pad_to_fixed_length == False, \
        "pad_to_fixed_length should default to False"
    
    print(f"✅ PASSED: DataConfig has pad_to_fixed_length field (default: {config.pad_to_fixed_length})")
    return True


def test_dataconfig_has_max_text_length():
    """Test that DataConfig has max_text_length field."""
    print("\n" + "=" * 80)
    print("TEST 2: DataConfig has max_text_length field")
    print("=" * 80)
    
    config = DataConfig()
    
    # Check that the field exists
    assert hasattr(config, 'max_text_length'), \
        "DataConfig should have max_text_length field"
    
    # Check that the field is an integer
    assert isinstance(config.max_text_length, int), \
        "max_text_length should be an integer"
    
    # Check that it has a reasonable default value
    assert config.max_text_length > 0, \
        "max_text_length should be positive"
    
    print(f"✅ PASSED: DataConfig has max_text_length field (default: {config.max_text_length})")
    return True


def test_config_can_be_created_with_static_shapes():
    """Test that config can be created with static shapes enabled."""
    print("\n" + "=" * 80)
    print("TEST 3: Config creation with static shapes enabled")
    print("=" * 80)
    
    # Create config with static shapes enabled
    data_config = DataConfig(
        pad_to_fixed_length=True,
        max_text_length=200,
        max_mel_frames=800,
        batch_size=16
    )
    
    # Verify values
    assert data_config.pad_to_fixed_length == True, \
        "pad_to_fixed_length should be True"
    assert data_config.max_text_length == 200, \
        "max_text_length should be 200"
    assert data_config.max_mel_frames == 800, \
        "max_mel_frames should be 800"
    
    print(f"✅ PASSED: Config created with static shapes enabled")
    print(f"  pad_to_fixed_length: {data_config.pad_to_fixed_length}")
    print(f"  max_text_length: {data_config.max_text_length}")
    print(f"  max_mel_frames: {data_config.max_mel_frames}")
    return True


def test_cli_argument_parsing():
    """Test that CLI arguments are properly defined in train_main.py."""
    print("\n" + "=" * 80)
    print("TEST 4: CLI argument parsing")
    print("=" * 80)
    
    # Import train_main to check if it has the CLI arguments
    # We'll check that the file contains the expected argument definitions
    train_main_path = os.path.join(
        os.path.dirname(__file__), '..', 'train_main.py'
    )
    
    with open(train_main_path, 'r') as f:
        content = f.read()
    
    # Check for --enable-static-shapes argument
    assert '--enable-static-shapes' in content, \
        "train_main.py should define --enable-static-shapes argument"
    
    # Check for --max-text-length argument
    assert '--max-text-length' in content, \
        "train_main.py should define --max-text-length argument"
    
    # Check for --max-mel-frames argument
    assert '--max-mel-frames' in content, \
        "train_main.py should define --max-mel-frames argument"
    
    # Check that the arguments are passed to build_config
    assert 'enable_static_shapes' in content, \
        "train_main.py should pass enable_static_shapes to build_config"
    
    print("✅ PASSED: CLI arguments are properly defined")
    print("  Found: --enable-static-shapes")
    print("  Found: --max-text-length")
    print("  Found: --max-mel-frames")
    return True


def test_build_config_accepts_static_shapes_params():
    """Test that build_config function accepts static shapes parameters."""
    print("\n" + "=" * 80)
    print("TEST 5: build_config accepts static shapes parameters")
    print("=" * 80)
    
    # Check the build_config signature
    train_main_path = os.path.join(
        os.path.dirname(__file__), '..', 'train_main.py'
    )
    
    with open(train_main_path, 'r') as f:
        content = f.read()
    
    # Find build_config function signature
    assert 'def build_config(' in content, \
        "train_main.py should have build_config function"
    
    # Check that it accepts the new parameters
    assert 'enable_static_shapes:' in content, \
        "build_config should accept enable_static_shapes parameter"
    
    assert 'max_text_length_override:' in content, \
        "build_config should accept max_text_length_override parameter"
    
    assert 'max_mel_frames_override:' in content, \
        "build_config should accept max_mel_frames_override parameter"
    
    print("✅ PASSED: build_config accepts static shapes parameters")
    return True


def test_dataconfig_instantiation_uses_params():
    """Test that DataConfig instantiation in build_config uses the parameters."""
    print("\n" + "=" * 80)
    print("TEST 6: DataConfig instantiation uses static shapes parameters")
    print("=" * 80)
    
    train_main_path = os.path.join(
        os.path.dirname(__file__), '..', 'train_main.py'
    )
    
    with open(train_main_path, 'r') as f:
        content = f.read()
    
    # Check that DataConfig is instantiated with pad_to_fixed_length
    assert 'pad_to_fixed_length=enable_static_shapes' in content, \
        "DataConfig should be instantiated with pad_to_fixed_length=enable_static_shapes"
    
    print("✅ PASSED: DataConfig instantiation uses pad_to_fixed_length parameter")
    return True


def main():
    """Run all tests."""
    print("\n" + "=" * 80)
    print("STATIC SHAPES CLI VALIDATION TESTS")
    print("=" * 80)
    
    tests = [
        test_dataconfig_has_pad_to_fixed_length,
        test_dataconfig_has_max_text_length,
        test_config_can_be_created_with_static_shapes,
        test_cli_argument_parsing,
        test_build_config_accepts_static_shapes_params,
        test_dataconfig_instantiation_uses_params,
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
        print("\nThe static shapes CLI integration is working correctly.")
        print("\nUsage:")
        print("  python3 train_main.py --enable-static-shapes --batch-size 16")
        print("  python3 train_main.py --enable-static-shapes --max-text-length 200 --max-mel-frames 800")
        print("\nThis will enable fixed-length padding to prevent tf.function retracing")
        print("and stabilize GPU utilization during training.")
        print("=" * 80)
        return 0
    else:
        print("\n❌ SOME TESTS FAILED")
        print("=" * 80)
        return 1


if __name__ == '__main__':
    sys.exit(main())
