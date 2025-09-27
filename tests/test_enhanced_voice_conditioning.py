#!/usr/bin/env python3
"""
Test script for enhanced voice conditioning implementation.

This script validates that the enhanced voice conditioning components
can be imported and initialized correctly without requiring TensorFlow.
"""

import sys
import os

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def test_imports():
    """Test that all new components can be imported."""
    print("Testing imports...")
    
    try:
        # Test config imports
        from myxtts.config.config import ModelConfig
        print("✅ ModelConfig imported successfully")
        
        # Test that we can create a config instance
        config = ModelConfig()
        print("✅ ModelConfig instance created successfully")
        
        # Test new configuration parameters
        assert hasattr(config, 'use_pretrained_speaker_encoder'), "Missing use_pretrained_speaker_encoder"
        assert hasattr(config, 'speaker_encoder_type'), "Missing speaker_encoder_type"
        assert hasattr(config, 'contrastive_loss_temperature'), "Missing contrastive_loss_temperature"
        assert hasattr(config, 'contrastive_loss_margin'), "Missing contrastive_loss_margin"
        print("✅ New configuration parameters are present")
        
        # Test default values
        assert config.use_pretrained_speaker_encoder == False, "Wrong default for use_pretrained_speaker_encoder"
        assert config.speaker_encoder_type == "ecapa_tdnn", "Wrong default for speaker_encoder_type"
        assert config.contrastive_loss_temperature == 0.1, "Wrong default for contrastive_loss_temperature"
        assert config.contrastive_loss_margin == 0.2, "Wrong default for contrastive_loss_margin"
        print("✅ Configuration defaults are correct")
        
        print("✅ All imports and configurations tested successfully!")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def test_syntax():
    """Test that all new Python files have valid syntax."""
    print("\nTesting syntax...")
    
    import py_compile
    
    files_to_test = [
        'myxtts/models/speaker_encoder.py',
        'myxtts/models/xtts.py',
        'inference_main.py',
        'train_main.py'
    ]
    
    all_valid = True
    for file_path in files_to_test:
        try:
            py_compile.compile(file_path, doraise=True)
            print(f"✅ {file_path} - syntax valid")
        except py_compile.PyCompileError as e:
            print(f"❌ {file_path} - syntax error: {e}")
            all_valid = False
        except FileNotFoundError:
            print(f"⚠️ {file_path} - file not found")
            all_valid = False
    
    if all_valid:
        print("✅ All files have valid syntax!")
    
    return all_valid


def test_configuration_consistency():
    """Test that train_main.py and inference_main.py have consistent configuration."""
    print("\nTesting configuration consistency...")
    
    try:
        # Check that train_main.py contains enhanced voice conditioning settings
        with open('train_main.py', 'r') as f:
            train_content = f.read()
        
        required_in_train = [
            'use_pretrained_speaker_encoder',
            'speaker_encoder_type',
            'contrastive_loss_temperature',
            'contrastive_loss_margin',
            'voice_similarity_loss_weight'
        ]
        
        for setting in required_in_train:
            if setting in train_content:
                print(f"✅ train_main.py contains {setting}")
            else:
                print(f"❌ train_main.py missing {setting}")
                return False
        
        # Check that inference_main.py contains command line arguments
        with open('inference_main.py', 'r') as f:
            inference_content = f.read()
        
        required_in_inference = [
            '--use-pretrained-speaker-encoder',
            '--speaker-encoder-type',
            '--voice-conditioning-strength',
            '--voice-cloning-temperature'
        ]
        
        for arg in required_in_inference:
            if arg in inference_content:
                print(f"✅ inference_main.py contains {arg}")
            else:
                print(f"❌ inference_main.py missing {arg}")
                return False
        
        print("✅ Configuration consistency validated!")
        return True
        
    except Exception as e:
        print(f"❌ Error testing configuration consistency: {e}")
        return False


def main():
    """Run all tests."""
    print("Enhanced Voice Conditioning Implementation Test")
    print("=" * 50)
    
    all_tests_passed = True
    
    # Run tests
    all_tests_passed &= test_imports()
    all_tests_passed &= test_syntax()
    all_tests_passed &= test_configuration_consistency()
    
    print("\n" + "=" * 50)
    if all_tests_passed:
        print("🎉 All tests passed! Enhanced voice conditioning implementation is ready.")
        print("\nTo enable enhanced voice conditioning:")
        print("1. In training: Set use_pretrained_speaker_encoder=True in train_main.py")
        print("2. In inference: Use --use-pretrained-speaker-encoder flag")
        print("3. Choose encoder type with --speaker-encoder-type [ecapa_tdnn|resemblyzer|coqui]")
    else:
        print("❌ Some tests failed. Please check the implementation.")
        sys.exit(1)


if __name__ == "__main__":
    main()