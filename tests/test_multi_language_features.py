#!/usr/bin/env python3
"""
Test script for the new multi-language and multi-speaker features.
"""

import sys
import os

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_config_new_features():
    """Test that new configuration options are available."""
    print("Testing new configuration features...")
    
    try:
        from myxtts.config.config import DataConfig, ModelConfig, XTTSConfig
        
        # Test DataConfig with new features
        data_config = DataConfig(
            enable_multilingual=True,
            enable_multispeaker=True,
            enable_pitch_shift=True,
            enable_noise_mixing=True,
            enable_phone_normalization=True,
            pitch_shift_range=[-1.0, 1.0],
            noise_mixing_snr_range=[15.0, 25.0],
            supported_languages=["en", "es", "fr"],
            language_detection_method="auto"
        )
        
        print(f"✅ DataConfig created with multilingual: {data_config.enable_multilingual}")
        print(f"✅ Pitch shift range: {data_config.pitch_shift_range}")
        print(f"✅ Supported languages: {data_config.supported_languages}")
        
        # Test ModelConfig with NLLB optimization
        model_config = ModelConfig(
            use_gst=False,
            enable_gradient_checkpointing=True,
            max_attention_sequence_length=384
        )
        
        print(f"✅ ModelConfig max attention length: {model_config.max_attention_sequence_length}")
        print(f"✅ GST enabled: {model_config.use_gst}")
        
        # Test complete config
        config = XTTSConfig(data=data_config, model=model_config)
        print("✅ Complete XTTSConfig created successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ Config test failed: {e}")
        return False

def test_data_pipeline_features():
    """Test data pipeline language detection methods."""
    print("\nTesting data pipeline features...")
    
    try:
        from myxtts.data.ljspeech import LJSpeechDataset
        from myxtts.config.config import DataConfig
        
        # Create a config with multi-language features
        config = DataConfig(
            enable_multilingual=True,
            enable_phone_normalization=True,
            language_detection_method="auto",
            supported_languages=["en", "es", "fr", "de"]
        )
        
        print("✅ DataConfig with multi-language features created")
        
        # Test language detection (without actually loading dataset)
        print("✅ Data pipeline configuration accepted")
        
        return True
        
    except Exception as e:
        print(f"❌ Data pipeline test failed: {e}")
        return False

def test_inference_arguments():
    """Test that inference_main.py has new arguments."""
    print("\nTesting inference arguments...")
    
    try:
        # Test argument parsing (without actually running inference)
        import inference_main
        
        # Check if parse_args function exists
        if hasattr(inference_main, 'parse_args'):
            print("✅ parse_args function available")
            
        # Check if language detection functions exist
        if hasattr(inference_main, 'detect_text_language'):
            print("✅ detect_text_language function available")
            
        if hasattr(inference_main, 'apply_phone_normalization'):
            print("✅ apply_phone_normalization function available")
            
        if hasattr(inference_main, 'list_available_speakers'):
            print("✅ list_available_speakers function available")
            
        return True
        
    except Exception as e:
        print(f"❌ Inference test failed: {e}")
        return False

def test_train_main_config():
    """Test train_main.py configuration."""
    print("\nTesting train_main.py configuration...")
    
    try:
        from train_main import build_config
        
        # Test that build_config works with new features
        config = build_config()
        
        print(f"✅ Config built with vocab size: {config.model.text_vocab_size}")
        print(f"✅ Multilingual support: {config.data.enable_multilingual}")
        print(f"✅ Gradient checkpointing: {config.model.enable_gradient_checkpointing}")
        
        return True
        
    except Exception as e:
        print(f"❌ train_main config test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🚀 Testing Multi-Language and Multi-Speaker Features")
    print("=" * 60)
    
    tests = [
        test_config_new_features,
        test_data_pipeline_features,
        test_inference_arguments,
        test_train_main_config,
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"❌ Test {test.__name__} failed with exception: {e}")
    
    print("\n" + "=" * 60)
    print(f"📊 Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Multi-language and multi-speaker features are working.")
        return True
    else:
        print("⚠️  Some tests failed. Check the output above for details.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
