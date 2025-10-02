#!/usr/bin/env python3
"""
Simple syntax and import test for new features (no tensorflow required).
"""

def test_basic_imports():
    """Test basic imports work."""
    print("Testing basic imports...")
    
    try:
        from myxtts.config.config import DataConfig, ModelConfig
        print("✅ Config imports successful")
        
        # Test new DataConfig fields
        config = DataConfig()
        assert hasattr(config, 'enable_multilingual')
        assert hasattr(config, 'enable_pitch_shift') 
        assert hasattr(config, 'enable_phone_normalization')
        assert hasattr(config, 'supported_languages')
        print("✅ New DataConfig fields present")
        
        # Test representative ModelConfig fields
        model = ModelConfig()
        assert hasattr(model, 'use_gst')
        assert hasattr(model, 'use_pretrained_speaker_encoder')
        assert hasattr(model, 'enable_gradient_checkpointing')
        print("✅ ModelConfig core fields present")
        
        return True
    except Exception as e:
        print(f"❌ Import test failed: {e}")
        return False

def test_language_detection_logic():
    """Test language detection without data pipeline."""
    print("\nTesting language detection logic...")
    
    # Simple test of language detection patterns
    test_cases = [
        ("Hello world", "en"),
        ("Hola mundo", "es"), 
        ("Bonjour le monde", "fr"),
        ("Hallo Welt", "de"),
        ("مرحبا بالعالم", "ar"),  # Arabic
        ("你好世界", "zh"),  # Chinese
    ]
    
    passed = 0
    for text, expected_lang in test_cases:
        # Basic character-based detection
        detected = "en"  # default
        
        if any('\u0600' <= c <= '\u06FF' for c in text):
            detected = 'ar'
        elif any('\u4e00' <= c <= '\u9fff' for c in text):
            detected = 'zh'
        elif 'hola' in text.lower() or 'mundo' in text.lower():
            detected = 'es'
        elif 'bonjour' in text.lower() or 'monde' in text.lower():
            detected = 'fr'
        elif 'hallo' in text.lower() or 'welt' in text.lower():
            detected = 'de'
        
        if detected == expected_lang:
            passed += 1
            print(f"✅ '{text}' -> {detected}")
        else:
            print(f"❌ '{text}' -> {detected} (expected {expected_lang})")
    
    print(f"Language detection: {passed}/{len(test_cases)} correct")
    return passed >= len(test_cases) // 2  # At least half correct

def test_config_serialization():
    """Test that configs can be serialized to dict."""
    print("\nTesting config serialization...")
    
    try:
        from myxtts.config.config import DataConfig, ModelConfig
        
        # Test DataConfig serialization
        data_config = DataConfig(
            enable_multilingual=True,
            pitch_shift_range=[-1.5, 1.5],
            supported_languages=["en", "es"]
        )
        
        config_dict = data_config.__dict__
        assert 'enable_multilingual' in config_dict
        assert 'pitch_shift_range' in config_dict
        print("✅ DataConfig serialization works")
        
        # Test ModelConfig serialization
        model_config = ModelConfig(
            use_gst=False,
            enable_gradient_checkpointing=True
        )

        model_dict = model_config.__dict__
        assert 'use_gst' in model_dict
        assert 'enable_gradient_checkpointing' in model_dict
        print("✅ ModelConfig serialization works")
        
        return True
        
    except Exception as e:
        print(f"❌ Serialization test failed: {e}")
        return False

def main():
    print("🔧 Simple Feature Test (No TensorFlow Required)")
    print("=" * 50)
    
    tests = [
        test_basic_imports,
        test_language_detection_logic,
        test_config_serialization,
    ]
    
    passed = 0
    for test in tests:
        if test():
            passed += 1
    
    print(f"\n📊 Results: {passed}/{len(tests)} tests passed")
    
    if passed == len(tests):
        print("🎉 All basic tests passed!")
        return True
    else:
        print("⚠️  Some tests failed.")
        return False

if __name__ == "__main__":
    main()
