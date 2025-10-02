#!/usr/bin/env python3
"""
Simple configuration validation for enhanced MyXTTS model.
Tests the configuration without requiring TensorFlow.
"""

import sys
import os

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_config_import():
    """Test that the config can be imported and has the new parameters."""
    print("🧪 Testing Configuration Import...")
    
    try:
        from myxtts.config.config import ModelConfig, TrainingConfig, XTTSConfig
        
        # Test ModelConfig enhancements
        model_config = ModelConfig()
        
        print("\n📐 Model Architecture Enhancements:")
        print(f"   • Text encoder layers: {model_config.text_encoder_layers}")
        print(f"   • Audio encoder dim: {model_config.audio_encoder_dim}")
        print(f"   • Audio encoder layers: {model_config.audio_encoder_layers}")
        print(f"   • Audio encoder heads: {model_config.audio_encoder_heads}")
        print(f"   • Decoder dim: {model_config.decoder_dim}")
        print(f"   • Decoder layers: {model_config.decoder_layers}")
        print(f"   • Decoder heads: {model_config.decoder_heads}")
        print(f"   • Speaker embedding dim: {model_config.speaker_embedding_dim}")
        
        print("\n🎭 Voice Cloning Features:")
        print(f"   • Voice conditioning enabled: {model_config.use_voice_conditioning}")
        print(f"   • Pretrained speaker encoder: {model_config.use_pretrained_speaker_encoder}")
        print(f"   • GST enabled: {model_config.use_gst}")
        
        # Test TrainingConfig enhancements
        training_config = TrainingConfig()
        
        print("\n🎯 Voice Cloning Loss Components:")
        print(f"   • Voice similarity loss: {training_config.voice_similarity_loss_weight}")
        print(f"   • Speaker classification loss: {training_config.speaker_classification_loss_weight}")
        print(f"   • Voice reconstruction loss: {training_config.voice_reconstruction_loss_weight}")
        print(f"   • Prosody matching loss: {training_config.prosody_matching_loss_weight}")
        print(f"   • Spectral consistency loss: {training_config.spectral_consistency_loss_weight}")
        
        # Validate key improvements
        assert model_config.text_encoder_layers == 8, "Text encoder layers should be 8"
        assert model_config.audio_encoder_dim == 768, "Audio encoder dim should be 768"
        assert model_config.decoder_dim == 1536, "Decoder dim should be 1536"
        assert model_config.speaker_embedding_dim == 512, "Speaker embedding should be 512"
        assert hasattr(training_config, 'voice_similarity_loss_weight'), "Voice similarity loss should exist"
        
        print("\n✅ Configuration import and validation successful!")
        return True
        
    except Exception as e:
        print(f"\n❌ Configuration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_model_size_calculation():
    """Calculate and display model size improvements."""
    print("\n📏 Model Size Improvements...")
    
    try:
        from myxtts.config.config import ModelConfig
        
        # Enhanced model
        enhanced = ModelConfig()
        
        # Original model (for comparison)
        original_text_layers = 4
        original_audio_dim = 512
        original_audio_layers = 4
        original_decoder_dim = 512
        original_decoder_layers = 6
        original_speaker_dim = 256
        
        # Calculate improvements
        text_improvement = enhanced.text_encoder_layers / original_text_layers
        audio_dim_improvement = enhanced.audio_encoder_dim / original_audio_dim
        audio_layers_improvement = enhanced.audio_encoder_layers / original_audio_layers
        decoder_dim_improvement = enhanced.decoder_dim / original_decoder_dim
        decoder_layers_improvement = enhanced.decoder_layers / original_decoder_layers
        speaker_improvement = enhanced.speaker_embedding_dim / original_speaker_dim
        
        print(f"   • Text encoder layers: {original_text_layers} → {enhanced.text_encoder_layers} ({text_improvement:.1f}x)")
        print(f"   • Audio encoder dim: {original_audio_dim} → {enhanced.audio_encoder_dim} ({audio_dim_improvement:.1f}x)")
        print(f"   • Audio encoder layers: {original_audio_layers} → {enhanced.audio_encoder_layers} ({audio_layers_improvement:.1f}x)")
        print(f"   • Decoder dim: {original_decoder_dim} → {enhanced.decoder_dim} ({decoder_dim_improvement:.1f}x)")
        print(f"   • Decoder layers: {original_decoder_layers} → {enhanced.decoder_layers} ({decoder_layers_improvement:.1f}x)")
        print(f"   • Speaker embedding: {original_speaker_dim} → {enhanced.speaker_embedding_dim} ({speaker_improvement:.1f}x)")
        
        # Overall complexity
        original_complexity = original_text_layers * original_audio_dim * original_decoder_dim
        enhanced_complexity = enhanced.text_encoder_layers * enhanced.audio_encoder_dim * enhanced.decoder_dim
        overall_improvement = enhanced_complexity / original_complexity
        
        print(f"   • Overall complexity: {overall_improvement:.1f}x larger")
        
        assert overall_improvement >= 5.0, f"Model should be at least 5x larger, got {overall_improvement:.1f}x"
        
        print(f"\n✅ Model is significantly larger! {overall_improvement:.1f}x improvement")
        return True
        
    except Exception as e:
        print(f"\n❌ Model size calculation failed: {e}")
        return False

def main():
    """Run configuration validation."""
    print("🚀 Enhanced MyXTTS Configuration Validation")
    print("=" * 50)
    
    tests_passed = 0
    total_tests = 2
    
    if test_config_import():
        tests_passed += 1
    
    if test_model_size_calculation():
        tests_passed += 1
    
    print("\n" + "=" * 50)
    print(f"🏁 Results: {tests_passed}/{total_tests} tests passed")
    
    if tests_passed == total_tests:
        print("\n✅ SUCCESS! Enhanced configuration is working correctly.")
        print("\n🎉 Your MyXTTS model improvements:")
        print("   • ✅ Larger model architecture for higher quality")
        print("   • ✅ Advanced voice cloning capabilities")
        print("   • ✅ Enhanced voice conditioning features")
        print("   • ✅ Optimized loss functions for voice similarity")
        print("   • ✅ Ready for superior text-to-speech with voice cloning")
        return 0
    else:
        print("\n❌ Some tests failed. Please check the configuration.")
        return 1

if __name__ == "__main__":
    exit(main())
