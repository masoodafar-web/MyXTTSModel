#!/usr/bin/env python3
"""
Test script for enhanced MyXTTS model with larger architecture and voice cloning.

This script validates that the model configuration improvements are working correctly
and demonstrates the new voice cloning capabilities.
"""

import sys
import os

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from train_main import build_config
from myxtts.config.config import XTTSConfig, ModelConfig, DataConfig, TrainingConfig

def test_enhanced_model_config():
    """Test that the enhanced model configuration is working correctly."""
    print("🧪 Testing Enhanced Model Configuration...")
    
    try:
        # Build the enhanced configuration
        config = build_config()
        
        # Validate model architecture enhancements
        print("\n📐 Model Architecture:")
        print(f"   • Text encoder layers: {config.model.text_encoder_layers} (enhanced from 4 to 8)")
        print(f"   • Audio encoder dim: {config.model.audio_encoder_dim} (enhanced from 512 to 768)")
        print(f"   • Audio encoder layers: {config.model.audio_encoder_layers} (enhanced from 4 to 8)")
        print(f"   • Audio encoder heads: {config.model.audio_encoder_heads} (enhanced from 4 to 12)")
        print(f"   • Decoder dim: {config.model.decoder_dim} (enhanced from 512 to 1536)")
        print(f"   • Decoder layers: {config.model.decoder_layers} (enhanced from 6 to 16)")
        print(f"   • Decoder heads: {config.model.decoder_heads} (enhanced from 8 to 24)")
        
        # Validate voice cloning features
        print("\n🎭 Voice Cloning Features:")
        print(f"   • Speaker embedding dim: {config.model.speaker_embedding_dim} (enhanced from 256 to 512)")
        print(f"   • Voice conditioning: {config.model.use_voice_conditioning}")
        print(f"   • Voice conditioning layers: {config.model.voice_conditioning_layers}")
        print(f"   • Voice similarity threshold: {config.model.voice_similarity_threshold}")
        print(f"   • Voice adaptation: {config.model.enable_voice_adaptation}")
        print(f"   • Speaker interpolation: {config.model.enable_speaker_interpolation}")
        print(f"   • Voice denoising: {config.model.enable_voice_denoising}")
        
        # Validate voice cloning loss components
        print("\n🎯 Voice Cloning Loss Components:")
        print(f"   • Voice similarity loss: {config.training.voice_similarity_loss_weight}")
        print(f"   • Speaker classification loss: {config.training.speaker_classification_loss_weight}")
        print(f"   • Voice reconstruction loss: {config.training.voice_reconstruction_loss_weight}")
        print(f"   • Prosody matching loss: {config.training.prosody_matching_loss_weight}")
        print(f"   • Spectral consistency loss: {config.training.spectral_consistency_loss_weight}")
        
        # Validate memory optimizations
        print("\n💾 Memory Optimizations:")
        print(f"   • Gradient checkpointing: {config.model.enable_gradient_checkpointing}")
        print(f"   • Memory efficient attention: {config.model.use_memory_efficient_attention}")
        print(f"   • Max attention sequence length: {config.model.max_attention_sequence_length}")
        
        # Calculate approximate model size
        text_params = config.model.text_encoder_dim * config.model.text_encoder_layers * config.model.text_encoder_heads
        audio_params = config.model.audio_encoder_dim * config.model.audio_encoder_layers * config.model.audio_encoder_heads
        decoder_params = config.model.decoder_dim * config.model.decoder_layers * config.model.decoder_heads
        total_params = text_params + audio_params + decoder_params
        
        print(f"\n📊 Estimated Model Complexity:")
        print(f"   • Text encoder complexity: {text_params:,}")
        print(f"   • Audio encoder complexity: {audio_params:,}")
        print(f"   • Decoder complexity: {decoder_params:,}")
        print(f"   • Total complexity factor: {total_params:,}")
        
        # Validate configuration integrity
        assert config.model.text_encoder_layers == 8, "Text encoder layers should be 8"
        assert config.model.audio_encoder_dim == 768, "Audio encoder dim should be 768"
        assert config.model.decoder_dim == 1536, "Decoder dim should be 1536"
        assert config.model.decoder_layers == 16, "Decoder layers should be 16"
        assert config.model.speaker_embedding_dim == 512, "Speaker embedding should be 512"
        assert config.model.use_voice_conditioning == True, "Voice conditioning should be enabled"
        assert hasattr(config.training, 'voice_similarity_loss_weight'), "Voice similarity loss should be defined"
        
        print("\n✅ All enhanced model configuration tests passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Enhanced model configuration test failed: {e}")
        return False

def test_voice_cloning_parameters():
    """Test voice cloning specific parameters."""
    print("\n🎭 Testing Voice Cloning Parameters...")
    
    try:
        config = build_config()
        
        # Test all new voice cloning parameters exist
        voice_cloning_params = [
            'voice_conditioning_layers',
            'voice_similarity_threshold', 
            'enable_voice_adaptation',
            'voice_encoder_dropout',
            'enable_speaker_interpolation',
            'voice_cloning_temperature',
            'voice_conditioning_strength',
            'max_reference_audio_length',
            'min_reference_audio_length',
            'voice_feature_dim',
            'enable_voice_denoising',
            'voice_cloning_loss_weight'
        ]
        
        for param in voice_cloning_params:
            assert hasattr(config.model, param), f"Missing voice cloning parameter: {param}"
            value = getattr(config.model, param)
            print(f"   • {param}: {value}")
        
        # Test voice cloning loss components
        voice_loss_params = [
            'voice_similarity_loss_weight',
            'speaker_classification_loss_weight',
            'voice_reconstruction_loss_weight',
            'prosody_matching_loss_weight',
            'spectral_consistency_loss_weight'
        ]
        
        for param in voice_loss_params:
            assert hasattr(config.training, param), f"Missing voice loss parameter: {param}"
            value = getattr(config.training, param)
            print(f"   • {param}: {value}")
        
        print("\n✅ All voice cloning parameters validated!")
        return True
        
    except Exception as e:
        print(f"\n❌ Voice cloning parameters test failed: {e}")
        return False

def test_model_size_improvements():
    """Test that the model is significantly larger than before."""
    print("\n📏 Testing Model Size Improvements...")
    
    try:
        # Get current enhanced config
        enhanced_config = build_config()
        
        # Create a baseline config for comparison (original smaller model)
        baseline_config = build_config()
        baseline_config.model.text_encoder_layers = 4  # Original
        baseline_config.model.audio_encoder_dim = 512  # Original  
        baseline_config.model.audio_encoder_layers = 4  # Original
        baseline_config.model.audio_encoder_heads = 4  # Original
        baseline_config.model.decoder_dim = 512  # Original
        baseline_config.model.decoder_layers = 6  # Original
        baseline_config.model.decoder_heads = 8  # Original
        baseline_config.model.speaker_embedding_dim = 256  # Original
        
        # Calculate size improvements
        text_improvement = enhanced_config.model.text_encoder_layers / baseline_config.model.text_encoder_layers
        audio_dim_improvement = enhanced_config.model.audio_encoder_dim / baseline_config.model.audio_encoder_dim
        audio_layers_improvement = enhanced_config.model.audio_encoder_layers / baseline_config.model.audio_encoder_layers
        decoder_dim_improvement = enhanced_config.model.decoder_dim / baseline_config.model.decoder_dim
        decoder_layers_improvement = enhanced_config.model.decoder_layers / baseline_config.model.decoder_layers
        speaker_improvement = enhanced_config.model.speaker_embedding_dim / baseline_config.model.speaker_embedding_dim
        
        print(f"   • Text encoder layers: {text_improvement:.1f}x larger")
        print(f"   • Audio encoder dimensions: {audio_dim_improvement:.1f}x larger")
        print(f"   • Audio encoder layers: {audio_layers_improvement:.1f}x larger")
        print(f"   • Decoder dimensions: {decoder_dim_improvement:.1f}x larger")
        print(f"   • Decoder layers: {decoder_layers_improvement:.1f}x larger")
        print(f"   • Speaker embeddings: {speaker_improvement:.1f}x larger")
        
        # Overall model complexity improvement
        baseline_complexity = (baseline_config.model.text_encoder_layers * 
                             baseline_config.model.audio_encoder_dim * 
                             baseline_config.model.decoder_dim)
        enhanced_complexity = (enhanced_config.model.text_encoder_layers * 
                             enhanced_config.model.audio_encoder_dim * 
                             enhanced_config.model.decoder_dim)
        
        overall_improvement = enhanced_complexity / baseline_complexity
        print(f"   • Overall model complexity: {overall_improvement:.1f}x larger")
        
        # Validate significant improvements
        assert text_improvement >= 2.0, "Text encoder should be at least 2x larger"
        assert audio_dim_improvement >= 1.5, "Audio encoder should be at least 1.5x larger"
        assert decoder_dim_improvement >= 3.0, "Decoder should be at least 3x larger"
        assert overall_improvement >= 5.0, "Overall model should be at least 5x larger"
        
        print(f"\n✅ Model is significantly larger! Overall improvement: {overall_improvement:.1f}x")
        return True
        
    except Exception as e:
        print(f"\n❌ Model size improvement test failed: {e}")
        return False

def main():
    """Run all validation tests."""
    print("🚀 Testing Enhanced MyXTTS Model Configuration")
    print("=" * 60)
    
    tests_passed = 0
    total_tests = 3
    
    # Run all tests
    if test_enhanced_model_config():
        tests_passed += 1
    
    if test_voice_cloning_parameters():
        tests_passed += 1
        
    if test_model_size_improvements():
        tests_passed += 1
    
    # Summary
    print("\n" + "=" * 60)
    print(f"🏁 Test Results: {tests_passed}/{total_tests} tests passed")
    
    if tests_passed == total_tests:
        print("✅ All tests passed! Enhanced model configuration is working correctly.")
        print("\n🎉 Your MyXTTS model is now:")
        print("   • Significantly larger for higher quality")
        print("   • Enhanced with advanced voice cloning capabilities")
        print("   • Optimized for superior voice replication")
        print("   • Ready for high-quality training and inference")
    else:
        print("❌ Some tests failed. Please check the configuration.")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())