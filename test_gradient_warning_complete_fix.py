#!/usr/bin/env python3
"""
Complete test to verify the gradient warning fix.

This test ensures that all model variables participate in gradient computation
even when prosody targets are not provided, using the regularization approach.
"""

import sys
import os
sys.path.append('.')

import tensorflow as tf
import numpy as np
from myxtts.config.config import ModelConfig, DataConfig, TrainingConfig, XTTSConfig
from myxtts.models.xtts import XTTS
from myxtts.training.losses import XTTSLoss

def test_gradient_warning_fix_minimal_targets():
    """Test gradient fix with minimal targets (original problematic scenario)."""
    print("🧪 Testing gradient warning fix with minimal targets...")
    
    # Create model with all prosody features enabled
    model_config = ModelConfig(
        text_encoder_dim=128,
        text_encoder_layers=2,
        text_encoder_heads=4,
        text_vocab_size=100,
        audio_encoder_dim=128,
        audio_encoder_layers=2,
        audio_encoder_heads=4,
        decoder_dim=256,
        decoder_layers=2,
        decoder_heads=4,
        speaker_embedding_dim=64,
        use_voice_conditioning=True,
        use_duration_predictor=True,
        
        # Enable GST and prosody features
        use_gst=True,
        gst_num_style_tokens=10,
        gst_style_token_dim=256,
        gst_style_embedding_dim=256,
        gst_num_heads=4,
        gst_reference_encoder_dim=128,
        
        n_mels=80,
        sample_rate=22050
    )
    
    model = XTTS(model_config)
    
    # Create loss with regularization enabled
    criterion = XTTSLoss(
        duration_loss_weight=0.8,
        pitch_loss_weight=0.5,
        energy_loss_weight=0.5,
        prosody_pitch_loss_weight=0.3,
        prosody_energy_loss_weight=0.3,
        speaking_rate_loss_weight=0.3
    )
    
    # Create minimal training data (the problematic scenario)
    batch_size = 2
    text_len = 10
    mel_len = 50
    
    text_inputs = tf.random.uniform((batch_size, text_len), maxval=100, dtype=tf.int32)
    mel_inputs = tf.random.normal((batch_size, mel_len, 80))
    audio_conditioning = tf.random.normal((batch_size, 30, 80))
    text_lengths = tf.constant([text_len, text_len-2], dtype=tf.int32)
    mel_lengths = tf.constant([mel_len, mel_len-5], dtype=tf.int32)
    
    print("   Testing minimal targets scenario (should not cause gradient warnings)...")
    
    with tf.GradientTape() as tape:
        # Forward pass
        outputs = model(
            text_inputs=text_inputs,
            mel_inputs=mel_inputs,
            audio_conditioning=audio_conditioning,
            reference_mel=audio_conditioning,
            text_lengths=text_lengths,
            mel_lengths=mel_lengths,
            training=True
        )
        
        print(f"   Model outputs: {list(outputs.keys())}")
        
        # Create minimal y_true (no prosody targets - the problematic case)
        y_true = {
            "mel_target": mel_inputs,
            "stop_target": tf.random.uniform((batch_size, mel_len, 1), maxval=2, dtype=tf.int32),
            "text_lengths": text_lengths,
            "mel_lengths": mel_lengths,
            # Note: Missing all prosody targets - this used to cause gradient warnings
        }
        
        # Create y_pred with ALL model outputs (key fix)
        y_pred = {
            "mel_output": outputs["mel_output"],
            "stop_tokens": outputs["stop_tokens"],
        }
        
        # Include ALL prosody outputs in y_pred so they participate in gradient computation
        prosody_outputs = ["duration_pred", "pitch_output", "energy_output", 
                          "prosody_pitch", "prosody_energy", "prosody_speaking_rate"]
        
        included_outputs = []
        for key in prosody_outputs:
            if key in outputs:
                y_pred[key] = outputs[key]
                included_outputs.append(key)
        
        print(f"   ✓ Included prosody outputs: {included_outputs}")
        
        # Compute loss with regularization
        loss = criterion(y_true, y_pred)
        print(f"   Loss computed: {float(loss):.4f}")
    
    # Check gradients
    gradients = tape.gradient(loss, model.trainable_variables)
    
    # Count variables without gradients
    missing_gradients = []
    for var, grad in zip(model.trainable_variables, gradients):
        if grad is None:
            missing_gradients.append(var.name)
    
    print(f"   Variables without gradients: {len(missing_gradients)}")
    
    if missing_gradients:
        print("   ❌ Some variables still missing gradients:")
        for var_name in missing_gradients[:5]:  # Show first 5
            print(f"     ✗ {var_name}")
        if len(missing_gradients) > 5:
            print(f"     ... and {len(missing_gradients) - 5} more")
        return False
    else:
        print("   ✅ All variables have gradients!")
        return True

def test_gradient_warning_fix_comprehensive_targets():
    """Test with comprehensive targets to ensure normal training still works."""
    print("\n🧪 Testing with comprehensive targets (normal training scenario)...")
    
    model_config = ModelConfig(
        text_encoder_dim=128, text_encoder_layers=2, text_encoder_heads=4, text_vocab_size=100,
        audio_encoder_dim=128, audio_encoder_layers=2, audio_encoder_heads=4,
        decoder_dim=256, decoder_layers=2, decoder_heads=4, speaker_embedding_dim=64,
        use_voice_conditioning=True, use_duration_predictor=True,
        use_gst=True, gst_num_style_tokens=10, gst_style_token_dim=256,
        gst_style_embedding_dim=256, gst_num_heads=4, gst_reference_encoder_dim=128,
        n_mels=80, sample_rate=22050
    )
    
    model = XTTS(model_config)
    criterion = XTTSLoss(duration_loss_weight=0.8, pitch_loss_weight=0.5, energy_loss_weight=0.5,
                        prosody_pitch_loss_weight=0.3, prosody_energy_loss_weight=0.3,
                        speaking_rate_loss_weight=0.3)
    
    batch_size = 2
    text_len = 10
    mel_len = 50
    
    text_inputs = tf.random.uniform((batch_size, text_len), maxval=100, dtype=tf.int32)
    mel_inputs = tf.random.normal((batch_size, mel_len, 80))
    audio_conditioning = tf.random.normal((batch_size, 30, 80))
    text_lengths = tf.constant([text_len, text_len-2], dtype=tf.int32)
    mel_lengths = tf.constant([mel_len, mel_len-5], dtype=tf.int32)
    
    with tf.GradientTape() as tape:
        outputs = model(text_inputs, mel_inputs, audio_conditioning, reference_mel=audio_conditioning,
                       text_lengths=text_lengths, mel_lengths=mel_lengths, training=True)
        
        # Create comprehensive targets
        y_true = {
            "mel_target": mel_inputs,
            "stop_target": tf.random.uniform((batch_size, mel_len, 1), maxval=2, dtype=tf.int32),
            "text_lengths": text_lengths,
            "mel_lengths": mel_lengths,
            "duration_target": tf.random.uniform((batch_size, text_len), minval=1.0, maxval=10.0),
            "pitch_target": tf.random.uniform((batch_size, mel_len, 1), minval=80.0, maxval=400.0),
            "energy_target": tf.random.uniform((batch_size, mel_len, 1), minval=0.0, maxval=1.0),
            "prosody_pitch_target": tf.random.uniform((batch_size, text_len, 1), minval=80.0, maxval=400.0),
            "prosody_energy_target": tf.random.uniform((batch_size, text_len, 1), minval=0.0, maxval=1.0),
            "prosody_speaking_rate_target": tf.random.uniform((batch_size, text_len, 1), minval=0.5, maxval=2.0),
        }
        
        y_pred = {"mel_output": outputs["mel_output"], "stop_tokens": outputs["stop_tokens"]}
        for key in ["duration_pred", "pitch_output", "energy_output", 
                   "prosody_pitch", "prosody_energy", "prosody_speaking_rate"]:
            if key in outputs:
                y_pred[key] = outputs[key]
        
        loss = criterion(y_true, y_pred)
        print(f"   Loss computed: {float(loss):.4f}")
    
    gradients = tape.gradient(loss, model.trainable_variables)
    missing_gradients = sum(1 for grad in gradients if grad is None)
    
    print(f"   Variables without gradients: {missing_gradients}")
    
    if missing_gradients == 0:
        print("   ✅ All variables have gradients!")
        return True
    else:
        print("   ❌ Some variables missing gradients!")
        return False

if __name__ == "__main__":
    print("Testing Complete Gradient Warning Fix")
    print("=" * 60)
    
    success1 = test_gradient_warning_fix_minimal_targets()
    success2 = test_gradient_warning_fix_comprehensive_targets()
    
    print("\n" + "=" * 60)
    if success1 and success2:
        print("🎉 COMPLETE SUCCESS! All gradient warnings should be fixed.")
        print("   ✅ Minimal targets (problematic case): FIXED")
        print("   ✅ Comprehensive targets (normal case): WORKING")
    else:
        print("💥 FAILURE: Some gradient warnings may still occur.")
        print(f"   Minimal targets: {'✅' if success1 else '❌'}")
        print(f"   Comprehensive targets: {'✅' if success2 else '❌'}")
        sys.exit(1)