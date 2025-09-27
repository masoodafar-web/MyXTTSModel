#!/usr/bin/env python3
"""
Comprehensive test script to verify all gradient warnings are fixed.

This script tests whether ALL model variables are properly included in gradient 
computation to avoid warnings for:
- Duration predictor variables  
- Prosody predictor variables (style_projection, prosody_layers, pitch/energy/speaking_rate predictors)
- Mel decoder prosody variables (pitch_projection, energy_projection)
"""

import sys
import os
sys.path.append('.')

import tensorflow as tf
import numpy as np
from myxtts.config.config import ModelConfig, DataConfig, TrainingConfig, XTTSConfig
from myxtts.models.xtts import XTTS
from myxtts.training.losses import XTTSLoss

def test_comprehensive_gradient_fix():
    """Test that ALL model variables get gradients during training."""
    print("🧪 Testing comprehensive gradient fix for all model components...")
    
    # Create comprehensive config with all features enabled
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
        use_duration_predictor=True,  # Enable duration predictor
        
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
    
    data_config = DataConfig(
        batch_size=2,
        sample_rate=22050
    )
    
    training_config = TrainingConfig(
        learning_rate=1e-4,
        duration_loss_weight=0.8,
        pitch_loss_weight=0.5,
        energy_loss_weight=0.5,
        prosody_pitch_loss_weight=0.3,
        prosody_energy_loss_weight=0.3,
        speaking_rate_loss_weight=0.3
    )
    
    config = XTTSConfig(
        model=model_config,
        data=data_config,
        training=training_config
    )
    
    # Create model
    model = XTTS(config.model)
    
    # Create comprehensive loss function with all prosody loss weights
    criterion = XTTSLoss(
        duration_loss_weight=0.8,
        pitch_loss_weight=0.5,
        energy_loss_weight=0.5,
        prosody_pitch_loss_weight=0.3,
        prosody_energy_loss_weight=0.3,
        speaking_rate_loss_weight=0.3
    )
    
    # Create optimizer
    optimizer = tf.keras.optimizers.AdamW(learning_rate=1e-4)
    
    # Create dummy batch data
    batch_size = 2
    text_len = 10
    mel_len = 50
    
    text_inputs = tf.random.uniform((batch_size, text_len), maxval=100, dtype=tf.int32)
    mel_inputs = tf.random.normal((batch_size, mel_len, 80))
    audio_conditioning = tf.random.normal((batch_size, 30, 80))  # Reference audio
    text_lengths = tf.constant([text_len, text_len-2], dtype=tf.int32)
    mel_lengths = tf.constant([mel_len, mel_len-5], dtype=tf.int32)
    
    # Create ALL target values to ensure all model components participate in loss
    duration_targets = tf.random.uniform((batch_size, text_len), minval=1.0, maxval=10.0)
    pitch_targets = tf.random.uniform((batch_size, mel_len, 1), minval=80.0, maxval=400.0)  # Mel-level pitch
    energy_targets = tf.random.uniform((batch_size, mel_len, 1), minval=0.0, maxval=1.0)  # Mel-level energy
    
    # Prosody targets (text-level)
    prosody_pitch_targets = tf.random.uniform((batch_size, text_len, 1), minval=80.0, maxval=400.0)
    prosody_energy_targets = tf.random.uniform((batch_size, text_len, 1), minval=0.0, maxval=1.0)
    prosody_speaking_rate_targets = tf.random.uniform((batch_size, text_len, 1), minval=0.5, maxval=2.0)
    
    print(f"   Model has GST: {model.gst is not None}")
    print(f"   Model has prosody predictor: {model.prosody_predictor is not None}")
    print(f"   Model has duration predictor: {hasattr(model.text_encoder, 'duration_predictor')}")
    print(f"   Duration predictor enabled: {model.text_encoder.use_duration_predictor}")
    
    # Training step with gradient tape
    with tf.GradientTape() as tape:
        # Forward pass with reference mel for prosody
        outputs = model(
            text_inputs=text_inputs,
            mel_inputs=mel_inputs,
            audio_conditioning=audio_conditioning,
            reference_mel=audio_conditioning,  # Use audio_conditioning as reference for prosody
            text_lengths=text_lengths,
            mel_lengths=mel_lengths,
            training=True
        )
        
        print(f"   Model outputs: {list(outputs.keys())}")
        
        # Create comprehensive y_true with ALL target values
        y_true = {
            "mel_target": mel_inputs,
            "stop_target": tf.random.uniform((batch_size, mel_len, 1), maxval=2, dtype=tf.int32),
            "text_lengths": text_lengths,
            "mel_lengths": mel_lengths,
            
            # Duration targets
            "duration_target": duration_targets,
            
            # Mel-level prosody targets (from mel decoder)
            "pitch_target": pitch_targets,
            "energy_target": energy_targets,
            
            # Text-level prosody targets (from prosody predictor)
            "prosody_pitch_target": prosody_pitch_targets,
            "prosody_energy_target": prosody_energy_targets,
            "prosody_speaking_rate_target": prosody_speaking_rate_targets,
        }
        
        # Create y_pred with all available outputs
        y_pred = {
            "mel_output": outputs["mel_output"],
            "stop_tokens": outputs["stop_tokens"],
        }
        
        # Add all prosody-related outputs that are available
        prosody_outputs_found = []
        if "duration_pred" in outputs:
            y_pred["duration_pred"] = outputs["duration_pred"]
            prosody_outputs_found.append("duration_pred")
            
        if "pitch_output" in outputs:
            y_pred["pitch_output"] = outputs["pitch_output"]
            prosody_outputs_found.append("pitch_output")
            
        if "energy_output" in outputs:
            y_pred["energy_output"] = outputs["energy_output"]
            prosody_outputs_found.append("energy_output")
            
        if "prosody_pitch" in outputs:
            y_pred["prosody_pitch"] = outputs["prosody_pitch"]
            prosody_outputs_found.append("prosody_pitch")
            
        if "prosody_energy" in outputs:
            y_pred["prosody_energy"] = outputs["prosody_energy"]
            prosody_outputs_found.append("prosody_energy")
            
        if "prosody_speaking_rate" in outputs:
            y_pred["prosody_speaking_rate"] = outputs["prosody_speaking_rate"]
            prosody_outputs_found.append("prosody_speaking_rate")
        
        print(f"   ✓ Prosody outputs found: {prosody_outputs_found}")
        
        # Compute comprehensive loss
        loss = criterion(y_true, y_pred)
        print(f"   Loss computed: {float(loss):.4f}")
    
    # Get gradients
    gradients = tape.gradient(loss, model.trainable_variables)
    
    # Check for gradient warnings by examining specific variable patterns
    problematic_vars = []
    prosody_vars = []
    duration_vars = []
    
    print("   Checking for gradient warnings in specific components:")
    for i, (var, grad) in enumerate(zip(model.trainable_variables, gradients)):
        var_name = var.name.lower()
        
        # Check for duration predictor variables
        if "duration_predictor" in var_name:
            duration_vars.append((var.name, grad is not None))
            if grad is None:
                problematic_vars.append(var.name)
            print(f"     Duration: {var.name} -> {'✓' if grad is not None else '✗'}")
        
        # Check for prosody predictor variables
        elif ("prosody_predictor" in var_name or 
              "style_projection" in var_name or
              "prosody_layer" in var_name or
              "pitch_predictor" in var_name or
              "energy_predictor" in var_name or
              "speaking_rate_predictor" in var_name):
            prosody_vars.append((var.name, grad is not None))
            if grad is None:
                problematic_vars.append(var.name)
            print(f"     Prosody: {var.name} -> {'✓' if grad is not None else '✗'}")
        
        # Check for mel decoder prosody variables
        elif ("pitch_projection" in var_name or "energy_projection" in var_name):
            prosody_vars.append((var.name, grad is not None))
            if grad is None:
                problematic_vars.append(var.name)
            print(f"     Mel Prosody: {var.name} -> {'✓' if grad is not None else '✗'}")
    
    print(f"\n   Found {len(duration_vars)} duration predictor variables")
    print(f"   Found {len(prosody_vars)} prosody-related variables")
    
    # Summary
    if not problematic_vars:
        print("✅ SUCCESS: All prosody and duration variables have gradients!")
        print("   This should prevent ALL gradient warnings.")
        return True
    else:
        print(f"❌ FAILURE: {len(problematic_vars)} variables still missing gradients!")
        for var_name in problematic_vars:
            print(f"     ✗ {var_name}")
        print("   These variables may still cause gradient warnings.")
        return False

if __name__ == "__main__":
    print("Testing Comprehensive Gradient Warning Fix")
    print("=" * 50)
    
    success = test_comprehensive_gradient_fix()
    
    print("\n" + "=" * 50)
    if success:
        print("🎉 SUCCESS! All gradient warnings should be fixed.")
    else:
        print("💥 FAILURE: Some gradient warnings may still occur.")
        sys.exit(1)