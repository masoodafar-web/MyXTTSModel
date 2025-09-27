#!/usr/bin/env python3
"""
Detailed variable analysis to identify specific gradient warning patterns.
"""

import sys
import os
sys.path.append('.')

import tensorflow as tf
import numpy as np
from myxtts.config.config import ModelConfig, DataConfig, TrainingConfig, XTTSConfig
from myxtts.models.xtts import XTTS
from myxtts.training.losses import XTTSLoss

def analyze_model_variables():
    """Analyze all model variables to identify gradient warning patterns."""
    print("🔍 Analyzing model variables for gradient warning patterns...")
    
    # Create comprehensive config
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
    
    # Create model
    model = XTTS(model_config)
    
    # Create comprehensive loss function
    criterion = XTTSLoss(
        duration_loss_weight=0.8,
        pitch_loss_weight=0.5,
        energy_loss_weight=0.5,
        prosody_pitch_loss_weight=0.3,
        prosody_energy_loss_weight=0.3,
        speaking_rate_loss_weight=0.3
    )
    
    # Create dummy data
    batch_size = 2
    text_len = 10
    mel_len = 50
    
    text_inputs = tf.random.uniform((batch_size, text_len), maxval=100, dtype=tf.int32)
    mel_inputs = tf.random.normal((batch_size, mel_len, 80))
    audio_conditioning = tf.random.normal((batch_size, 30, 80))
    text_lengths = tf.constant([text_len, text_len-2], dtype=tf.int32)
    mel_lengths = tf.constant([mel_len, mel_len-5], dtype=tf.int32)
    
    # Create ALL target values
    duration_targets = tf.random.uniform((batch_size, text_len), minval=1.0, maxval=10.0)
    pitch_targets = tf.random.uniform((batch_size, mel_len, 1), minval=80.0, maxval=400.0)
    energy_targets = tf.random.uniform((batch_size, mel_len, 1), minval=0.0, maxval=1.0)
    prosody_pitch_targets = tf.random.uniform((batch_size, text_len, 1), minval=80.0, maxval=400.0)
    prosody_energy_targets = tf.random.uniform((batch_size, text_len, 1), minval=0.0, maxval=1.0)
    prosody_speaking_rate_targets = tf.random.uniform((batch_size, text_len, 1), minval=0.5, maxval=2.0)
    
    # Training step
    with tf.GradientTape() as tape:
        outputs = model(
            text_inputs=text_inputs,
            mel_inputs=mel_inputs,
            audio_conditioning=audio_conditioning,
            reference_mel=audio_conditioning,
            text_lengths=text_lengths,
            mel_lengths=mel_lengths,
            training=True
        )
        
        y_true = {
            "mel_target": mel_inputs,
            "stop_target": tf.random.uniform((batch_size, mel_len, 1), maxval=2, dtype=tf.int32),
            "text_lengths": text_lengths,
            "mel_lengths": mel_lengths,
            "duration_target": duration_targets,
            "pitch_target": pitch_targets,
            "energy_target": energy_targets,
            "prosody_pitch_target": prosody_pitch_targets,
            "prosody_energy_target": prosody_energy_targets,
            "prosody_speaking_rate_target": prosody_speaking_rate_targets,
        }
        
        y_pred = {
            "mel_output": outputs["mel_output"],
            "stop_tokens": outputs["stop_tokens"],
        }
        
        # Add all available outputs
        for key in ["duration_pred", "pitch_output", "energy_output", 
                   "prosody_pitch", "prosody_energy", "prosody_speaking_rate"]:
            if key in outputs:
                y_pred[key] = outputs[key]
        
        loss = criterion(y_true, y_pred)
    
    # Get gradients
    gradients = tape.gradient(loss, model.trainable_variables)
    
    print("\n📋 DETAILED VARIABLE ANALYSIS")
    print("=" * 60)
    
    # Analyze variables by component hierarchy
    components = {
        "text_encoder": [],
        "audio_encoder": [], 
        "gst": [],
        "prosody_predictor": [],
        "mel_decoder": [],
        "other": []
    }
    
    # Variables mentioned in the problem statement
    problem_vars = [
        'xtts/text_encoder/duration_predictor/kernel',
        'xtts/text_encoder/duration_predictor/bias',
        'xtts/prosody_predictor/style_projection/kernel',
        'xtts/prosody_predictor/style_projection/bias',
        'xtts/prosody_predictor/prosody_layer_0/kernel',
        'xtts/prosody_predictor/prosody_layer_0/bias',
        'xtts/prosody_predictor/prosody_layer_1/kernel',
        'xtts/prosody_predictor/prosody_layer_1/bias',
        'xtts/prosody_predictor/pitch_predictor/kernel',
        'xtts/prosody_predictor/pitch_predictor/bias',
        'xtts/prosody_predictor/energy_predictor/kernel',
        'xtts/prosody_predictor/energy_predictor/bias',
        'xtts/prosody_predictor/speaking_rate_predictor/kernel',
        'xtts/prosody_predictor/speaking_rate_predictor/bias',
        'xtts/mel_decoder/pitch_projection/kernel',
        'xtts/mel_decoder/pitch_projection/bias',
        'xtts/mel_decoder/energy_projection/kernel',
        'xtts/mel_decoder/energy_projection/bias'
    ]
    
    print(f"Variables from problem statement:")
    for var_name in problem_vars:
        print(f"  - {var_name}")
    
    print(f"\nActual model variables (first 50):")
    problematic_vars = []
    
    for i, (var, grad) in enumerate(zip(model.trainable_variables, gradients)):
        if i < 50:  # Show first 50
            status = "✓" if grad is not None else "✗"
            print(f"  {i:3d}: {var.name:<50} (shape: {str(var.shape):<15}) {status}")
        
        if grad is None:
            problematic_vars.append(var.name)
        
        # Categorize variables
        var_name_lower = var.name.lower()
        if "text_encoder" in var_name_lower:
            components["text_encoder"].append((var.name, grad is not None))
        elif "audio_encoder" in var_name_lower:
            components["audio_encoder"].append((var.name, grad is not None))
        elif "gst" in var_name_lower or "global_style" in var_name_lower:
            components["gst"].append((var.name, grad is not None))
        elif "prosody_predictor" in var_name_lower:
            components["prosody_predictor"].append((var.name, grad is not None))
        elif "mel_decoder" in var_name_lower:
            components["mel_decoder"].append((var.name, grad is not None))
        else:
            components["other"].append((var.name, grad is not None))
    
    print(f"\n📊 COMPONENT ANALYSIS")
    print("=" * 60)
    for component, vars_list in components.items():
        if vars_list:
            missing_grads = sum(1 for _, has_grad in vars_list if not has_grad)
            print(f"{component}: {len(vars_list)} variables, {missing_grads} missing gradients")
            
            if missing_grads > 0:
                print("  Missing gradients:")
                for var_name, has_grad in vars_list:
                    if not has_grad:
                        print(f"    ✗ {var_name}")
    
    print(f"\n🎯 GRADIENT PROBLEM SUMMARY")
    print("=" * 60)
    if problematic_vars:
        print(f"❌ Found {len(problematic_vars)} variables without gradients:")
        for var_name in problematic_vars:
            print(f"  ✗ {var_name}")
        return False
    else:
        print("✅ All variables have gradients!")
        return True

if __name__ == "__main__":
    success = analyze_model_variables()
    if not success:
        sys.exit(1)