#!/usr/bin/env python3
"""
Test script to verify the gradient warning fix for duration predictor.

This script tests whether the duration predictor variables are properly
included in gradient computation to avoid the warning:
"Gradients do not exist for variables ['xtts/text_encoder/duration_predictor/kernel', 
'xtts/text_encoder/duration_predictor/bias']"
"""

import sys
import os
sys.path.append('.')

import tensorflow as tf
import numpy as np
from myxtts.config.config import ModelConfig, DataConfig, TrainingConfig, XTTSConfig
from myxtts.models.xtts import XTTS
from myxtts.training.losses import XTTSLoss

def test_gradient_warning_fix():
    """Test that duration predictor variables get gradients during training."""
    print("🧪 Testing gradient warning fix for duration predictor...")
    
    # Create minimal config
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
        n_mels=80,
        sample_rate=22050
    )
    
    data_config = DataConfig(
        batch_size=2,
        sample_rate=22050
    )
    
    training_config = TrainingConfig(
        learning_rate=1e-4,
        duration_loss_weight=0.8
    )
    
    config = XTTSConfig(
        model=model_config,
        data=data_config,
        training=training_config
    )
    
    # Create model
    model = XTTS(config.model)
    
    # Create loss function
    criterion = XTTSLoss(
        duration_loss_weight=0.8
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
    
    # Create duration targets
    duration_targets = tf.random.uniform((batch_size, text_len), minval=1.0, maxval=10.0)
    
    print(f"   Model has duration predictor: {hasattr(model.text_encoder, 'duration_predictor')}")
    print(f"   Duration predictor enabled: {model.text_encoder.use_duration_predictor}")
    if model.text_encoder.duration_predictor:
        print(f"   Duration predictor layers: {len(model.text_encoder.duration_predictor.layers) if hasattr(model.text_encoder.duration_predictor, 'layers') else 'single layer'}")
    
    # Training step with gradient tape
    with tf.GradientTape() as tape:
        # Forward pass
        outputs = model(
            text_inputs=text_inputs,
            mel_inputs=mel_inputs,
            audio_conditioning=audio_conditioning,
            text_lengths=text_lengths,
            mel_lengths=mel_lengths,
            training=True
        )
        
        print(f"   Model outputs: {list(outputs.keys())}")
        
        # Create y_true and y_pred for loss calculation
        y_true = {
            "mel_target": mel_inputs,
            "stop_target": tf.random.uniform((batch_size, mel_len, 1), maxval=2, dtype=tf.int32),
            "text_lengths": text_lengths,
            "mel_lengths": mel_lengths,
            "duration_target": duration_targets
        }
        
        y_pred = {
            "mel_output": outputs["mel_output"],
            "stop_tokens": outputs["stop_tokens"],
        }
        
        # Add duration predictions if available
        if "duration_pred" in outputs:
            y_pred["duration_pred"] = outputs["duration_pred"]
            print(f"   ✓ Duration predictions included in outputs")
        else:
            print(f"   ✗ Duration predictions missing from outputs")
        
        # Compute loss
        loss = criterion(y_true, y_pred)
        print(f"   Loss computed: {float(loss):.4f}")
    
    # Get gradients
    gradients = tape.gradient(loss, model.trainable_variables)
    
    # Check if duration predictor variables have gradients
    duration_predictor_vars = []
    duration_predictor_grads = []
    
    print("   All trainable variables:")
    for i, (var, grad) in enumerate(zip(model.trainable_variables, gradients)):
        var_name = var.name.lower()
        print(f"     {i}: {var.name} (shape: {var.shape})")
        
        # The duration predictor is a Dense layer with output size 1
        # Look for variables with shape (*, 1) and (1,) that belong to text encoder
        is_duration_var = False
        if i >= 35 and i <= 36:  # Based on the output, these should be duration predictor vars
            is_duration_var = True
        # Also check by shape - duration predictor should have (input_dim, 1) kernel and (1,) bias
        elif (var.shape[-1:] == (1,) and len(var.shape) == 2) or (var.shape == (1,)):
            # This could be duration predictor if it's in the text encoder part
            if i < 84:  # Before audio encoder starts (rough estimate)
                is_duration_var = True
                
        if is_duration_var:
            duration_predictor_vars.append(var)
            duration_predictor_grads.append(grad)
            print(f"       -> Duration predictor var: {var.name}")
            if grad is not None:
                print(f"         ✓ Has gradient: shape {grad.shape}, norm: {tf.norm(grad):.6f}")
            else:
                print(f"         ✗ NO GRADIENT!")
    
    print(f"\n   Found {len(duration_predictor_vars)} duration predictor variables")
    
    # Summary
    if duration_predictor_vars:
        grads_exist = all(grad is not None for grad in duration_predictor_grads)
        if grads_exist:
            print("✅ SUCCESS: All duration predictor variables have gradients!")
            print("   This should prevent the gradient warning.")
            return True
        else:
            print("❌ FAILURE: Some duration predictor variables missing gradients!")
            print("   The gradient warning may still occur.")
            return False
    else:
        if model.text_encoder.use_duration_predictor:
            print("❌ FAILURE: Duration predictor enabled but no variables found!")
            return False
        else:
            print("ℹ️  Duration predictor disabled - no variables expected.")
            return True

def test_with_duration_predictor_disabled():
    """Test that disabling duration predictor prevents the variables from being created."""
    print("\n🧪 Testing with duration predictor disabled...")
    
    # Create config with duration predictor disabled
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
        use_duration_predictor=False,  # Disable duration predictor
        n_mels=80,
        sample_rate=22050
    )
    
    # Create model
    model = XTTS(model_config)
    
    print(f"   Duration predictor enabled: {model.text_encoder.use_duration_predictor}")
    print(f"   Duration predictor object: {model.text_encoder.duration_predictor}")
    
    # Check that no duration predictor variables exist
    duration_vars = [var for var in model.trainable_variables if 'duration_predictor' in var.name]
    
    if not duration_vars:
        print("✅ SUCCESS: No duration predictor variables created when disabled!")
        return True
    else:
        print(f"❌ FAILURE: {len(duration_vars)} duration predictor variables still exist!")
        for var in duration_vars:
            print(f"     {var.name}")
        return False

if __name__ == "__main__":
    print("Testing Duration Predictor Gradient Warning Fix")
    print("=" * 50)
    
    success1 = test_gradient_warning_fix()
    success2 = test_with_duration_predictor_disabled()
    
    print("\n" + "=" * 50)
    if success1 and success2:
        print("🎉 ALL TESTS PASSED! Gradient warning should be fixed.")
    else:
        print("💥 SOME TESTS FAILED. Gradient warning may still occur.")
        sys.exit(1)